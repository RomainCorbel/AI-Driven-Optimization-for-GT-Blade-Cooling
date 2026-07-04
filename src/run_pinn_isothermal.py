"""
run_pinn_isothermal.py — single-DP PINN for the isothermal dp11 benchmark.

This is a specialized sibling of run_pinn.py, not a generalization of it. It exists because
preProcessedData/isothermal/dp11/ is a *different* domain from the multi-pass serpentine dataset
that run_pinn.py / config.py are built around:
  - single straight rib-pass geometry (x:[0,0.953] y:[0,0.177] z:[0,0.0224] m), mesh
    Baseline_ML4Science.stl, vs. the 3-pass serpentine geometry used elsewhere in this repo.
  - T is exactly constant everywhere (T_inlet == T_wall == 298.15 K, std ~1e-13 K — pure floating
    point noise). run_pinn.py's θ = (T - T_wall)/(T_inlet - T_wall) normalization divides by zero
    here, which is exactly why it raises RuntimeError below ISOTHERMAL_DELTA_T_THRESHOLD.

Consequently this script drops:
  - the 5 z-scored design parameters + 2 wall-sigmoid input features (this domain isn't part of
    the DP_CONFIGS parametric family, and has no internal partition walls to distinguish)
  - the ±420mm STL inlet/outlet buffer trim (this mesh has no extension buffer: STL bounds match
    the CFD data extent exactly)
  - the θ-based T normalization

...and instead z-scores T directly from data (mean/std, with a floor on std) exactly like vx/vy/vz
already are, mirroring how the very first (pre-multi-DP) version of this PINN handled temperature.
Velocity/pressure non-dimensionalization (V_IN, P_REF, P_SCALE) is otherwise unchanged from
run_pinn.py. Everything downstream (Sutherland viscosity, momentum/heat residuals) is fed the
model's real, un-standardized T output directly, so it degrades gracefully to a constant when T
truly carries no signal.
"""

import argparse
import glob
import trimesh
import numpy as np
import random
import torch
import torch.nn as nn
import os
import sys
import time
import wandb
from torch.optim import Adam

torch.set_default_dtype(torch.float64)

from config import M, R, K, CP, T_REF_SUTH, MU_REF, S_SUTH
from models import FFNN
from utils import _Tee, plot_fields

DATA_DIR  = "./preProcessedData/isothermal/dp11/"
STL_GLOB  = DATA_DIR + "Baseline_ML4Science.stl"
T_STD_FLOOR = 1.0   # [K] — floor for the T z-score std; this DP has real std ~1e-13 K

# ═══════════════════════════════════════════════════════════════
# GEOMETRY SAMPLING  (labels: 0=interior, 1=inlet, 2=outlet, 3=wall)
# Identical to run_pinn.py — fully generic w.r.t. mesh + x-bounds, no changes needed.
# ═══════════════════════════════════════════════════════════════

def build_wall_pool(mesh, x_stl_min_mm, x_stl_max_mm, n_pool=200_000):
    pts, _ = trimesh.sample.sample_surface(mesh, n_pool * 3)
    x    = pts[:, 0]
    mask = (
        (np.abs(x - x_stl_min_mm) > 0.1) &
        (np.abs(x - x_stl_max_mm) > 0.1) &
        (x >= x_stl_min_mm) &
        (x <= x_stl_max_mm)
    )
    wall_pts = torch.tensor(pts[mask], dtype=torch.float64, device='cpu')
    if len(wall_pts) > n_pool:
        wall_pts = wall_pts[torch.randperm(len(wall_pts), device='cpu')[:n_pool]]
    return wall_pts


def sample_wall(wall_pool, n):
    idx = torch.randperm(len(wall_pool), device=wall_pool.device)[:n]
    return wall_pool[idx], 3 * torch.ones(n, dtype=torch.int64, device=wall_pool.device)


def build_volume_pool(mesh, x_stl_min_mm, x_stl_max_mm,
                      n_pool=800_000, wall_fraction=0.5,
                      delta_horiz_mm=1.0, delta_vert_mm=1.0, vert_frac=0.7):
    n_near    = int(n_pool * wall_fraction)
    n_uniform = n_pool - n_near

    raw  = trimesh.sample.volume_mesh(mesh, n_uniform * 3)
    mask = (raw[:, 0] >= x_stl_min_mm) & (raw[:, 0] <= x_stl_max_mm)
    raw  = raw[mask]
    if len(raw) > n_uniform:
        raw = raw[np.random.choice(len(raw), n_uniform, replace=False)]

    def _near(face_ids, n, delta_mm):
        sub = mesh.submesh([face_ids], only_watertight=False)
        if not sub:
            return np.zeros((0, 3))
        s_pts, s_fids = trimesh.sample.sample_surface(sub[0], n * 2)
        eps  = np.random.uniform(0, delta_mm, size=(len(s_pts), 1))
        near = s_pts - eps * sub[0].face_normals[s_fids]
        keep = (near[:, 0] >= x_stl_min_mm) & (near[:, 0] <= x_stl_max_mm)
        near = near[keep]
        if len(near) > n:
            near = near[np.random.choice(len(near), n, replace=False)]
        return near

    horiz_fids  = np.where(np.abs(mesh.face_normals[:, 2]) >= 0.7)[0]
    vert_fids   = np.where(np.abs(mesh.face_normals[:, 2]) <  0.7)[0]
    near_horiz  = _near(horiz_fids, int(n_near * (1 - vert_frac)), delta_horiz_mm)
    near_vert   = _near(vert_fids,  int(n_near * vert_frac),       delta_vert_mm)
    all_pts     = np.vstack([raw, near_horiz, near_vert])
    print(f"    uniform={len(raw)}  near-horiz={len(near_horiz)}  near-vert={len(near_vert)}")
    return torch.tensor(all_pts, dtype=torch.float64, device='cpu'), len(raw), len(near_horiz)


def sample_volume(vol_pool, n):
    idx = torch.randperm(len(vol_pool), device=vol_pool.device)[:n]
    return vol_pool[idx], torch.zeros(n, dtype=torch.int64, device=vol_pool.device)


def sample_collocation(vol_pool, wall_pool, n_vol, n_wall):
    p, l_vol  = sample_volume(vol_pool, n_vol)
    w, l_wall = sample_wall(wall_pool, n_wall)
    pts  = torch.cat([p, w])
    lbls = torch.cat([l_vol, l_wall])
    perm = torch.randperm(pts.size(0), device=pts.device)
    return pts[perm], lbls[perm]


# ═══════════════════════════════════════════════════════════════
# MODEL — coordinate-only (no design params, no wall-sigmoid features)
# ═══════════════════════════════════════════════════════════════

class NormalizedPINN(nn.Module):
    """Z-scores (x,y,z) input; un-standardizes all 5 outputs (vx,vy,vz,p,T) to physical units.

    Unlike models.NormalizedPINN, this does not append wall-sigmoid features — this domain
    has no internal partition walls (single straight pass) and isn't part of the DP_CONFIGS
    parametric family, so there's nothing to condition on beyond raw coordinates.
    """
    def __init__(self, net, coord_mean, coord_std, out_mean, out_std):
        super().__init__()
        self.net = net
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std",  coord_std)
        self.register_buffer("out_mean",   out_mean)
        self.register_buffer("out_std",    out_std)

    def forward(self, x):
        x_norm   = (x - self.coord_mean) / self.coord_std
        y_norm   = self.net(x_norm)
        safe_std = torch.where(self.out_std == 0, torch.ones_like(self.out_std), self.out_std)
        return y_norm * safe_std + self.out_mean


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def dynamic_viscosity(T_K):
    """Sutherland's law, real Kelvin in -> Pa.s out."""
    return MU_REF * (T_K.abs() / T_REF_SUTH) ** 1.5 * (T_REF_SUTH + S_SUTH) / (T_K.abs() + S_SUTH)


def _grad(f, pts):
    return torch.autograd.grad(f, pts, torch.ones_like(f),
                                retain_graph=True, create_graph=True)[0]


def _grad2(f, pts, dim):
    g = _grad(f, pts)[:, dim:dim+1]
    return torch.autograd.grad(g, pts, torch.ones_like(g),
                                retain_graph=True, create_graph=True)[0][:, dim:dim+1]


def compute_derivatives(fields, pts_xyz):
    """All spatial derivatives. pts_xyz [N,3] must have requires_grad=True.
    fields are already un-standardized physical values (vx,vy,vz,p,T) except vx,vy,vz,p
    remain in their non-dim (V_IN/P_SCALE) form — only T is real Kelvin (see load_data)."""
    vx = fields[:, 0:1]; vy = fields[:, 1:2]; vz = fields[:, 2:3]
    p  = fields[:, 3:4]; T  = fields[:, 4:5]

    gvx = _grad(vx, pts_xyz); vx_x, vx_y, vx_z = gvx[:,0:1], gvx[:,1:2], gvx[:,2:3]
    vx_xx = _grad2(vx, pts_xyz, 0); vx_yy = _grad2(vx, pts_xyz, 1); vx_zz = _grad2(vx, pts_xyz, 2)

    gvy = _grad(vy, pts_xyz); vy_x, vy_y, vy_z = gvy[:,0:1], gvy[:,1:2], gvy[:,2:3]
    vy_xx = _grad2(vy, pts_xyz, 0); vy_yy = _grad2(vy, pts_xyz, 1); vy_zz = _grad2(vy, pts_xyz, 2)

    gvz = _grad(vz, pts_xyz); vz_x, vz_y, vz_z = gvz[:,0:1], gvz[:,1:2], gvz[:,2:3]
    vz_xx = _grad2(vz, pts_xyz, 0); vz_yy = _grad2(vz, pts_xyz, 1); vz_zz = _grad2(vz, pts_xyz, 2)

    gp  = _grad(p,  pts_xyz); p_x,  p_y,  p_z  = gp[:,0:1],  gp[:,1:2],  gp[:,2:3]
    gT  = _grad(T,  pts_xyz); T_x,  T_y,  T_z  = gT[:,0:1],  gT[:,1:2],  gT[:,2:3]
    T_xx = _grad2(T, pts_xyz, 0); T_yy = _grad2(T, pts_xyz, 1); T_zz = _grad2(T, pts_xyz, 2)

    return (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    )


def compute_losses(
    vx, vy, vz, p, T,
    vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
    vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
    vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
    p_x, p_y, p_z,
    T_x, T_y, T_z, T_xx, T_yy, T_zz,
    labels,
    vx_in, vy_in, vz_in,             # inlet BC velocities (non-dim)
    ctx,                               # scalings dict (see load_data)
    div_scale, mom_scale_x, mom_scale_y, mom_scale_z,
):
    interior = labels == 0
    inlet    = labels == 1
    outlet   = labels == 2
    wall     = labels == 3

    # T here is in the same once-normalized scale as vx/vy/vz/p (T_nd = (T_K - t_mean)/t_std),
    # NOT real Kelvin — reconstruct real T_K before evaluating Sutherland's law.
    T_K         = T * ctx["t_std"] + ctx["t_mean"]
    mu_ratio    = dynamic_viscosity(T_K) / MU_REF   # dimensionless ratio; MU_REF cancels below
    visc_coeff  = ctx["visc_coeff"]                # (MU_REF/rho)/V_IN  [m]
    therm_coeff = ctx["therm_coeff"]               # alpha_th/V_IN      [m]

    vx_var = ctx["vx_var"]; vy_var = ctx["vy_var"]; vz_var = ctx["vz_var"]
    p_var  = ctx["p_var"];  t_var  = ctx["t_var"]

    # ── PDE losses ────────────────────────────────────────────
    l_div = torch.mean(((vx_x + vy_y + vz_z) / div_scale) ** 2)

    def navier_stokes(u_x, u_y, u_z, u_xx, u_yy, u_zz, p_grad, mom_scale):
        advec = (vx[interior]*u_x[interior] + vy[interior]*u_y[interior] + vz[interior]*u_z[interior])
        lap   = u_xx[interior] + u_yy[interior] + u_zz[interior]
        res   = advec + p_grad[interior] - visc_coeff * mu_ratio[interior] * lap
        return torch.mean((res / mom_scale) ** 2)

    l_mom_x = navier_stokes(vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, p_x, mom_scale_x)
    l_mom_y = navier_stokes(vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, p_y, mom_scale_y)
    l_mom_z = navier_stokes(vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_z, mom_scale_z)

    l_heat = torch.mean((
        vx[interior]*T_x[interior] + vy[interior]*T_y[interior] + vz[interior]*T_z[interior]
        - therm_coeff * (T_xx[interior] + T_yy[interior] + T_zz[interior])
    ) ** 2) / t_var

    # ── BC losses ─────────────────────────────────────────────
    l_inlet_vx = torch.mean((vx[inlet].squeeze(1) - vx_in) ** 2) / vx_var
    l_inlet_vy = torch.mean((vy[inlet].squeeze(1) - vy_in) ** 2) / vy_var
    l_inlet_vz = torch.mean((vz[inlet].squeeze(1) - vz_in) ** 2) / vz_var
    l_inlet_T  = torch.mean((T[inlet]  - ctx["t_inlet_val_nd"]) ** 2) / t_var
    l_outlet_p = torch.mean((p[outlet] - 0.0)          ** 2) / p_var   # p̃_outlet = 0 by P_REF
    l_wall_vx  = torch.mean( vx[wall] ** 2)                  / vx_var
    l_wall_vy  = torch.mean( vy[wall] ** 2)                  / vy_var
    l_wall_vz  = torch.mean( vz[wall] ** 2)                  / vz_var
    l_wall_T   = torch.mean((T[wall]  - ctx["t_wall_val_nd"]) ** 2)    / t_var

    return (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
        l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
    )


# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="PINN — single-DP training for the isothermal dp11 benchmark")

parser.add_argument("--run-path",   default=None,               help="Output dir (auto-derived if omitted)")
parser.add_argument("--project",    default="PINN_isothermal",  help="W&B project name")
parser.add_argument("--device",     default="cuda")
parser.add_argument("--debug",      action="store_false", help="Skip W&B logging")

parser.add_argument("--hidden-dim", type=int,   default=20)
parser.add_argument("--n-layers",   type=int,   default=4)

parser.add_argument("--epochs",     type=int,   default=6_000)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--lr",         type=float, default=3e-3)
parser.add_argument("--lr-end",     type=float, default=1e-4)

parser.add_argument("--n-train",       type=int,   default=5_000)
parser.add_argument("--n-test",        type=int,   default=10_000)
parser.add_argument("--n-sup",         type=int,   default=500,
                    help="Supervised CFD points per epoch (0 = physics-only)")
parser.add_argument("--n-snapshots",   type=int,   default=10_000)
parser.add_argument("--wall-fraction", type=float, default=0.5)
parser.add_argument("--n-pool",        type=int,   default=200_000)
parser.add_argument("--pool-frac-vol", type=float, default=0.5)
parser.add_argument("--pool-frac-wall",type=float, default=0.5)
parser.add_argument("--delta-mm-horiz",type=float, default=1.0)
parser.add_argument("--delta-mm-vert", type=float, default=20.0)
parser.add_argument("--w-pde",   type=float, default=1.0)
parser.add_argument("--w-bc",    type=float, default=1.0)
parser.add_argument("--w-sup",   type=float, default=1.0)
parser.add_argument("--w-sup-p", type=float, default=1.0)

args = parser.parse_args()

DEVICE     = args.device
PROJECT    = args.project
DEBUG      = args.debug
EPOCHS     = args.epochs
HIDDEN_DIM = args.hidden_dim
N_LAYERS   = args.n_layers
SEED       = args.seed
LR         = args.lr
LR_END     = args.lr_end
GAMMA      = (LR_END / LR) ** (1.0 / EPOCHS)
N_TEST              = args.n_test
N_TOTAL_TRAIN       = args.n_train
N_SUP               = args.n_sup
N_POINT_SNAPSHOTS   = args.n_snapshots
WALL_FRACTION       = args.wall_fraction
N_TOTAL_POOL        = args.n_pool
POOL_FRAC_VOL       = args.pool_frac_vol
POOL_FRAC_WALL      = args.pool_frac_wall
DELTA_HORIZ_MM      = args.delta_mm_horiz
DELTA_VERT_MM       = args.delta_mm_vert
W_PDE   = args.w_pde
W_BC    = args.w_bc
W_SUP   = args.w_sup
W_SUP_P = args.w_sup_p

IN_DIM = 3   # (x, y, z) only — no design params, no wall-sigmoid features

FRAC_VOL    = 0.20
FRAC_WALL   = 0.70
FRAC_INLET  = 0.05
FRAC_OUTLET = 0.05
n_vol_train    = int(FRAC_VOL    * N_TOTAL_TRAIN)
n_wall_train   = int(FRAC_WALL   * N_TOTAL_TRAIN)
n_inlet_train  = int(FRAC_INLET  * N_TOTAL_TRAIN)
n_outlet_train = int(FRAC_OUTLET * N_TOTAL_TRAIN)

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_default_device(DEVICE)
print(f"Device : {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def _var(raw_var, floor, name):
    v = raw_var.clamp(min=floor)
    if raw_var.item() < floor:
        print(f"    [variance floor] {name}: raw={raw_var.item():.2e} → {floor:.2e}")
    return v


def load_data(args):
    print(f"\n{'─'*60}\n  Loading isothermal dp11  ({DATA_DIR})")

    _in_vx = np.load(DATA_DIR + "vel_x_inlet.npy")
    _in_vy = np.load(DATA_DIR + "vel_y_inlet.npy")
    _in_vz = np.load(DATA_DIR + "vel_z_inlet.npy")

    _vx_raw = np.load(DATA_DIR + "vel_x.npy")
    _temp   = np.load(DATA_DIR + "temp.npy")
    cfd_pts = torch.tensor(_vx_raw[:, :3], device='cpu')
    cfd_vx  = torch.tensor(_vx_raw[:, 3],                           device='cpu')
    cfd_vy  = torch.tensor(np.load(DATA_DIR + "vel_y.npy")[:, 3],   device='cpu')
    cfd_vz  = torch.tensor(np.load(DATA_DIR + "vel_z.npy")[:, 3],   device='cpu')
    cfd_p   = torch.tensor(np.load(DATA_DIR + "press.npy")[:, 3],   device='cpu')
    cfd_T   = torch.tensor(_temp[:, 3],                              device='cpu')

    # ── Domain bounds — no STL buffer: this mesh matches the CFD extent exactly ──
    x_min_mm = float(cfd_pts[:, 0].min()) * 1000.0
    x_max_mm = float(cfd_pts[:, 0].max()) * 1000.0
    stl_buffer_mm = 0.0
    x_stl_min_mm  = x_min_mm
    x_stl_max_mm  = x_max_mm
    print(f"    Physical/STL x=[{x_min_mm:.1f}, {x_max_mm:.1f}] mm  (no buffer)")

    # ── T BCs — real physical values, used directly (no ΔT-based non-dim) ──
    t_wall_val  = float(cfd_T.min())
    _x_min_temp = float(_temp[:, 0].min())
    t_inlet_val = float(_temp[_temp[:, 0] == _x_min_temp][:, 3].mean())
    delta_t     = t_inlet_val - t_wall_val
    print(f"    T_WALL={t_wall_val:.6f} K   T_inlet={t_inlet_val:.6f} K   ΔT={delta_t:.2e} K"
          f"  (isothermal — T is z-scored directly, not by ΔT)")

    # ── Outlet mask (CFD slice near x_max, matching run_pinn.py) ──
    out_mask = cfd_pts[:, 0] > (x_max_mm / 1000.0 - 0.005)

    # ── Physical scales ────────────────────────────────────────
    V_IN     = float(torch.tensor(_in_vx[:, 3]).abs().mean())
    P_REF    = float(cfd_p[out_mask].mean())
    t_ref    = float(cfd_T.mean())
    rho      = P_REF * M / (R * t_ref)
    alpha_th = K / (rho * CP)
    nu_ref   = MU_REF / rho              # arbitrary but consistent reference — cancels in mu_ratio
    visc_coeff  = nu_ref     / V_IN       # [m]
    therm_coeff = alpha_th   / V_IN       # [m]
    P_SCALE  = float(cfd_p.std().clamp(min=1.0))
    print(f"    V_IN={V_IN:.3f} m/s  P_REF={P_REF:.0f} Pa  P_SCALE={P_SCALE:.1f} Pa"
          f"  rho={rho:.4f} kg/m3  ν_ref/V_IN={visc_coeff:.2e} m")

    # ── Non-dim fields — vx/vy/vz/p exactly as run_pinn.py; T is a plain z-score ──
    cfd_vx_nd = cfd_vx / V_IN
    cfd_vy_nd = cfd_vy / V_IN
    cfd_vz_nd = cfd_vz / V_IN
    cfd_p_nd  = (cfd_p - P_REF) / P_SCALE
    t_mean    = float(cfd_T.mean())
    t_std     = max(float(cfd_T.std()), T_STD_FLOOR)
    cfd_T_nd  = (cfd_T - t_mean) / t_std
    print(f"    T z-score: mean={t_mean:.4f} K  std={float(cfd_T.std()):.2e} K"
          f"  → using floor={t_std:.4f} K")

    inlet_pool = (
        torch.tensor(_in_vx[:, :3],  device='cpu'),
        torch.tensor(_in_vx[:, 3],   device='cpu') / V_IN,
        torch.tensor(_in_vy[:, 3],   device='cpu') / V_IN,
        torch.tensor(_in_vz[:, 3],   device='cpu') / V_IN,
    )
    outlet_pool = cfd_pts[out_mask]

    # ── STL mesh and pools — built fresh in memory every run, no disk cache ──
    stl_files = glob.glob(STL_GLOB)
    if not stl_files:
        raise FileNotFoundError(f"No STL found: {STL_GLOB}")
    print(f"    STL: {os.path.basename(stl_files[0])}")
    mesh = trimesh.load(stl_files[0])
    print(f"    Building volume pool…")
    vol_pool, _n_uni, _n_horiz = build_volume_pool(
        mesh, x_stl_min_mm, x_stl_max_mm,
        n_pool=int(POOL_FRAC_VOL * N_TOTAL_POOL),
        wall_fraction=WALL_FRACTION,
        delta_horiz_mm=DELTA_HORIZ_MM, delta_vert_mm=DELTA_VERT_MM)
    print(f"    Building wall pool…")
    wall_pool = build_wall_pool(
        mesh, x_stl_min_mm, x_stl_max_mm,
        n_pool=int(POOL_FRAC_WALL * N_TOTAL_POOL))
    del mesh

    # ── Train / test split ─────────────────────────────────────
    n_total = cfd_pts.shape[0]
    n_test  = min(N_TEST, n_total - 1)
    perm    = torch.randperm(n_total, device='cpu')
    t_idx   = perm[:n_test]
    tr_idx  = perm[n_test:]

    test_pts  = cfd_pts[t_idx];    train_pts = cfd_pts[tr_idx]
    test_vx   = cfd_vx_nd[t_idx];  train_vx  = cfd_vx_nd[tr_idx]
    test_vy   = cfd_vy_nd[t_idx];  train_vy  = cfd_vy_nd[tr_idx]
    test_vz   = cfd_vz_nd[t_idx];  train_vz  = cfd_vz_nd[tr_idx]
    test_p    = cfd_p_nd[t_idx];   train_p   = cfd_p_nd[tr_idx]
    test_T    = cfd_T_nd[t_idx];   train_T   = cfd_T_nd[tr_idx]

    # ── Supervised points ─────────────────────────────────────
    n_sup = min(N_SUP, train_pts.shape[0])
    if n_sup > 0:
        _sup_idx = torch.randint(0, train_pts.shape[0], (n_sup,), device='cpu')
        sup_pts  = train_pts[_sup_idx]
        sup_vx   = train_vx[_sup_idx]; sup_vy = train_vy[_sup_idx]; sup_vz = train_vz[_sup_idx]
        sup_p    = train_p[_sup_idx];  sup_T  = train_T[_sup_idx]
    else:
        sup_pts = sup_vx = sup_vy = sup_vz = sup_p = sup_T = None

    # ── Loss normalisation variances ──────────────────────────
    vx_raw = torch.maximum(train_vx.var(), inlet_pool[1].var())
    vy_raw = torch.maximum(train_vy.var(), inlet_pool[2].var())
    vz_raw = torch.maximum(train_vz.var(), inlet_pool[3].var())
    vx_var = _var(vx_raw, 1e-6, "vx").to(DEVICE)
    vy_var = _var(vy_raw, 1e-6, "vy").to(DEVICE)
    vz_var = _var(vz_raw, 1e-6, "vz").to(DEVICE)
    p_var  = _var(train_p.var(), 1e-6, "p").to(DEVICE)
    t_var  = _var(train_T.var(), 1e-2, "T").to(DEVICE)

    # ── Snap points ───────────────────────────────────────────
    n_snap   = min(N_POINT_SNAPSHOTS, n_total)
    snap_idx = np.random.choice(n_total, n_snap, replace=False)
    snap_pts = cfd_pts[snap_idx]
    snap_vx  = cfd_vx_nd[snap_idx]; snap_vy = cfd_vy_nd[snap_idx]; snap_vz = cfd_vz_nd[snap_idx]
    snap_p   = cfd_p_nd[snap_idx];  snap_T  = cfd_T_nd[snap_idx]

    print(f"    vx_var={vx_var.item():.3e}  vy_var={vy_var.item():.3e}"
          f"  vz_var={vz_var.item():.3e}  p_var={p_var.item():.3e}  T_var={t_var.item():.3e}")
    print(f"    CFD pts={n_total:,}  train={len(tr_idx):,}  test={n_test:,}  sup={n_sup}")

    return {
        "vol_pool": vol_pool, "vol_pool_n_uni": _n_uni, "vol_pool_n_horiz": _n_horiz,
        "wall_pool": wall_pool, "inlet_pool": inlet_pool, "outlet_pool": outlet_pool,
        "stl_buffer_mm": stl_buffer_mm,
        "x_min_mm": x_min_mm, "x_max_mm": x_max_mm,
        "x_stl_min_mm": x_stl_min_mm, "x_stl_max_mm": x_stl_max_mm,
        "ctx": {
            "t_mean": t_mean, "t_std": t_std,
            "t_inlet_val_nd": (t_inlet_val - t_mean) / t_std,
            "t_wall_val_nd":  (t_wall_val  - t_mean) / t_std,
            "visc_coeff": visc_coeff, "therm_coeff": therm_coeff,
            "vx_var": vx_var, "vy_var": vy_var, "vz_var": vz_var,
            "p_var": p_var, "t_var": t_var,
        },
        "V_IN": V_IN, "P_SCALE": P_SCALE, "P_REF": P_REF,
        "t_mean": t_mean, "t_std": t_std,
        "cfd_pts_nd": cfd_pts,
        "cfd_fields_nd": torch.stack([cfd_vx_nd, cfd_vy_nd, cfd_vz_nd, cfd_p_nd, cfd_T_nd], dim=1),
        "test_idx": t_idx,
        "test_pts": test_pts, "test_vx": test_vx, "test_vy": test_vy,
        "test_vz": test_vz,  "test_p": test_p,   "test_T": test_T,
        "n_sup": n_sup,
        "sup_pts": sup_pts, "sup_vx": sup_vx, "sup_vy": sup_vy,
        "sup_vz": sup_vz,   "sup_p": sup_p,   "sup_T": sup_T,
        "snap_pts": snap_pts, "snap_vx": snap_vx, "snap_vy": snap_vy,
        "snap_vz": snap_vz,   "snap_p": snap_p,   "snap_T": snap_T,
    }


# ═══════════════════════════════════════════════════════════════
# RUN PATH / LOGGING
# ═══════════════════════════════════════════════════════════════

IS_SWEEP = bool(os.environ.get("WANDB_SWEEP_ID"))

if args.run_path:
    RUN_PATH = args.run_path
elif IS_SWEEP:
    _sweep_short = os.environ.get("WANDB_SWEEP_ID", "sweep")[-8:]
    _trial_uid   = os.environ.get("SLURM_ARRAY_TASK_ID", str(os.getpid()))
    RUN_PATH = f"../pinn_isothermal_sweep_runs/{_sweep_short}/trial_{_trial_uid}"
else:
    RUN_PATH = (f"../pinn_isothermal_runs/"
                f"dp11_h{HIDDEN_DIM}_l{N_LAYERS}_e{EPOCHS}_lr{LR:.0e}_lrend{LR_END:.0e}"
                f"_ntrain{N_TOTAL_TRAIN}_sup{N_SUP}_s{SEED}")
os.makedirs(RUN_PATH, exist_ok=True)

_log_path = os.path.join(RUN_PATH, "training.log")
_log_handle = open(_log_path, "w", buffering=1, encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__,  _log_handle)
sys.stderr  = _Tee(sys.__stderr__,  _log_handle)

dp = load_data(args)

print(f"\nRun path : {RUN_PATH}")
print(f"Config   : h={HIDDEN_DIM}  l={N_LAYERS}  epochs={EPOCHS}")
print(f"           lr={LR}  lr_end={LR_END}  gamma={GAMMA:.6f}")
print(f"           n_train={N_TOTAL_TRAIN}  n_sup={N_SUP}  n_test={N_TEST}")
print(f"           w_pde={W_PDE}  w_bc={W_BC}  w_sup={W_SUP}  w_sup_p={W_SUP_P}")

if not DEBUG or IS_SWEEP:
    api_key = "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"
    wandb.login(key=api_key)
    run = wandb.init(
        project=PROJECT,
        name=os.path.basename(RUN_PATH),
        config={
            "epochs": EPOCHS, "lr": LR, "lr_end": LR_END,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "in_dim": IN_DIM,
            "seed": SEED, "n_train": N_TOTAL_TRAIN, "n_sup": N_SUP, "n_test": N_TEST,
            "w_pde": W_PDE, "w_bc": W_BC, "w_sup": W_SUP, "w_sup_p": W_SUP_P,
        },
    )

if IS_SWEEP:
    _sc = wandb.config
    HIDDEN_DIM    = int(_sc.get("hidden_dim",  HIDDEN_DIM))
    N_LAYERS      = int(_sc.get("n_layers",    N_LAYERS))
    EPOCHS        = int(_sc.get("epochs",      EPOCHS))
    LR            = float(_sc.get("lr",        LR))
    LR_END        = LR * float(_sc.get("lr_decay_ratio", LR_END / LR))
    W_PDE         = float(_sc.get("w_pde",     W_PDE))
    W_BC          = float(_sc.get("w_bc",      W_BC))
    W_SUP         = float(_sc.get("w_sup",     W_SUP))
    W_SUP_P       = float(_sc.get("w_sup_p",   W_SUP_P))
    N_TOTAL_TRAIN = int(_sc.get("n_train",     N_TOTAL_TRAIN))
    GAMMA         = (LR_END / LR) ** (1.0 / EPOCHS)
    n_vol_train    = int(FRAC_VOL    * N_TOTAL_TRAIN)
    n_wall_train   = int(FRAC_WALL   * N_TOTAL_TRAIN)
    n_inlet_train  = int(FRAC_INLET  * N_TOTAL_TRAIN)
    n_outlet_train = int(FRAC_OUTLET * N_TOTAL_TRAIN)
    print(f"[SWEEP] h={HIDDEN_DIM}  l={N_LAYERS}  epochs={EPOCHS}"
          f"  lr={LR}  w_pde={W_PDE}  w_sup={W_SUP}  n_train={N_TOTAL_TRAIN}")

# ═══════════════════════════════════════════════════════════════
# GLOBAL NORMALIZATION
# ═══════════════════════════════════════════════════════════════

coord_mean = dp["cfd_pts_nd"].mean(dim=0).to(DEVICE)
coord_std  = dp["cfd_pts_nd"].std(dim=0).clamp(min=1e-6).to(DEVICE)
out_mean   = dp["cfd_fields_nd"].mean(dim=0).to(DEVICE)
out_std    = dp["cfd_fields_nd"].std(dim=0).clamp(min=1e-3).to(DEVICE)

MOM_SCALE_X = 1.0 / float(coord_std[0])
MOM_SCALE_Y = 1.0 / float(coord_std[1])
MOM_SCALE_Z = 1.0 / float(coord_std[2])
DIV_SCALE   = float(max(out_std[0] / coord_std[0],
                        out_std[1] / coord_std[1],
                        out_std[2] / coord_std[2]))

print(f"\nGlobal stats")
print(f"  coord_mean : {coord_mean.tolist()}")
print(f"  coord_std  : {coord_std.tolist()}")
print(f"  out_mean   : {out_mean.tolist()}")
print(f"  out_std    : {out_std.tolist()}")
print(f"  MOM_SCALE  : X={MOM_SCALE_X:.1f}  Y={MOM_SCALE_Y:.1f}  Z={MOM_SCALE_Z:.1f}")
print(f"  DIV_SCALE  : {DIV_SCALE:.1f}")

del dp["cfd_pts_nd"], dp["cfd_fields_nd"]

# ═══════════════════════════════════════════════════════════════
# MODEL + OPTIMIZER
# ═══════════════════════════════════════════════════════════════

net   = FFNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS).to(DEVICE).double()
model = NormalizedPINN(net, coord_mean, coord_std, out_mean, out_std)

opt_model = Adam(model.parameters(), lr=LR, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt_model, gamma=GAMMA)


def safe_log10(x):
    v = x.item() if isinstance(x, torch.Tensor) else float(x)
    return float(np.log10(v)) if v > 0 else float("nan")


plot_epochs = set(np.linspace(0, EPOCHS - 1, 4, dtype=int).tolist())

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

start = time.time()
ctx = dp["ctx"]

for epoch in range(EPOCHS):
    model.train()
    opt_model.zero_grad()

    coll_pts, coll_lbls = sample_collocation(dp["vol_pool"], dp["wall_pool"], n_vol_train, n_wall_train)
    coll_pts = coll_pts.clone()
    coll_pts[:, 0] -= dp["stl_buffer_mm"]
    coll_pts /= 1000.0   # mm → m

    _in_idx     = torch.randperm(len(dp["inlet_pool"][0]), device='cpu')[:n_inlet_train]
    inlet_pts   = dp["inlet_pool"][0][_in_idx]
    vx_inlet_bc = dp["inlet_pool"][1][_in_idx].to(DEVICE)
    vy_inlet_bc = dp["inlet_pool"][2][_in_idx].to(DEVICE)
    vz_inlet_bc = dp["inlet_pool"][3][_in_idx].to(DEVICE)
    _out_idx    = torch.randperm(len(dp["outlet_pool"]), device='cpu')[:n_outlet_train]
    outlet_pts  = dp["outlet_pool"][_out_idx]

    pts_xyz = torch.cat([coll_pts, inlet_pts, outlet_pts]).to(DEVICE)
    pts_xyz.requires_grad_(True)
    lbls = torch.cat([
        coll_lbls,
        torch.ones( inlet_pts.shape[0],  dtype=torch.long, device='cpu'),
        2 * torch.ones(outlet_pts.shape[0], dtype=torch.long, device='cpu'),
    ]).to(DEVICE)

    fields = model(pts_xyz)
    derivs = compute_derivatives(fields, pts_xyz)

    (l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
     l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
     l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
    ) = compute_losses(
        *derivs, lbls,
        vx_inlet_bc, vy_inlet_bc, vz_inlet_bc,
        ctx,
        DIV_SCALE, MOM_SCALE_X, MOM_SCALE_Y, MOM_SCALE_Z,
    )

    if dp["n_sup"] > 0:
        sup_f    = model(dp["sup_pts"].to(DEVICE))
        l_sup_vx = torch.mean((sup_f[:, 0] - dp["sup_vx"].to(DEVICE)) ** 2) / ctx["vx_var"]
        l_sup_vy = torch.mean((sup_f[:, 1] - dp["sup_vy"].to(DEVICE)) ** 2) / ctx["vy_var"]
        l_sup_vz = torch.mean((sup_f[:, 2] - dp["sup_vz"].to(DEVICE)) ** 2) / ctx["vz_var"]
        l_sup_p  = torch.mean((sup_f[:, 3] - dp["sup_p"].to(DEVICE))  ** 2) / ctx["p_var"]
        l_sup_T  = torch.mean((sup_f[:, 4] - dp["sup_T"].to(DEVICE))  ** 2) / ctx["t_var"]
    else:
        z = torch.zeros(1, dtype=torch.float64, device=DEVICE).squeeze()
        l_sup_vx = l_sup_vy = l_sup_vz = l_sup_p = l_sup_T = z

    l_pde   = l_div + l_mom_x + l_mom_y + l_mom_z + l_heat
    l_bc    = (l_inlet_vx + l_inlet_vy + l_inlet_vz + l_inlet_T + l_outlet_p
               + l_wall_vx + l_wall_vy + l_wall_vz + l_wall_T)
    l_sup   = l_sup_vx + l_sup_vy + l_sup_vz + l_sup_p + l_sup_T
    l_total = W_PDE * l_pde + W_BC * l_bc + W_SUP * (l_sup - l_sup_p) + W_SUP_P * l_sup_p

    l_total.backward()
    opt_model.step()
    scheduler.step()

    # ── Eval ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred = model(dp["test_pts"].to(DEVICE))
    mse_vx = torch.mean((pred[:, 0] - dp["test_vx"].to(DEVICE)) ** 2)
    mse_vy = torch.mean((pred[:, 1] - dp["test_vy"].to(DEVICE)) ** 2)
    mse_vz = torch.mean((pred[:, 2] - dp["test_vz"].to(DEVICE)) ** 2)
    mse_p  = torch.mean((pred[:, 3] - dp["test_p"].to(DEVICE))  ** 2)
    mse_T  = torch.mean((pred[:, 4] - dp["test_T"].to(DEVICE))  ** 2)
    mse_total = (mse_vx + mse_vy + mse_vz + mse_p + mse_T).item()

    # ── Snapshot plots ───────────────────────────────────────
    if epoch in plot_epochs:
        snap_dir = os.path.join(RUN_PATH, f"snap_{epoch+1}_of_{EPOCHS}")
        model.eval()
        with torch.no_grad():
            pred_snap = model(dp["snap_pts"].to(DEVICE))
        plot_fields(
            dp["snap_pts"].cpu().numpy(),
            [
                ("vx", dp["snap_vx"].cpu().numpy() * dp["V_IN"],
                       pred_snap[:, 0].cpu().numpy() * dp["V_IN"]),
                ("vy", dp["snap_vy"].cpu().numpy() * dp["V_IN"],
                       pred_snap[:, 1].cpu().numpy() * dp["V_IN"]),
                ("vz", dp["snap_vz"].cpu().numpy() * dp["V_IN"],
                       pred_snap[:, 2].cpu().numpy() * dp["V_IN"]),
                ("p",  dp["snap_p"].cpu().numpy() * dp["P_SCALE"] + dp["P_REF"],
                       pred_snap[:, 3].cpu().numpy() * dp["P_SCALE"] + dp["P_REF"]),
                ("T",  dp["snap_T"].cpu().numpy() * dp["t_std"] + dp["t_mean"],
                       pred_snap[:, 4].cpu().numpy() * dp["t_std"] + dp["t_mean"]),
            ],
            output_dir=snap_dir,
        )

    # ── Logging ──────────────────────────────────────────────
    log = {
        "pde/divergence": safe_log10(l_div), "pde/momentum_x": safe_log10(l_mom_x),
        "pde/momentum_y": safe_log10(l_mom_y), "pde/momentum_z": safe_log10(l_mom_z),
        "pde/heat":       safe_log10(l_heat),
        "bc/inlet_vx":    safe_log10(l_inlet_vx), "bc/inlet_vy": safe_log10(l_inlet_vy),
        "bc/inlet_vz":    safe_log10(l_inlet_vz), "bc/inlet_T":  safe_log10(l_inlet_T),
        "bc/outlet_p":    safe_log10(l_outlet_p),
        "bc/wall_vx":     safe_log10(l_wall_vx), "bc/wall_vy": safe_log10(l_wall_vy),
        "bc/wall_vz":     safe_log10(l_wall_vz), "bc/wall_T":  safe_log10(l_wall_T),
        "sup/vx": safe_log10(l_sup_vx), "sup/vy": safe_log10(l_sup_vy),
        "sup/vz": safe_log10(l_sup_vz), "sup/p":  safe_log10(l_sup_p), "sup/T": safe_log10(l_sup_T),
        "loss/pde": safe_log10(l_pde), "loss/bc": safe_log10(l_bc),
        "loss/sup": safe_log10(l_sup), "loss/total": safe_log10(l_total),
        "eval/mse_vx": safe_log10(mse_vx), "eval/mse_vy": safe_log10(mse_vy),
        "eval/mse_vz": safe_log10(mse_vz), "eval/mse_p":  safe_log10(mse_p),
        "eval/mse_T":  safe_log10(mse_T),  "eval/mse_total": safe_log10(mse_total),
        "train/lr": scheduler.get_last_lr()[0],
    }
    log = {k: v for k, v in log.items() if not (isinstance(v, float) and np.isnan(v))}

    print(
        f"\n[{epoch+1:>5}/{EPOCHS}]\n"
        f"PDE   div={l_div.item():.3e}  mom_x={l_mom_x.item():.3e}  mom_y={l_mom_y.item():.3e}"
        f"  mom_z={l_mom_z.item():.3e}  heat={l_heat.item():.3e}  | total={l_pde.item():.3e}\n"
        f"BC    in_vx={l_inlet_vx.item():.3e}  in_T={l_inlet_T.item():.3e}"
        f"  out_p={l_outlet_p.item():.3e}  wall_T={l_wall_T.item():.3e}  | total={l_bc.item():.3e}\n"
        f"SUP   vx={l_sup_vx.item():.3e}  p={l_sup_p.item():.3e}  T={l_sup_T.item():.3e}"
        f"  | total={l_sup.item():.3e}\n"
        f"LOSS  total={l_total.item():.3e}  lr={scheduler.get_last_lr()[0]:.3e}\n"
        f"MSE   vx={mse_vx.item():.3e}  vy={mse_vy.item():.3e}  vz={mse_vz.item():.3e}"
        f"  p={mse_p.item():.3e}  T={mse_T.item():.3e}  | total={mse_total:.3e}"
    )

    if not DEBUG or IS_SWEEP:
        wandb.log(log, step=epoch)

print(f"\nTraining done in {time.time() - start:.1f}s")
torch.save(model.state_dict(), os.path.join(RUN_PATH, "pinn_model.pt"))
torch.save({
    "coord_mean": coord_mean.cpu(), "coord_std": coord_std.cpu(),
    "out_mean":   out_mean.cpu(),   "out_std":   out_std.cpu(),
    "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "in_dim": IN_DIM,
    "V_IN": dp["V_IN"], "P_SCALE": dp["P_SCALE"], "P_REF": dp["P_REF"],
    "t_mean": dp["t_mean"], "t_std": dp["t_std"],
}, os.path.join(RUN_PATH, "normalization.pt"))

# ═══════════════════════════════════════════════════════════════
# INFERENCE  (CPU, full domain)
# See run_pinn.py — PINN inference is forced onto CPU: float64 CUDA causes
# numerical overflow/NaN on the IZAR GPU at inference time (training is unaffected).
# ═══════════════════════════════════════════════════════════════

torch.set_default_device("cpu")
net_inf   = FFNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS)
model_inf = NormalizedPINN(net_inf, coord_mean.cpu(), coord_std.cpu(), out_mean.cpu(), out_std.cpu())
model_inf.load_state_dict(
    torch.load(os.path.join(RUN_PATH, "pinn_model.pt"), weights_only=True, map_location="cpu"))
model_inf.eval()

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

print(f"\n{'─'*60}\nInference — isothermal dp11")
inf_dir = os.path.join(RUN_PATH, "inference")
os.makedirs(inf_dir, exist_ok=True)

_vx_raw = np.load(DATA_DIR + "vel_x.npy")
cfd_pts_cpu = torch.tensor(_vx_raw[:, :3], dtype=torch.float64)
cfd_vx  = _vx_raw[:, 3]
cfd_vy  = np.load(DATA_DIR + "vel_y.npy")[:, 3]
cfd_vz  = np.load(DATA_DIR + "vel_z.npy")[:, 3]
cfd_p   = np.load(DATA_DIR + "press.npy")[:, 3]
cfd_T   = np.load(DATA_DIR + "temp.npy")[:, 3]
del _vx_raw

with torch.no_grad():
    pred_all = model_inf(cfd_pts_cpu)

vx_pred = pred_all[:, 0].numpy() * dp["V_IN"]
vy_pred = pred_all[:, 1].numpy() * dp["V_IN"]
vz_pred = pred_all[:, 2].numpy() * dp["V_IN"]
p_pred  = pred_all[:, 3].numpy() * dp["P_SCALE"] + dp["P_REF"]
T_pred  = pred_all[:, 4].numpy() * dp["t_std"]   + dp["t_mean"]

te_np = dp["test_idx"].cpu().numpy()
for true_arr, pred_arr, name in [
    (cfd_vx, vx_pred, "vx"), (cfd_vy, vy_pred, "vy"), (cfd_vz, vz_pred, "vz"),
    (cfd_p,  p_pred,  "p"),  (cfd_T,  T_pred,  "T"),
]:
    print(f"  {name}  overall RMSE={rmse(true_arr, pred_arr):.4e}"
          f"  test RMSE={rmse(true_arr[te_np], pred_arr[te_np]):.4e}")

pts_np   = cfd_pts_cpu.numpy()
idx_plot = np.random.choice(len(pts_np), min(100_000, len(pts_np)), replace=False)
plot_fields(
    pts_np[idx_plot],
    [
        ("vx", cfd_vx[idx_plot], vx_pred[idx_plot]),
        ("vy", cfd_vy[idx_plot], vy_pred[idx_plot]),
        ("vz", cfd_vz[idx_plot], vz_pred[idx_plot]),
        ("p",  cfd_p[idx_plot],  p_pred[idx_plot]),
        ("T",  cfd_T[idx_plot],  T_pred[idx_plot]),
    ],
    output_dir=inf_dir,
)

_log_handle.close()
