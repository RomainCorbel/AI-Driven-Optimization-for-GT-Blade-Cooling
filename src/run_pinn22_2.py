import argparse
import glob
import trimesh
import numpy as np
import random
import torch
import torch.nn as nn
import os
import sys
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import wandb
from torch.optim import Adam

torch.set_default_dtype(torch.float64)

# ═══════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS  (SI throughout — no bars, no kK)
# ═══════════════════════════════════════════════════════════════

T_INLET  = 298.15      # [K]   inlet / wall temperature BC
T_WALL   = 298.15      # [K]
P_OUTLET_PA = 0.835e5  # [Pa]  outlet static pressure BC

M  = 28.96e-3          # [kg/mol] molar mass of air
R  = 8.314             # [J/(mol·K)]
K  = 2.61e-2           # [W/(m·K)] thermal conductivity
CP = 1.00e3            # [J/(kg·K)] specific heat
PR = None              # Prandtl number — computed below after rho

T_REF_SUTH = 278.15    # [K]   Sutherland reference temperature
MU_REF     = 1.716e-5  # [Pa·s]
S_SUTH     = 110.4     # [K]

rho = P_OUTLET_PA * M / (R * T_REF_SUTH)   # [kg/m³]
PR  = MU_REF * CP / K                       # Prandtl number (~0.66), dimensionless

# ═══════════════════════════════════════════════════════════════
# GEOMETRY SAMPLING  (labels: 0=interior, 1=inlet, 2=outlet, 3=wall)
# ═══════════════════════════════════════════════════════════════

def _submesh_pts(mesh, face_ids, n):
    pts, _ = trimesh.sample.sample_surface(
        mesh.submesh([face_ids], only_watertight=False)[0], n
    )
    return torch.tensor(pts)


def sample_inlet(mesh, n):
    thresh = 1e-5
    faces  = [i for i, f in enumerate(mesh.faces)
               if np.all(np.abs(mesh.vertices[f, 0]) < thresh)]
    pts    = _submesh_pts(mesh, faces, n)
    return pts, torch.ones(len(pts), dtype=torch.int64)


def build_wall_pool(mesh, n_pool=200_000):
    """Called once. Returns (wall_pts_mm, wall_normals) for physical-domain walls."""
    pts, face_ids = trimesh.sample.sample_surface(mesh, n_pool * 3)
    x = pts[:, 0]
    mask = (
        (np.abs(x - X_STL_MIN_MM) > 0.1) &
        (np.abs(x - X_STL_MAX_MM) > 0.1) &
        (x >= X_STL_MIN_MM) &
        (x <= X_STL_MAX_MM)
    )
    wall_pts     = torch.tensor(pts[mask],                        dtype=torch.float64)
    wall_normals = torch.tensor(mesh.face_normals[face_ids[mask]], dtype=torch.float64)
    if len(wall_pts) > n_pool:
        idx = torch.randperm(len(wall_pts))[:n_pool]
        wall_pts, wall_normals = wall_pts[idx], wall_normals[idx]
    return wall_pts, wall_normals


def sample_wall(wall_pool, n):
    wall_pts, wall_normals = wall_pool
    idx     = torch.randperm(len(wall_pts))[:n]
    return wall_pts[idx], 3 * torch.ones(n, dtype=torch.int64), wall_normals[idx]


def build_volume_pool(mesh, n_pool=800_000, wall_fraction=0.5,
                      delta_horiz_mm=1.0, delta_vert_mm=1.0, vert_frac=0.7):
    """Called once. Returns pool of interior collocation points (mm, STL frame)."""
    n_near    = int(n_pool * wall_fraction)
    n_uniform = n_pool - n_near

    raw  = trimesh.sample.volume_mesh(mesh, n_uniform * 3)
    mask = (raw[:, 0] >= X_STL_MIN_MM) & (raw[:, 0] <= X_STL_MAX_MM)
    raw  = raw[mask]
    if len(raw) > n_uniform:
        raw = raw[np.random.choice(len(raw), n_uniform, replace=False)]

    def _sample_submesh_near(face_ids, n, delta_mm):
        sub = mesh.submesh([face_ids], only_watertight=False)
        if not sub:
            return np.zeros((0, 3))
        s_pts, s_fids = trimesh.sample.sample_surface(sub[0], n * 2)
        eps  = np.random.uniform(0, delta_mm, size=(len(s_pts), 1))
        near = s_pts - eps * sub[0].face_normals[s_fids]
        keep = (near[:, 0] >= X_STL_MIN_MM) & (near[:, 0] <= X_STL_MAX_MM)
        near = near[keep]
        if len(near) > n:
            near = near[np.random.choice(len(near), n, replace=False)]
        return near

    horiz_fids = np.where(np.abs(mesh.face_normals[:, 2]) >= 0.7)[0]
    vert_fids  = np.where(np.abs(mesh.face_normals[:, 2]) <  0.7)[0]
    near_horiz = _sample_submesh_near(horiz_fids, int(n_near * (1 - vert_frac)), delta_horiz_mm)
    near_vert  = _sample_submesh_near(vert_fids,  int(n_near * vert_frac),       delta_vert_mm)
    all_pts = np.vstack([raw, near_horiz, near_vert])
    print(f"  uniform: {len(raw)}  near-horiz: {len(near_horiz)}  near-vert: {len(near_vert)}")
    return torch.tensor(all_pts, dtype=torch.float64), len(raw), len(near_horiz)


def sample_volume(vol_pool, n):
    idx = torch.randperm(len(vol_pool))[:n]
    return vol_pool[idx], torch.zeros(n, dtype=torch.int64)


def sample_collocation(vol_pool, wall_pool, n_vol, n_wall):
    parts_pts, parts_lbl, parts_normals = [], [], []
    p, l = sample_volume(vol_pool, n_vol)
    parts_pts.append(p);  parts_lbl.append(l)
    parts_normals.append(torch.zeros(len(p), 3, dtype=torch.float64))
    w_pts, w_lbl, w_normals = sample_wall(wall_pool, n_wall)
    parts_pts.append(w_pts);   parts_lbl.append(w_lbl);   parts_normals.append(w_normals)
    pts, lbl, normals = torch.cat(parts_pts), torch.cat(parts_lbl), torch.cat(parts_normals)
    perm = torch.randperm(pts.size(0))
    return pts[perm], lbl[perm], normals[perm]


# ═══════════════════════════════════════════════════════════════
# PHYSICS  (fully dimensionless)
# ═══════════════════════════════════════════════════════════════

def dynamic_viscosity_nd(theta_star):
    """Sutherland viscosity, dimensionless output: mu* = mu(T) / mu(T_ref).
    theta_star = T / T_ref, so T = theta_star * T_ref."""
    T = theta_star.abs() * T_REF
    return (T / T_REF_SUTH) ** 1.5 * (T_REF_SUTH + S_SUTH) / (T + S_SUTH)
    # Returns mu(T)/MU_REF  (dimensionless viscosity ratio)


def _grad(f, pts):
    return torch.autograd.grad(
        f, pts, torch.ones_like(f), retain_graph=True, create_graph=True
    )[0]


def _grad2(f, pts, dim):
    g = _grad(f, pts)[:, dim:dim+1]
    return torch.autograd.grad(
        g, pts, torch.ones_like(g), retain_graph=True, create_graph=True
    )[0][:, dim:dim+1]


def compute_derivatives(fields, pts):
    """
    fields[:,0..4] = [u*, v*, w*, p*, T*]  — all dimensionless.
    pts            = x*                    — dimensionless (z-scored coords).
    autograd gives ∂(·)/∂x_i* directly — no chain-rule conversion needed.
    """
    u = fields[:, 0:1]
    v = fields[:, 1:2]
    w = fields[:, 2:3]
    p = fields[:, 3:4]
    T = fields[:, 4:5]

    gu = _grad(u, pts); u_x, u_y, u_z = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
    u_xx = _grad2(u, pts, 0); u_yy = _grad2(u, pts, 1); u_zz = _grad2(u, pts, 2)

    gv = _grad(v, pts); v_x, v_y, v_z = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
    v_xx = _grad2(v, pts, 0); v_yy = _grad2(v, pts, 1); v_zz = _grad2(v, pts, 2)

    gw = _grad(w, pts); w_x, w_y, w_z = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3]
    w_xx = _grad2(w, pts, 0); w_yy = _grad2(w, pts, 1); w_zz = _grad2(w, pts, 2)

    gp = _grad(p, pts); p_x, p_y, p_z = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]
    p_xx = p_yy = p_zz = None   # PPE removed — second derivatives of p not needed

    gT = _grad(T, pts); T_x, T_y, T_z = gT[:, 0:1], gT[:, 1:2], gT[:, 2:3]
    T_xx = _grad2(T, pts, 0); T_yy = _grad2(T, pts, 1); T_zz = _grad2(T, pts, 2)

    return (
        u, v, w, p, T,
        u_x, u_y, u_z, u_xx, u_yy, u_zz,
        v_x, v_y, v_z, v_xx, v_yy, v_zz,
        w_x, w_y, w_z, w_xx, w_yy, w_zz,
        p_x, p_y, p_z, p_xx, p_yy, p_zz,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    )


def compute_losses(
    u, v, w, p, T,
    u_x, u_y, u_z, u_xx, u_yy, u_zz,
    v_x, v_y, v_z, v_xx, v_yy, v_zz,
    w_x, w_y, w_z, w_xx, w_yy, w_zz,
    p_x, p_y, p_z, p_xx, p_yy, p_zz,
    T_x, T_y, T_z, T_xx, T_yy, T_zz,
    labels, wall_normals,
    p_out_nd, u_in_nd, v_in_nd, w_in_nd, T_in_nd, T_wall_nd,
    u_var, v_var, w_var, p_var, T_var,
):
    """
    All arguments are fully dimensionless:
      - fields:         u*, v*, w*, p*, T*
      - derivatives:    ∂u*/∂x_i*
      - BCs:            target values already in dimensionless units
      - variances:      computed on dimensionless CFD data (all the cfd data)

    PDE RESIDUALS in x* space (all O(1)):

    Continuity:   ∂u*/∂x* + ∂v*/∂y* + ∂w*/∂z* = 0

    Momentum-x:   (u*·∇*)u* + ∂p*/∂x* − Σ_j (1/Re_j) ∂²u*/∂x_j*² = 0
                  Re_j = rho * U_ref * L_ref_j / (mu* · MU_REF)
                  where mu* = mu(T)/MU_REF = dynamic_viscosity_nd(T*)

    Heat:         (u*·∇*)T* − Σ_j (1/(Re_j·Pr)) ∂²T*/∂x_j*² = 0

    PPE:          ∇*²p* = −Σ_{i,j} ∂u_i*/∂x_j* · ∂u_j*/∂x_i*
                  Coefficient = −1 exactly because P_ref = rho · U_ref²
    """
    interior = labels == 0
    inlet    = labels == 1
    outlet   = labels == 2
    wall     = labels == 3

    # Dimensionless viscosity at interior points: mu(T) / MU_REF
    mu_nd = dynamic_viscosity_nd(T[interior])   # dimensionless, > 0

    # Per-direction Reynolds numbers (dimensionless)
    # Re_j = rho * U_ref * L_ref_j / (mu* · MU_REF)
    # mu* varies per point, so Re_j_inv = MU_REF / (rho * U_ref * L_ref_j) * mu_nd⁻¹
    # Pre-computed scalars: INV_RE_BASE_j = MU_REF / (rho * U_ref * L_ref_j)
    inv_re_x = INV_RE_BASE_X * mu_nd   # 1/Re_x(T) per interior point
    inv_re_y = INV_RE_BASE_Y * mu_nd
    inv_re_z = INV_RE_BASE_Z * mu_nd

    # ── Continuity ────────────────────────────────────────────
    l_div = torch.mean((u_x + v_y + w_z) ** 2)

    # ── Momentum (each direction) ─────────────────────────────
    def momentum_residual(q_x, q_y, q_z, q_xx, q_yy, q_zz, dp_dq):
        advec = (u[interior] * q_x[interior]
                 + v[interior] * q_y[interior]
                 + w[interior] * q_z[interior])
        visc  = (inv_re_x * q_xx[interior]
                 + inv_re_y * q_yy[interior]
                 + inv_re_z * q_zz[interior])
        return torch.mean((advec + dp_dq[interior] - visc) ** 2)

    l_mom_x = momentum_residual(u_x, u_y, u_z, u_xx, u_yy, u_zz, p_x)
    l_mom_y = momentum_residual(v_x, v_y, v_z, v_xx, v_yy, v_zz, p_y)
    l_mom_z = momentum_residual(w_x, w_y, w_z, w_xx, w_yy, w_zz, p_z)

    # ── Heat equation ─────────────────────────────────────────
    # 1/(Re_j·Pr)  — Pr is constant (laminar, constant properties)
    inv_rePr_x = inv_re_x / PR
    inv_rePr_y = inv_re_y / PR
    inv_rePr_z = inv_re_z / PR
    l_heat = torch.mean((
        u[interior] * T_x[interior]
        + v[interior] * T_y[interior]
        + w[interior] * T_z[interior]
        - inv_rePr_x * T_xx[interior]
        - inv_rePr_y * T_yy[interior]
        - inv_rePr_z * T_zz[interior]
    ) ** 2)

    # PPE removed: it has a degenerate trivial minimum at constant velocity
    # (∇u*=0 → rhs=0, ∇²p*=0 → lhs=0 → l_ppe=0 for any linear pressure field),
    # which causes the network to collapse to a flat velocity solution early in
    # training and never recover. NS momentum already constrains pressure via ∂p*/∂x*.
    l_ppe = torch.zeros(1, dtype=u.dtype, device=u.device).squeeze()

    # ── BC losses — all normalised by field variance ──────────
    # Everything in dimensionless units; variances from dimensionless CFD data.
    l_inlet_u  = torch.mean((u[inlet].squeeze(1) - u_in_nd)  ** 2) / u_var
    l_inlet_v  = torch.mean((v[inlet].squeeze(1) - v_in_nd)  ** 2) / v_var
    l_inlet_w  = torch.mean((w[inlet].squeeze(1) - w_in_nd)  ** 2) / w_var
    l_inlet_T  = torch.mean((T[inlet].squeeze(1) - T_in_nd)  ** 2) / T_var
    l_outlet_p = torch.mean((p[outlet].squeeze(1)- p_out_nd) ** 2) / p_var
    l_wall_u   = torch.mean(u[wall] ** 2)                            / u_var
    l_wall_v   = torch.mean(v[wall] ** 2)                            / v_var
    l_wall_w   = torch.mean(w[wall] ** 2)                            / w_var
    l_wall_T   = torch.mean((T[wall].squeeze(1)  - T_wall_nd) ** 2) / T_var

    # ∂p*/∂n* = 0 at walls — in x* space, normals are unit vectors in STL frame
    # (unchanged by isotropic scaling within each direction — normals are already
    # in the STL coordinate system; coord_std anisotropy would distort them,
    # but since we only need the zero-gradient condition, the sign is enough)
    n_x = wall_normals[wall, 0:1]
    n_y = wall_normals[wall, 1:2]
    n_z = wall_normals[wall, 2:3]
    dp_dn = p_x[wall] * n_x + p_y[wall] * n_y + p_z[wall] * n_z
    # Normalise by expected variance of ∂p*/∂n* ≈ p_var (since n is unit vector)
    l_wall_dp_dn = torch.mean(dp_dn ** 2) / p_var

    return (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat, l_ppe,
        l_inlet_u, l_inlet_v, l_inlet_w, l_inlet_T, l_outlet_p,
        l_wall_u,  l_wall_v,  l_wall_w,  l_wall_T,
        l_wall_dp_dn,
    )


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FFNN(nn.Module):
    """Fully-connected network with Sin activations (standard for PINNs)."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), Sin()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ═══════════════════════════════════════════════════════════════
# PLOTTING  (converts dimensionless fields back to physical units)
# ═══════════════════════════════════════════════════════════════

def plot_fields(pts, fields, output_dir, tag="", slice_frac=0.10):
    """One figure per field: 3D scatter + x-y midplane cut.
    pts    : physical [m] coordinates (numpy array)
    fields : list of (name, data_physical, pred_physical)
    """
    os.makedirs(output_dir, exist_ok=True)

    def to_np(x):
        if x is None:
            return None
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pts_np    = to_np(pts)
    ranges    = pts_np.max(axis=0) - pts_np.min(axis=0)
    box_aspect = (ranges / ranges.max()).tolist()

    z_vals  = pts_np[:, 2]
    z_mid   = 0.5 * (z_vals.min() + z_vals.max())
    z_tol   = slice_frac * (z_vals.max() - z_vals.min())
    cut_mask = np.abs(z_vals - z_mid) < z_tol
    p_cut   = pts_np[cut_mask]
    cut_label = f"x-y  at  z = {z_mid*1000:.1f} mm  (n={cut_mask.sum()})"

    for name, data_raw, pred_raw in fields:
        data, pred = to_np(data_raw), to_np(pred_raw)
        has_data   = data is not None
        vmin = float(min(data.min(), pred.min()) if has_data else pred.min())
        vmax = float(max(data.max(), pred.max()) if has_data else pred.max())

        subplots = []
        if has_data:
            subplots.append((f"Data – {name}", data,        vmin, vmax))
        subplots.append(    (f"Pred – {name}", pred,        vmin, vmax))
        if has_data:
            subplots.append((f"Diff – {name}", data - pred, None, None))

        n_sub = len(subplots)
        fig   = plt.figure(figsize=(6 * n_sub, 10))

        for i, (title, color, cmin, cmax) in enumerate(subplots):
            ax = fig.add_subplot(2, n_sub, i + 1, projection="3d")
            sc = ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                            c=color, cmap="viridis",
                            vmin=cmin, vmax=cmax, s=1, rasterized=True)
            ax.set_box_aspect(box_aspect)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("X"); ax.set_yticklabels([]); ax.set_zticklabels([])
            plt.colorbar(sc, ax=ax, shrink=0.5, label=name)

        for i, (title, color, cmin, cmax) in enumerate(subplots):
            ax  = fig.add_subplot(2, n_sub, n_sub + i + 1)
            col = color[cut_mask]
            sc  = ax.scatter(p_cut[:, 0], p_cut[:, 1],
                             c=col, cmap="viridis",
                             vmin=cmin, vmax=cmax, s=4, rasterized=True)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
            ax.set_title(f"{title}\n{cut_label}", fontsize=9)
            plt.colorbar(sc, ax=ax, label=name, shrink=0.5, aspect=15)

        fig.tight_layout()
        safe  = name.replace("/", "_per_").replace(" ", "_").replace("[", "").replace("]", "")
        fname = f"{safe}{'_' + tag if tag else ''}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="PINN — dimensionless NS + heat for GT blade cooling")

parser.add_argument("--folder",     default="dp00")
parser.add_argument("--run-path",   default=None)
parser.add_argument("--project",    default="PINN22_2")
parser.add_argument("--device",     default="cuda")
parser.add_argument("--debug",      action="store_true",  help="Skip W&B logging")

parser.add_argument("--hidden-dim", type=int,   default=20)
parser.add_argument("--n-layers",   type=int,   default=4)

parser.add_argument("--epochs",     type=int,   default=6_000)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--lr",         type=float, default=3e-3)
parser.add_argument("--lr-end",     type=float, default=1e-4)

parser.add_argument("--n-train",        type=int,   default=5_000)
parser.add_argument("--n-test",         type=int,   default=10_000)
parser.add_argument("--n-sup",          type=int,   default=500)
parser.add_argument("--n-snapshots",    type=int,   default=20_000)
parser.add_argument("--wall-fraction",  type=float, default=0.5)
parser.add_argument("--n-pool",         type=int,   default=500_000)
parser.add_argument("--pool-frac-vol",  type=float, default=0.5)
parser.add_argument("--pool-frac-wall", type=float, default=0.5)
parser.add_argument("--delta-mm-horiz", type=float, default=1.0)
parser.add_argument("--delta-mm-vert",  type=float, default=20.0)
parser.add_argument("--w-pde",    type=float, default=1.0)
parser.add_argument("--w-bc",     type=float, default=1.0)
parser.add_argument("--w-sup",    type=float, default=1.0)
parser.add_argument("--w-sup-p",  type=float, default=1.0)
parser.add_argument("--outlet-y-min-frac", type=float, default=0.0)
parser.add_argument("--inlet-y-max-frac",  type=float, default=1.0)

args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

FOLDER     = args.folder
PROJECT    = args.project
DEVICE     = args.device
DEBUG      = args.debug

EPOCHS     = args.epochs
HIDDEN_DIM = args.hidden_dim
N_LAYERS   = args.n_layers
SEED       = args.seed
LR         = args.lr
LR_END     = args.lr_end
GAMMA      = (LR_END / LR) ** (1.0 / args.epochs)

N_TEST            = args.n_test
N_TOTAL_TRAIN     = args.n_train
N_SUP             = args.n_sup
N_POINT_SNAPSHOTS = args.n_snapshots
WALL_FRACTION     = args.wall_fraction
DELTA_HORIZ_MM    = args.delta_mm_horiz
DELTA_VERT_MM     = args.delta_mm_vert
W_PDE    = args.w_pde
W_BC     = args.w_bc
W_SUP    = args.w_sup
W_SUP_P  = args.w_sup_p
OUTLET_Y_MIN_FRAC = args.outlet_y_min_frac
INLET_Y_MAX_FRAC  = args.inlet_y_max_frac
N_TOTAL_POOL   = args.n_pool
POOL_FRAC_VOL  = args.pool_frac_vol
POOL_FRAC_WALL = args.pool_frac_wall

DATA_DIR = f"./preProcessedData/with_T/{FOLDER}/"

if args.run_path:
    RUN_PATH = args.run_path
else:
    _name = (f"f{FOLDER}"
             f"_h{HIDDEN_DIM}_l{N_LAYERS}"
             f"_e{EPOCHS}"
             f"_lr{LR:.0e}_lrend{LR_END:.0e}"
             f"_ntrain{N_TOTAL_TRAIN}"
             f"_sup{N_SUP}"
             f"_wpde{W_PDE}_wbc{W_BC}_wsup{W_SUP}_wsupp{W_SUP_P}"
             f"_s{SEED}")
    RUN_PATH = f"../pinn20_runs/{_name}"

os.makedirs(RUN_PATH, exist_ok=True)
_log_path = os.path.join(RUN_PATH, "training.log")

class _Tee(io.TextIOBase):
    def __init__(self, stream, logfile):
        self._stream  = stream
        self._logfile = logfile
    def write(self, s):
        self._stream.write(s); self._logfile.write(s); return len(s)
    def flush(self):
        self._stream.flush(); self._logfile.flush()

_log_handle = open(_log_path, "w", buffering=1, encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__,  _log_handle)
sys.stderr  = _Tee(sys.__stderr__,  _log_handle)

FRAC_VOL    = 0.20
FRAC_WALL   = 0.70
FRAC_INLET  = 0.05
FRAC_OUTLET = 0.05
n_vol_train    = int(FRAC_VOL    * N_TOTAL_TRAIN)
n_outlet_train = int(FRAC_OUTLET * N_TOTAL_TRAIN)
n_wall_train   = int(FRAC_WALL   * N_TOTAL_TRAIN)
n_inlet_train  = int(FRAC_INLET  * N_TOTAL_TRAIN)

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

print(f"Run path : {RUN_PATH}")
print(f"Log file : {_log_path}")
print(f"Config   : folder={FOLDER}  h={HIDDEN_DIM}  l={N_LAYERS}  epochs={EPOCHS}")
print(f"           lr={LR}  lr_end={LR_END}  gamma={GAMMA:.6f}")
print(f"           n_train={N_TOTAL_TRAIN}  n_sup={N_SUP}  seed={SEED}")
print(f"           w_pde={W_PDE}  w_bc={W_BC}  w_sup={W_SUP}  w_sup_p={W_SUP_P}")

if not DEBUG:
    api_key = "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"
    wandb.login(key=api_key)
    run = wandb.init(
        project=PROJECT,
        name=os.path.basename(RUN_PATH),
        config={
            "folder": FOLDER, "run_path": RUN_PATH,
            "epochs": EPOCHS, "lr": LR, "lr_end": LR_END,
            "scheduler": "exponential", "gamma": GAMMA,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "seed": SEED,
            "n_train": N_TOTAL_TRAIN, "n_test": N_TEST, "n_supervised": N_SUP,
            "w_pde": W_PDE, "w_bc": W_BC, "w_sup": W_SUP, "w_sup_p": W_SUP_P,
            "n_vol_train": n_vol_train, "n_outlet_train": n_outlet_train,
            "n_wall_train": n_wall_train, "n_inlet_train": n_inlet_train,
        },
    )

np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
torch.set_default_device(DEVICE)
print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# DATA LOADING  (raw physical units)
# ═══════════════════════════════════════════════════════════════

_inlet_vx_raw = np.load(DATA_DIR + "vel_x_inlet.npy")
_inlet_vy_raw = np.load(DATA_DIR + "vel_y_inlet.npy")
_inlet_vz_raw = np.load(DATA_DIR + "vel_z_inlet.npy")
if INLET_Y_MAX_FRAC < 1.0:
    _y_max_inlet  = float(_inlet_vx_raw[:, 1].max())
    _y_max_cut    = INLET_Y_MAX_FRAC * _y_max_inlet
    _inlet_mask   = _inlet_vx_raw[:, 1] <= _y_max_cut
    _inlet_vx_raw = _inlet_vx_raw[_inlet_mask]
    _inlet_vy_raw = _inlet_vy_raw[_inlet_mask]
    _inlet_vz_raw = _inlet_vz_raw[_inlet_mask]
    print(f"Inlet BC y-filter : y <= {_y_max_cut:.4f} m  (frac={INLET_Y_MAX_FRAC},"
          f" kept {_inlet_mask.sum()} / {_inlet_mask.size} pts)")

_temp_all  = np.load(DATA_DIR + "temp.npy")
_x_min_T   = float(_temp_all[:, 0].min())
_temp_face = _temp_all[_temp_all[:, 0] == _x_min_T]
if INLET_Y_MAX_FRAC < 1.0:
    _temp_face = _temp_face[_temp_face[:, 1] <= INLET_Y_MAX_FRAC * float(_temp_all[:, 1].max())]
T_inlet_cfd = float(_temp_face[:, 3].mean())   # [K]
print(f"Inlet T from CFD  : mean={T_inlet_cfd:.2f} K  (T_INLET_code={T_INLET:.2f} K)")

# Physical CFD data
cfd_pts = torch.tensor(np.load(DATA_DIR + "vel_x.npy")[:, :3])   # [m]
cfd_vx  = torch.tensor(np.load(DATA_DIR + "vel_x.npy")[:, 3])    # [m/s]
cfd_vy  = torch.tensor(np.load(DATA_DIR + "vel_y.npy")[:, 3])    # [m/s]
cfd_vz  = torch.tensor(np.load(DATA_DIR + "vel_z.npy")[:, 3])    # [m/s]
cfd_p   = torch.tensor(np.load(DATA_DIR + "press.npy")[:, 3])    # [Pa]
cfd_T   = torch.tensor(np.load(DATA_DIR + "temp.npy")[:, 3])     # [K]

# ── Domain bounds ────────────────────────────────────────────
X_MIN_MM = float(cfd_pts[:, 0].min()) * 1000
X_MAX_MM = float(cfd_pts[:, 0].max()) * 1000
STL_INLET_BUFFER_MM = 0.0 if FOLDER == "dp11" else 420.0
X_STL_MIN_MM = X_MIN_MM + STL_INLET_BUFFER_MM
X_STL_MAX_MM = X_MAX_MM + STL_INLET_BUFFER_MM
print(f"Physical domain  x : [{X_MIN_MM:.2f}, {X_MAX_MM:.2f}] mm"
      f"  =  [{X_MIN_MM/1000:.4f}, {X_MAX_MM/1000:.4f}] m")

# ── Outlet BC pool ───────────────────────────────────────────
_out_mask = cfd_pts[:, 0] > (X_MAX_MM / 1000 - 0.005)
if OUTLET_Y_MIN_FRAC > 0.0:
    _y_max    = float(cfd_pts[:, 1].max())
    _y_min_out = OUTLET_Y_MIN_FRAC * _y_max
    _out_mask  = _out_mask & (cfd_pts[:, 1] >= _y_min_out)
    print(f"Outlet BC y-filter: y >= {_y_min_out:.4f} m")
outlet_pool = cfd_pts[_out_mask]   # [m] — coords only, label=2 enforced at training
print(f"Outlet pool      : {len(outlet_pool)} CFD points, sampling {n_outlet_train} per epoch")
# ── REFERENCE SCALES ───────────────────────────────────────────
# x* = (x - x_min) / L_ref
# u* = u / U_REF
# p* = (p - p_outlet) / (ρ U_REF²)
# T* = T / T_REF

U_REF = 50.0        # [m/s]
T_REF = T_inlet_cfd     # [K]

P_REF = rho * U_REF**2   # [Pa]

# ── COORDINATE SCALING (PHYSICAL, NOT STATISTICAL) ─────────────
coord_min = cfd_pts.min(dim=0).values   # [m]
coord_max = cfd_pts.max(dim=0).values   # [m]

L_ref = coord_max - coord_min           # [m]
coord_mean = 0.5 * (coord_min + coord_max)

def to_nd(pts):
    """Physical [m] coords → dimensionless x* = (x - x_min) / L_ref."""
    return (pts - coord_min) / L_ref

# ── DIMENSIONLESS CFD FIELDS ───────────────────────────────────
nd_vx = cfd_vx / U_REF
nd_vy = cfd_vy / U_REF
nd_vz = cfd_vz / U_REF

nd_p  = (cfd_p - P_OUTLET_PA) / P_REF
nd_T  = cfd_T / T_REF

# ── INLET POOL ────────────────────────────────────────────────
nd_inlet_vx = torch.tensor(_inlet_vx_raw[:, 3]) / U_REF
nd_inlet_vy = torch.tensor(_inlet_vy_raw[:, 3]) / U_REF
nd_inlet_vz = torch.tensor(_inlet_vz_raw[:, 3]) / U_REF

inlet_pool = (
    torch.tensor(_inlet_vx_raw[:, :3]),  # coords [m]
    nd_inlet_vx,
    nd_inlet_vy,
    nd_inlet_vz,
)

print(f"Inlet pool       : {len(inlet_pool[0])} CFD points, sampling {n_inlet_train} per epoch")
print(f"P_REF            : {P_REF:.2f} Pa ({P_REF/1e5:.4f} bar)")
print(f"p* definition    : (p - P_OUTLET) / P_REF → outlet BC p* = 0")
print(f"p* stats         : mean={nd_p.mean().item():.3f}, std={nd_p.std().item():.3f}")
print(f"T_INLET_nd       : {T_inlet_cfd/T_REF:.4f}")

# ── OUTPUT STATISTICS (DIMENSIONLESS SPACE) ───────────────────
out_mean = torch.stack([
    nd_vx.mean(), nd_vy.mean(), nd_vz.mean(),
    nd_p.mean(),  nd_T.mean()
])

out_std = torch.stack([
    nd_vx.std(), nd_vy.std(), nd_vz.std(),
    nd_p.std(),  nd_T.std()
]).clamp(min=1e-3)

# ── REYNOLDS NUMBER SCALING (PHYSICAL, ANISOTROPIC) ───────────
# Re_i = ρ U_REF L_ref_i / μ
# 1/Re_i = μ / (ρ U_REF L_ref_i)

INV_RE_BASE_X = float(MU_REF / (rho * U_REF * L_ref[0]))
INV_RE_BASE_Y = float(MU_REF / (rho * U_REF * L_ref[1]))
INV_RE_BASE_Z = float(MU_REF / (rho * U_REF * L_ref[2]))

# ── LOGGING ────────────────────────────────────────────────────
print(f"coord_min [m]    : {coord_min.tolist()}")
print(f"coord_max [m]    : {coord_max.tolist()}")
print(f"L_ref [m]        : {L_ref.tolist()}")
print(f"coord_mean [m]   : {coord_mean.tolist()}")

print(f"out_mean (nd)    : {out_mean.tolist()}")
print(f"out_std (nd)     : {out_std.tolist()}")

print(f"1/Re_base X      : {INV_RE_BASE_X:.2e}")
print(f"1/Re_base Y      : {INV_RE_BASE_Y:.2e}")
print(f"1/Re_base Z      : {INV_RE_BASE_Z:.2e}")

print(f"Pr               : {PR:.4f}")
# ── Train / test split ────────────────────────────────────────
perm      = torch.randperm(cfd_pts.shape[0])
test_idx  = perm[:N_TEST]
train_idx = perm[N_TEST:]

test_pts  = cfd_pts[test_idx];   train_pts = cfd_pts[train_idx]
test_u    = nd_vx[test_idx];     train_u   = nd_vx[train_idx]
test_v    = nd_vy[test_idx];     train_v   = nd_vy[train_idx]
test_w    = nd_vz[test_idx];     train_w   = nd_vz[train_idx]
test_p    = nd_p[test_idx];      train_p   = nd_p[train_idx]
test_T    = nd_T[test_idx];      train_T   = nd_T[train_idx]

# Fixed supervised subset (dimensionless targets)
if N_SUP > 0:
    _fixed_sup_idx = torch.randint(0, train_pts.shape[0], (N_SUP,))
    fixed_sup_pts  = train_pts[_fixed_sup_idx]
    fixed_sup_u    = train_u[_fixed_sup_idx]
    fixed_sup_v    = train_v[_fixed_sup_idx]
    fixed_sup_w    = train_w[_fixed_sup_idx]
    fixed_sup_p    = train_p[_fixed_sup_idx]
    fixed_sup_T    = train_T[_fixed_sup_idx]
    print(f"Fixed supervised : {N_SUP} points (from {train_pts.shape[0]} training pts)")

# ── Field variances (on dimensionless data) ───────────────────
def _var(raw_var, floor, name):
    v = raw_var.clamp(min=floor)
    if raw_var.item() < floor:
        print(f"  [variance floor hit] {name}: raw={raw_var.item():.2e} → floor={floor:.2e}")
    return v

u_var = nd_vx.var()
v_var = nd_vy.var()
w_var = nd_vz.var()
p_var = nd_p.var()
# T is nearly isothermal (inlet ≈ wall ≈ T_REF), so nd_T.var() ≈ 0.
# Use a floor equivalent to ~1 K in physical space.
T_var = nd_T.var().clamp(min=(1.0 / T_REF) ** 2)

print(f"\nField variances (dimensionless):")
print(f"  u*: {u_var.item():.4e}  v*: {v_var.item():.4e}  w*: {w_var.item():.4e}")
print(f"  p*: {p_var.item():.4e}  T*: {T_var.item():.4e}")

# ── Dimensionless BC targets ──────────────────────────────────
P_OUT_ND   = 0               # scalar
T_INLET_ND = T_inlet_cfd / T_REF               # scalar
T_WALL_ND  = T_WALL      / T_REF               # scalar

# ── STL / geometry ───────────────────────────────────────────
stl_files = glob.glob(DATA_DIR + "*.stl")
print(f"Loading STL: {stl_files[0]}")
mesh = trimesh.load(stl_files[0])
_stl_xmin = float(mesh.vertices[:, 0].min())
_stl_xmax = float(mesh.vertices[:, 0].max())
print(f"STL mesh         x : [{_stl_xmin:.2f}, {_stl_xmax:.2f}] mm"
      f"  (physical in STL: [{X_STL_MIN_MM:.1f}, {X_STL_MAX_MM:.1f}] mm)")

print("Building volume pool (one-time trimesh call)...")
vol_pool, _n_uniform, _n_near_horiz = build_volume_pool(
    mesh, n_pool=int(POOL_FRAC_VOL * N_TOTAL_POOL),
    wall_fraction=WALL_FRACTION,
    delta_horiz_mm=DELTA_HORIZ_MM, delta_vert_mm=DELTA_VERT_MM)

print("Building wall pool (one-time trimesh call)...")
wall_pool = build_wall_pool(mesh, n_pool=int(POOL_FRAC_WALL * N_TOTAL_POOL))

# ── Snapshot points for plots (physical [m] coords) ──────────
_snap_n   = min(N_POINT_SNAPSHOTS, cfd_pts.shape[0])
_snap_idx = np.random.choice(cfd_pts.shape[0], _snap_n, replace=False)
snap_pts  = cfd_pts[_snap_idx]
snap_u    = nd_vx[_snap_idx];  snap_v = nd_vy[_snap_idx];  snap_w = nd_vz[_snap_idx]
snap_p    = nd_p[_snap_idx];   snap_T = nd_T[_snap_idx]

# ── Visualisation pools ───────────────────────────────────────
_pool_np     = vol_pool.cpu().numpy()
_uni         = _pool_np[:_n_uniform]
_near_horiz  = _pool_np[_n_uniform : _n_uniform + _n_near_horiz]
_near_vert   = _pool_np[_n_uniform + _n_near_horiz:]
_subsample   = 20_000
_ui = np.random.choice(len(_uni),        min(_subsample, len(_uni)),        replace=False)
_hi = np.random.choice(len(_near_horiz), min(_subsample, len(_near_horiz)), replace=False)
_vi = np.random.choice(len(_near_vert),  min(_subsample, len(_near_vert)),  replace=False)

fig_pool = plt.figure(figsize=(18, 5))
fig_pool.suptitle(
    f"Volume pool  —  uniform (blue, n={len(_uni)})  "
    f"near-horiz (red, n={len(_near_horiz)})  "
    f"near-vert (green, n={len(_near_vert)})  "
    f"δ_horiz={DELTA_HORIZ_MM} mm  δ_vert={DELTA_VERT_MM} mm", fontsize=10)
ax3d = fig_pool.add_subplot(141, projection="3d")
ax3d.scatter(_uni[_ui,0],        _uni[_ui,1],        _uni[_ui,2],        s=.5, alpha=.2, c="steelblue", label="uniform")
ax3d.scatter(_near_horiz[_hi,0], _near_horiz[_hi,1], _near_horiz[_hi,2], s=.5, alpha=.3, c="crimson",   label="near-horiz")
ax3d.scatter(_near_vert[_vi,0],  _near_vert[_vi,1],  _near_vert[_vi,2],  s=.5, alpha=.3, c="limegreen", label="near-vert")
ax3d.set_xlabel("x [mm]"); ax3d.set_ylabel("y [mm]"); ax3d.set_zlabel("z [mm]")
ax3d.set_title("3D view"); ax3d.legend(markerscale=8, fontsize=7)
for _idx, (xl, yl, xi, yi) in enumerate([("x [mm]","y [mm]",0,1),("x [mm]","z [mm]",0,2),("y [mm]","z [mm]",1,2)], start=2):
    ax = fig_pool.add_subplot(1,4,_idx)
    ax.scatter(_uni[_ui,xi],        _uni[_ui,yi],        s=.5, alpha=.2, c="steelblue")
    ax.scatter(_near_horiz[_hi,xi], _near_horiz[_hi,yi], s=.5, alpha=.3, c="crimson")
    ax.scatter(_near_vert[_vi,xi],  _near_vert[_vi,yi],  s=.5, alpha=.3, c="limegreen")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}")
plt.tight_layout()
fig_pool.savefig(os.path.join(RUN_PATH, "viz_volume_pool.png"), dpi=150, bbox_inches="tight")
plt.close(fig_pool)

# Convert pool to CFD frame [m] for geometry visualisation
_pool_cfd_m = _pool_np.copy().astype(float)
_pool_cfd_m[:, 0] -= STL_INLET_BUFFER_MM
_pool_cfd_m /= 1000.0
_bg_i = np.random.choice(len(_pool_cfd_m), min(20_000, len(_pool_cfd_m)), replace=False)
_bg   = _pool_cfd_m[_bg_i]

_in_np  = inlet_pool[0].cpu().numpy()
_out_np = outlet_pool.cpu().numpy()
fig_geo = plt.figure(figsize=(18, 5))
fig_geo.suptitle(
    f"BC geometry  —  vol pool (grey)  inlet (green, n={len(_in_np)})  outlet (red, n={len(_out_np)})", fontsize=10)
ax3d_g = fig_geo.add_subplot(141, projection="3d")
ax3d_g.scatter(_bg[:,0],    _bg[:,1],    _bg[:,2],    s=.5, alpha=.15, c="grey",      label="vol pool")
ax3d_g.scatter(_in_np[:,0],  _in_np[:,1],  _in_np[:,2],  s=4,  alpha=1., c="limegreen", label="inlet")
ax3d_g.scatter(_out_np[:,0], _out_np[:,1], _out_np[:,2], s=4,  alpha=1., c="crimson",   label="outlet")
ax3d_g.set_xlabel("x [m]"); ax3d_g.set_ylabel("y [m]"); ax3d_g.set_zlabel("z [m]")
ax3d_g.set_title("3D view"); ax3d_g.legend(markerscale=5, fontsize=7)
for _gi, (xl, yl, xi, yi) in enumerate([("x [m]","y [m]",0,1),("x [m]","z [m]",0,2),("y [m]","z [m]",1,2)], start=2):
    ax = fig_geo.add_subplot(1,4,_gi)
    ax.scatter(_bg[:,xi],    _bg[:,yi],    s=.5, alpha=.15, c="grey")
    ax.scatter(_in_np[:,xi],  _in_np[:,yi],  s=4,  alpha=1., c="limegreen")
    ax.scatter(_out_np[:,xi], _out_np[:,yi], s=4,  alpha=1., c="crimson")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}")
plt.tight_layout()
fig_geo.savefig(os.path.join(RUN_PATH, "viz_geometry_bc.png"), dpi=150, bbox_inches="tight")
plt.close(fig_geo)

if N_SUP > 0:
    _sup_np = fixed_sup_pts.cpu().numpy()
    fig_sv = plt.figure(figsize=(18, 5))
    fig_sv.suptitle(f"Supervised CFD pts  —  supervised (red, n={len(_sup_np)})  vol pool (grey)", fontsize=10)
    ax3d_sv = fig_sv.add_subplot(141, projection="3d")
    ax3d_sv.scatter(_bg[:,0], _bg[:,1], _bg[:,2], s=.5, alpha=.15, c="grey", label="vol pool")
    ax3d_sv.scatter(_sup_np[:,0], _sup_np[:,1], _sup_np[:,2], s=1.5, alpha=1., c="red", label="supervised")
    ax3d_sv.set_xlabel("x [m]"); ax3d_sv.set_ylabel("y [m]"); ax3d_sv.set_zlabel("z [m]")
    ax3d_sv.legend(markerscale=8, fontsize=7)
    for _svi, (xl, yl, xi, yi) in enumerate([("x [m]","y [m]",0,1),("x [m]","z [m]",0,2),("y [m]","z [m]",1,2)], start=2):
        ax = fig_sv.add_subplot(1,4,_svi)
        ax.scatter(_bg[:,xi], _bg[:,yi], s=.5, alpha=.15, c="grey")
        ax.scatter(_sup_np[:,xi], _sup_np[:,yi], s=1.5, alpha=1., c="red")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}")
    plt.tight_layout()
    fig_sv.savefig(os.path.join(RUN_PATH, "viz_supervised_points.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_sv)

# ═══════════════════════════════════════════════════════════════
# MODEL + OPTIMIZER
# ═══════════════════════════════════════════════════════════════
model = FFNN(in_dim=3, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS).to(DEVICE).double()
# Bias the last layer toward known BC values so training starts from a reasonable
# range without using any CFD data: u*=v*=w*=0 (no-slip), p*=0 (outlet), T*=T_WALL_ND.
with torch.no_grad():
    model.net[-1].bias.data[4] = T_WALL_ND   # T* → wall temperature
opt_model = Adam(model.parameters(), lr=LR, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt_model, gamma=GAMMA)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def safe_log10(x):
    v = x.item() if isinstance(x, torch.Tensor) else float(x)
    return float(np.log10(v)) if v > 0 else float("nan")

plot_epochs = set(np.linspace(0, EPOCHS - 1, 10, dtype=int).tolist())

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

_n_near_vert_cnt = len(_near_vert)
_sup_str = f"+{N_SUP:,} supervised" if N_SUP > 0 else "physics-only (N_SUP=0)"
_sep  = "═" * 65
_thin = "─" * 63
print(f"""
{_sep}
  DIMENSIONLESS SCALES
{_sep}
  U_REF = {U_REF} m/s    T_REF = {T_REF} K    P_REF = {P_REF:.1f} Pa
  Pr    = {PR:.4f}
  1/Re_base : X={INV_RE_BASE_X:.2e}  Y={INV_RE_BASE_Y:.2e}  Z={INV_RE_BASE_Z:.2e}
  P_out_nd  = {P_OUT_ND:.4f}    T_inlet_nd = {T_INLET_ND:.4f}    T_wall_nd = {T_WALL_ND:.4f}
{_sep}

  POOLS  (built once — N_TOTAL_POOL={N_TOTAL_POOL:,})
  {_thin}
  Volume pool      {len(vol_pool):>10,} pts  total
    uniform        {_n_uniform:>10,} pts  ({100*_n_uniform//len(vol_pool):2d}%)
    near-horiz     {_n_near_horiz:>10,} pts  ({100*_n_near_horiz//len(vol_pool):2d}%)
    near-vert      {_n_near_vert_cnt:>10,} pts  ({100*_n_near_vert_cnt//len(vol_pool):2d}%)
  Wall pool        {len(wall_pool[0]):>10,} pts
  Inlet pool       {len(inlet_pool[0]):>10,} pts
  Outlet pool      {len(outlet_pool):>10,} pts
{_sep}

  N_TOTAL_TRAIN = {N_TOTAL_TRAIN:,} pts/pass  |  supervised: {_sup_str}
  {_thin}
  Interior (label=0)  PDE      {n_vol_train:>8,} pts ({int(FRAC_VOL*100):2d}%)
  Wall     (label=3)  no-slip  {n_wall_train:>8,} pts ({int(FRAC_WALL*100):2d}%)
  Inlet    (label=1)  u,v,w,T  {n_inlet_train:>8,} pts ({int(FRAC_INLET*100):2d}%)
  Outlet   (label=2)  p        {n_outlet_train:>8,} pts ({int(FRAC_OUTLET*100):2d}%)
""")

start = time.time()

for epoch in range(EPOCHS):
    # ── Collocation points: STL mm → physical m ───────────────
    coll_pts, coll_lbls, coll_normals = sample_collocation(
        vol_pool, wall_pool, n_vol_train, n_wall_train
    )
    coll_pts = coll_pts.clone()
    coll_pts[:, 0] -= STL_INLET_BUFFER_MM   # shift to CFD x-frame
    coll_pts /= 1000.0                        # mm → m

    # ── Inlet / outlet points ─────────────────────────────────
    _in_idx     = torch.randperm(len(inlet_pool[0]))[:n_inlet_train]
    inlet_pts   = inlet_pool[0][_in_idx]      # [m]
    u_inlet_bc  = inlet_pool[1][_in_idx]      # u* (dimensionless)
    v_inlet_bc  = inlet_pool[2][_in_idx]
    w_inlet_bc  = inlet_pool[3][_in_idx]

    _out_idx   = torch.randperm(len(outlet_pool))[:n_outlet_train]
    outlet_pts = outlet_pool[_out_idx]        # [m]

    # ── Assemble batch ────────────────────────────────────────
    pts  = torch.cat([coll_pts, inlet_pts, outlet_pts]).requires_grad_(True)
    lbls = torch.cat([
        coll_lbls,
        torch.ones(n_inlet_train,  dtype=torch.long),
        2 * torch.ones(n_outlet_train, dtype=torch.long),
    ])
    wall_normals_batch = torch.cat([
        coll_normals,
        torch.zeros(n_inlet_train,  3, dtype=torch.float64),
        torch.zeros(n_outlet_train, 3, dtype=torch.float64),
    ])

    # ── Forward + PDE/BC losses ───────────────────────────────
    model.train()
    opt_model.zero_grad()

    fields = model(to_nd(pts))   # [u*, v*, w*, p*, T*]
    derivs = compute_derivatives(fields, pts)

    (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat, l_ppe,
        l_inlet_u, l_inlet_v, l_inlet_w, l_inlet_T, l_outlet_p,
        l_wall_u,  l_wall_v,  l_wall_w,  l_wall_T,
        l_wall_dp_dn,
    ) = compute_losses(
        *derivs, lbls, wall_normals_batch,
        P_OUT_ND, u_inlet_bc, v_inlet_bc, w_inlet_bc, T_INLET_ND, T_WALL_ND,
        u_var, v_var, w_var, p_var, T_var,
    )

    # ── Supervised losses ─────────────────────────────────────
    if N_SUP > 0:
        sup_fields = model(to_nd(fixed_sup_pts))    # [u*, v*, w*, p*, T*]
        l_sup_u = torch.mean((sup_fields[:, 0] - fixed_sup_u) ** 2) / u_var
        l_sup_v = torch.mean((sup_fields[:, 1] - fixed_sup_v) ** 2) / v_var
        l_sup_w = torch.mean((sup_fields[:, 2] - fixed_sup_w) ** 2) / w_var
        l_sup_p = torch.mean((sup_fields[:, 3] - fixed_sup_p) ** 2) / p_var
        l_sup_T = torch.mean((sup_fields[:, 4] - fixed_sup_T) ** 2) / T_var
    else:
        z = torch.zeros(1, dtype=torch.float64, device=DEVICE).squeeze()
        l_sup_u = l_sup_v = l_sup_w = l_sup_p = l_sup_T = z

    l_pde_total = l_div + l_mom_x + l_mom_y + l_mom_z + l_heat + l_ppe
    l_bc_total  = (l_inlet_u + l_inlet_v + l_inlet_w + l_inlet_T + l_outlet_p
                   + l_wall_u + l_wall_v + l_wall_w + l_wall_T + l_wall_dp_dn)
    l_sup_total = l_sup_u + l_sup_v + l_sup_w + l_sup_p + l_sup_T

    l_total = (W_PDE  * l_pde_total
               + W_BC * l_bc_total
               + W_SUP   * (l_sup_u + l_sup_v + l_sup_w + l_sup_T)
               + W_SUP_P * l_sup_p)

    l_total.backward()
    opt_model.step()
    scheduler.step()

    # ── Evaluation MSE (dimensionless) ────────────────────────
    model.eval()
    with torch.no_grad():
        pred = model(to_nd(test_pts))   # [u*, v*, w*, p*, T*]
    mse_u = torch.mean((pred[:, 0] - test_u) ** 2)
    mse_v = torch.mean((pred[:, 1] - test_v) ** 2)
    mse_w = torch.mean((pred[:, 2] - test_w) ** 2)
    mse_p = torch.mean((pred[:, 3] - test_p) ** 2)
    mse_T = torch.mean((pred[:, 4] - test_T) ** 2)
    mse_total = mse_u + mse_v + mse_w + mse_p + mse_T

    # ── Snapshot plots ────────────────────────────────────────
    if epoch in plot_epochs:
        snap_dir = os.path.join(RUN_PATH, f"snap_{epoch+1}_of_{EPOCHS}")
        with torch.no_grad():
            pred_snap = model(to_nd(snap_pts))
        # Convert dimensionless → physical for plots
        plot_fields(
            snap_pts.cpu().numpy(),
            [
                ("vx [m/s]", (snap_u * U_REF).cpu().numpy(),        (pred_snap[:, 0] * U_REF).cpu().numpy()),
                ("vy [m/s]", (snap_v * U_REF).cpu().numpy(),        (pred_snap[:, 1] * U_REF).cpu().numpy()),
                ("vz [m/s]", (snap_w * U_REF).cpu().numpy(),        (pred_snap[:, 2] * U_REF).cpu().numpy()),
                ("p [Pa]",   (snap_p * P_REF + P_OUTLET_PA).cpu().numpy(),  (pred_snap[:, 3] * P_REF + P_OUTLET_PA).cpu().numpy()),
                ("T [K]",    (snap_T * T_REF).cpu().numpy(),        (pred_snap[:, 4] * T_REF).cpu().numpy()),
            ],
            output_dir=snap_dir,
        )

    # ── Logging ───────────────────────────────────────────────
    log = {
        "pde/divergence":  safe_log10(l_div),
        "pde/momentum_x":  safe_log10(l_mom_x),
        "pde/momentum_y":  safe_log10(l_mom_y),
        "pde/momentum_z":  safe_log10(l_mom_z),
        "pde/heat":        safe_log10(l_heat),
        "pde/ppe":         safe_log10(l_ppe),
        "bc/inlet_u":      safe_log10(l_inlet_u),
        "bc/inlet_v":      safe_log10(l_inlet_v),
        "bc/inlet_w":      safe_log10(l_inlet_w),
        "bc/inlet_T":      safe_log10(l_inlet_T),
        "bc/outlet_p":     safe_log10(l_outlet_p),
        "bc/wall_u":       safe_log10(l_wall_u),
        "bc/wall_v":       safe_log10(l_wall_v),
        "bc/wall_w":       safe_log10(l_wall_w),
        "bc/wall_T":       safe_log10(l_wall_T),
        "bc/wall_dp_dn":   safe_log10(l_wall_dp_dn),
        "sup/u":           safe_log10(l_sup_u),
        "sup/v":           safe_log10(l_sup_v),
        "sup/w":           safe_log10(l_sup_w),
        "sup/p":           safe_log10(l_sup_p),
        "sup/T":           safe_log10(l_sup_T),
        "loss/pde_total":  safe_log10(l_pde_total),
        "loss/bc_total":   safe_log10(l_bc_total),
        "loss/sup_total":  safe_log10(l_sup_total),
        "loss/total":      safe_log10(l_total),
        "eval/mse_u":      safe_log10(mse_u),
        "eval/mse_v":      safe_log10(mse_v),
        "eval/mse_w":      safe_log10(mse_w),
        "eval/mse_p":      safe_log10(mse_p),
        "eval/mse_T":      safe_log10(mse_T),
        "eval/mse_total":  safe_log10(mse_total),
        "train/lr":        scheduler.get_last_lr()[0],
    }
    log = {k: v for k, v in log.items() if not (isinstance(v, float) and np.isnan(v))}

    print(
        f"\n\n\n\n[{epoch+1:>5}/{EPOCHS}]\n"
        f"PDE   div={l_div.item():.3e}  mom_x={l_mom_x.item():.3e}  mom_y={l_mom_y.item():.3e}  "
        f"mom_z={l_mom_z.item():.3e}  heat={l_heat.item():.3e}  ppe={l_ppe.item():.3e}  | total={l_pde_total.item():.3e}\n"
        f"BC    inlet_u={l_inlet_u.item():.3e}  inlet_v={l_inlet_v.item():.3e}  inlet_w={l_inlet_w.item():.3e}  "
        f"inlet_T={l_inlet_T.item():.3e}  outlet_p={l_outlet_p.item():.3e}\n"
        f"      wall_u={l_wall_u.item():.3e}   wall_v={l_wall_v.item():.3e}   wall_w={l_wall_w.item():.3e}   "
        f"wall_T={l_wall_T.item():.3e}   wall_dp_dn={l_wall_dp_dn.item():.3e}  | total={l_bc_total.item():.3e}\n"
        f"SUP   u={l_sup_u.item():.3e}  v={l_sup_v.item():.3e}  w={l_sup_w.item():.3e}  "
        f"p={l_sup_p.item():.3e}  T={l_sup_T.item():.3e}  | total={l_sup_total.item():.3e}\n"
        f"LOSS  total={l_total.item():.3e}  (w_pde={W_PDE}  w_bc={W_BC}  w_sup={W_SUP}  w_sup_p={W_SUP_P})  "
        f"lr={scheduler.get_last_lr()[0]:.3e}\n"
        f"MSE   u={mse_u.item():.3e}  v={mse_v.item():.3e}  w={mse_w.item():.3e}  "
        f"p={mse_p.item():.3e}  T={mse_T.item():.3e}  | total={mse_total.item():.3e}"
    )

    if not DEBUG:
        wandb.log(log, step=epoch)

print(f"\nTraining done in {time.time() - start:.1f}s")
torch.save(model.state_dict(), os.path.join(RUN_PATH, "pinn_model.pt"))

# ═══════════════════════════════════════════════════════════════
# INFERENCE  (CPU, full domain — convert back to physical units)
# ═══════════════════════════════════════════════════════════════

torch.set_default_device("cpu")
model_inf = FFNN(in_dim=3, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS)
model_inf.load_state_dict(
    torch.load(os.path.join(RUN_PATH, "pinn_model.pt"),
               weights_only=True, map_location="cpu")
)
model_inf.eval()

vx_in_full = np.load(DATA_DIR + "vel_x_inlet.npy")
vy_in_full = np.load(DATA_DIR + "vel_y_inlet.npy")
vz_in_full = np.load(DATA_DIR + "vel_z_inlet.npy")
vx_full    = np.load(DATA_DIR + "vel_x.npy")
vy_full    = np.load(DATA_DIR + "vel_y.npy")
vz_full    = np.load(DATA_DIR + "vel_z.npy")
p_full     = np.load(DATA_DIR + "press.npy")
T_full     = np.load(DATA_DIR + "temp.npy")

all_pts_np = np.concatenate((vx_in_full[:, :3], vx_full[:, :3]))
_coord_min_cpu = coord_min.cpu()
_L_ref_cpu     = L_ref.cpu()
with torch.no_grad():
    _pts_nd = (torch.tensor(all_pts_np) - _coord_min_cpu) / _L_ref_cpu
    pred_nd = model_inf(_pts_nd)   # [u*, v*, w*, p*, T*]

# Convert dimensionless → physical
vx_pred = pred_nd[:, 0].numpy() * U_REF       # [m/s]
vy_pred = pred_nd[:, 1].numpy() * U_REF
vz_pred = pred_nd[:, 2].numpy() * U_REF
p_pred  = pred_nd[:, 3].numpy() * P_REF + P_OUTLET_PA      # [Pa]
T_pred  = pred_nd[:, 4].numpy() * T_REF       # [K]

n_inlet      = vx_in_full.shape[0]
train_idx_np = train_idx.cpu().numpy()
test_idx_np  = test_idx.cpu().numpy()

# ── Velocity plots & RMSE (physical units) ───────────────────
idx_vel    = np.random.choice(all_pts_np.shape[0],
                               min(100_000, all_pts_np.shape[0]), replace=False)
vel_fields = []
for in_arr, body_arr, pred_arr, name in [
    (vx_in_full, vx_full, vx_pred, "vx [m/s]"),
    (vy_in_full, vy_full, vy_pred, "vy [m/s]"),
    (vz_in_full, vz_full, vz_pred, "vz [m/s]"),
]:
    true_all = np.concatenate((in_arr[:, 3], body_arr[:, 3]))
    vel_fields.append((name, true_all[idx_vel], pred_arr[idx_vel]))
    body_pred = pred_arr[n_inlet:]
    print(
        f"{name}  train RMSE: "
        f"{np.sqrt(np.mean((body_arr[:, 3][train_idx_np] - body_pred[train_idx_np])**2)):.4e} m/s"
        f"  test RMSE: "
        f"{np.sqrt(np.mean((body_arr[:, 3][test_idx_np]  - body_pred[test_idx_np])**2)):.4e} m/s"
    )
plot_fields(all_pts_np[idx_vel], vel_fields, output_dir=os.path.join(RUN_PATH, "inference"))

# ── Scalar (p, T) plots & RMSE ───────────────────────────────
idx_sc    = np.random.choice(vx_full.shape[0], min(50_000, vx_full.shape[0]), replace=False)
sc_fields = []
for tag, true_arr, pred_slice, unit in [
    ("p [Pa]", p_full[:, 3],  p_pred[n_inlet:], "Pa"),
    ("T [K]",  T_full[:, 3],  T_pred[n_inlet:], "K"),
]:
    sc_fields.append((tag, true_arr[idx_sc], pred_slice[idx_sc]))
    print(
        f"{tag}  train RMSE: "
        f"{np.sqrt(np.mean((true_arr[train_idx_np] - pred_slice[train_idx_np])**2)):.4e} {unit}"
        f"  test RMSE: "
        f"{np.sqrt(np.mean((true_arr[test_idx_np]  - pred_slice[test_idx_np])**2)):.4e} {unit}"
    )
plot_fields(vx_full[:, :3][idx_sc], sc_fields, output_dir=os.path.join(RUN_PATH, "inference"))

_log_handle.close()
