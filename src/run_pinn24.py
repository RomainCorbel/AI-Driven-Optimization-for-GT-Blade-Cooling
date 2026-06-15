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
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

T_INLET  = 298.15      # [K]   inlet temperature BC (overridden from CFD)
T_WALL   = 298.15      # [K]   wall temperature BC  (overridden from CFD min)
P_OUTLET = 0.835       # [bar] outlet static pressure BC (overridden from CFD mean)

M  = 28.96e-3          # [kg/mol] molar mass of air
R  = 8.314             # [J/mol/K] universal gas constant
K  = 2.61e-2           # [W/m/K] thermal conductivity
CP = 1.00e3            # [J/kg/K] specific heat

T_REF_SUTH = 278.15    # [K]   Sutherland reference temperature
MU_REF     = 1.716e-5  # [Pa·s] dynamic viscosity at T_REF_SUTH
S_SUTH     = 110.4     # [K]   Sutherland constant

rho   = (P_OUTLET * 1e5) * M / (R * T_REF_SUTH)  # [kg/m³] approximate air density

# Normalization anchors for parametric inputs (AR, e/Dh, P/e, alpha, Re).
# Maps typical rib-cooled duct values to O(1). Adjust when full 50-DP range is known.
PARAM_NAMES = ["AR",  "e/Dh", "P/e", "alpha", "Re"   ]
PARAM_MEANS = torch.tensor([ 6.0,   0.15,  10.0,   60.0,  75000.0], dtype=torch.float64)
PARAM_STDS  = torch.tensor([ 5.0,   0.10,   7.0,   30.0,  70000.0], dtype=torch.float64)

# Physical scales for residual normalization — computed from coord_std after data loading.
# DELTA_T, T_WALL, T_INLET_ND (globals) are set after data loading and used in losses/dynamic_viscosity_tilde.

# ═══════════════════════════════════════════════════════════════
# GEOMETRY SAMPLING  (labels: 0=interior, 1=inlet, 2=outlet, 3=wall)
# ═══════════════════════════════════════════════════════════════

def _submesh_pts(mesh, face_ids, n):
    pts, _ = trimesh.sample.sample_surface(
        mesh.submesh([face_ids], only_watertight=False)[0], n
    )
    return torch.tensor(pts)


def build_wall_pool(mesh, n_pool=200_000):
    pts, _ = trimesh.sample.sample_surface(mesh, n_pool * 3)
    x    = pts[:, 0]
    mask = (
        (np.abs(x - X_STL_MIN_MM) > 0.1) &
        (np.abs(x - X_STL_MAX_MM) > 0.1) &
        (x >= X_STL_MIN_MM) &
        (x <= X_STL_MAX_MM)
    )
    wall_pts = torch.tensor(pts[mask], dtype=torch.float64)
    if len(wall_pts) > n_pool:
        wall_pts = wall_pts[torch.randperm(len(wall_pts))[:n_pool]]
    return wall_pts


def sample_wall(wall_pool, n):
    idx = torch.randperm(len(wall_pool))[:n]
    pts = wall_pool[idx]
    return pts, 3 * torch.ones(len(pts), dtype=torch.int64)


def build_volume_pool(mesh, n_pool=800_000, wall_fraction=0.5,
                      delta_horiz_mm=1.0, delta_vert_mm=1.0, vert_frac=0.7):
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
    pts = vol_pool[idx]
    return pts, torch.zeros(len(pts), dtype=torch.int64)


def sample_collocation(vol_pool, wall_pool, n_vol, n_wall):
    p, l_vol  = sample_volume(vol_pool, n_vol)
    w, l_wall = sample_wall(wall_pool, n_wall)
    pts = torch.cat([p, w])
    lbl = torch.cat([l_vol, l_wall])
    perm = torch.randperm(pts.size(0))
    return pts[perm], lbl[perm]


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def dynamic_viscosity_tilde(T_tilde):
    """Return mu/MU_REF (dimensionless) given dimensionless temperature T̃."""
    T_K = T_tilde * DELTA_T + T_WALL
    mu  = MU_REF * (T_K.abs() / T_REF_SUTH) ** 1.5 * (T_REF_SUTH + S_SUTH) / (T_K.abs() + S_SUTH)
    return mu / MU_REF


def _grad(f, pts):
    return torch.autograd.grad(
        f, pts, torch.ones_like(f), retain_graph=True, create_graph=True
    )[0]


def _grad2(f, pts, dim):
    g = _grad(f, pts)[:, dim:dim+1]
    return torch.autograd.grad(
        g, pts, torch.ones_like(g), retain_graph=True, create_graph=True
    )[0][:, dim:dim+1]


def compute_derivatives(fields, pts_xyz):
    """Compute spatial derivatives. pts_xyz is [N,3] (x,y,z only, requires_grad=True)."""
    vx = fields[:, 0:1]
    vy = fields[:, 1:2]
    vz = fields[:, 2:3]
    p  = fields[:, 3:4]
    T  = fields[:, 4:5]

    gvx = _grad(vx, pts_xyz); vx_x, vx_y, vx_z = gvx[:, 0:1], gvx[:, 1:2], gvx[:, 2:3]
    vx_xx = _grad2(vx, pts_xyz, 0); vx_yy = _grad2(vx, pts_xyz, 1); vx_zz = _grad2(vx, pts_xyz, 2)

    gvy = _grad(vy, pts_xyz); vy_x, vy_y, vy_z = gvy[:, 0:1], gvy[:, 1:2], gvy[:, 2:3]
    vy_xx = _grad2(vy, pts_xyz, 0); vy_yy = _grad2(vy, pts_xyz, 1); vy_zz = _grad2(vy, pts_xyz, 2)

    gvz = _grad(vz, pts_xyz); vz_x, vz_y, vz_z = gvz[:, 0:1], gvz[:, 1:2], gvz[:, 2:3]
    vz_xx = _grad2(vz, pts_xyz, 0); vz_yy = _grad2(vz, pts_xyz, 1); vz_zz = _grad2(vz, pts_xyz, 2)

    gp = _grad(p, pts_xyz); p_x, p_y, p_z = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]
    gT = _grad(T, pts_xyz); T_x, T_y, T_z = gT[:, 0:1], gT[:, 1:2], gT[:, 2:3]
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
    labels, p_out_nd, vx_in, vy_in, vz_in,
    vx_var, vy_var, vz_var, p_var, T_var,
):
    interior = labels == 0
    inlet    = labels == 1
    outlet   = labels == 2
    wall     = labels == 3
    mu_tilde = dynamic_viscosity_tilde(T)

    visc_coeff  = (nu / V_IN)     # ν/V_in  [m]
    therm_coeff = (alpha / V_IN)  # α/V_in  [m]

    # ── PDE losses ───────────────────────────────────────────
    l_div = torch.mean(((vx_x + vy_y + vz_z) / DIV_SCALE) ** 2)

    def navier_stokes(u_x, u_y, u_z, u_xx, u_yy, u_zz, p_grad, mom_scale):
        advec = (vx[interior] * u_x[interior]
                 + vy[interior] * u_y[interior]
                 + vz[interior] * u_z[interior])
        lap   = u_xx[interior] + u_yy[interior] + u_zz[interior]
        res   = advec + p_grad[interior] - visc_coeff * mu_tilde[interior] * lap
        return torch.mean((res / mom_scale) ** 2)

    l_mom_x = navier_stokes(vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, p_x, MOM_SCALE_X)
    l_mom_y = navier_stokes(vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, p_y, MOM_SCALE_Y)
    l_mom_z = navier_stokes(vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_z, MOM_SCALE_Z)

    l_heat = torch.mean((
        vx[interior] * T_x[interior]
        + vy[interior] * T_y[interior]
        + vz[interior] * T_z[interior]
        - therm_coeff * (T_xx[interior] + T_yy[interior] + T_zz[interior])
    ) ** 2)

    # ── BC losses ────────────────────────────────────────────
    l_inlet_vx = torch.mean((vx[inlet].squeeze(1) - vx_in) ** 2) / vx_var
    l_inlet_vy = torch.mean((vy[inlet].squeeze(1) - vy_in) ** 2) / vy_var
    l_inlet_vz = torch.mean((vz[inlet].squeeze(1) - vz_in) ** 2) / vz_var
    l_inlet_T  = torch.mean((T[inlet] - T_INLET_ND) ** 2) / T_var  # 1.0 non-iso, 0.0 isothermal
    l_outlet_p = torch.mean((p[outlet] - p_out_nd) ** 2)          / p_var
    l_wall_vx  = torch.mean(vx[wall] ** 2)                        / vx_var
    l_wall_vy  = torch.mean(vy[wall] ** 2)                        / vy_var
    l_wall_vz  = torch.mean(vz[wall] ** 2)                        / vz_var
    l_wall_T   = torch.mean((T[wall] - 0.0) ** 2) / T_var

    return (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
        l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
    )


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FFNN(nn.Module):
    """Fully-connected network with Sin activations."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), Sin()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for idx, lin in enumerate(linears):
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return self.net(x)


class NormalizedPINN(nn.Module):
    """Z-scores inputs (coords + pre-normalized params); un-standardizes outputs.

    Input layout: [x, y, z, AR_nd, EDH_nd, PE_nd, ALPHA_nd, RE_nd] (8 dims).
    Wall sigmoid features s1, s2 are appended from x[:,1] (physical y) if enabled.
    coord_mean/coord_std cover all dims including params (params: mean=0, std=1).
    """

    def __init__(self, net, coord_mean, coord_std, out_mean, out_std,
                 wall_y1=None, wall_y2=None, wall_eps=0.002):
        super().__init__()
        self.net = net
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std",  coord_std)
        self.register_buffer("out_mean",   out_mean)
        self.register_buffer("out_std",    out_std)
        self.wall_y1  = wall_y1
        self.wall_y2  = wall_y2
        self.wall_eps = wall_eps

    def forward(self, x):
        # x[:,1] is always physical y regardless of extra columns
        if self.wall_y1 is not None and self.wall_y2 is not None:
            s1 = torch.sigmoid((x[:, 1:2] - self.wall_y1) / self.wall_eps)
            s2 = torch.sigmoid((x[:, 1:2] - self.wall_y2) / self.wall_eps)
            x = torch.cat([x, s1, s2], dim=1)
        x_norm   = (x - self.coord_mean) / self.coord_std
        y_norm   = self.net(x_norm)
        safe_std = torch.where(self.out_std == 0,
                               torch.ones_like(self.out_std), self.out_std)
        return y_norm * safe_std + self.out_mean


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_fields(pts, fields, output_dir, tag="", slice_frac=0.10):
    """One figure per field: top row = 3D scatter, bottom row = x-y cut at z=z_mid."""
    os.makedirs(output_dir, exist_ok=True)

    def to_np(x):
        if x is None:
            return None
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pts_np = to_np(pts)

    ranges     = pts_np.max(axis=0) - pts_np.min(axis=0)
    box_aspect = (ranges / ranges.max()).tolist()

    z_vals   = pts_np[:, 2]
    z_mid    = 0.5 * (z_vals.min() + z_vals.max())
    z_tol    = slice_frac * (z_vals.max() - z_vals.min())
    cut_mask = np.abs(z_vals - z_mid) < z_tol
    p_cut    = pts_np[cut_mask]
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
            ax.set_xlabel("X")
            ax.set_yticklabels([]); ax.set_ylabel("")
            ax.set_zticklabels([]); ax.set_zlabel("")
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
        fname = f"{name}{'_' + tag if tag else ''}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_point_clouds(title, path, datasets, wall_ys=None, unit="m"):
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(title, fontsize=10)

    ax3d = fig.add_subplot(141, projection="3d")
    for pts, color, s, alpha, label in datasets:
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, alpha=alpha, c=color, label=label)
    ax3d.set_xlabel(f"x [{unit}]"); ax3d.set_ylabel(f"y [{unit}]"); ax3d.set_zlabel(f"z [{unit}]")
    ax3d.set_title("3D view"); ax3d.legend(markerscale=8, fontsize=7)
    if wall_ys:
        _xr = [min(d[0][:, 0].min() for d in datasets), max(d[0][:, 0].max() for d in datasets)]
        _zr = [min(d[0][:, 2].min() for d in datasets), max(d[0][:, 2].max() for d in datasets)]
        for wy in wall_ys:
            ax3d.plot(_xr, [wy, wy], [_zr[0], _zr[0]], c="orange", lw=1.0, ls="--")
            ax3d.plot(_xr, [wy, wy], [_zr[1], _zr[1]], c="orange", lw=1.0, ls="--")

    for i, (xl, yl, xi, yi) in enumerate(
            [(f"x [{unit}]", f"y [{unit}]", 0, 1),
             (f"x [{unit}]", f"z [{unit}]", 0, 2),
             (f"y [{unit}]", f"z [{unit}]", 1, 2)], start=2):
        ax = fig.add_subplot(1, 4, i)
        for pts, color, s, alpha, _ in datasets:
            ax.scatter(pts[:, xi], pts[:, yi], s=s, alpha=alpha, c=color)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}")
        if wall_ys:
            for wy in wall_ys:
                if yi == 1:
                    ax.axhline(wy, color="orange", lw=1.2, ls="--")
                elif xi == 1:
                    ax.axvline(wy, color="orange", lw=1.2, ls="--")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{os.path.basename(path)}  → {path}")


# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="PINN24 — GT blade cooling, parametric inputs (x,y,z,AR,e/Dh,P/e,alpha,Re)")

# I/O
parser.add_argument("--folder",     default="dp00",   help="Data subfolder under preProcessedData/with_T/")
parser.add_argument("--run-path",   default=None,     help="Output dir (default: auto-derived)")
parser.add_argument("--project",    default="PINN24", help="W&B project name")
parser.add_argument("--device",     default="cuda",   help="Torch device")
parser.add_argument("--debug",      action="store_false", help="Skip W&B logging")

# Architecture
parser.add_argument("--hidden-dim", type=int,   default=20)
parser.add_argument("--n-layers",   type=int,   default=4)

# Training
parser.add_argument("--epochs",     type=int,   default=6_000)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--lr",         type=float, default=3e-3)
parser.add_argument("--lr-end",     type=float, default=1e-4)

# Sampling
parser.add_argument("--n-train",       type=int,   default=5_000)
parser.add_argument("--n-test",        type=int,   default=10_000)
parser.add_argument("--n-sup",         type=int,   default=500)
parser.add_argument("--n-snapshots",   type=int,   default=20_000)
parser.add_argument("--wall-fraction",  type=float, default=0.5)
parser.add_argument("--n-pool",         type=int,   default=500_000)
parser.add_argument("--pool-frac-vol",  type=float, default=0.5)
parser.add_argument("--pool-frac-wall", type=float, default=0.5)
parser.add_argument("--delta-mm-horiz", type=float, default=1.0)
parser.add_argument("--delta-mm-vert",  type=float, default=20.0)
parser.add_argument("--w-pde",   type=float, default=1.0)
parser.add_argument("--w-bc",    type=float, default=1.0)
parser.add_argument("--w-sup",   type=float, default=1.0)
parser.add_argument("--w-sup-p", type=float, default=1.0)
parser.add_argument("--outlet-y-min-frac", type=float, default=0.0)
parser.add_argument("--inlet-y-max-frac",  type=float, default=1.0)

# Internal dividing walls
parser.add_argument("--wall-y1", type=float, default=-1.0)
parser.add_argument("--wall-y2", type=float, default=-1.0)

# Parametric inputs — geometric and flow parameters for this DP
parser.add_argument("--ar",        type=float, default=7.5,     help="Aspect ratio AR = W/H")
parser.add_argument("--e-dh",      type=float, default=0.074,   help="Relative rib height e/Dh")
parser.add_argument("--p-e",       type=float, default=8.0,     help="Rib pitch-to-height ratio P/e")
parser.add_argument("--alpha-rib", type=float, default=60.0,    help="Rib angle alpha [degrees]")
parser.add_argument("--re",        type=float, default=100000.0, help="Reynolds number Re")

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

N_TEST              = args.n_test
N_TOTAL_TRAIN       = args.n_train
N_SUP               = args.n_sup
N_POINT_SNAPSHOTS   = args.n_snapshots
WALL_FRACTION       = args.wall_fraction
DELTA_HORIZ_MM      = args.delta_mm_horiz
DELTA_VERT_MM       = args.delta_mm_vert
W_PDE   = args.w_pde
W_BC    = args.w_bc
W_SUP   = args.w_sup
W_SUP_P = args.w_sup_p
OUTLET_Y_MIN_FRAC   = args.outlet_y_min_frac
INLET_Y_MAX_FRAC    = args.inlet_y_max_frac
N_TOTAL_POOL        = args.n_pool
POOL_FRAC_VOL       = args.pool_frac_vol
POOL_FRAC_WALL      = args.pool_frac_wall

# Wall-distance features
WALL_Y1  = args.wall_y1 if args.wall_y1 > 0 else None
WALL_Y2  = args.wall_y2 if args.wall_y2 > 0 else None
WALL_EPS = 0.002
USE_WALL_FEATURES = (WALL_Y1 is not None) and (WALL_Y2 is not None)

# Parametric inputs for this DP
AR        = args.ar
E_DH      = args.e_dh
P_E       = args.p_e
ALPHA_RIB = args.alpha_rib
RE_PARAM  = args.re

# Pre-normalize params to O(1) using fixed anchors
PARAMS_RAW = torch.tensor([AR, E_DH, P_E, ALPHA_RIB, RE_PARAM], dtype=torch.float64)
params_nd  = (PARAMS_RAW - PARAM_MEANS) / PARAM_STDS   # [5], O(1), no requires_grad

# Input dimensionality: 3 coords + 5 params + optional 2 wall features
N_PARAMS = 5
IN_DIM   = 3 + N_PARAMS + (2 if USE_WALL_FEATURES else 0)

DATA_DIR = f"./preProcessedData/with_T/{FOLDER}/"

if args.run_path:
    RUN_PATH = args.run_path
else:
    _wall_tag = f"_wy1{WALL_Y1:.3f}_wy2{WALL_Y2:.3f}" if USE_WALL_FEATURES else ""
    _name = (f"f{FOLDER}"
             f"_AR{AR}_eDh{E_DH}_Pe{P_E}_al{ALPHA_RIB}_Re{int(RE_PARAM)}"
             f"_h{HIDDEN_DIM}_l{N_LAYERS}"
             f"_e{EPOCHS}"
             f"_lr{LR:.0e}_lrend{LR_END:.0e}"
             f"_ntrain{N_TOTAL_TRAIN}"
             f"_sup{N_SUP}"
             f"_wpde{W_PDE}_wbc{W_BC}_wsup{W_SUP}_wsupp{W_SUP_P}"
             f"{_wall_tag}"
             f"_s{SEED}")
    RUN_PATH = f"../pinn24_runs/{_name}"

os.makedirs(RUN_PATH, exist_ok=True)
_log_path = os.path.join(RUN_PATH, "training.log")

class _Tee(io.TextIOBase):
    def __init__(self, stream, logfile):
        self._stream  = stream
        self._logfile = logfile
    def write(self, s):
        self._stream.write(s)
        self._logfile.write(s)
        return len(s)
    def flush(self):
        self._stream.flush()
        self._logfile.flush()

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
print(f"Params   : AR={AR}  e/Dh={E_DH}  P/e={P_E}  alpha={ALPHA_RIB}°  Re={RE_PARAM:.0f}")
print(f"           params_nd = {params_nd.tolist()}")
if USE_WALL_FEATURES:
    print(f"Wall features : ON  wall_y1={WALL_Y1:.4f} m  wall_y2={WALL_Y2:.4f} m  in_dim={IN_DIM}")
else:
    print(f"Wall features : OFF  in_dim={IN_DIM}")

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
            "wall_y1": WALL_Y1, "wall_y2": WALL_Y2,
            "wall_eps": WALL_EPS, "use_wall_features": USE_WALL_FEATURES,
            "AR": AR, "e_dh": E_DH, "p_e": P_E, "alpha_rib": ALPHA_RIB, "Re": RE_PARAM,
        },
    )

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_default_device(DEVICE)
print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# DATA
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
    print(f"Inlet BC y-filter : y <= {_y_max_cut:.4f} m  (frac={INLET_Y_MAX_FRAC}, kept {_inlet_mask.sum()} / {_inlet_mask.size} pts)")

_temp_all   = np.load(DATA_DIR + "temp.npy")
_x_min_temp = float(_temp_all[:, 0].min())
_temp_face  = _temp_all[_temp_all[:, 0] == _x_min_temp]
if INLET_Y_MAX_FRAC < 1.0:
    _y_max_temp = float(_temp_all[:, 1].max())
    _temp_face  = _temp_face[_temp_face[:, 1] <= INLET_Y_MAX_FRAC * _y_max_temp]
T_inlet_bc  = float(_temp_face[:, 3].mean())
print(f"Inlet T from CFD  : mean={T_inlet_bc:.2f} K")

_vx_raw = np.load(DATA_DIR + "vel_x.npy")
cfd_pts = torch.tensor(_vx_raw[:, :3])
cfd_vx  = torch.tensor(_vx_raw[:, 3])
cfd_vy  = torch.tensor(np.load(DATA_DIR + "vel_y.npy")[:, 3])
cfd_vz  = torch.tensor(np.load(DATA_DIR + "vel_z.npy")[:, 3])
cfd_p   = torch.tensor(np.load(DATA_DIR + "press.npy")[:, 3])
cfd_T   = torch.tensor(_temp_all[:, 3])

T_WALL = float(cfd_T.min())
print(f"T_WALL (from CFD min) = {T_WALL:.2f} K")

# ── Domain bounds ──────────────────────────────────────────────
X_MIN_MM = float(cfd_pts[:, 0].min()) * 1000
X_MAX_MM = float(cfd_pts[:, 0].max()) * 1000
STL_INLET_BUFFER_MM = 0.0 if FOLDER == "dp11" else 420.0
X_STL_MIN_MM = X_MIN_MM + STL_INLET_BUFFER_MM
X_STL_MAX_MM = X_MAX_MM + STL_INLET_BUFFER_MM
print(f"Physical domain  x : [{X_MIN_MM:.2f}, {X_MAX_MM:.2f}] mm"
      f"  =  [{X_MIN_MM/1000:.4f}, {X_MAX_MM/1000:.4f}] m")

# ── Outlet BC points ───────────────────────────────────────────
_out_mask  = cfd_pts[:, 0] > (X_MAX_MM / 1000 - 0.005)
if OUTLET_Y_MIN_FRAC > 0.0:
    _y_max     = float(cfd_pts[:, 1].max())
    _y_min_out = OUTLET_Y_MIN_FRAC * _y_max
    _out_mask  = _out_mask & (cfd_pts[:, 1] >= _y_min_out)
    print(f"Outlet BC y-filter: y >= {_y_min_out:.4f} m  (frac={OUTLET_Y_MIN_FRAC}, y_max={_y_max:.4f})")

# ── Output non-dimensionalization scales ───────────────────────
V_IN    = float(torch.tensor(_inlet_vx_raw[:, 3]).abs().mean())
nu      = MU_REF / rho
alpha   = K / (rho * CP)
P_DYN   = rho * V_IN**2
P_REF   = float(cfd_p[_out_mask].mean())
DELTA_T     = max(T_inlet_bc - T_WALL, 1.0)   # clamped to ≥1 K: prevents /0 for isothermal
T_INLET_ND  = (T_inlet_bc - T_WALL) / DELTA_T  # = 1.0 non-iso, = 0.0 isothermal
P_OUTLET_ND = 0.0
print(f"Output non-dim   : V_IN={V_IN:.3f} m/s  P_DYN={P_DYN:.2f} Pa  P_REF={P_REF:.0f} Pa ({P_REF/1e5:.4f} bar)  DELTA_T={DELTA_T:.2f} K  T_INLET_ND={T_INLET_ND:.3f}")
print(f"                   ν/V_in={nu/V_IN:.2e} m  α/V_in={alpha/V_IN:.2e} m")

# Non-dim outputs; coords unchanged
cfd_pts_nd = cfd_pts
cfd_vx_nd  = cfd_vx / V_IN
cfd_vy_nd  = cfd_vy / V_IN
cfd_vz_nd  = cfd_vz / V_IN
cfd_p_nd   = (cfd_p - P_REF) / P_DYN
cfd_T_nd   = (cfd_T - T_WALL) / DELTA_T   # = 0 everywhere for isothermal (DELTA_T clamped)

inlet_pool = (
    torch.tensor(_inlet_vx_raw[:, :3]),
    torch.tensor(_inlet_vx_raw[:, 3]) / V_IN,
    torch.tensor(_inlet_vy_raw[:, 3]) / V_IN,
    torch.tensor(_inlet_vz_raw[:, 3]) / V_IN,
)
print(f"Inlet pool       : {len(inlet_pool[0])} CFD points, sampling {n_inlet_train} per epoch")

outlet_pool = cfd_pts_nd[_out_mask]
print(f"Outlet pool      : {len(outlet_pool)} CFD points near x={X_MAX_MM/1000:.4f} m, sampling {n_outlet_train} per epoch")

coord_mean = cfd_pts_nd.mean(dim=0)
coord_std  = cfd_pts_nd.std(dim=0)
out_mean   = torch.stack([cfd_vx_nd.mean(), cfd_vy_nd.mean(), cfd_vz_nd.mean(),
                           cfd_p_nd.mean(), cfd_T_nd.mean()])
out_std    = torch.stack([cfd_vx_nd.std(),  cfd_vy_nd.std(),  cfd_vz_nd.std(),
                           cfd_p_nd.std(),  cfd_T_nd.std()])
out_std    = out_std.clamp(min=1e-3)

# ── Extend coord stats: coords (data-derived) + params (fixed 0/1) + wall features ──
# params are already pre-normalized to O(1), so mean=0, std=1 for those dims.
_param_mean_ext = torch.zeros(N_PARAMS, dtype=torch.float64)
_param_std_ext  = torch.ones(N_PARAMS,  dtype=torch.float64)

if USE_WALL_FEATURES:
    _y      = cfd_pts[:, 1]
    _s1     = torch.sigmoid((_y - WALL_Y1) / WALL_EPS)
    _s2     = torch.sigmoid((_y - WALL_Y2) / WALL_EPS)
    _s1_mean, _s1_std = float(_s1.mean()), float(_s1.std().clamp(min=1e-3))
    _s2_mean, _s2_std = float(_s2.mean()), float(_s2.std().clamp(min=1e-3))
    coord_mean_net = torch.cat([coord_mean, _param_mean_ext,
                                 torch.tensor([_s1_mean, _s2_mean], dtype=torch.float64)])
    coord_std_net  = torch.cat([coord_std,  _param_std_ext,
                                 torch.tensor([_s1_std,  _s2_std],  dtype=torch.float64)])
    print(f"Wall sigmoid features (eps={WALL_EPS} m):"
          f"  s1 mean={_s1_mean:.3f} std={_s1_std:.3f}"
          f"  s2 mean={_s2_mean:.3f} std={_s2_std:.3f}")
else:
    coord_mean_net = torch.cat([coord_mean, _param_mean_ext])
    coord_std_net  = torch.cat([coord_std,  _param_std_ext])

# ── Anisotropic PDE scales ─────────────────────────────────────
MOM_SCALE_X = 1.0 / float(coord_std[0])
MOM_SCALE_Y = 1.0 / float(coord_std[1])
MOM_SCALE_Z = 1.0 / float(coord_std[2])
DIV_SCALE   = float(max(out_std[0] / coord_std[0],
                        out_std[1] / coord_std[1],
                        out_std[2] / coord_std[2]))
print(f"MOM_SCALE  : X={MOM_SCALE_X:.1f}  Y={MOM_SCALE_Y:.1f}  Z={MOM_SCALE_Z:.1f}")
print(f"DIV_SCALE  : {DIV_SCALE:.1f}")
print(f"coord_mean : {coord_mean.tolist()}")
print(f"coord_std  : {coord_std.tolist()}")
print(f"out_mean   : {out_mean.tolist()}")
print(f"out_std    : {out_std.tolist()}")

perm      = torch.randperm(cfd_pts_nd.shape[0])
test_idx  = perm[:N_TEST]
train_idx = perm[N_TEST:]

test_pts  = cfd_pts_nd[test_idx];  train_pts = cfd_pts_nd[train_idx]
test_vx   = cfd_vx_nd[test_idx];   train_vx  = cfd_vx_nd[train_idx]
test_vy   = cfd_vy_nd[test_idx];   train_vy  = cfd_vy_nd[train_idx]
test_vz   = cfd_vz_nd[test_idx];   train_vz  = cfd_vz_nd[train_idx]
test_p    = cfd_p_nd[test_idx];    train_p   = cfd_p_nd[train_idx]
test_T    = cfd_T_nd[test_idx];    train_T   = cfd_T_nd[train_idx]

if N_SUP > 0:
    _fixed_sup_idx = torch.randint(0, train_pts.shape[0], (N_SUP,))
    fixed_sup_pts  = train_pts[_fixed_sup_idx]
    fixed_sup_vx   = train_vx[_fixed_sup_idx]
    fixed_sup_vy   = train_vy[_fixed_sup_idx]
    fixed_sup_vz   = train_vz[_fixed_sup_idx]
    fixed_sup_p    = train_p[_fixed_sup_idx]
    fixed_sup_T    = train_T[_fixed_sup_idx]
    print(f"Fixed supervised : {N_SUP} points sampled once (from {train_pts.shape[0]} training pts)")

def _var(raw_var, floor, name):
    v = raw_var.clamp(min=floor)
    if raw_var.item() < floor:
        print(f"  [variance floor hit] {name}: raw={raw_var.item():.2e} → using floor={floor:.2e}")
    return v

vx_raw = torch.maximum(train_vx.var(), inlet_pool[1].var())
vy_raw = torch.maximum(train_vy.var(), inlet_pool[2].var())
vz_raw = torch.maximum(train_vz.var(), inlet_pool[3].var())
vx_var = _var(vx_raw, 1e-6, "vx")
vy_var = _var(vy_raw, 1e-6, "vy")
vz_var = _var(vz_raw, 1e-6, "vz")
p_var  = _var(train_p.var(), 1e-6, "p")
T_var  = _var(train_T.var(), 1e-2, "T")   # floor 1e-2: avoids /0 for isothermal (T̃=0 everywhere)

print(f"\nField variances (used for loss normalization):")
print(f"  vx: {vx_var.item():.4e}  vy: {vy_var.item():.4e}  vz: {vz_var.item():.4e}")
print(f"  p:  {p_var.item():.4e}   T:  {T_var.item():.4e}")

stl_files = glob.glob(DATA_DIR + "*.stl")
print(f"Loading STL: {stl_files[0]}")
mesh = trimesh.load(stl_files[0])
_stl_xmin = float(mesh.vertices[:, 0].min())
_stl_xmax = float(mesh.vertices[:, 0].max())
print(f"STL mesh         x : [{_stl_xmin:.2f}, {_stl_xmax:.2f}] mm"
      f"  (inlet buffer = {STL_INLET_BUFFER_MM:.0f} mm,"
      f"  physical in STL: [{X_STL_MIN_MM:.1f}, {X_STL_MAX_MM:.1f}] mm)")

print("Building volume pool...")
vol_pool, _n_uniform, _n_near_horiz = build_volume_pool(
    mesh, n_pool=int(POOL_FRAC_VOL * N_TOTAL_POOL),
    wall_fraction=WALL_FRACTION,
    delta_horiz_mm=DELTA_HORIZ_MM, delta_vert_mm=DELTA_VERT_MM)
print("Building wall pool...")
wall_pool = build_wall_pool(mesh, n_pool=int(POOL_FRAC_WALL * N_TOTAL_POOL))

_pool_np    = vol_pool.cpu().numpy()
_uni        = _pool_np[:_n_uniform]
_near_horiz = _pool_np[_n_uniform : _n_uniform + _n_near_horiz]
_near_vert  = _pool_np[_n_uniform + _n_near_horiz :]
_sub = 20_000
_ui  = np.random.choice(len(_uni),        min(_sub, len(_uni)),        replace=False)
_hi  = np.random.choice(len(_near_horiz), min(_sub, len(_near_horiz)), replace=False)
_vi  = np.random.choice(len(_near_vert),  min(_sub, len(_near_vert)),  replace=False)
plot_point_clouds(
    title=(f"Volume pool  —  uniform (blue, n={len(_uni)})  "
           f"near-horiz (red, n={len(_near_horiz)})  near-vert (green, n={len(_near_vert)})  "
           f"δ_horiz={DELTA_HORIZ_MM} mm  δ_vert={DELTA_VERT_MM} mm"),
    path=os.path.join(RUN_PATH, "viz_volume_pool.png"),
    datasets=[
        (_uni[_ui],        "steelblue", 0.5, 0.2, "uniform"),
        (_near_horiz[_hi], "crimson",   0.5, 0.3, "near-horiz"),
        (_near_vert[_vi],  "limegreen", 0.5, 0.3, "near-vert"),
    ],
    unit="mm",
)

_pool_cfd_m = _pool_np.copy().astype(float)
_pool_cfd_m[:, 0] -= STL_INLET_BUFFER_MM
_pool_cfd_m /= 1000.0
_bg_n = 20_000
_bg_i = np.random.choice(len(_pool_cfd_m), min(_bg_n, len(_pool_cfd_m)), replace=False)
_bg   = _pool_cfd_m[_bg_i]

_in_np  = inlet_pool[0].cpu().numpy()
_out_np = outlet_pool.cpu().numpy()
_wall_suptitle = f"  |  walls y={WALL_Y1:.4f} / {WALL_Y2:.4f} m" if USE_WALL_FEATURES else ""
plot_point_clouds(
    title=(f"BC geometry  —  vol pool (grey)  inlet (green, n={len(_in_np)})  outlet (red, n={len(_out_np)})" + _wall_suptitle),
    path=os.path.join(RUN_PATH, "viz_geometry_bc.png"),
    datasets=[
        (_bg,     "grey",      0.5, 0.15, "vol pool"),
        (_in_np,  "limegreen", 4,   1.0,  "inlet"),
        (_out_np, "crimson",   4,   1.0,  "outlet"),
    ],
    wall_ys=[WALL_Y1, WALL_Y2] if USE_WALL_FEATURES else None,
)

_sup_np = fixed_sup_pts.cpu().numpy() if N_SUP > 0 else None
_sv_datasets = [(_bg, "grey", 0.5, 0.15, "vol pool")]
if N_SUP > 0:
    _sv_datasets.append((_sup_np, "red", 1.5, 1.0, "supervised"))
plot_point_clouds(
    title=(f"Supervised pts (red, n={N_SUP})  vol pool (grey)" if N_SUP > 0 else "Vol pool — no supervised pts"),
    path=os.path.join(RUN_PATH, "viz_supervised_points.png"),
    datasets=_sv_datasets,
)

_snap_n   = min(N_POINT_SNAPSHOTS, cfd_pts_nd.shape[0])
_snap_idx = np.random.choice(cfd_pts_nd.shape[0], _snap_n, replace=False)
snap_pts  = cfd_pts_nd[_snap_idx]
snap_vx   = cfd_vx_nd[_snap_idx]
snap_vy   = cfd_vy_nd[_snap_idx]
snap_vz   = cfd_vz_nd[_snap_idx]
snap_p    = cfd_p_nd[_snap_idx]
snap_T    = cfd_T_nd[_snap_idx]

# ═══════════════════════════════════════════════════════════════
# PARAMETRIC INPUT HELPER
# ═══════════════════════════════════════════════════════════════

def with_params(pts_xyz):
    """Append pre-normalized param vector to coordinate tensor [N,3] → [N,8].
    The param row is detached so autograd flows only through pts_xyz[:,:3].
    Moves params to the same device as pts_xyz (handles CPU inference / CUDA training).
    """
    row = params_nd.detach().to(pts_xyz.device).unsqueeze(0).expand(pts_xyz.shape[0], -1)
    return torch.cat([pts_xyz, row], dim=1)

# ═══════════════════════════════════════════════════════════════
# MODEL + OPTIMIZER
# ═══════════════════════════════════════════════════════════════

net   = FFNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS).to(DEVICE).double()
model = NormalizedPINN(net, coord_mean_net, coord_std_net, out_mean, out_std,
                       wall_y1=WALL_Y1 if WALL_Y1 else None,
                       wall_y2=WALL_Y2 if WALL_Y2 else None,
                       wall_eps=WALL_EPS)

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
  POINT SETUP SUMMARY
{_sep}

  POOLS  (built once — N_TOTAL_POOL={N_TOTAL_POOL:,}, vol {POOL_FRAC_VOL:.0%} / wall {POOL_FRAC_WALL:.0%} — mm, STL frame)
  {_thin}
  Volume pool            {len(vol_pool):>10,} pts  total
    ├─ uniform                      {_n_uniform:>10,} pts  ({100*_n_uniform//len(vol_pool):2d}%)
    ├─ near-horiz  δ ≤ {DELTA_HORIZ_MM:5.1f} mm  |nz|≥0.7  {_n_near_horiz:>10,} pts  ({100*_n_near_horiz//len(vol_pool):2d}%)
    └─ near-vert   δ ≤ {DELTA_VERT_MM:5.1f} mm  |nz|<0.7  {_n_near_vert_cnt:>10,} pts  ({100*_n_near_vert_cnt//len(vol_pool):2d}%)
  Wall pool              {len(wall_pool):>10,} pts  (surface, STL frame)
  Inlet pool             {len(inlet_pool[0]):>10,} pts
  Outlet pool            {len(outlet_pool):>10,} pts
{_sep}

  N_TOTAL_TRAIN = {N_TOTAL_TRAIN:,} pts per forward pass, split as:
    FRAC_VOL={FRAC_VOL:.0%}  FRAC_WALL={FRAC_WALL:.0%}  FRAC_INLET={FRAC_INLET:.0%}  FRAC_OUTLET={FRAC_OUTLET:.0%}
  {_thin}
  Volume interior  (label=0)  PDE residuals    {n_vol_train:>8,} pts  ({int(FRAC_VOL   *100):2d}%)
  Wall BC          (label=3)  no-slip + T      {n_wall_train:>8,} pts  ({int(FRAC_WALL  *100):2d}%)
  Inlet BC         (label=1)  vx,vy,vz,T       {n_inlet_train:>8,} pts  ({int(FRAC_INLET *100):2d}%)
  Outlet BC        (label=2)  pressure          {n_outlet_train:>8,} pts  ({int(FRAC_OUTLET*100):2d}%)
  Supervised CFD             data loss          {N_SUP:>8,} pts  ({_sup_str})

""")
start = time.time()

for epoch in range(EPOCHS):
    coll_pts, coll_lbls = sample_collocation(
        vol_pool, wall_pool, n_vol_train, n_wall_train
    )
    coll_pts = coll_pts.clone()
    coll_pts[:, 0] = coll_pts[:, 0] - STL_INLET_BUFFER_MM
    coll_pts = coll_pts / 1000   # mm → m

    _in_idx  = torch.randperm(len(inlet_pool[0]))[:n_inlet_train]
    inlet_pts   = inlet_pool[0][_in_idx]
    vx_inlet_bc = inlet_pool[1][_in_idx]
    vy_inlet_bc = inlet_pool[2][_in_idx]
    vz_inlet_bc = inlet_pool[3][_in_idx]
    _out_idx = torch.randperm(len(outlet_pool))[:n_outlet_train]
    outlet_pts  = outlet_pool[_out_idx]

    # Split: xyz requires grad (for PDE), params appended without grad
    pts_xyz = torch.cat([coll_pts, inlet_pts, outlet_pts])
    pts_xyz.requires_grad_(True)
    lbls = torch.cat([coll_lbls,
                       torch.ones(inlet_pts.shape[0],      dtype=torch.long),
                       2 * torch.ones(outlet_pts.shape[0], dtype=torch.long)])

    model.train()
    opt_model.zero_grad()

    fields = model(with_params(pts_xyz))
    derivs = compute_derivatives(fields, pts_xyz)  # grad w.r.t. xyz only
    (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
        l_wall_vx,  l_wall_vy,  l_wall_vz,  l_wall_T,
    ) = compute_losses(
        *derivs, lbls,
        P_OUTLET_ND, vx_inlet_bc, vy_inlet_bc, vz_inlet_bc,
        vx_var, vy_var, vz_var, p_var, T_var,
    )

    if N_SUP > 0:
        sup_fields = model(with_params(fixed_sup_pts))
        l_sup_vx = torch.mean((sup_fields[:, 0] - fixed_sup_vx) ** 2) / vx_var
        l_sup_vy = torch.mean((sup_fields[:, 1] - fixed_sup_vy) ** 2) / vy_var
        l_sup_vz = torch.mean((sup_fields[:, 2] - fixed_sup_vz) ** 2) / vz_var
        l_sup_p  = torch.mean((sup_fields[:, 3] - fixed_sup_p)  ** 2) / p_var
        l_sup_T  = torch.mean((sup_fields[:, 4] - fixed_sup_T)  ** 2) / T_var
    else:
        z = torch.zeros(1, dtype=torch.float64, device=DEVICE).squeeze()
        l_sup_vx = l_sup_vy = l_sup_vz = l_sup_p = l_sup_T = z

    l_pde_total = l_div + l_mom_x + l_mom_y + l_mom_z + l_heat
    l_bc_total  = (l_inlet_vx + l_inlet_vy + l_inlet_vz + l_inlet_T + l_outlet_p
                   + l_wall_vx + l_wall_vy + l_wall_vz + l_wall_T)
    l_sup_total = l_sup_vx + l_sup_vy + l_sup_vz + l_sup_p + l_sup_T

    l_total = (W_PDE * l_pde_total
             + W_BC  * l_bc_total
             + W_SUP * (l_sup_total - l_sup_p)
             + W_SUP_P * l_sup_p)
    l_total.backward()
    opt_model.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        pred = model(with_params(test_pts))
    mse_vx    = torch.mean((pred[:, 0] - test_vx) ** 2)
    mse_vy    = torch.mean((pred[:, 1] - test_vy) ** 2)
    mse_vz    = torch.mean((pred[:, 2] - test_vz) ** 2)
    mse_p     = torch.mean((pred[:, 3] - test_p)  ** 2)
    mse_T     = torch.mean((pred[:, 4] - test_T)  ** 2)
    mse_total = mse_vx + mse_vy + mse_vz + mse_p + mse_T

    if epoch in plot_epochs:
        snap_dir = os.path.join(RUN_PATH, f"snap_{epoch + 1}_of_{EPOCHS}")
        with torch.no_grad():
            pred_snap = model(with_params(snap_pts))
        plot_fields(
            snap_pts.cpu().numpy(),
            [
                ("vx", snap_vx.cpu().numpy() * V_IN,                   pred_snap[:, 0].cpu().numpy() * V_IN),
                ("vy", snap_vy.cpu().numpy() * V_IN,                   pred_snap[:, 1].cpu().numpy() * V_IN),
                ("vz", snap_vz.cpu().numpy() * V_IN,                   pred_snap[:, 2].cpu().numpy() * V_IN),
                ("p",  (snap_p.cpu().numpy() * P_DYN + P_REF) / 1e5,  (pred_snap[:, 3].cpu().numpy() * P_DYN + P_REF) / 1e5),
                ("T",  snap_T.cpu().numpy() * DELTA_T + T_WALL,        pred_snap[:, 4].cpu().numpy() * DELTA_T + T_WALL),
            ],
            output_dir=snap_dir,
        )

    log = {
        "pde/divergence": safe_log10(l_div),
        "pde/momentum_x": safe_log10(l_mom_x),
        "pde/momentum_y": safe_log10(l_mom_y),
        "pde/momentum_z": safe_log10(l_mom_z),
        "pde/heat":       safe_log10(l_heat),
        "bc/inlet_vx":    safe_log10(l_inlet_vx),
        "bc/inlet_vy":    safe_log10(l_inlet_vy),
        "bc/inlet_vz":    safe_log10(l_inlet_vz),
        "bc/inlet_T":     safe_log10(l_inlet_T),
        "bc/outlet_p":    safe_log10(l_outlet_p),
        "bc/wall_vx":     safe_log10(l_wall_vx),
        "bc/wall_vy":     safe_log10(l_wall_vy),
        "bc/wall_vz":     safe_log10(l_wall_vz),
        "bc/wall_T":      safe_log10(l_wall_T),
        "sup/vx":         safe_log10(l_sup_vx),
        "sup/vy":         safe_log10(l_sup_vy),
        "sup/vz":         safe_log10(l_sup_vz),
        "sup/p":          safe_log10(l_sup_p),
        "sup/T":          safe_log10(l_sup_T),
        "loss/pde_total": safe_log10(l_pde_total),
        "loss/bc_total":  safe_log10(l_bc_total),
        "loss/sup_total": safe_log10(l_sup_total),
        "loss/total":     safe_log10(l_total),
        "eval/mse_vx":    safe_log10(mse_vx),
        "eval/mse_vy":    safe_log10(mse_vy),
        "eval/mse_vz":    safe_log10(mse_vz),
        "eval/mse_p":     safe_log10(mse_p),
        "eval/mse_T":     safe_log10(mse_T),
        "eval/mse_total": safe_log10(mse_total),
        "train/lr": scheduler.get_last_lr()[0],
    }
    log = {k: v for k, v in log.items() if not (isinstance(v, float) and np.isnan(v))}

    print(
        f"\n\n\n\n[{epoch+1:>5}/{EPOCHS}]\n"
        f"PDE   div={l_div.item():.3e}  mom_x={l_mom_x.item():.3e}  mom_y={l_mom_y.item():.3e}  mom_z={l_mom_z.item():.3e}  heat={l_heat.item():.3e}  | total={l_pde_total.item():.3e}\n"
        f"BC    inlet_vx={l_inlet_vx.item():.3e}  inlet_vy={l_inlet_vy.item():.3e}  inlet_vz={l_inlet_vz.item():.3e}  inlet_T={l_inlet_T.item():.3e}  outlet_p={l_outlet_p.item():.3e}\n"
        f"      wall_vx={l_wall_vx.item():.3e}   wall_vy={l_wall_vy.item():.3e}   wall_vz={l_wall_vz.item():.3e}   wall_T={l_wall_T.item():.3e}   | total={l_bc_total.item():.3e}\n"
        f"SUP   vx={l_sup_vx.item():.3e}  vy={l_sup_vy.item():.3e}  vz={l_sup_vz.item():.3e}  p={l_sup_p.item():.3e}  T={l_sup_T.item():.3e}  | total={l_sup_total.item():.3e}\n"
        f"LOSS  total={l_total.item():.3e}  (w_pde={W_PDE}  w_bc={W_BC}  w_sup={W_SUP}  w_sup_p={W_SUP_P})  lr={scheduler.get_last_lr()[0]:.3e}\n"
        f"MSE   vx={mse_vx.item():.3e}  vy={mse_vy.item():.3e}  vz={mse_vz.item():.3e}  p={mse_p.item():.3e}  T={mse_T.item():.3e}  | total={mse_total.item():.3e}"
    )

    if not DEBUG:
        wandb.log(log, step=epoch)

print(f"\nTraining done in {time.time() - start:.1f}s")
torch.save(model.state_dict(), os.path.join(RUN_PATH, "pinn_model.pt"))

# ═══════════════════════════════════════════════════════════════
# INFERENCE  (CPU, full domain)
# ═══════════════════════════════════════════════════════════════

torch.set_default_device("cpu")
net_inf   = FFNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS)
model_inf = NormalizedPINN(net_inf,
                            coord_mean_net.cpu(), coord_std_net.cpu(),
                            out_mean.cpu(),       out_std.cpu(),
                            wall_y1=WALL_Y1 if WALL_Y1 else None,
                            wall_y2=WALL_Y2 if WALL_Y2 else None,
                            wall_eps=WALL_EPS)
model_inf.load_state_dict(
    torch.load(os.path.join(RUN_PATH, "pinn_model.pt"),
               weights_only=True, map_location="cpu")
)
model_inf.eval()

pts_full_np    = cfd_pts.cpu().numpy()
vx_full_np     = cfd_vx.cpu().numpy()
vy_full_np     = cfd_vy.cpu().numpy()
vz_full_np     = cfd_vz.cpu().numpy()
p_full_np      = cfd_p.cpu().numpy()
T_full_np      = cfd_T.cpu().numpy()

_pts_tensor = torch.tensor(pts_full_np)

with torch.no_grad():
    pred_all = model_inf(with_params(_pts_tensor))

vx_pred = pred_all[:, 0].numpy() * V_IN
vy_pred = pred_all[:, 1].numpy() * V_IN
vz_pred = pred_all[:, 2].numpy() * V_IN
p_pred  = (pred_all[:, 3].numpy() * P_DYN + P_REF) / 1e5
T_pred  = pred_all[:, 4].numpy() * DELTA_T + T_WALL

train_idx_np = train_idx.cpu().numpy()
test_idx_np  = test_idx.cpu().numpy()

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

idx_plot  = np.random.choice(len(pts_full_np), min(100_000, len(pts_full_np)), replace=False)
all_fields = []
for true_vals, pred_arr, name in [
    (vx_full_np,       vx_pred, "vx"),
    (vy_full_np,       vy_pred, "vy"),
    (vz_full_np,       vz_pred, "vz"),
    (p_full_np / 1e5,  p_pred,  "p"),
    (T_full_np,        T_pred,  "T"),
]:
    all_fields.append((name, true_vals[idx_plot], pred_arr[idx_plot]))
    print(f"{name}  train RMSE: {rmse(true_vals[train_idx_np], pred_arr[train_idx_np]):.4e}"
          f"  test RMSE:  {rmse(true_vals[test_idx_np],  pred_arr[test_idx_np]):.4e}")
plot_fields(pts_full_np[idx_plot], all_fields, output_dir=os.path.join(RUN_PATH, "inference"))

_log_handle.close()
