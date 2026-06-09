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

T_INLET  = 298.15      # [K]   inlet temperature BC
T_WALL   = 298.15      # [K]   wall temperature BC
P_OUTLET = 0.835       # [bar] outlet static pressure BC

M  = 28.96e-3          # [kg/mol] molar mass of air
R  = 8.314             # [J/mol/K] universal gas constant
K  = 2.61e-2           # [W/m/K] thermal conductivity
CP = 1.00e3            # [J/kg/K] specific heat

T_REF_SUTH = 278.15    # [K]   Sutherland reference temperature
MU_REF     = 1.716e-5  # [Pa·s] dynamic viscosity at T_REF_SUTH
S_SUTH     = 110.4     # [K]   Sutherland constant

rho   = (P_OUTLET * 1e5) * M / (R * T_REF_SUTH)  # [kg/m³]
alpha = K / (rho * CP)                              # [m²/s] thermal diffusivity

# Physical scales for residual normalization
U_SCALE   = 50.0    # [m/s]  characteristic velocity
# MOM_SCALE_X/Y/Z and DIV_SCALE are anisotropic — computed from coord_std after data loading.

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


def sample_wall(mesh, n):
    # X_STL_MIN_MM / X_STL_MAX_MM are physical-domain bounds in STL coords (offset by buffer).
    pts, face_ids = trimesh.sample.sample_surface(mesh, n)
    x = pts[:, 0]
    mask = (
        (np.abs(x - X_STL_MIN_MM) > 0.1) &   # not the inlet face
        (np.abs(x - X_STL_MAX_MM) > 0.1) &   # not the outlet face
        (x >= X_STL_MIN_MM) &                 # discard inlet buffer geometry
        (x <= X_STL_MAX_MM)                   # discard outlet buffer geometry
    )
    normals = torch.tensor(mesh.face_normals[face_ids[mask]])
    pts     = torch.tensor(pts[mask])
    return pts, 3 * torch.ones(len(pts), dtype=torch.int64), normals


def build_volume_pool(mesh, n_pool=800_000, wall_fraction=0.5, delta_mm=1.0, vert_frac=0.5):
    # Called once before training. Returns a fixed pool of collocation points:
    #   - (1 - wall_fraction) * n_pool uniform points clipped to the physical domain
    #   - wall_fraction * n_pool points within delta_mm of a wall face (inward displacement)
    # Each epoch draws n_vol_train random indices — trimesh is never called again.
    n_near    = int(n_pool * wall_fraction)
    n_uniform = n_pool - n_near

    # Uniform part — oversample to account for buffer-region clipping
    raw  = trimesh.sample.volume_mesh(mesh, n_uniform * 3)
    mask = (raw[:, 0] >= X_STL_MIN_MM) & (raw[:, 0] <= X_STL_MAX_MM)
    raw  = raw[mask]
    if len(raw) > n_uniform:
        raw = raw[np.random.choice(len(raw), n_uniform, replace=False)]

    # Near-wall part — sample horizontal (|n_z| >= 0.7) and vertical (|n_z| < 0.7)
    # faces separately with equal budgets, then concatenate.
    def _sample_submesh_near(face_ids, n):
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
    near_horiz = _sample_submesh_near(horiz_fids, int(n_near * (1 - vert_frac)))
    near_vert  = _sample_submesh_near(vert_fids,  int(n_near * vert_frac))
    all_pts = np.vstack([raw, near_horiz, near_vert])
    print(f"  uniform: {len(raw)}  near-horiz: {len(near_horiz)}  near-vert: {len(near_vert)}")
    return torch.tensor(all_pts, dtype=torch.float64), len(raw), len(near_horiz)


def sample_volume(vol_pool, n):
    idx = torch.randperm(len(vol_pool))[:n]
    pts = vol_pool[idx]
    return pts, torch.zeros(len(pts), dtype=torch.int64)


def sample_collocation(vol_pool, n_vol, n_wall):
    parts_pts, parts_lbl, parts_normals = [], [], []
    p, l = sample_volume(vol_pool, n_vol)
    parts_pts.append(p);   parts_lbl.append(l)
    parts_normals.append(torch.zeros(len(p), 3, dtype=torch.float64))
    w_pts, w_lbl, w_normals = sample_wall(mesh, n_wall)
    parts_pts.append(w_pts);     parts_lbl.append(w_lbl);     parts_normals.append(w_normals)
    pts, lbl, normals = torch.cat(parts_pts), torch.cat(parts_lbl), torch.cat(parts_normals)
    perm = torch.randperm(pts.size(0))
    return pts[perm], lbl[perm], normals[perm]


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def dynamic_viscosity(T):
    return MU_REF * (T.abs() / T_REF_SUTH) ** 1.5 * (T_REF_SUTH + S_SUTH) / (T.abs() + S_SUTH)


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
    vx = fields[:, 0:1]
    vy = fields[:, 1:2]
    vz = fields[:, 2:3]
    p  = fields[:, 3:4]
    T  = fields[:, 4:5] * 1000   # un-scale kK → K

    gvx = _grad(vx, pts); vx_x, vx_y, vx_z = gvx[:, 0:1], gvx[:, 1:2], gvx[:, 2:3]
    vx_xx = _grad2(vx, pts, 0); vx_yy = _grad2(vx, pts, 1); vx_zz = _grad2(vx, pts, 2)

    gvy = _grad(vy, pts); vy_x, vy_y, vy_z = gvy[:, 0:1], gvy[:, 1:2], gvy[:, 2:3]
    vy_xx = _grad2(vy, pts, 0); vy_yy = _grad2(vy, pts, 1); vy_zz = _grad2(vy, pts, 2)

    gvz = _grad(vz, pts); vz_x, vz_y, vz_z = gvz[:, 0:1], gvz[:, 1:2], gvz[:, 2:3]
    vz_xx = _grad2(vz, pts, 0); vz_yy = _grad2(vz, pts, 1); vz_zz = _grad2(vz, pts, 2)

    gp = _grad(p, pts); p_x, p_y, p_z = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]
    gT = _grad(T, pts); T_x, T_y, T_z = gT[:, 0:1], gT[:, 1:2], gT[:, 2:3]
    T_xx = _grad2(T, pts, 0); T_yy = _grad2(T, pts, 1); T_zz = _grad2(T, pts, 2)

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
    labels, wall_normals, p_out, vx_in, vy_in, vz_in, T_in, T_w,
    vx_var, vy_var, vz_var, p_var, T_var,
):
    interior = labels == 0
    inlet    = labels == 1
    outlet   = labels == 2
    wall     = labels == 3
    mu       = dynamic_viscosity(T)

    # ── PDE losses ───────────────────────────────────────────
    l_div = torch.mean(((vx_x + vy_y + vz_z) / DIV_SCALE) ** 2)

    def navier_stokes(u_x, u_y, u_z, u_xx, u_yy, u_zz, p_grad, mom_scale):
        advec = (vx[interior] * u_x[interior]
                 + vy[interior] * u_y[interior]
                 + vz[interior] * u_z[interior])
        lap   = u_xx[interior] + u_yy[interior] + u_zz[interior]
        res   = advec + (1e5 / rho) * p_grad[interior] - (mu[interior] / rho) * lap
        return torch.mean((res / mom_scale) ** 2)

    l_mom_x = navier_stokes(vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, p_x, MOM_SCALE_X)
    l_mom_y = navier_stokes(vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, p_y, MOM_SCALE_Y)
    l_mom_z = navier_stokes(vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_z, MOM_SCALE_Z)

    HEAT_SCALE = U_SCALE * 1000.0
    l_heat = torch.mean((
        (alpha * (T_xx[interior] + T_yy[interior] + T_zz[interior])
         + vx[interior] * T_x[interior]
         + vy[interior] * T_y[interior]
         + vz[interior] * T_z[interior]) / HEAT_SCALE
    ) ** 2)

    # ── BC losses ────────────────────────────────────────────
    l_inlet_vx = torch.mean((vx[inlet].squeeze(1) - vx_in) ** 2) / vx_var
    l_inlet_vy = torch.mean((vy[inlet].squeeze(1) - vy_in) ** 2) / vy_var
    l_inlet_vz = torch.mean((vz[inlet].squeeze(1) - vz_in) ** 2) / vz_var
    l_inlet_T  = torch.mean((T[inlet]              - T_in) ** 2) / T_var
    l_outlet_p = torch.mean((p[outlet]             - p_out) ** 2) / p_var
    l_wall_vx  = torch.mean(vx[wall] ** 2)                        / vx_var
    l_wall_vy  = torch.mean(vy[wall] ** 2)                        / vy_var
    l_wall_vz  = torch.mean(vz[wall] ** 2)                        / vz_var
    l_wall_T   = torch.mean((T[wall]               - T_w) ** 2)  / T_var

    n_x = wall_normals[wall, 0:1]
    n_y = wall_normals[wall, 1:2]
    n_z = wall_normals[wall, 2:3]
    dp_dn = p_x[wall] * n_x + p_y[wall] * n_y + p_z[wall] * n_z
    grad_p_var = p_var / coord_std.mean() ** 2
    l_wall_dp_dn = torch.mean(dp_dn ** 2) / grad_p_var

    return (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
        l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
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

        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for idx, lin in enumerate(linears):
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return self.net(x)


class NormalizedPINN(nn.Module):
    """Z-scores inputs; un-standardizes outputs to physical units."""

    def __init__(self, net, coord_mean, coord_std, out_mean, out_std):
        super().__init__()
        self.net = net
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std",  coord_std)
        self.register_buffer("out_mean",   out_mean)
        self.register_buffer("out_std",    out_std)

    def forward(self, x):
        x_norm    = (x - self.coord_mean) / self.coord_std
        y_norm    = self.net(x_norm)
        safe_std  = torch.where(self.out_std == 0,
                                torch.ones_like(self.out_std), self.out_std)
        return y_norm * safe_std + self.out_mean


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_fields(pts, fields, output_dir, tag=""):
    os.makedirs(output_dir, exist_ok=True)

    def to_np(x):
        if x is None:
            return None
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pts_np = to_np(pts)
    idx    = np.arange(len(pts_np))
    p_sub  = pts_np

    # Physical aspect ratio so geometry isn't distorted (x >> z for this duct)
    ranges = p_sub.max(axis=0) - p_sub.min(axis=0)
    box_aspect = (ranges / ranges.max()).tolist()

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
        fig, axes = plt.subplots(1, n_sub, figsize=(6 * n_sub, 5),
                                  subplot_kw={"projection": "3d"})
        if n_sub == 1:
            axes = [axes]
        for ax, (title, color, cmin, cmax) in zip(axes, subplots):
            sc = ax.scatter(p_sub[:, 0], p_sub[:, 1], p_sub[:, 2],
                            c=color[idx], cmap="viridis",
                            vmin=cmin, vmax=cmax, s=1, rasterized=True)
            ax.set_box_aspect(box_aspect)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            plt.colorbar(sc, ax=ax, shrink=0.5, label=name)
        fig.tight_layout()
        fname = f"{name}{'_' + tag if tag else ''}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="PINN13 — physics-informed NN for GT blade cooling")

# I/O
parser.add_argument("--folder",     default="dp11",   help="Data subfolder under preProcessedData/with_T/")
parser.add_argument("--run-path",   default=None,     help="Output dir (default: auto-derived from all params)")
parser.add_argument("--project",    default="PINN10", help="W&B project name")
parser.add_argument("--device",     default="cuda",   help="Torch device")
parser.add_argument("--debug",      action="store_true", help="Skip W&B logging")

# Architecture
parser.add_argument("--hidden-dim", type=int,   default=20)
parser.add_argument("--n-layers",   type=int,   default=4)

# Training
parser.add_argument("--epochs",     type=int,   default=10_000)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--lr-weights", type=float, default=1e-4)
parser.add_argument("--lr-min",     type=float, default=1e-6)
parser.add_argument("--t0",         type=int,   default=2000,  help="CosineAnnealingWarmRestarts T_0")
parser.add_argument("--n-mult",     type=int,   default=2,     help="CosineAnnealingWarmRestarts T_mult")

# Sampling
parser.add_argument("--n-train",       type=int,   default=5_000)
parser.add_argument("--n-test",        type=int,   default=10_000)
parser.add_argument("--n-sup",         type=int,   default=500,   help="Supervised CFD points per epoch (0 = physics-only)")
parser.add_argument("--n-snapshots",   type=int,   default=20_000, help="Points used for snapshot plots")
parser.add_argument("--wall-fraction", type=float, default=0.5,   help="Fraction of volume pool sampled near walls")
parser.add_argument("--delta-mm",      type=float, default=1.0,   help="Max inward displacement from wall surface [mm]")

args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════
# CONFIG  (derived from args — single source of truth)
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
LR_WEIGHTS = args.lr_weights
LR_MIN     = args.lr_min
T_0        = args.t0
N_MULT     = args.n_mult

N_TEST     = args.n_test
N_TRAIN    = args.n_train
N_SUP      = args.n_sup
N_POINT_SNAPSHOTS = args.n_snapshots
WALL_FRACTION = args.wall_fraction
DELTA_MM      = args.delta_mm

DATA_DIR = f"./preProcessedData/with_T/{FOLDER}/"

# Auto-derive run path from all key params — every unique config lands in its own folder
if args.run_path:
    RUN_PATH = args.run_path
else:
    _name = (f"f{FOLDER}"
             f"_h{HIDDEN_DIM}_l{N_LAYERS}"
             f"_e{EPOCHS}"
             f"_lr{LR:.0e}_lrw{LR_WEIGHTS:.0e}"
             f"_t0{T_0}_nm{N_MULT}"
             f"_ntrain{N_TRAIN}"
             f"_sup{N_SUP}"
             f"_s{SEED}")
    RUN_PATH = f"../pinn13_runs/{_name}"

# Create run directory and redirect all output there
os.makedirs(RUN_PATH, exist_ok=True)
_log_path = os.path.join(RUN_PATH, "training.log")

class _Tee(io.TextIOBase):
    """Mirrors writes to a stream and a log file simultaneously."""
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

n_vol_train    = int(0.60 * N_TRAIN)
n_outlet_train = int(0.05 * N_TRAIN)
n_wall_train   = int(0.30 * N_TRAIN)
n_inlet_train  = int(0.05 * N_TRAIN)

LOSS_NAMES = [
    "divergence",
    "momentum_x", "momentum_y", "momentum_z",
    "heat",
    "bc_inlet_vx", "bc_inlet_vy", "bc_inlet_vz", "bc_inlet_T", "bc_outlet_p",
    "bc_wall_vx",  "bc_wall_vy",  "bc_wall_vz",  "bc_wall_T",
    "bc_wall_dp_dn",
    "sup_vx", "sup_vy", "sup_vz", "sup_p", "sup_T",
]
N_LOSSES = len(LOSS_NAMES)

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

print(f"Run path : {RUN_PATH}")
print(f"Log file : {_log_path}")
print(f"Config   : folder={FOLDER}  h={HIDDEN_DIM}  l={N_LAYERS}  epochs={EPOCHS}")
print(f"           lr={LR}  lr_weights={LR_WEIGHTS}  lr_min={LR_MIN}")
print(f"           T0={T_0}  N_mult={N_MULT}  n_train={N_TRAIN}  n_sup={N_SUP}  seed={SEED}")

if not DEBUG:
    api_key = "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"
    wandb.login(key=api_key)
    run = wandb.init(
        project=PROJECT,
        name=os.path.basename(RUN_PATH),
        config={
            "folder": FOLDER, "run_path": RUN_PATH,
            "epochs": EPOCHS, "lr": LR, "lr_weights": LR_WEIGHTS,
            "lr_min": LR_MIN, "scheduler": "cosine_warm_restarts",
            "T_0": T_0, "T_mult": N_MULT,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "seed": SEED,
            "n_train": N_TRAIN, "n_test": N_TEST, "n_supervised": N_SUP,
            "n_vol_train": n_vol_train, "n_outlet_train": n_outlet_train,
            "n_wall_train": n_wall_train, "n_inlet_train": n_inlet_train,
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

inlet_raw   = np.load(DATA_DIR + "vel_x_inlet.npy")
inlet_perm  = np.random.permutation(inlet_raw.shape[0])[:n_inlet_train]
inlet_pts   = torch.tensor(inlet_raw[inlet_perm, :3])
vx_inlet_bc = torch.tensor(np.load(DATA_DIR + "vel_x_inlet.npy")[inlet_perm, 3])
vy_inlet_bc = torch.tensor(np.load(DATA_DIR + "vel_y_inlet.npy")[inlet_perm, 3])
vz_inlet_bc = torch.tensor(np.load(DATA_DIR + "vel_z_inlet.npy")[inlet_perm, 3])

cfd_pts = torch.tensor(np.load(DATA_DIR + "vel_x.npy")[:, :3])
cfd_vx  = torch.tensor(np.load(DATA_DIR + "vel_x.npy")[:, 3])
cfd_vy  = torch.tensor(np.load(DATA_DIR + "vel_y.npy")[:, 3])
cfd_vz  = torch.tensor(np.load(DATA_DIR + "vel_z.npy")[:, 3])
cfd_p   = torch.tensor(np.load(DATA_DIR + "press.npy")[:, 3])
cfd_T   = torch.tensor(np.load(DATA_DIR + "temp.npy")[:, 3])

# ── Domain bounds — derived from CFD, works for any dp ──────
# CFD files are always exported to the physical domain only, so their
# x-range is the ground truth. sample_wall / sample_volume use these
# globals to clip away any buffer geometry in the STL.
X_MIN_MM = float(cfd_pts[:, 0].min()) * 1000   # [mm] physical domain start (CFD frame)
X_MAX_MM = float(cfd_pts[:, 0].max()) * 1000   # [mm] physical domain end   (CFD frame)
STL_INLET_BUFFER_MM = 420.0   # [mm] inlet (and outlet) buffer present in the STL mesh
X_STL_MIN_MM = X_MIN_MM + STL_INLET_BUFFER_MM  # physical domain start in STL coords
X_STL_MAX_MM = X_MAX_MM + STL_INLET_BUFFER_MM  # physical domain end   in STL coords
print(f"Physical domain  x : [{X_MIN_MM:.2f}, {X_MAX_MM:.2f}] mm"
      f"  =  [{X_MIN_MM/1000:.4f}, {X_MAX_MM/1000:.4f}] m")

# ── Outlet BC points — CFD slice near x = X_MAX_MM ──────────
# STLs may not have an internal face at the physical outlet (e.g. dp00
# mesh runs to 1975 mm with no face at 1135 mm). Using the CFD slice
# mirrors how inlet_pts works and is robust for any dp.
_out_mask  = cfd_pts[:, 0] > (X_MAX_MM / 1000 - 0.005)   # 5 mm slice
_out_pool  = cfd_pts[_out_mask]
_out_perm  = torch.randperm(len(_out_pool))[:n_outlet_train]
outlet_pts = _out_pool[_out_perm]
print(f"Outlet BC        : {len(_out_pool)} CFD points near x={X_MAX_MM/1000:.4f} m,"
      f" using {len(outlet_pts)}")

coord_mean = cfd_pts.mean(dim=0)
coord_std  = cfd_pts.std(dim=0)
out_mean   = torch.stack([cfd_vx.mean(), cfd_vy.mean(), cfd_vz.mean(),
                           (cfd_p / 1e5).mean(), (cfd_T / 1000).mean()])
out_std    = torch.stack([cfd_vx.std(),  cfd_vy.std(),  cfd_vz.std(),
                           (cfd_p / 1e5).std(),  (cfd_T / 1000).std()])
out_std    = out_std.clamp(min=1e-3)

# ── Anisotropic PDE scales (account for thin z-dimension) ────────
# MOM_SCALE_i = U² / L_i: each direction uses its own coordinate std as length scale.
# DIV_SCALE   = max(v_std_i / L_i): dominated by the thinnest direction (z).
MOM_SCALE_X = U_SCALE ** 2 / float(coord_std[0])
MOM_SCALE_Y = U_SCALE ** 2 / float(coord_std[1])
MOM_SCALE_Z = U_SCALE ** 2 / float(coord_std[2])
DIV_SCALE   = float(max(out_std[0] / coord_std[0],
                        out_std[1] / coord_std[1],
                        out_std[2] / coord_std[2]))
print(f"MOM_SCALE  : X={MOM_SCALE_X:.1f}  Y={MOM_SCALE_Y:.1f}  Z={MOM_SCALE_Z:.1f}")
print(f"DIV_SCALE  : {DIV_SCALE:.1f}")

print(f"coord_mean : {coord_mean.tolist()}")
print(f"coord_std  : {coord_std.tolist()}")
print(f"out_mean   : {out_mean.tolist()}")
print(f"out_std    : {out_std.tolist()}")

perm      = torch.randperm(cfd_pts.shape[0])
test_idx  = perm[:N_TEST]
train_idx = perm[N_TEST:]

test_pts = cfd_pts[test_idx];  train_pts = cfd_pts[train_idx]
test_vx  = cfd_vx[test_idx];   train_vx  = cfd_vx[train_idx]
test_vy  = cfd_vy[test_idx];   train_vy  = cfd_vy[train_idx]
test_vz  = cfd_vz[test_idx];   train_vz  = cfd_vz[train_idx]
test_p   = cfd_p[test_idx] / 1e5;   train_p = cfd_p[train_idx] / 1e5
test_T   = cfd_T[test_idx];    train_T  = cfd_T[train_idx]

def _var(raw_var, floor, name):
    v = raw_var.clamp(min=floor)
    if raw_var.item() < floor:
        print(f"  [variance floor hit] {name}: raw={raw_var.item():.2e} → using floor={floor:.2e}")
    return v

vx_raw = torch.maximum(train_vx.var(), vx_inlet_bc.var())
vy_raw = torch.maximum(train_vy.var(), vy_inlet_bc.var())
vz_raw = torch.maximum(train_vz.var(), vz_inlet_bc.var())
vx_var = _var(vx_raw, 1e-6,                  "vx")
vy_var = _var(vy_raw, 1e-6,                  "vy")
vz_var = _var(vz_raw, 1e-6,                  "vz")
p_var  = _var(train_p.var(), (P_OUTLET*0.01)**2, "p")
T_var  = _var(train_T.var(), 1.0,            "T")

print(f"\nField variances (used for all loss normalization):")
print(f"  vx: {vx_var.item():.4e}  vy: {vy_var.item():.4e}  vz: {vz_var.item():.4e}")
print(f"  p:  {p_var.item():.4e}   T:  {T_var.item():.4e}")

stl_files = glob.glob(DATA_DIR + "*.stl")
print(f"Loading STL: {stl_files[0]}")
mesh = trimesh.load(stl_files[0])
_stl_xmin = float(mesh.vertices[:, 0].min())
_stl_xmax = float(mesh.vertices[:, 0].max())
print(f"STL mesh         x : [{_stl_xmin:.2f}, {_stl_xmax:.2f}] mm"
      f"  (inlet buffer = {STL_INLET_BUFFER_MM:.0f} mm,"
      f"  outlet buffer = {STL_INLET_BUFFER_MM:.0f} mm,"
      f"  physical in STL: [{X_STL_MIN_MM:.1f}, {X_STL_MAX_MM:.1f}] mm)")

print("Building volume pool (one-time trimesh call)...")
vol_pool, _n_uniform, _n_near_horiz = build_volume_pool(mesh, n_pool=500_000,
                                                        wall_fraction=WALL_FRACTION, delta_mm=DELTA_MM)
print(f"Volume pool      : {len(vol_pool)} points in physical domain"
      f"  (drawing {n_vol_train} per epoch)")

# ── Volume pool visualization ─────────────────────────────────
_pool_np     = vol_pool.cpu().numpy()
_uni         = _pool_np[:_n_uniform]
_near_horiz  = _pool_np[_n_uniform : _n_uniform + _n_near_horiz]
_near_vert   = _pool_np[_n_uniform + _n_near_horiz :]
_subsample = 5_000
_ui  = np.random.choice(len(_uni),        min(_subsample, len(_uni)),        replace=False)
_hi  = np.random.choice(len(_near_horiz), min(_subsample, len(_near_horiz)), replace=False)
_vi  = np.random.choice(len(_near_vert),  min(_subsample, len(_near_vert)),  replace=False)

fig_pool = plt.figure(figsize=(18, 5))
fig_pool.suptitle(
    f"Volume pool  —  uniform (blue, n={len(_uni)})  "
    f"near-horiz (red, n={len(_near_horiz)})  "
    f"near-vert (green, n={len(_near_vert)})  δ={DELTA_MM} mm",
    fontsize=10)

ax3d = fig_pool.add_subplot(141, projection="3d")
ax3d.scatter(_uni[_ui, 0],        _uni[_ui, 1],        _uni[_ui, 2],        s=0.5, alpha=0.2, c="steelblue", label="uniform")
ax3d.scatter(_near_horiz[_hi, 0], _near_horiz[_hi, 1], _near_horiz[_hi, 2], s=0.5, alpha=0.3, c="crimson",   label="near-horiz")
ax3d.scatter(_near_vert[_vi, 0],  _near_vert[_vi, 1],  _near_vert[_vi, 2],  s=0.5, alpha=0.3, c="limegreen", label="near-vert")
ax3d.set_xlabel("x [mm]"); ax3d.set_ylabel("y [mm]"); ax3d.set_zlabel("z [mm]")
ax3d.set_title("3D view"); ax3d.legend(markerscale=8, fontsize=7)

for idx, (xlabel, ylabel, xi, yi) in enumerate(
        [("x [mm]", "y [mm]", 0, 1),
         ("x [mm]", "z [mm]", 0, 2),
         ("y [mm]", "z [mm]", 1, 2)], start=2):
    ax = fig_pool.add_subplot(1, 4, idx)
    ax.scatter(_uni[_ui, xi],        _uni[_ui, yi],        s=0.5, alpha=0.2, c="steelblue")
    ax.scatter(_near_horiz[_hi, xi], _near_horiz[_hi, yi], s=0.5, alpha=0.3, c="crimson")
    ax.scatter(_near_vert[_vi, xi],  _near_vert[_vi, yi],  s=0.5, alpha=0.3, c="limegreen")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(f"{xlabel} vs {ylabel}")

plt.tight_layout()
_pool_viz_path = os.path.join(RUN_PATH, "viz_volume_pool.png")
fig_pool.savefig(_pool_viz_path, dpi=150, bbox_inches="tight")
plt.close(fig_pool)
print(f"Volume pool viz  → {_pool_viz_path}")

_snap_n   = min(N_POINT_SNAPSHOTS, cfd_pts.shape[0])
_snap_idx = np.random.choice(cfd_pts.shape[0], _snap_n, replace=False)
snap_pts  = cfd_pts[_snap_idx]
snap_vx   = cfd_vx[_snap_idx]
snap_vy   = cfd_vy[_snap_idx]
snap_vz   = cfd_vz[_snap_idx]
snap_p    = cfd_p[_snap_idx] / 1e5
snap_T    = cfd_T[_snap_idx]

# ═══════════════════════════════════════════════════════════════
# MODEL + OPTIMIZERS
# ═══════════════════════════════════════════════════════════════

net   = FFNN(in_dim=3, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS).to(DEVICE).double()
model = NormalizedPINN(net, coord_mean, coord_std, out_mean, out_std)

weights     = nn.Parameter(torch.ones(N_LOSSES, dtype=torch.float64, device=DEVICE))
opt_model   = Adam(model.parameters(), lr=LR,         betas=(0.99, 0.999))
opt_weights = Adam([weights],          lr=LR_WEIGHTS, betas=(0.99, 0.999), maximize=True)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt_model, T_0=T_0, T_mult=N_MULT, eta_min=LR_MIN)

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

start = time.time()
print("=" * 60)

for epoch in range(EPOCHS):
    coll_pts, coll_lbls, coll_normals = sample_collocation(
        vol_pool, n_vol_train, n_wall_train
    )
    coll_pts = coll_pts.clone()
    coll_pts[:, 0] = coll_pts[:, 0] - STL_INLET_BUFFER_MM  # shift x to CFD frame (remove buffer)
    coll_pts = coll_pts / 1000   # mm → m

    # inlet_pts (label=1) and outlet_pts (label=2) come from CFD data (already in m)
    pts  = torch.cat([coll_pts, inlet_pts, outlet_pts]).requires_grad_(True)
    lbls = torch.cat([coll_lbls,
                       torch.ones(inlet_pts.shape[0],      dtype=torch.long),
                       2 * torch.ones(outlet_pts.shape[0], dtype=torch.long)])
    wall_normals = torch.cat([coll_normals,
                               torch.zeros(inlet_pts.shape[0],  3, dtype=torch.float64),
                               torch.zeros(outlet_pts.shape[0], 3, dtype=torch.float64)])

    model.train()
    opt_model.zero_grad()
    opt_weights.zero_grad()

    fields = model(pts)
    derivs = compute_derivatives(fields, pts)
    (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
        l_wall_vx,  l_wall_vy,  l_wall_vz,  l_wall_T,
        l_wall_dp_dn,
    ) = compute_losses(
        *derivs, lbls, wall_normals,
        P_OUTLET, vx_inlet_bc, vy_inlet_bc, vz_inlet_bc, T_INLET, T_WALL,
        vx_var, vy_var, vz_var, p_var, T_var,
    )

    if N_SUP > 0:
        sup_idx    = torch.randint(0, train_pts.shape[0], (N_SUP,))
        sup_fields = model(train_pts[sup_idx])
        l_sup_vx = torch.mean((sup_fields[:, 0]        - train_vx[sup_idx]) ** 2) / vx_var
        l_sup_vy = torch.mean((sup_fields[:, 1]        - train_vy[sup_idx]) ** 2) / vy_var
        l_sup_vz = torch.mean((sup_fields[:, 2]        - train_vz[sup_idx]) ** 2) / vz_var
        l_sup_p  = torch.mean((sup_fields[:, 3]        - train_p[sup_idx])  ** 2) / p_var
        l_sup_T  = torch.mean((sup_fields[:, 4] * 1000 - train_T[sup_idx])  ** 2) / T_var
    else:
        z = torch.zeros(1, dtype=torch.float64, device=DEVICE).squeeze()
        l_sup_vx = l_sup_vy = l_sup_vz = l_sup_p = l_sup_T = z

    l_pde_total = l_div + l_mom_x + l_mom_y + l_mom_z + l_heat
    l_bc_total  = (l_inlet_vx + l_inlet_vy + l_inlet_vz + l_inlet_T + l_outlet_p
                   + l_wall_vx + l_wall_vy + l_wall_vz + l_wall_T + l_wall_dp_dn)
    l_sup_total = l_sup_vx + l_sup_vy + l_sup_vz + l_sup_p + l_sup_T

    all_losses = torch.stack([
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T,
        l_outlet_p,
        l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
        l_wall_dp_dn,
        l_sup_vx, l_sup_vy, l_sup_vz, l_sup_p, l_sup_T,
    ])
    l_unweighted = all_losses.sum()
    l_weighted   = torch.sum(weights * all_losses) - weights.sum()

    l_weighted.backward()
    opt_model.step()
    opt_weights.step()
    with torch.no_grad():
        weights.clamp_(min=1e-6)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        pred  = model(test_pts)
    mse_vx    = torch.mean((pred[:, 0]        - test_vx) ** 2)
    mse_vy    = torch.mean((pred[:, 1]        - test_vy) ** 2)
    mse_vz    = torch.mean((pred[:, 2]        - test_vz) ** 2)
    mse_p     = torch.mean((pred[:, 3]        - test_p)  ** 2)
    mse_T     = torch.mean((pred[:, 4] * 1000 - test_T)  ** 2)
    mse_total = mse_vx + mse_vy + mse_vz + mse_p + mse_T

    if epoch in plot_epochs:
        snap_dir = os.path.join(RUN_PATH, f"snap_{epoch + 1}_of_{EPOCHS}")
        with torch.no_grad():
            pred_snap = model(snap_pts)
        plot_fields(
            snap_pts.cpu().numpy(),
            [
                ("vx", snap_vx.cpu().numpy(), pred_snap[:, 0].cpu().numpy()),
                ("vy", snap_vy.cpu().numpy(), pred_snap[:, 1].cpu().numpy()),
                ("vz", snap_vz.cpu().numpy(), pred_snap[:, 2].cpu().numpy()),
                ("p",  snap_p.cpu().numpy(),  pred_snap[:, 3].cpu().numpy()),
                ("T",  snap_T.cpu().numpy(),  (pred_snap[:, 4] * 1000).cpu().numpy()),
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
        "bc/wall_dp_dn":  safe_log10(l_wall_dp_dn),
        "sup/vx":         safe_log10(l_sup_vx),
        "sup/vy":         safe_log10(l_sup_vy),
        "sup/vz":         safe_log10(l_sup_vz),
        "sup/p":          safe_log10(l_sup_p),
        "sup/T":          safe_log10(l_sup_T),
        "loss/pde_total":        safe_log10(l_pde_total),
        "loss/bc_total":         safe_log10(l_bc_total),
        "loss/sup_total":        safe_log10(l_sup_total),
        "loss/unweighted_total": safe_log10(l_unweighted),
        "loss/weighted_total":   safe_log10(l_weighted),
        "eval/mse_vx":    safe_log10(mse_vx),
        "eval/mse_vy":    safe_log10(mse_vy),
        "eval/mse_vz":    safe_log10(mse_vz),
        "eval/mse_p":     safe_log10(mse_p),
        "eval/mse_T":     safe_log10(mse_T),
        "eval/mse_total": safe_log10(mse_total),
        **{f"weight/{name}": weights[i].item() for i, name in enumerate(LOSS_NAMES)},
        "train/lr": scheduler.get_last_lr()[0],
    }
    log = {k: v for k, v in log.items() if not (isinstance(v, float) and np.isnan(v))}

    w = {name: weights[i].item() for i, name in enumerate(LOSS_NAMES)}
    print(
        f"\n\n\n\n[{epoch+1:>5}/{EPOCHS}]\n"
        f"PDE   div={l_div.item():.3e}  mom_x={l_mom_x.item():.3e}  mom_y={l_mom_y.item():.3e}  mom_z={l_mom_z.item():.3e}  heat={l_heat.item():.3e}  | total={l_pde_total.item():.3e}\n"
        f"      w: div={w['divergence']:.3e}  mom_x={w['momentum_x']:.3e}  mom_y={w['momentum_y']:.3e}  mom_z={w['momentum_z']:.3e}  heat={w['heat']:.3e}\n"
        f"BC    inlet_vx={l_inlet_vx.item():.3e}  inlet_vy={l_inlet_vy.item():.3e}  inlet_vz={l_inlet_vz.item():.3e}  inlet_T={l_inlet_T.item():.3e}  outlet_p={l_outlet_p.item():.3e}\n"
        f"      wall_vx={l_wall_vx.item():.3e}   wall_vy={l_wall_vy.item():.3e}   wall_vz={l_wall_vz.item():.3e}   wall_T={l_wall_T.item():.3e}   wall_dp_dn={l_wall_dp_dn.item():.3e}  | total={l_bc_total.item():.3e}\n"
        f"      w: inlet_vx={w['bc_inlet_vx']:.3e}  inlet_vy={w['bc_inlet_vy']:.3e}  inlet_vz={w['bc_inlet_vz']:.3e}  inlet_T={w['bc_inlet_T']:.3e}  outlet_p={w['bc_outlet_p']:.3e}\n"
        f"      w: wall_vx={w['bc_wall_vx']:.3e}   wall_vy={w['bc_wall_vy']:.3e}   wall_vz={w['bc_wall_vz']:.3e}   wall_T={w['bc_wall_T']:.3e}   wall_dp_dn={w['bc_wall_dp_dn']:.3e}\n"
        f"SUP   vx={l_sup_vx.item():.3e}  vy={l_sup_vy.item():.3e}  vz={l_sup_vz.item():.3e}  p={l_sup_p.item():.3e}  T={l_sup_T.item():.3e}  | total={l_sup_total.item():.3e}\n"
        f"      w: vx={w['sup_vx']:.3e}  vy={w['sup_vy']:.3e}  vz={w['sup_vz']:.3e}  p={w['sup_p']:.3e}  T={w['sup_T']:.3e}\n"
        f"MSE   vx={mse_vx.item():.3e}  vy={mse_vy.item():.3e}  vz={mse_vz.item():.3e}  p={mse_p.item():.3e}  T={mse_T.item():.3e}  | total={mse_total.item():.3e}"
    )

    if not DEBUG:
        wandb.log(log, step=epoch)

print(f"\nTraining done in {time.time() - start:.1f}s")
torch.save(model.state_dict(),     os.path.join(RUN_PATH, "pinn_model.pt"))
torch.save(weights.detach().cpu(), os.path.join(RUN_PATH, "loss_weights.pt"))

# ═══════════════════════════════════════════════════════════════
# INFERENCE  (CPU, full domain)
# ═══════════════════════════════════════════════════════════════

torch.set_default_device("cpu")
net_inf   = FFNN(in_dim=3, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS)
model_inf = NormalizedPINN(net_inf,
                            coord_mean.cpu(), coord_std.cpu(),
                            out_mean.cpu(),   out_std.cpu())
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
with torch.no_grad():
    pred_all = model_inf(torch.tensor(all_pts_np))

vx_pred = pred_all[:, 0].numpy()
vy_pred = pred_all[:, 1].numpy()
vz_pred = pred_all[:, 2].numpy()
p_pred  = pred_all[:, 3].numpy()
T_pred  = (pred_all[:, 4] * 1000).numpy()

n_inlet      = vx_in_full.shape[0]
train_idx_np = train_idx.cpu().numpy()
test_idx_np  = test_idx.cpu().numpy()

idx_vel    = np.random.choice(all_pts_np.shape[0],
                               min(100_000, all_pts_np.shape[0]), replace=False)
vel_fields = []
for in_arr, body_arr, pred_arr, name in [
    (vx_in_full, vx_full, vx_pred, "vx"),
    (vy_in_full, vy_full, vy_pred, "vy"),
    (vz_in_full, vz_full, vz_pred, "vz"),
]:
    true_all  = np.concatenate((in_arr[:, 3], body_arr[:, 3]))
    vel_fields.append((name, true_all[idx_vel], pred_arr[idx_vel]))
    body_pred = pred_arr[n_inlet:]
    print(
        f"{name}  train RMSE: "
        f"{np.sqrt(np.mean((body_arr[:, 3][train_idx_np] - body_pred[train_idx_np])**2)):.4e}"
        f"  test RMSE: "
        f"{np.sqrt(np.mean((body_arr[:, 3][test_idx_np]  - body_pred[test_idx_np])**2)):.4e}"
    )
plot_fields(all_pts_np[idx_vel], vel_fields, output_dir=os.path.join(RUN_PATH, "inference"))

idx_sc    = np.random.choice(vx_full.shape[0], min(50_000, vx_full.shape[0]), replace=False)
sc_fields = []
for tag, true_arr, pred_slice in [
    ("p", p_full[:, 3] / 1e5, p_pred[n_inlet:]),
    ("T", T_full[:, 3],       T_pred[n_inlet:]),
]:
    sc_fields.append((tag, true_arr[idx_sc], pred_slice[idx_sc]))
    print(
        f"{tag}  train RMSE: "
        f"{np.sqrt(np.mean((true_arr[train_idx_np] - pred_slice[train_idx_np])**2)):.4e}"
        f"  test RMSE: "
        f"{np.sqrt(np.mean((true_arr[test_idx_np]  - pred_slice[test_idx_np])**2)):.4e}"
    )
plot_fields(vx_full[:, :3][idx_sc], sc_fields, output_dir=os.path.join(RUN_PATH, "inference"))

_log_handle.close()
