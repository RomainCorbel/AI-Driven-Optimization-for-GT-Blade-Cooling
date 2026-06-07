import trimesh
import numpy as np
import random
import torch
import torch.nn as nn
import os
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
L_SCALE   = 1.0     # [m]    characteristic length  (domain x: 0 → 0.953 m)
MOM_SCALE = U_SCALE ** 2 / L_SCALE   # [m/s²]  ~ 2500  advection scale
DIV_SCALE = U_SCALE / L_SCALE        # [1/s]   ~ 50    continuity scale

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


def sample_outlet(mesh, n):
    thresh = 1e-5
    x_max  = mesh.vertices[:, 0].max()
    faces  = [i for i, f in enumerate(mesh.faces)
               if np.all(np.abs(mesh.vertices[f, 0] - x_max) < thresh)]
    pts    = _submesh_pts(mesh, faces, n)
    return pts, 2 * torch.ones(len(pts), dtype=torch.int64)


def sample_wall(mesh, n):
    thresh = 1e-5
    x_max  = mesh.vertices[:, 0].max()
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    mask   = (np.abs(pts[:, 0]) > thresh) & \
              ~((x_max - thresh <= pts[:, 0]) & (pts[:, 0] <= x_max + thresh))
    pts    = torch.tensor(pts[mask])
    return pts, 3 * torch.ones(len(pts), dtype=torch.int64)


def sample_volume(mesh, n):
    pts = torch.tensor(trimesh.sample.volume_mesh(mesh, n))
    return pts, torch.zeros(len(pts), dtype=torch.int64)


def sample_collocation(mesh, n_vol, n_outlet, n_wall):
    parts_pts, parts_lbl = [], []
    for fn, nn_ in [(sample_outlet, n_outlet), (sample_wall, n_wall), (sample_volume, n_vol)]:
        p, l = fn(mesh, nn_)
        parts_pts.append(p)
        parts_lbl.append(l)
    pts, lbl = torch.cat(parts_pts), torch.cat(parts_lbl)
    perm = torch.randperm(pts.size(0))
    return pts[perm], lbl[perm]


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
    labels, p_out, vx_in, vy_in, vz_in, T_in, T_w,
    # variance scales (precomputed, passed in for normalization)
    vx_var, vy_var, vz_var, p_var, T_var,
):
    interior = labels == 0
    inlet    = labels == 1
    outlet   = labels == 2
    wall     = labels == 3
    mu       = dynamic_viscosity(T)

    # ── PDE losses ───────────────────────────────────────────
    # Continuity: normalized by (U/L)²
    l_div = torch.mean(((vx_x + vy_y + vz_z) / DIV_SCALE) ** 2)

    # Navier-Stokes: normalized by (U²/L)²
    def navier_stokes(u_x, u_y, u_z, u_xx, u_yy, u_zz, p_grad):
        advec = (vx[interior] * u_x[interior]
                 + vy[interior] * u_y[interior]
                 + vz[interior] * u_z[interior])
        lap   = u_xx[interior] + u_yy[interior] + u_zz[interior]
        res   = advec + (1e5 / rho) * p_grad[interior] - (mu[interior] / rho) * lap
        return torch.mean((res / MOM_SCALE) ** 2)

    l_mom_x = navier_stokes(vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, p_x)
    l_mom_y = navier_stokes(vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, p_y)
    l_mom_z = navier_stokes(vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_z)

    # Heat equation: convection term ~ U * dT/dx ~ 50 * 1000/1 = 5e4 K/s
    HEAT_SCALE = U_SCALE * 1000.0 / L_SCALE   # ~ 5e4
    l_heat = torch.mean((
        (alpha * (T_xx[interior] + T_yy[interior] + T_zz[interior])
         + vx[interior] * T_x[interior]
         + vy[interior] * T_y[interior]
         + vz[interior] * T_z[interior]) / HEAT_SCALE
    ) ** 2)

    # ── BC losses — normalized by field variance ─────────────
    # Velocity BCs normalized by vx_var (same physical scale for all velocities
    # keeps the no-slip wall loss comparable to the inlet BC loss)
    l_inlet_vx = torch.mean((vx[inlet].squeeze(1) - vx_in) ** 2) / vx_var
    l_inlet_vy = torch.mean((vy[inlet].squeeze(1) - vy_in) ** 2) / vy_var
    l_inlet_vz = torch.mean((vz[inlet].squeeze(1) - vz_in) ** 2) / vz_var
    l_inlet_T  = torch.mean((T[inlet]              - T_in) ** 2) / T_var
    l_outlet_p = torch.mean((p[outlet]             - p_out) ** 2) / p_var
    l_wall_vx  = torch.mean(vx[wall] ** 2)                        / vx_var
    l_wall_vy  = torch.mean(vy[wall] ** 2)                        / vy_var
    l_wall_vz  = torch.mean(vz[wall] ** 2)                        / vz_var
    l_wall_T   = torch.mean((T[wall]               - T_w) ** 2)  / T_var

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
    """Fully-connected network with Sin activations (standard for PINNs)."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), Sin()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Layer-appropriate initialization
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
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            plt.colorbar(sc, ax=ax, shrink=0.5, label=name)
        fig.tight_layout()
        fname = f"{name}{'_' + tag if tag else ''}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)




# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

FOLDER       = "dp11"
PROJECT_NAME = "PINNs-ALL_scitas_dp11"
DEVICE       = "cuda"
DEBUG        = False

DATA_DIR = f"./preProcessedData/with_T/{FOLDER}/"
RUN_PATH = f"../RUN_MoreWalls_claude_no_sup"

EPOCHS     = 10_000
HIDDEN_DIM = 20
N_LAYERS   = 4
SEED       = 42
LR         = 1e-3
LR_WEIGHTS = 1e-4
LR_MIN     = 1e-6
T_0        = 2000
N_MULT     = 2

N_TEST  = 10_000
N_TRAIN = 5_000
N_SUP   = 0      # supervised CFD points drawn each epoch

N_POINT_SNAPSHOTS = 20_000

n_vol_train    = int(0.60 * N_TRAIN)
n_outlet_train = int(0.05 * N_TRAIN)
n_wall_train   = int(0.30 * N_TRAIN)
n_inlet_train  = int(0.05 * N_TRAIN)

# indices 0-4  → PDE losses   (divergence, mom_x, mom_y, mom_z, heat)
# indices 5-13 → BC losses
# indices 14-18 → supervised losses
LOSS_NAMES = [
    "divergence",
    "momentum_x", "momentum_y", "momentum_z",
    "heat",
    "bc_inlet_vx", "bc_inlet_vy", "bc_inlet_vz", "bc_inlet_T", "bc_outlet_p",
    "bc_wall_vx",  "bc_wall_vy",  "bc_wall_vz",  "bc_wall_T",
    "sup_vx", "sup_vy", "sup_vz", "sup_p", "sup_T",
]
N_LOSSES   = len(LOSS_NAMES)
N_PDE      = 5   # first N_PDE entries are PDE terms

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

if not DEBUG:
    api_key = "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"
    wandb.login(key=api_key)
    run = wandb.init(
        project=PROJECT_NAME,
        config={
            "epochs": EPOCHS, "lr": LR, "lr_weights": LR_WEIGHTS,
            "lr_min": LR_MIN, "scheduler": "cosine_warm_restarts",
            "T_0": T_0, "T_mult": N_MULT,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "seed": SEED,
            "n_train": N_TRAIN, "n_test": N_TEST, "n_supervised": N_SUP,
            "n_vol_train": n_vol_train, "n_outlet_train": n_outlet_train,
            "n_wall_train": n_wall_train, "n_inlet_train": n_inlet_train,
            "pde_warmup_epochs": PDE_WARMUP_EPOCHS,
            "mom_scale": MOM_SCALE, "div_scale": DIV_SCALE,
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

# ── Normalization stats ──────────────────────────────────────
coord_mean = cfd_pts.mean(dim=0)
coord_std  = cfd_pts.std(dim=0)
out_mean   = torch.stack([cfd_vx.mean(), cfd_vy.mean(), cfd_vz.mean(),
                           (cfd_p / 1e5).mean(), (cfd_T / 1000).mean()])
out_std    = torch.stack([cfd_vx.std(),  cfd_vy.std(),  cfd_vz.std(),
                           (cfd_p / 1e5).std(),  (cfd_T / 1000).std()])
out_std    = out_std.clamp(min=1e-3)

print(f"coord_mean : {coord_mean.tolist()}")
print(f"coord_std  : {coord_std.tolist()}")
print(f"out_mean   : {out_mean.tolist()}")
print(f"out_std    : {out_std.tolist()}")

# ── Variance scales for loss normalization (precomputed, fixed) ──
# All losses become dimensionless O(1) quantities — critical for
# the adaptive weights and for keeping vx on equal footing with vy/vz.
perm      = torch.randperm(cfd_pts.shape[0])
test_idx  = perm[:N_TEST]
train_idx = perm[N_TEST:]

test_pts = cfd_pts[test_idx];  train_pts = cfd_pts[train_idx]
test_vx  = cfd_vx[test_idx];   train_vx  = cfd_vx[train_idx]
test_vy  = cfd_vy[test_idx];   train_vy  = cfd_vy[train_idx]
test_vz  = cfd_vz[test_idx];   train_vz  = cfd_vz[train_idx]
test_p   = cfd_p[test_idx] / 1e5;   train_p = cfd_p[train_idx] / 1e5
test_T   = cfd_T[test_idx];    train_T  = cfd_T[train_idx]

# Variance of each field on the training split — used to normalize losses
vx_var = train_vx.var().clamp(min=1e-6)
vy_var = train_vy.var().clamp(min=1e-6)
vz_var = train_vz.var().clamp(min=1e-6)
p_var  = train_p.var().clamp(min=1e-6)
T_var  = train_T.var().clamp(min=1e-6)

print(f"\nField variances (used for loss normalization):")
print(f"  vx: {vx_var.item():.4e}  vy: {vy_var.item():.4e}  vz: {vz_var.item():.4e}")
print(f"  p:  {p_var.item():.4e}   T:  {T_var.item():.4e}")

mesh = trimesh.load("./Baseline_ML4Science.stl")

# Snapshot subset (fixed across epochs → plots show learning, not sampling noise)
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
    # ── Collocation points ───────────────────────────────────
    coll_pts, coll_lbls = sample_collocation(
        mesh, n_vol_train, n_outlet_train, n_wall_train
    )
    coll_pts = coll_pts / 1000   # STL in mm → m

    # Append inlet points from CSV (already in meters)
    pts  = torch.cat([coll_pts, inlet_pts]).requires_grad_(True)
    lbls = torch.cat([coll_lbls,
                       torch.ones(inlet_pts.shape[0], dtype=torch.long)])

    # ── Forward + PDE/BC losses ──────────────────────────────
    model.train()
    opt_model.zero_grad()
    opt_weights.zero_grad()

    fields = model(pts)
    derivs = compute_derivatives(fields, pts)
    (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
        l_wall_vx,  l_wall_vy,  l_wall_vz,  l_wall_T,
    ) = compute_losses(
        *derivs, lbls,
        P_OUTLET, vx_inlet_bc, vy_inlet_bc, vz_inlet_bc, T_INLET, T_WALL,
        vx_var, vy_var, vz_var, p_var, T_var,
    )

    # ── Supervised losses — variance-normalized ──────────────
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

    # ── Aggregate & weighted loss ─────────────────────────────
    l_pde_total = l_div + l_mom_x + l_mom_y + l_mom_z + l_heat
    l_bc_total  = (l_inlet_vx + l_inlet_vy + l_inlet_vz + l_inlet_T + l_outlet_p
                   + l_wall_vx + l_wall_vy + l_wall_vz + l_wall_T)
    l_sup_total = l_sup_vx + l_sup_vy + l_sup_vz + l_sup_p + l_sup_T

    all_losses = torch.stack([
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,          # 0-4  PDE
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T,     # 5-8  inlet BC
        l_outlet_p,                                          # 9    outlet BC
        l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,         # 10-13 wall BC
        l_sup_vx, l_sup_vy, l_sup_vz, l_sup_p, l_sup_T,   # 14-18 supervised
    ])
    l_unweighted = all_losses.sum()
    l_weighted   = torch.sum(weights * all_losses) - weights.sum()

    l_weighted.backward()
    opt_model.step()
    opt_weights.step()
    with torch.no_grad():
        weights.clamp_(min=1e-6)
    scheduler.step()

    # ── Test MSE ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred  = model(test_pts)
    mse_vx    = torch.mean((pred[:, 0]        - test_vx) ** 2)
    mse_vy    = torch.mean((pred[:, 1]        - test_vy) ** 2)
    mse_vz    = torch.mean((pred[:, 2]        - test_vz) ** 2)
    mse_p     = torch.mean((pred[:, 3]        - test_p)  ** 2)
    mse_T     = torch.mean((pred[:, 4] * 1000 - test_T)  ** 2)
    mse_total = mse_vx + mse_vy + mse_vz + mse_p + mse_T

    # ── Snapshot plots ───────────────────────────────────────
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

    # ── Logging ──────────────────────────────────────────────
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
        "loss/pde_total":        safe_log10(l_pde_total),
        "loss/bc_total":         safe_log10(l_bc_total),
        "loss/sup_total":        safe_log10(l_sup_total),
        "loss/unweighted_total": safe_log10(l_unweighted),
        "loss/weighted_total":   safe_log10(l_weighted),
        "loss/pde_weight":       pde_weight,
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

    print(
        f"[{epoch+1:>5}/{EPOCHS}] pde_w={pde_weight:.2f}  "
        f"PDE: {l_pde_total.item():.3e}  BC: {l_bc_total.item():.3e}  "
        f"SUP: {l_sup_total.item():.3e}  "
        f"MSE vx/vy/vz/p/T: "
        f"{mse_vx.item():.2e}/{mse_vy.item():.2e}/{mse_vz.item():.2e}"
        f"/{mse_p.item():.2e}/{mse_T.item():.2e}"
    )

    if not DEBUG:
        wandb.log(log, step=epoch)

print(f"\nTraining done in {time.time() - start:.1f}s")
os.makedirs(RUN_PATH, exist_ok=True)
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

# Velocity fields (inlet + volume)
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

# Scalar fields (volume only)
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
