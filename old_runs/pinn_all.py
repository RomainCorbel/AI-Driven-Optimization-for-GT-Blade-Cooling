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
from git import Repo
from torch.optim import Adam

torch.set_default_dtype(torch.float64)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
# =============================================================================
# CONSTANTS
# =============================================================================

# --- Boundary conditions (fixed inputs) ---
T_inlet  = 298.15          # [K]  Inlet temperature (25 deg)
T_wall   = 298.15          # [K]  Wall temperature (25 deg)

P_ATM    = 101_325         # [Pa] Standard atmospheric pressure
P_GAUGE  = 17_825          # [Pa] Gauge pressure offset
p_outlet = 0.835           # [bar]  Outlet static pressure boundary condition (= 83500 Pa)

# --- Air thermophysical properties (constant, from literature) ---
M                    = 28.96e-3    # [kg/mol]  Molar mass of air
R                    = 8.314       # [J/mol/K] Universal gas constant
thermal_conductivity = 2.61e-2     # [W/m/K]   Air thermal conductivity
specific_heat        = 1.00e3      # [J/kg/K]  Air specific heat capacity (cp)

# --- Sutherland's law reference values ---
T_ref_const = 278.15   # [K]       Reference temperature
mu_ref      = 1.716e-5 # [Pa·s]    Dynamic viscosity at T_ref
S           = 110.4    # [K]       Sutherland constant

# --- Derived constant (computed once from the above) ---
# Ideal gas law: rho = p / (r_specific * T), with r_specific = R/M
r_specific = R / M                                      # [J/kg/K]
rho    = (p_outlet * 1e5) * M / (R * T_ref_const)  # [kg/m³]  ≈ 0.9718
alpha = thermal_conductivity / (rho * specific_heat)

# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY SAMPLING  (labels: 0=interior, 1=inlet, 2=outlet, 3=wall)
# ═════════════════════════════════════════════════════════════════════════════
def get_inlet_surface_points(obj, num_points):
    threshold = 1e-5
    faces_x_zero = [
        i for i, face in enumerate(obj.faces)
        if np.all(np.abs(obj.vertices[face, 0]) < threshold)
    ]
    subset_mesh = obj.submesh([faces_x_zero], only_watertight=False)[0]
    points, _ = trimesh.sample.sample_surface(subset_mesh, count=num_points)
    pts    = torch.tensor(points, dtype=torch.float64)
    labels = torch.ones(pts.size(0), dtype=torch.int64)          # label 1
    return pts, labels


def get_outlet_surface_points(obj, num_points):
    threshold = 1e-5
    x_max = obj.vertices[:, 0].max()
    faces_x_max = [
        i for i, face in enumerate(obj.faces)
        if np.all(np.abs(obj.vertices[face, 0] - x_max) < threshold)
    ]
    subset_mesh = obj.submesh([faces_x_max], only_watertight=False)[0]
    points, _ = trimesh.sample.sample_surface(subset_mesh, count=num_points)
    pts    = torch.tensor(points, dtype=torch.float64)
    labels = 2 * torch.ones(pts.size(0), dtype=torch.int64)      # label 2  (was 3 — bug fix)
    return pts, labels


def get_wall_surface_points(obj, num_points):
    threshold = 1e-5
    x_max = obj.vertices[:, 0].max()
    points, _ = trimesh.sample.sample_surface(obj, count=num_points)
    mask = (
        (np.abs(points[:, 0]) > threshold)
        & ~((x_max - threshold <= points[:, 0]) & (points[:, 0] <= x_max + threshold))
    )
    pts    = torch.tensor(points[mask], dtype=torch.float64)
    labels = 3 * torch.ones(pts.size(0), dtype=torch.int64)      # label 3  (was 2 — bug fix)
    return pts, labels


def get_volume_points(obj, num_points):
    pts    = torch.tensor(trimesh.sample.volume_mesh(obj, num_points), dtype=torch.float64)
    labels = torch.zeros(pts.size(0), dtype=torch.int64)          # label 0
    return pts, labels


def sample_points(obj, num_volume_points, num_outlet_surface_points,
                  num_wall_surface_points, num_inlet_surface_points=0, shuffle=True):
    parts_pts, parts_lbl = [], []

    if num_inlet_surface_points > 0:
        p, l = get_inlet_surface_points(obj, num_inlet_surface_points)
        parts_pts.append(p); parts_lbl.append(l)

    p, l = get_outlet_surface_points(obj, num_outlet_surface_points)
    parts_pts.append(p); parts_lbl.append(l)

    p, l = get_wall_surface_points(obj, num_wall_surface_points)
    parts_pts.append(p); parts_lbl.append(l)

    p, l = get_volume_points(obj, num_volume_points)
    parts_pts.append(p); parts_lbl.append(l)

    all_points = torch.cat(parts_pts, dim=0)
    all_labels = torch.cat(parts_lbl, dim=0)

    print(
        f"Number of Outlet surface points: {num_outlet_surface_points}, "
        f"Number of volume points: {num_volume_points}, "
        f"Number of Wall surface points: {num_wall_surface_points}"
    )

    if shuffle:
        perm = torch.randperm(all_points.size(0))
        return all_points[perm], all_labels[perm]
    return all_points, all_labels


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def dynamic_viscosity(T, mu_ref=1.716e-5, T_ref=273.15, S=110.4):
    mu = (
        mu_ref
        * ((torch.abs(T) / T_ref) ** 1.5)
        * ((T_ref + S) / (torch.abs(T) + S))
    )
    return mu


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_fields(pts, fields, output_dir="../run", suffix_tag=""):
    os.makedirs(output_dir, exist_ok=True)

    def to_np(x):
        if x is None:
            return None
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pts_np = to_np(pts)
    n_pts  = min(15000, len(pts_np))
    idx    = np.random.choice(len(pts_np), n_pts, replace=False)
    p      = pts_np[idx]

    for name, data_raw, pred_raw in fields:
        data     = to_np(data_raw)
        pred     = to_np(pred_raw)
        has_data = data is not None

        vmin = float(min(data.min(), pred.min()) if has_data else pred.min())
        vmax = float(max(data.max(), pred.max()) if has_data else pred.max())

        subplots = []
        if has_data:
            subplots.append((f"Data – {name}", data,        vmin, vmax))
        subplots.append(    (f"Pred – {name}", pred,        vmin, vmax))
        if has_data:
            subplots.append((f"Diff – {name}", data - pred, None, None))

        n = len(subplots)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), subplot_kw={"projection": "3d"})
        if n == 1:
            axes = [axes]

        for ax, (title, color, cmin, cmax) in zip(axes, subplots):
            sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                            c=color[idx], cmap="viridis",
                            vmin=cmin, vmax=cmax, s=1, rasterized=True)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            plt.colorbar(sc, ax=ax, shrink=0.5, label=name)

        fig.tight_layout()
        tag = f"_{suffix_tag}" if suffix_tag else ""
        fig.savefig(os.path.join(output_dir, f"{name}{tag}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════════

class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super().__init__()
        layers = []
        for i in range(num_layer - 1):
            in_f  = in_dim if i == 0 else hidden_dim
            layers += [nn.Linear(in_f, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class NormalizedPINNs(nn.Module):
    """Z-scores inputs; unstandardizes outputs. Physical units in, physical units out."""

    def __init__(self, pinn, coord_mean, coord_std, out_mean, out_std):
        super().__init__()
        self.pinn = pinn
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std",  coord_std)
        self.register_buffer("out_mean",   out_mean)
        self.register_buffer("out_std",    out_std)

    def forward(self, x):
        x_norm   = (x - self.coord_mean) / self.coord_std
        out_norm = self.pinn(x_norm)
        safe_std = torch.where(self.out_std == 0, torch.ones_like(self.out_std), self.out_std)
        return out_norm * safe_std + self.out_mean
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        a = 0.7 # 2 * 1.0 / m.weight.size(1) ** 0.5         # let s try to initial the weights uniformly instead of putting 0, that was satisfying some losses for no reason!
        nn.init.uniform_(m.weight, -a, a)
        nn.init.uniform_(m.bias,   -a, a)

def make_loss_weights(n, device):
    return nn.Parameter(torch.ones(n, dtype=torch.float64, device=device))


# ═════════════════════════════════════════════════════════════════════════════
# DERIVATIVES + LOSSES
# ═════════════════════════════════════════════════════════════════════════════

def get_values_and_derivatives(fields, points):
    vx = fields[:, 0:1]
    vy = fields[:, 1:2]
    vz = fields[:, 2:3]
    p  = fields[:, 3:4]
    T  = fields[:, 4:5] * 1000

    def grad1(f):
        return torch.autograd.grad(f, points, torch.ones_like(f),
                                   retain_graph=True, create_graph=True)[0]
    def grad2(f, dim):
        g = grad1(f)[:, dim:dim+1]
        return torch.autograd.grad(g, points, torch.ones_like(g),
                                   retain_graph=True, create_graph=True)[0][:, dim:dim+1]

    gvx = grad1(vx);  vx_x, vx_y, vx_z = gvx[:, 0:1], gvx[:, 1:2], gvx[:, 2:3]
    vx_xx, vx_yy, vx_zz = grad2(vx, 0), grad2(vx, 1), grad2(vx, 2)

    gvy = grad1(vy);  vy_x, vy_y, vy_z = gvy[:, 0:1], gvy[:, 1:2], gvy[:, 2:3]
    vy_xx, vy_yy, vy_zz = grad2(vy, 0), grad2(vy, 1), grad2(vy, 2)

    gvz = grad1(vz);  vz_x, vz_y, vz_z = gvz[:, 0:1], gvz[:, 1:2], gvz[:, 2:3]
    vz_xx, vz_yy, vz_zz = grad2(vz, 0), grad2(vz, 1), grad2(vz, 2)

    gp = grad1(p);    p_x, p_y, p_z = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]

    gT = grad1(T);    T_x, T_y, T_z = gT[:, 0:1], gT[:, 1:2], gT[:, 2:3]
    T_xx, T_yy, T_zz = grad2(T, 0), grad2(T, 1), grad2(T, 2)

    return (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    )


def get_loss(
    vx, vy, vz, p, T,
    vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
    vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
    vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
    p_x, p_y, p_z,
    T_x, T_y, T_z, T_xx, T_yy, T_zz,
    labels,
    p_outlet_val,
    vx_inlet, vy_inlet, vz_inlet, T_inlet_val,
    T_wall_val, alpha
):
    interior = labels == 0   # PDE residuals everywhere inside
    inlet    = labels == 1
    outlet   = labels == 2   # pressure BC
    wall     = labels == 3   # no-slip + temperature BC

    loss_divergence = torch.mean((vx_x + vy_y + vz_z) ** 2)

    mu = dynamic_viscosity(T)

    loss_momentum_x = torch.mean((
        vx[interior] * vx_x[interior] + vy[interior] * vx_y[interior] + vz[interior] * vx_z[interior]
        + (10**5 / rho) * p_x[interior]
        - (mu[interior] / rho) * (vx_xx[interior] + vx_yy[interior] + vx_zz[interior])
    ) ** 2)

    loss_momentum_y = torch.mean((
        vx[interior] * vy_x[interior] + vy[interior] * vy_y[interior] + vz[interior] * vy_z[interior]
        + (10**5 / rho) * p_y[interior]
        - (mu[interior] / rho) * (vy_xx[interior] + vy_yy[interior] + vy_zz[interior])
    ) ** 2)

    loss_momentum_z = torch.mean((
        vx[interior] * vz_x[interior] + vy[interior] * vz_y[interior] + vz[interior] * vz_z[interior]
        + (10**5 / rho) * p_z[interior]
        - (mu[interior] / rho) * (vz_xx[interior] + vz_yy[interior] + vz_zz[interior])
    ) ** 2)
    
    loss_heat = torch.mean((
        alpha * (T_xx[interior] + T_yy[interior] + T_zz[interior])
        + vx[interior] * T_x[interior] + vy[interior] * T_y[interior] + vz[interior] * T_z[interior]
    ) ** 2) / 10**6 # (to keep it in a similar range as the other losses, ex: vx * T_x  ≈  50 m/s  ×  1000 K/m  =  50,000 K/s so residual² ≈ (50,000)² = 2.5×10⁹ / 10⁶ = 2,500 comparable to momentum losses)

    loss_inlet_vx = torch.mean((vx[inlet].squeeze(1) - vx_inlet) ** 2)
    loss_inlet_vy = torch.mean((vy[inlet].squeeze(1) - vy_inlet) ** 2)
    loss_inlet_vz = torch.mean((vz[inlet].squeeze(1) - vz_inlet) ** 2)
    loss_inlet_T  = torch.mean((T[inlet]  - T_inlet_val) ** 2) # deleted /10^6 to keep it in a similar range as the other losses (inlet T error is typically around 10 K, so squared error is around 100, which is comparable to momentum losses), dividing by 10^6 is too much and makes the loss very small and hard to optimize

    loss_outlet_p = torch.mean((p[outlet] - p_outlet_val) ** 2)

    loss_wall_vx = torch.mean(vx[wall] ** 2)
    loss_wall_vy = torch.mean(vy[wall] ** 2)
    loss_wall_vz = torch.mean(vz[wall] ** 2)
    loss_wall_T  = torch.mean((T[wall] - T_wall_val) ** 2)

    return (
        loss_divergence,
        loss_momentum_x, loss_momentum_y, loss_momentum_z,
        loss_heat,
        loss_inlet_vx, loss_inlet_vy, loss_inlet_vz, loss_outlet_p, loss_inlet_T,
        loss_wall_vx, loss_wall_vy, loss_wall_vz, loss_wall_T,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RUN CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

folder           = "dp11"
Project_name     = "PINNs-ALL_scitas"
device           = "cuda"
debug            = False

data_folder      = "./preProcessedData/with_T/" + folder + "/"
Full_Project_name = Project_name + "_" + folder

epochs_adam    = 100
hidden_dim     = 50
num_layer      = 4
seed           = 42
lr_adam        = 1e-3
lr_weights     = 1e-4
lr_final       = 1e-7
lr_decay_rate  = (lr_final / lr_adam) ** (1.0 / epochs_adam)

testing_factor = 100

Num_points_used_for_testing    = 100 * testing_factor
Num_points_used_for_training_per_epoch = 500 * testing_factor # per epoch because all the points that are not in the test set are in the training set, and we sample a new subset of them every epoch

num_supervised_train_points    = 0 # int(0.1  * Num_points_used_for_training_per_epoch)
Num_points_used_for_validation_per_epoch = 200 * testing_factor

n_volume_pts_4_training          = int(0.5  * Num_points_used_for_training_per_epoch)
n_outlet_surface_pts_4_training  = int(0.1  * Num_points_used_for_training_per_epoch)
n_surface_pts_4_training         = int(0.3  * Num_points_used_for_training_per_epoch)
n_inlet_pts_4_training           = int(0.1  * Num_points_used_for_training_per_epoch)

n_volume_pts_4_validation          = int(0.5  * Num_points_used_for_validation_per_epoch)
n_outlet_surface_pts_4_validation  = int(0.1  * Num_points_used_for_validation_per_epoch)
n_surface_pts_4_validation         = int(0.3  * Num_points_used_for_validation_per_epoch)
n_inlet_pts_4_validation           = int(0.1  * Num_points_used_for_validation_per_epoch)

LOSSES = [
    "divergence", "momentum_x", "momentum_y", "momentum_z", "heat_equation",
    "inlet_vx_boundary", "inlet_vy_boundary", "inlet_vz_boundary", "outlet_p_boundary", "inlet_temp_boundary",
    "wall_vx_boundary", "wall_vy_boundary", "wall_vz_boundary", "wall_temp_boundary",
    "supervised_vx", "supervised_vy", "supervised_vz", "supervised_p", "supervised_T",
]

# ═════════════════════════════════════════════════════════════════════════════
# INIT
# ═════════════════════════════════════════════════════════════════════════════

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo = Repo(parent_dir)

if not debug:
    run = wandb.init(
        project=Full_Project_name,
        config={
            "optimizer": "Adam", "architecture": "FFNN",
            "epochs_adam": epochs_adam, "lr_adam": lr_adam,
            "lr_weights": lr_weights, "lr_final": lr_final,
            "lr_decay_rate": lr_decay_rate, "adam_betas": (0.99, 0.999),
            "seed": seed, "hidden_dim": hidden_dim, "num_layers": num_layer,
            "input_standardization": True, "output_standardization": True,
            "testing_factor": testing_factor, 
            "num_supervised_train_points": num_supervised_train_points,
            "n_volume_pts_4_training": n_volume_pts_4_training,
            "n_outlet_surface_pts_4_training": n_outlet_surface_pts_4_training,
            "n_surface_pts_4_training": n_surface_pts_4_training,
            "n_inlet_pts_4_training": n_inlet_pts_4_training,
            "n_volume_pts_4_validation": n_volume_pts_4_validation,
            "n_outlet_surface_pts_4_validation": n_outlet_surface_pts_4_validation,
            "n_surface_pts_4_validation": n_surface_pts_4_validation,
            "n_inlet_pts_4_validation": n_inlet_pts_4_validation,
        },
    )

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_default_device(device)
print(f"Using device: {device}")

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

# Inlet: coordinates + ground-truth velocities from CSV (same permutation → aligned)
inlet_raw     = np.load(data_folder + "vel_x_inlet.npy")
inlet_perm    = np.random.permutation(inlet_raw.shape[0])[:n_inlet_pts_4_training]
inlet_points  = torch.tensor(inlet_raw[inlet_perm, 0:3])
vx_inlet_data = torch.tensor(np.load(data_folder + "vel_x_inlet.npy")[inlet_perm, 3])
vy_inlet_data = torch.tensor(np.load(data_folder + "vel_y_inlet.npy")[inlet_perm, 3])
vz_inlet_data = torch.tensor(np.load(data_folder + "vel_z_inlet.npy")[inlet_perm, 3])

# Volume data for supervised loss and MSE tracking
data_points = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 0:3])
vx_data     = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 3])
vy_data     = torch.tensor(np.load(data_folder + "vel_y.npy")[:, 3])
vz_data     = torch.tensor(np.load(data_folder + "vel_z.npy")[:, 3])
p_data      = torch.tensor(np.load(data_folder + "press.npy")[:, 3])
temp_data   = torch.tensor(np.load(data_folder + "temp.npy")[:, 3])

# Normalisation statistics
coord_mean = data_points.mean(dim=0)
coord_std  = data_points.std(dim=0)
print(f"Coord mean: {coord_mean.tolist()}")
print(f"Coord std:  {coord_std.tolist()}")

out_mean = torch.stack([
    vx_data.mean(), vy_data.mean(), vz_data.mean(),
    (p_data / 1e5).mean(), (temp_data / 1000).mean(),
])
out_std = torch.stack([
    vx_data.std(), vy_data.std(), vz_data.std(),
    (p_data / 1e5).std(), (temp_data / 1000).std(),
])
print(f"Output mean: {out_mean.tolist()}")
print(f"Output std:  {out_std.tolist()}")

# Train / test split
perm = torch.randperm(data_points.shape[0])

test_idx       = perm[:Num_points_used_for_testing]   # first Num_points_used_for_testing for testing
train_pool_idx = perm[Num_points_used_for_testing:]   # everything else → training pool

train_data_points = data_points[train_pool_idx]
train_vx_data     = vx_data[train_pool_idx]
train_vy_data     = vy_data[train_pool_idx]
train_vz_data     = vz_data[train_pool_idx]
train_p_data      = p_data[train_pool_idx] / 10**5
train_temp_data   = temp_data[train_pool_idx]

test_data_points = data_points[test_idx]
test_vx_data     = vx_data[test_idx]
test_vy_data     = vy_data[test_idx]
test_vz_data     = vz_data[test_idx]
test_p_data      = p_data[test_idx] / 10**5
test_temp_data   = temp_data[test_idx]

obj = trimesh.load("./Baseline_ML4Science.stl")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL + OPTIMIZERS
# ═════════════════════════════════════════════════════════════════════════════

base_pinn  = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
base_pinn.apply(init_weights)
base_pinn  = base_pinn.double()
pinn_model = NormalizedPINNs(base_pinn, coord_mean, coord_std, out_mean, out_std)

loss_weights = make_loss_weights(19, device)

adam_optimizer    = Adam(pinn_model.parameters(), lr=lr_adam, betas=(0.99, 0.999))
adam_weights      = Adam([loss_weights],           lr=lr_weights, betas=(0.99, 0.999), maximize=True)
adam_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=lr_decay_rate)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def weighted_total_loss(losses_list):
    losses_stack = torch.stack(losses_list)
    return torch.sum(loss_weights * losses_stack) - torch.sum(loss_weights)


def weights_log_dict():
    return {name: loss_weights[i].item() for i, name in enumerate(LOSSES)}


def safe_log10(x):
    v = x.item() if hasattr(x, "item") else float(x)
    return float(np.log10(v)) if v > 0 else None


# Fixed validation collocation points (no STL inlet — handled by inlet_points from CSV)
validation_points_raw, validation_labels = sample_points(
    obj,
    n_volume_pts_4_validation,
    n_outlet_surface_pts_4_validation,
    n_surface_pts_4_validation,
    num_inlet_surface_points=0,   # inlet BCs come from CSV inlet_points only
)
validation_points = validation_points_raw / 1000

inlet_labels_v       = torch.ones(inlet_points.shape[0], dtype=torch.long)
val_pts_combined     = torch.cat([validation_points, inlet_points], dim=0)
val_labels_combined  = torch.cat([validation_labels, inlet_labels_v], dim=0)

global_step = 0


def run_validation():
    pinn_model.eval()
    adam_optimizer.zero_grad()

    pts = val_pts_combined.clone().requires_grad_(True)
    val_fields = pinn_model(pts)

    (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    ) = get_values_and_derivatives(val_fields, pts)

    (
        val_loss_divergence,
        val_loss_momentum_x, val_loss_momentum_y, val_loss_momentum_z,
        val_loss_heat,
        val_loss_inlet_vx, val_loss_inlet_vy, val_loss_inlet_vz, val_loss_outlet_p, val_loss_inlet_T,
        val_loss_wall_vx, val_loss_wall_vy, val_loss_wall_vz, val_loss_wall_T,
    ) = get_loss(
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
        val_labels_combined,
        p_outlet,
        vx_inlet_data, vy_inlet_data, vz_inlet_data, T_inlet, T_wall, alpha
    )

    with torch.no_grad():
        test_fields = pinn_model(test_data_points)
    val_mse_vx = torch.mean((test_fields[:, 0] - test_vx_data) ** 2)
    val_mse_vy = torch.mean((test_fields[:, 1] - test_vy_data) ** 2)
    val_mse_vz = torch.mean((test_fields[:, 2] - test_vz_data) ** 2)
    val_mse_p  = torch.mean((test_fields[:, 3] - test_p_data)  ** 2)
    val_mse_T  = torch.mean((test_fields[:, 4] * 1000 - test_temp_data) ** 2) / 10**6
    val_mse_total = val_mse_vx + val_mse_vy + val_mse_vz + val_mse_p + val_mse_T

    val_log = {
        "Validation Divergence Loss":  safe_log10(val_loss_divergence),
        "Validation X Momentum Loss":  safe_log10(val_loss_momentum_x),
        "Validation Y Momentum Loss":  safe_log10(val_loss_momentum_y),
        "Validation Z Momentum Loss":  safe_log10(val_loss_momentum_z),
        "Validation Heat Loss":        safe_log10(val_loss_heat),
        "total_validation_PDE_loss":   safe_log10(val_loss_divergence + val_loss_momentum_x + val_loss_momentum_y + val_loss_momentum_z + val_loss_heat),
        
        "Validation Inlet vx Loss":    safe_log10(val_loss_inlet_vx),
        "Validation Inlet vy Loss":    safe_log10(val_loss_inlet_vy),
        "Validation Inlet vz Loss":    safe_log10(val_loss_inlet_vz),
        "Validation Inlet T Loss":     safe_log10(val_loss_inlet_T),
        "Validation Outlet p Loss":    safe_log10(val_loss_outlet_p),
        "Validation Wall vx Loss":     safe_log10(val_loss_wall_vx),
        "Validation Wall vy Loss":     safe_log10(val_loss_wall_vy),
        "Validation Wall vz Loss":     safe_log10(val_loss_wall_vz),
        "Validation Wall T Loss":      safe_log10(val_loss_wall_T),
        "total_validation_boundary_loss":  safe_log10(val_loss_inlet_vx + val_loss_inlet_vy + val_loss_inlet_vz + val_loss_inlet_T + val_loss_outlet_p + val_loss_wall_vx + val_loss_wall_vy + val_loss_wall_vz + val_loss_wall_T),
        
        "Val MSE vx":                  safe_log10(val_mse_vx),
        "Val MSE vy":                  safe_log10(val_mse_vy),
        "Val MSE vz":                  safe_log10(val_mse_vz),
        "Val MSE p":                   safe_log10(val_mse_p),
        "Val MSE T":                   safe_log10(val_mse_T),
        "Val MSE Total":               safe_log10(val_mse_total),
        "Validation supervised Total Loss":       safe_log10(val_loss_total),

        "validation total loss": safe_log10(va
    }
    val_log = {k: v for k, v in val_log.items() if v is not None}

    print(
        f"  Val Div: {val_loss_divergence.item():.4e}  "
        f"X-Mom: {val_loss_momentum_x.item():.4e}  "
        f"Outlet p: {val_loss_outlet_p.item():.4e}  "
        f"Wall: {(val_loss_wall_vx + val_loss_wall_vy + val_loss_wall_vz + val_loss_wall_T).item():.4e}  "
        f"Total: {val_loss_total.item():.4e}  "
        f"Val MSE(vx/vy/vz/p/T): "
        f"{val_mse_vx.item():.2e}/{val_mse_vy.item():.2e}/{val_mse_vz.item():.2e}"
        f"/{val_mse_p.item():.2e}/{val_mse_T.item():.2e}"
    )
    return val_log


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

start_time = time.time()
print("=" * 60)
print("Adam optimizer")
print("=" * 60)

for epoch in range(epochs_adam):
    print(f"[Adam] {epoch+1}/{epochs_adam}")

    val_log = run_validation()

    # Collocation points from STL (no STL inlet — CSV inlet_points carry BCs)
    train_points, train_labels = sample_points(
        obj,
        n_volume_pts_4_training,
        n_outlet_surface_pts_4_training,
        n_surface_pts_4_training,
        num_inlet_surface_points=0,
    )
    train_points = train_points / 1000

    inlet_labels_t  = torch.ones(inlet_points.shape[0], dtype=torch.long)
    all_train_pts   = torch.cat([train_points, inlet_points], dim=0)
    all_train_labels = torch.cat([train_labels, inlet_labels_t], dim=0)
    all_train_pts.requires_grad_(True)

    # Random batch from CFD data for supervised loss
    indices           = torch.randint(0, train_data_points.shape[0], (num_supervised_train_points,))
    sampled_points    = train_data_points[indices]
    vx_sampled_data   = train_vx_data[indices]
    vy_sampled_data   = train_vy_data[indices]
    vz_sampled_data   = train_vz_data[indices]
    p_sampled_data    = train_p_data[indices]
    temp_sampled_data = train_temp_data[indices]

    pinn_model.train()
    adam_optimizer.zero_grad()
    adam_weights.zero_grad()

    train_fields = pinn_model(all_train_pts)
    (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    ) = get_values_and_derivatives(train_fields, all_train_pts)

    (
        loss_divergence,
        loss_momentum_x, loss_momentum_y, loss_momentum_z,
        loss_heat,
        loss_inlet_vx, loss_inlet_vy, loss_inlet_vz, loss_outlet_p, loss_inlet_T,
        loss_wall_vx, loss_wall_vy, loss_wall_vz, loss_wall_T,
    ) = get_loss(
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
        all_train_labels,
        p_outlet,
        vx_inlet_data, vy_inlet_data, vz_inlet_data, T_inlet, T_wall, alpha
    )

    if num_supervised_train_points > 0:
        fields_supervised = pinn_model(sampled_points)
        sup_vx = torch.mean((fields_supervised[:, 0] - vx_sampled_data) ** 2)
        sup_vy = torch.mean((fields_supervised[:, 1] - vy_sampled_data) ** 2)
        sup_vz = torch.mean((fields_supervised[:, 2] - vz_sampled_data) ** 2)
        sup_p  = torch.mean((fields_supervised[:, 3] - p_sampled_data)  ** 2)
        sup_T  = torch.mean((fields_supervised[:, 4] * 1000 - temp_sampled_data) ** 2)
    else:
        _zero = torch.zeros(1, dtype=torch.float64, device=device).squeeze().detach()
        sup_vx = sup_vy = sup_vz = sup_p = sup_T = _zero

    loss_total = weighted_total_loss([
        loss_divergence,
        loss_momentum_x, loss_momentum_y, loss_momentum_z,
        loss_heat,
        loss_inlet_vx, loss_inlet_vy, loss_inlet_vz, loss_outlet_p, loss_inlet_T,
        loss_wall_vx, loss_wall_vy, loss_wall_vz, loss_wall_T,
        sup_vx, sup_vy, sup_vz, sup_p, sup_T,
    ])

    loss_total.backward()
    adam_optimizer.step()
    adam_weights.step()
    with torch.no_grad():
        loss_weights.clamp_(min=1e-6)
    adam_lr_scheduler.step()

    train_log = {
        "loss_divergence":    np.log10(loss_divergence.item()),
        "loss_momentum_x":    np.log10(loss_momentum_x.item()),
        "loss_momentum_y":    np.log10(loss_momentum_y.item()),
        "loss_momentum_z":    np.log10(loss_momentum_z.item()),
        "loss_heat":          np.log10(loss_heat.item()),
        "total_physical_loss":       np.log10((
            loss_divergence
            + loss_momentum_x + loss_momentum_y + loss_momentum_z
            + loss_heat).item()),
        "loss_inlet_vx":      np.log10(loss_inlet_vx.item()),
        "loss_inlet_vy":      np.log10(loss_inlet_vy.item()),
        "loss_inlet_vz":      np.log10(loss_inlet_vz.item()),
        "loss_inlet_T":       np.log10(loss_inlet_T.item()),
        "loss_outlet_p":      np.log10(loss_outlet_p.item()),
        "loss_wall_vx":       np.log10(loss_wall_vx.item()),
        "loss_wall_vy":       np.log10(loss_wall_vy.item()),
        "loss_wall_vz":       np.log10(loss_wall_vz.item()),
        "loss_wall_T":        np.log10(loss_wall_T.item()),
        "total_boundary_loss": np.log10((
            loss_inlet_vx + loss_inlet_vy + loss_inlet_vz + loss_inlet_T
            + loss_outlet_p
            + loss_wall_vx + loss_wall_vy + loss_wall_vz + loss_wall_T).item()),
        "loss_supervised_vx": safe_log10(sup_vx),
        "loss_supervised_vy": safe_log10(sup_vy),
        "loss_supervised_vz": safe_log10(sup_vz),
        "loss_supervised_p":  safe_log10(sup_p),
        "loss_supervised_T":  safe_log10(sup_T),
        "total_supervised_loss": safe_log10((sup_vx + sup_vy + sup_vz + sup_p + sup_T).item()),
        "loss_total":         np.log10(loss_total.item()),
        "LR": adam_lr_scheduler.get_last_lr()[0],
        **weights_log_dict(),
    }
    train_log = {k: v for k, v in train_log.items() if v is not None}
    sup_str = (
        f"Sup(vx/vy/vz/p/T): "
        f"{sup_vx.item():.2e}/{sup_vy.item():.2e}/{sup_vz.item():.2e}"
        f"/{sup_p.item():.2e}/{sup_T.item():.2e}  "
        if num_supervised_train_points > 0 else ""
    )
    print(
        f"  Div: {loss_divergence.item():.4e}  "
        f"X-Mom: {loss_momentum_x.item():.4e}  "
        f"Wall vx: {loss_wall_vx.item():.4e}  "
        f"{sup_str}"
        f"Total: {loss_total.item():.4e}  "
        f"Weights: {[f'{w:.3f}' for w in loss_weights.detach().cpu().tolist()]}"
    )

    if not debug:
        wandb.log({**train_log, **val_log}, step=global_step)
    global_step += 1

stop_time = time.time()
print(f"Time taken for training: {stop_time - start_time:.1f}s")

torch.save(pinn_model.state_dict(), "../run/pinn_model_v5.pt")
torch.save(loss_weights.detach().cpu(), "../run/loss_weights_v5.pt")

# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE  (CPU, full domain)
# ═════════════════════════════════════════════════════════════════════════════

device = "cpu"
torch.set_default_device(device)
base_pinn_inf  = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
pinn_model_inf = NormalizedPINNs(
    base_pinn_inf,
    coord_mean.to(device), coord_std.to(device),
    out_mean.to(device),   out_std.to(device),
)
pinn_model_inf.load_state_dict(
    torch.load("../run/pinn_model_v5.pt", weights_only=True, map_location="cpu")
)

vx_full    = np.load(os.path.join(data_folder, "vel_x.npy"))
vx_in_full = np.load(os.path.join(data_folder, "vel_x_inlet.npy"))
vy_full    = np.load(os.path.join(data_folder, "vel_y.npy"))
vy_in_full = np.load(os.path.join(data_folder, "vel_y_inlet.npy"))
vz_full    = np.load(os.path.join(data_folder, "vel_z.npy"))
vz_in_full = np.load(os.path.join(data_folder, "vel_z_inlet.npy"))
p_full     = np.load(os.path.join(data_folder, "press.npy"))
T_full     = np.load(os.path.join(data_folder, "temp.npy"))

all_points = torch.tensor(np.concatenate((vx_in_full[:, :3], vx_full[:, :3])))
all_fields_pred = pinn_model_inf(all_points)
vx_pred = all_fields_pred[:, 0].detach().numpy()
vy_pred = all_fields_pred[:, 1].detach().numpy()
vz_pred = all_fields_pred[:, 2].detach().numpy()
p_pred  = all_fields_pred[:, 3].detach().numpy()
T_pred  = (all_fields_pred[:, 4] * 1000).detach().numpy()

sup_idx_np  = train_pool_idx.cpu().numpy()
test_idx_np = test_idx.cpu().numpy()
n_inlet     = vx_in_full.shape[0]

all_pts_np = np.concatenate((vx_in_full[:, :3], vx_full[:, :3]))
index_vel  = np.random.choice(all_pts_np.shape[0], min(100000, all_pts_np.shape[0]), replace=False)
vel_fields = []
for inlet, body, pred_arr, name in [
    (vx_in_full, vx_full, vx_pred, "vx"),
    (vy_in_full, vy_full, vy_pred, "vy"),
    (vz_in_full, vz_full, vz_pred, "vz"),
]:
    true_all = np.concatenate((inlet[:, 3], body[:, 3]))
    vel_fields.append((name, true_all[index_vel], pred_arr[index_vel]))
    print(f"{name} - Train RMSE: {np.sqrt(np.mean((body[:, 3][sup_idx_np]  - pred_arr[sup_idx_np])  ** 2)):.4e}  "
          f"Test RMSE: {np.sqrt(np.mean((body[:, 3][test_idx_np] - pred_arr[test_idx_np]) ** 2)):.4e}")
plot_fields(all_pts_np[index_vel], vel_fields)

index_sc = np.random.choice(vx_full.shape[0], min(50000, vx_full.shape[0]), replace=False)
sc_fields = []
for tag, scalar_raw, pred_slice in [
    ("p", p_full[:, 3] / 1e5, p_pred[n_inlet:]),
    ("T", T_full[:, 3],       T_pred[n_inlet:]),
]:
    sc_fields.append((tag, scalar_raw[index_sc], pred_slice[index_sc]))
    print(f"{tag} - Train RMSE: {np.sqrt(np.mean((scalar_raw[sup_idx_np]  - pred_slice[sup_idx_np])  ** 2)):.4e}  "
          f"Test RMSE: {np.sqrt(np.mean((scalar_raw[test_idx_np] - pred_slice[test_idx_np]) ** 2)):.4e}")
plot_fields(vx_full[:, :3][index_sc], sc_fields)

# ═════════════════════════════════════════════════════════════════════════════
# COMMIT + PUSH
# ═════════════════════════════════════════════════════════════════════════════
# if repo.is_dirty(untracked_files=True):
#     print("Repository has changes, preparing to commit.")
#     repo.git.add(A=True)
#     commit_message = (
#         f"Running job with run name: {run.name}, url: {run.url}"
#         if not debug else "Running job (debug mode)"
#     )
#     repo.index.commit(commit_message)
#     print(f"Committed changes with message: {commit_message}")
#     repo.remote(name="origin").push()
#     print("Pushed changes to the remote repository.")
# else:
#     print("No changes to commit.")
