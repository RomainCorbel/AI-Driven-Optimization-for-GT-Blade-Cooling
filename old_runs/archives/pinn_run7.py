# reprise de la v5, mais adaptation pour test train eval

import trimesh
import numpy as np
import random
import torch

torch.set_default_dtype(torch.float64)
from utils import *
from unet import *
from torch.optim import Adam
from pinn7 import PINNs, NormalizedPINNs, make_loss_weights, init_weights, get_values_and_derivatives, get_loss
import time
import wandb
import os
from git import Repo
from utils import *
from constants import *

folder = "dp11"
Project_name = "PINNs-5"
device = "cuda"
debug = False

data_folder = "./preProcessedData/with_T/" + folder + "/"
Full_Project_name = Project_name + "_" + folder

# Hyperparams
epochs_adam    = 200
hidden_dim     = 20 # 128
num_layer      = 4 # 4
seed           = 42
lr_adam        = 1e-3
lr_weights     = 1e-3
lr_final       = 1e-7
lr_decay_rate  = (lr_final / lr_adam) ** (1.0 / epochs_adam)

# Sampling
testing_factor          = 5   # multiply all sample counts by this factor to get more accurate validation curves
num_samples           = 100 * testing_factor
n_test                = 500 *  testing_factor

n_volume_pts_4_training         = 200 * testing_factor
n_outlet_surface_pts_4_training  = 20 * testing_factor
n_surface_pts_4_training   = 80 * testing_factor
n_inlet_pts_4_training           = 20 * testing_factor

n_volume_pts_4_validation = 200 * testing_factor
n_outlet_surface_pts_4_validation = 20 * testing_factor
n_surface_pts_4_validation = 80 * testing_factor
n_inlet_pts_4_validation           = 20 * testing_factor

LOSSES = [
    # Physical losses (PDE residuals)
    "divergence",            # continuity equation: du/dx + dv/dy + dw/dz = 0
    "momentum_x",            # x-momentum: u du/dx + v du/dy + w du/dz = -1/rho dp/dx + nu ∇²u
    "momentum_y",            # y-momentum: u dv/dx + v dv/dy + w dv/dz = -1/rho dp/dy + nu ∇²v
    "momentum_z",            # z-momentum: u dw/dx + v dw/dy + w dw/dz = -1/rho dp/dz + nu ∇²w
    "heat_equation",         # u dT/dx + v dT/dy + w dT/dz = alpha ∇²T

    # Boundary condition losses
    "inlet_vx_boundary",     # Dirichlet BC: vx at inlet (from CFD profile)
    "inlet_vy_boundary",     # Dirichlet BC: vy at inlet (from CFD profile)
    "inlet_vz_boundary",     # Dirichlet BC: vz at inlet (from CFD profile)
    "inlet_temp_boundary",   # Dirichlet BC: temperature at inlet (From CFD)
    "outlet_p_boundary",     # Dirichlet BC: avg static pressure at outlet (from CFD)
    "wall_vx_boundary",      # Dirichlet BC: vx = 0 at wall (no-slip)
    "wall_vy_boundary",      # Dirichlet BC: vy = 0 at wall (no-slip)
    "wall_vz_boundary",      # Dirichlet BC: vz = 0 at wall (no-slip)
    "wall_temp_boundary",    # Dirichlet BC: temperature at wall is T_wall (From CFD)

    # Supervised losses
    "supervised_vx",         # supervised loss: vx vs CFD
    "supervised_vy",         # supervised loss: vy vs CFD
    "supervised_vz",         # supervised loss: vz vs CFD
    "supervised_p",          # supervised loss: p vs CFD
    "supervised_T",          # supervised loss: T vs CFD
]

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo = Repo(parent_dir)

if not debug:
    run = wandb.init(
        project=Full_Project_name,
        config={
            "optimizer": "Adam",
            "architecture": "FFNN",
            "epochs_adam": epochs_adam,
            "lr_adam": lr_adam,
            "lr_weights": lr_weights,
            "lr_final": lr_final,
            "lr_decay_rate": lr_decay_rate,
            "adam_betas": (0.99, 0.999),
            "seed": seed,
            "hidden_dim": hidden_dim,
            "num_layers": num_layer,
            "input_standardization": True,
            "output_standardization": True,
            "testing_factor": testing_factor,
            "num_samples": num_samples,
            "n_test": n_test,
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

inlet = np.load(data_folder + "vel_x_inlet.npy")
inlet_perm    = np.random.permutation(inlet.shape[0])[:n_inlet_pts_4_training]
inlet_points  = torch.tensor(inlet[inlet_perm, 0:3]) # (x, y , z at inlet)
vx_inlet_data = torch.tensor(np.load(data_folder + "vel_x_inlet.npy")[inlet_perm, 3]) # (vx at inlet)
vy_inlet_data = torch.tensor(np.load(data_folder + "vel_y_inlet.npy")[inlet_perm, 3]) # (vy at inlet)
vz_inlet_data = torch.tensor(np.load(data_folder + "vel_z_inlet.npy")[inlet_perm, 3]) # (vz at inlet)

data_points = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 0:3]) # ((x,y,z) everywhere)
vx_data     = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 3]) # (vx everywhere)
vy_data     = torch.tensor(np.load(data_folder + "vel_y.npy")[:, 3]) # (vy everywhere)
vz_data     = torch.tensor(np.load(data_folder + "vel_z.npy")[:, 3]) # (vz everywhere)
p_data      = torch.tensor(np.load(data_folder + "press.npy")[:, 3]) # (p everywhere)
temp_data   = torch.tensor(np.load(data_folder + "temp.npy")[:, 3]) # (T everywhere)

# ─────────────────────────────────────────────────────────────────────────────
coord_mean = data_points.mean(dim=0)   # shape (3,)
coord_std  = data_points.std(dim=0)    # shape (3,)
print(f"Coord mean: {coord_mean.tolist()}")
print(f"Coord std:  {coord_std.tolist()}")

# Output stats (use all data so stats are representative of the full domain)
out_mean = torch.stack([
    vx_data.mean(),
    vy_data.mean(),
    vz_data.mean(),
    (p_data / 1e5).mean(),
    (temp_data / 1000).mean(),
])   # shape (5,)
out_std = torch.stack([
    vx_data.std(),
    vy_data.std(),
    vz_data.std(),
    (p_data / 1e5).std(),
    (temp_data / 1000).std(),
])   # shape (5,)
print(f"Output mean: {out_mean.tolist()}")
print(f"Output std:  {out_std.tolist()}")

perm        = torch.randperm(data_points.shape[0])

Num_points_used_for_testing    = 300000   # held out, evaluated once at the end
Num_points_used_for_evaluation = 1000   # fixed set, evaluated every epoch
# Per-epoch training batch sampled from the train pool:
Num_points_used_for_training   = 2000   # physics + BC losses applied on all of these
Num_training_points_supervised = 500    # supervised loss also applied on first N of the training batch

test_idx       = perm[:Num_points_used_for_testing]
eval_idx       = perm[Num_points_used_for_testing : Num_points_used_for_testing + Num_points_used_for_evaluation]
train_pool_idx = perm[Num_points_used_for_testing + Num_points_used_for_evaluation:]

# train_idx   = perm[n_test:]
# test_idx    = perm[:n_test]

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

base_pinn  = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
base_pinn.apply(init_weights)
base_pinn  = base_pinn.double()
pinn_model = NormalizedPINNs(base_pinn, coord_mean, coord_std, out_mean, out_std)

loss_weights = make_loss_weights(19, device)

adam_optimizer = Adam(
    pinn_model.parameters(),
    lr=lr_adam,
    betas=(0.99, 0.999),
)
adam_weights = Adam(
    [loss_weights],
    lr=lr_weights,
    betas=(0.99, 0.999),
    maximize=True,
)
adam_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    adam_optimizer, gamma=lr_decay_rate
)

start_time = time.time()
training_loss_track   = []
validation_loss_track = []

validation_points_raw, validation_labels = sample_points(obj, n_volume_pts_4_validation, n_outlet_surface_pts_4_validation, n_surface_pts_4_validation, n_inlet_pts_4_validation) # sample_points(obj, n_volume_pts, n_outlet_surface_pts, n_other_surface_pts)
validation_points = validation_points_raw / 1000
global_step = 0


def weighted_total_loss(losses_list):
    """
    losses_list: list of 15 scalar tensors in the same order as LOSS_NAMES.
    Returns: sum(w_i * l_i) - sum(w_i)
    """
    losses_stack = torch.stack(losses_list)
    return torch.sum(loss_weights * losses_stack) - torch.sum(loss_weights)


def weights_log_dict():
    return {name: loss_weights[i].item() for i, name in enumerate(LOSSES)}


def run_validation():
    pinn_model.eval()
    adam_optimizer.zero_grad()

    inlet_labels_v = torch.ones(inlet_points.shape[0], dtype=torch.long)
    val_pts_combined = torch.cat([validation_points, inlet_points], dim=0)
    val_labels_combined = torch.cat([validation_labels, inlet_labels_v], dim=0)
    val_pts_combined.requires_grad_(True)
    val_fields = pinn_model(val_pts_combined)

    (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    ) = get_values_and_derivatives(val_fields, val_pts_combined)

    (
        val_loss_divergence,
        val_loss_momentum_x,
        val_loss_momentum_y,
        val_loss_momentum_z,
        val_loss_heat,
        
        val_loss_inlet_vx,
        val_loss_inlet_vy,
        val_loss_inlet_vz,
        val_loss_inlet_T,
        val_loss_outlet_p,
        val_loss_wall_vx,
        val_loss_wall_vy,
        val_loss_wall_vz,
        val_loss_wall_T,
    ) = get_loss(
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
        val_labels_combined,
        p_outlet,
        vx_inlet_data, vy_inlet_data, vz_inlet_data, T_inlet, T_wall,
    )

    val_loss_total = (
        val_loss_divergence
        + val_loss_momentum_x
        + val_loss_momentum_y
        + val_loss_momentum_z
        + val_loss_heat
        + val_loss_inlet_vx
        + val_loss_inlet_vy
        + val_loss_inlet_vz
        + val_loss_inlet_T
        + val_loss_outlet_p
        + val_loss_wall_vx
        + val_loss_wall_vy
        + val_loss_wall_vz
        + val_loss_wall_T
    )

    with torch.no_grad():
        test_fields = pinn_model(test_data_points)
    val_mse_vx = torch.mean((test_fields[:, 0] - test_vx_data) ** 2)
    val_mse_vy = torch.mean((test_fields[:, 1] - test_vy_data) ** 2)
    val_mse_vz = torch.mean((test_fields[:, 2] - test_vz_data) ** 2)
    val_mse_p  = torch.mean((test_fields[:, 3] - test_p_data) ** 2)
    val_mse_T  = torch.mean((test_fields[:, 4] * 1000 - test_temp_data) ** 2) / 10**6
    val_mse_total = val_mse_vx + val_mse_vy + val_mse_vz + val_mse_p + val_mse_T

    def safe_log10(x):
        v = x.item() if hasattr(x, "item") else float(x)
        return float(np.log10(v)) if v > 0 else None

    val_log = {
        "Validation Divergence Loss":    safe_log10(val_loss_divergence),
        "Validation X Momentum Loss":    safe_log10(val_loss_momentum_x),
        "Validation Y Momentum Loss":    safe_log10(val_loss_momentum_y),
        "Validation Z Momentum Loss":    safe_log10(val_loss_momentum_z),
        "Validation Heat Loss":          safe_log10(val_loss_heat),
        "Validation Inlet vx Loss":      safe_log10(val_loss_inlet_vx),
        "Validation Inlet vy Loss":      safe_log10(val_loss_inlet_vy),
        "Validation Inlet vz Loss":      safe_log10(val_loss_inlet_vz),
        "Validation Inlet T Loss":       safe_log10(val_loss_inlet_T),
        "Validation Outlet p Loss":      safe_log10(val_loss_outlet_p),
        "Validation Wall vx Loss":       safe_log10(val_loss_wall_vx),
        "Validation Wall vy Loss":       safe_log10(val_loss_wall_vy),
        "Validation Wall vz Loss":       safe_log10(val_loss_wall_vz),
        "Validation Wall T Loss":        safe_log10(val_loss_wall_T),
        "Validation Total Loss":         safe_log10(val_loss_total),
        "Val MSE vx":                    safe_log10(val_mse_vx),
        "Val MSE vy":                    safe_log10(val_mse_vy),
        "Val MSE vz":                    safe_log10(val_mse_vz),
        "Val MSE p":                     safe_log10(val_mse_p),
        "Val MSE T":                     safe_log10(val_mse_T),
        "Val MSE Total":                 safe_log10(val_mse_total),
    }
    val_log = {k: v for k, v in val_log.items() if v is not None}

    validation_loss_track.append(val_loss_total.item())
    print(
        f"  Val Div: {val_loss_divergence.item():.4e}  "
        f"X-Mom: {val_loss_momentum_x.item():.4e}  "
        f"Outlet p: {val_loss_outlet_p.item():.4e}  "
        f"Wall: {(val_loss_wall_vx + val_loss_wall_vy + val_loss_wall_vz + val_loss_wall_T).item():.4e}  "
        f"Total: {val_loss_total.item():.4e}  "
        f"Val MSE(vx/vy/vz/p/T): {val_mse_vx.item():.2e}/{val_mse_vy.item():.2e}/{val_mse_vz.item():.2e}/{val_mse_p.item():.2e}/{val_mse_T.item():.2e}"
    )

    return val_log


# ─────────────────────────────────────────────────────────────────────────────
# Adam  (resample collocation points each epoch)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Adam optimizer")
print("=" * 60)

for epoch in range(epochs_adam):
    print(f"[Adam] {epoch+1}/{epochs_adam}")

    val_log = run_validation()

    train_points, train_labels = sample_points(obj, n_volume_pts_4_training, n_outlet_surface_pts_4_training, n_surface_pts_4_training, n_inlet_pts_4_training)
    train_points = train_points / 1000
    inlet_labels_t = torch.ones(inlet_points.shape[0], dtype=torch.long)
    all_train_pts = torch.cat([train_points, inlet_points], dim=0)
    all_train_labels = torch.cat([train_labels, inlet_labels_t], dim=0)
    all_train_pts.requires_grad_(True)

    indices           = torch.randint(0, train_data_points.shape[0], (num_samples,))
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
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        loss_heat,

        loss_inlet_vx,
        loss_inlet_vy,
        loss_inlet_vz,
        loss_inlet_T,
        loss_outlet_p,
        loss_wall_vx,
        loss_wall_vy,
        loss_wall_vz,
        loss_wall_T,
    ) = get_loss(
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
        all_train_labels,
        p_outlet,
        vx_inlet_data, vy_inlet_data, vz_inlet_data, T_inlet, T_wall,
    )

    fields_supervised = pinn_model(sampled_points)
    sup_vx = torch.mean((fields_supervised[:, 0] - vx_sampled_data) ** 2)
    sup_vy = torch.mean((fields_supervised[:, 1] - vy_sampled_data) ** 2)
    sup_vz = torch.mean((fields_supervised[:, 2] - vz_sampled_data) ** 2)
    sup_p  = torch.mean((fields_supervised[:, 3] - p_sampled_data) ** 2)
    sup_T  = torch.mean((fields_supervised[:, 4] * 1000 - temp_sampled_data) ** 2) / 10**6

    loss_total = weighted_total_loss([
        # physical
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        loss_heat,
        # bc
        loss_inlet_vx,
        loss_inlet_vy,
        loss_inlet_vz,
        loss_inlet_T,
        loss_outlet_p,
        loss_wall_vx,
        loss_wall_vy,
        loss_wall_vz,
        loss_wall_T,
        # supervised
        sup_vx,
        sup_vy,
        sup_vz,
        sup_p,
        sup_T,
    ])

    loss_total.backward()
    adam_optimizer.step()   # minimise L pour θ
    adam_weights.step()     # maximise L pour w
    with torch.no_grad():
        loss_weights.clamp_(min=1e-6)   # w_i > 0 : évite qu'un poids négatif inverse l'objectif
    adam_lr_scheduler.step()

    training_loss_track.append(loss_total.item())
    train_log = {
        "Divergence Loss":      np.log10(loss_divergence.item()),
        "X Momentum Loss":      np.log10(loss_momentum_x.item()),
        "Y Momentum Loss":      np.log10(loss_momentum_y.item()),
        "Z Momentum Loss":      np.log10(loss_momentum_z.item()),
        "Heat Loss":            np.log10(loss_heat.item()),
        "Inlet vx Loss":        np.log10(loss_inlet_vx.item()),
        "Inlet vy Loss":        np.log10(loss_inlet_vy.item()),
        "Inlet vz Loss":        np.log10(loss_inlet_vz.item()),
        "Inlet T Loss":         np.log10(loss_inlet_T.item()),
        "Outlet p Loss":        np.log10(loss_outlet_p.item()),
        "Wall vx Loss":         np.log10(loss_wall_vx.item()),
        "Wall vy Loss":         np.log10(loss_wall_vy.item()),
        "Wall vz Loss":         np.log10(loss_wall_vz.item()),
        "Wall T Loss":          np.log10(loss_wall_T.item()),
        "Supervised vx Loss":   np.log10(sup_vx.item()),
        "Supervised vy Loss":   np.log10(sup_vy.item()),
        "Supervised vz Loss":   np.log10(sup_vz.item()),
        "Supervised p Loss":    np.log10(sup_p.item()),
        "Supervised T Loss":    np.log10(sup_T.item()),
        "Total Loss":           np.log10(loss_total.item()),
        "LR": adam_lr_scheduler.get_last_lr()[0],
        **weights_log_dict(),
    }
    print(
        f"  Div: {loss_divergence.item():.4e}  "
        f"X-Mom: {loss_momentum_x.item():.4e}  "
        f"Wall vx: {loss_wall_vx.item():.4e}  "
        f"Sup(vx/vy/vz/p/T): {sup_vx.item():.2e}/{sup_vy.item():.2e}/{sup_vz.item():.2e}/{sup_p.item():.2e}/{sup_T.item():.2e}  "
        f"Total: {loss_total.item():.4e}  "
        f"Weights: {[f'{w:.3f}' for w in loss_weights.detach().cpu().tolist()]}"
    )

    if not debug:
        wandb.log({**train_log, **val_log}, step=global_step)
    global_step += 1


stop_time = time.time()
print(f"Time taken for training: {stop_time - start_time:.1f}s")
# Save the full NormalizedPINNs state dict (includes all normalization buffers)
torch.save(pinn_model.state_dict(), "../run/pinn_model_v5.pt")
torch.save(loss_weights.detach().cpu(), "../run/loss_weights_v5.pt")


## Plotting
pinn_model.eval()
with torch.no_grad():
    validation_fields = pinn_model(validation_points)

fields = [
    ("vx", validation_fields[:, 0].cpu().detach().numpy()),
    ("vy", validation_fields[:, 1].cpu().detach().numpy()),
    ("vz", validation_fields[:, 2].cpu().detach().numpy()),
    ("p",  validation_fields[:, 3].cpu().detach().numpy()),
    ("T",  validation_fields[:, 4].cpu().detach().numpy() * 1000),
]
plot_fields(fields, validation_points)


######## Inference
# NormalizedPINNs handles redimensionalization: physical coordinates go in,
# the wrapper normalizes them internally, and outputs are in physical units.

device = "cpu"
torch.set_default_device(device)
base_pinn_inf  = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
pinn_model_inf = NormalizedPINNs(
    base_pinn_inf,
    coord_mean.to(device), coord_std.to(device),
    out_mean.to(device),   out_std.to(device),
)
pinn_model_inf.load_state_dict(
    torch.load("../run/pinn_model_v5.pt", weights_only=True, map_location=torch.device("cpu"))
)

all_points = torch.tensor(
    np.concatenate(
        (
            np.load(os.path.join(data_folder, "vel_x_inlet.npy"))[:, :3],
            np.load(os.path.join(data_folder, "vel_x.npy"))[:, :3],
        )
    )
)

all_fields = pinn_model_inf(all_points)

vx_pred = all_fields[:, 0].cpu().detach().numpy()
vy_pred = all_fields[:, 1].cpu().detach().numpy()
vz_pred = all_fields[:, 2].cpu().detach().numpy()
p_pred  = all_fields[:, 3].cpu().detach().numpy()
T_pred  = (all_fields[:, 4] * 1000).cpu().detach().numpy()

plot_aginast_data(data_folder, vx_pred, vy_pred, vz_pred, p_pred, T_pred)

time.sleep(120)
if repo.is_dirty(untracked_files=True):
    print("Repository has changes, preparing to commit.")
    repo.git.add(A=True)
    commit_message = f"Running job with run name: {run.name}, url: {run.url}"
    repo.index.commit(commit_message)
    print(f"Committed changes with message: {commit_message}")
    origin = repo.remote(name="origin")
    origin.push()
    print("Pushed changes to the remote repository.")
else:
    print("No changes to commit.")