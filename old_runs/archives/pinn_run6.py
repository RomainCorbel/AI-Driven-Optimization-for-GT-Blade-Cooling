import numpy as np
import random
import torch

torch.set_default_dtype(torch.float64)
from utils6 import *
from unet import *
from torch.optim import Adam
from pinn7 import PINNs, NormalizedPINNs, make_loss_weights, init_weights, get_values_and_derivatives, get_loss
import time
import wandb
import os
from git import Repo
from constants import *
import trimesh

folder = "dp11"
Project_name = "PINNs-6"
device = "cuda"
debug = True

data_folder = "./preProcessedData/with_T/" + folder + "/"
Full_Project_name = Project_name + "_" + folder

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparams
# ─────────────────────────────────────────────────────────────────────────────
epochs_adam   = 200
hidden_dim    = 20
num_layer     = 4
seed          = 42
lr_adam       = 1e-3
lr_weights    = 1e-3
lr_final      = 1e-7
lr_decay_rate = (lr_final / lr_adam) ** (1.0 / epochs_adam)

# Dataset split
Num_points_used_for_testing    = 300000   # held out, evaluated once at the end
Num_points_used_for_evaluation = 1000   # fixed set, evaluated every epoch
# Per-epoch training batch sampled from the train pool:
Num_points_used_for_training   = 2000   # physics + BC losses applied on all of these
Num_training_points_supervised = 500    # supervised loss also applied on first N of the training batch

# The 15 losses passed to weighted_total_loss (order must match make_loss_weights count)
LOSS_NAMES = [
    "divergence",            # continuity equation: du/dx + dv/dy + dw/dz = 0
    "momentum_x",            # x-momentum: u du/dx + v du/dy + w du/dz = -1/rho dp/dx + nu ∇²u
    "momentum_y",            # y-momentum: u dv/dx + v dv/dy + w dv/dz = -1/rho dp/dy + nu ∇²v
    "momentum_z",            # z-momentum: u dw/dx + v dw/dy + w dw/dz = -1/rho dp/dz + nu ∇²w
    "heat_equation",         # u dT/dx + v dT/dy + w dT/dz = alpha ∇²T

    "outlet_p_boundary",     # Dirichlet BC: pressure at outlet
    "wall_no_slip_boundary", # no-slip BC: v = 0 at wall // dv/dn = 0 for velocity components parallel to the wall
    "inlet_temp_boundary",   # Dirichlet BC: temperature at inlet
    "inlet_vx_boundary",     # Dirichlet BC: vx at inlet
    "inlet_vy_boundary",     # Dirichlet BC: vy at inlet
    "inlet_vz_boundary",     # Dirichlet BC: vz at inlet
    "wall_temp_boundary",    # Dirichlet BC: temperature at wall

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
            "hidden_dim": hidden_dim,
            "num_layers": num_layer,
            "Num_points_used_for_testing": Num_points_used_for_testing,
            "Num_points_used_for_evaluation": Num_points_used_for_evaluation,
            "Num_points_used_for_training": Num_points_used_for_training,
            "Num_training_points_supervised": Num_training_points_supervised,
        },
    )

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_default_device(device)
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Load and merge 5 CSV files → data_full of shape (N, 8): x y z vx vy vz p T
# ─────────────────────────────────────────────────────────────────────────────
raw_vx = np.genfromtxt(data_folder + "vel_x.csv", delimiter=',')
raw_vy = np.genfromtxt(data_folder + "vel_y.csv", delimiter=',')
raw_vz = np.genfromtxt(data_folder + "vel_z.csv", delimiter=',')
raw_p  = np.genfromtxt(data_folder + "press.csv", delimiter=',')
raw_T  = np.genfromtxt(data_folder + "temp.csv",  delimiter=',')

coords = raw_vx[:, :3]
full_data = np.concatenate([
    coords,
    raw_vx[:, 3:4], raw_vy[:, 3:4], raw_vz[:, 3:4], raw_p[:, 3:4], raw_T[:, 3:4],
], axis=1)
data_full = torch.from_numpy(full_data).to(device)
print(f"Loaded {data_full.shape[0]} points.")

data_points = data_full[:, :3]
vx_data     = data_full[:, 3]
vy_data     = data_full[:, 4]
vz_data     = data_full[:, 5]
p_data      = data_full[:, 6]
temp_data   = data_full[:, 7]

# ─────────────────────────────────────────────────────────────────────────────
# Input/output standardization (z-score) for NormalizedPINNs
# ─────────────────────────────────────────────────────────────────────────────
# one possible futur fix, the means and std should not be computed on the whole dataset, but only on the training pool. 
means = torch.stack([data_full[:, i].mean() for i in range(data_full.shape[1])])
stds  = torch.stack([data_full[:, i].std()  for i in range(data_full.shape[1])])
coord_mean, out_mean = means[:3], means[3:]
coord_std,  out_std  = stds[:3],  stds[3:]
print(f"Coordinate mean: {coord_mean.tolist()}")
print(f"Coordinate std:  {coord_std.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# Three-way split: test | eval | train pool
#
#   test pool  — never seen during training; evaluated once at the end
#   eval pool  — fixed set evaluated every epoch (no gradient updates)
#   train pool — everything else; training batches are resampled from here each epoch
# ─────────────────────────────────────────────────────────────────────────────
obj = trimesh.load("./Baseline_ML4Science.stl")
train_points, train_labels = sample_points(obj,)
...
N    = data_full.shape[0]
perm = torch.randperm(N)

test_idx       = perm[:Num_points_used_for_testing]
eval_idx       = perm[Num_points_used_for_testing : Num_points_used_for_testing + Num_points_used_for_evaluation]
train_pool_idx = perm[Num_points_used_for_testing + Num_points_used_for_evaluation:]

print(f"Split — test: {test_idx.shape[0]}, eval: {eval_idx.shape[0]}, train pool: {train_pool_idx.shape[0]}")

# Fixed test data (held out until the final evaluation)
test_data   = data_full[test_idx]

# Fixed evaluation data (used every epoch)
eval_data   = data_full[eval_idx]
eval_labels = all_labels[eval_idx]
eval_points = eval_data[:, :3].clone().requires_grad_(True)

# ─────────────────────────────────────────────────────────────────────────────
# Model and optimizers
# ─────────────────────────────────────────────────────────────────────────────
base_pinn  = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
base_pinn.apply(init_weights)
base_pinn  = base_pinn.double()
pinn_model = NormalizedPINNs(base_pinn, coord_mean, coord_std, out_mean, out_std)

loss_weights = make_loss_weights(len(LOSS_NAMES), device)

adam_optimizer    = Adam(pinn_model.parameters(), lr=lr_adam)
adam_weights      = Adam([loss_weights], lr=lr_weights, maximize=True)
adam_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=lr_decay_rate)

start_time            = time.time()
training_loss_track   = []
evaluation_loss_track = []
global_step = 0


def weighted_total_loss(losses_list):
    losses_stack = torch.stack(losses_list)
    return torch.sum(loss_weights * losses_stack) - torch.sum(loss_weights)


def weights_log_dict():
    return {f"weight/{name}": loss_weights[i].item() for i, name in enumerate(LOSS_NAMES)}


def safe_log10(x):
    v = x.item() if hasattr(x, "item") else float(x)
    return float(np.log10(v)) if v > 0 else None


def compute_all_losses(points, labels, cfd_data):
    """
    Run the model on `points` and compute every loss component.

    points:   (N, 3) with requires_grad=True — coordinates only (no CFD data used as input)
    labels:   (N,)  int64 — 0=volume, 2=wall, 3=outlet
    cfd_data: (N, 8) — full CFD row [x,y,z,vx,vy,vz,p,T] used for supervised losses

    Returns a dict {loss_name: scalar_tensor} covering physics, BC, and supervised losses.
    Inlet BC is always evaluated on the global inlet_points (not part of `points`).
    """
    fields = pinn_model(points)

    (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    ) = get_values_and_derivatives(fields, points)

    (
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        loss_outlet_boundary,
        loss_wall_no_slip,
        loss_heat,
        loss_wall_temp,
    ) = get_loss(
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
        labels,
        p_outlet,
    )

    inlet_fields    = pinn_model(inlet_points)
    loss_inlet_vx   = torch.mean((inlet_fields[:, 0] - vx_inlet_data) ** 2)
    loss_inlet_vy   = torch.mean((inlet_fields[:, 1] - vy_inlet_data) ** 2)
    loss_inlet_vz   = torch.mean((inlet_fields[:, 2] - vz_inlet_data) ** 2)
    loss_inlet_temp = torch.mean((inlet_fields[:, 4] * 1000 - T_inlet) ** 2) / 1e6

    sup_vx = torch.mean((fields[:, 0] - cfd_data[:, 3]) ** 2)
    sup_vy = torch.mean((fields[:, 1] - cfd_data[:, 4]) ** 2)
    sup_vz = torch.mean((fields[:, 2] - cfd_data[:, 5]) ** 2)
    sup_p  = torch.mean((fields[:, 3] - cfd_data[:, 6]) ** 2)
    sup_T  = torch.mean((fields[:, 4] * 1000 - cfd_data[:, 7]) ** 2) / 1e6

    return {
        "divergence":            loss_divergence,
        "momentum_x":            loss_momentum_x,
        "momentum_y":            loss_momentum_y,
        "momentum_z":            loss_momentum_z,
        "heat_equation":         loss_heat,
        "outlet_p_boundary":     loss_outlet_boundary,
        "wall_no_slip_boundary": loss_wall_no_slip,
        "wall_temp_boundary":    loss_wall_temp,
        "inlet_vx_boundary":     loss_inlet_vx,
        "inlet_vy_boundary":     loss_inlet_vy,
        "inlet_vz_boundary":     loss_inlet_vz,
        "inlet_temp_boundary":   loss_inlet_temp,
        "supervised_vx":         sup_vx,
        "supervised_vy":         sup_vy,
        "supervised_vz":         sup_vz,
        "supervised_p":          sup_p,
        "supervised_T":          sup_T,
    }


_PHYSICS_BC_KEYS = [
    "divergence", "momentum_x", "momentum_y", "momentum_z", "heat_equation",
    "outlet_p_boundary", "wall_no_slip_boundary", "wall_temp_boundary",
    "inlet_vx_boundary", "inlet_vy_boundary", "inlet_vz_boundary", "inlet_temp_boundary",
]


def run_evaluation():
    """Evaluate on the fixed eval set every epoch (no gradient updates)."""
    pinn_model.eval()
    adam_optimizer.zero_grad()

    losses = compute_all_losses(eval_points, eval_labels, eval_data)

    eval_physics_bc_loss = sum(losses[k] for k in _PHYSICS_BC_KEYS)

    eval_log = {f"Eval/{k}": safe_log10(v) for k, v in losses.items()}
    eval_log["Eval/physics_bc_total"] = safe_log10(eval_physics_bc_loss)
    eval_log = {k: v for k, v in eval_log.items() if v is not None}

    evaluation_loss_track.append(eval_physics_bc_loss.item())
    print(
        f"  [Eval] Div: {losses['divergence'].item():.3e}  "
        f"Mom-X: {losses['momentum_x'].item():.3e}  "
        f"Wall: {losses['wall_no_slip_boundary'].item():.3e}  "
        f"Sup(vx/p/T): {losses['supervised_vx'].item():.2e}/"
        f"{losses['supervised_p'].item():.2e}/{losses['supervised_T'].item():.2e}  "
        f"PhyBC: {eval_physics_bc_loss.item():.3e}"
    )
    return eval_log


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
#
# Each epoch:
#   1. Evaluate on the fixed eval set (no updates).
#   2. Sample Num_points_used_for_training points from the train pool.
#   3. Compute physics + BC losses on ALL sampled points.
#   4. Compute supervised loss on the first Num_training_points_supervised points
#      of the batch (already random because of the per-epoch permutation).
#   5. Backprop the weighted total loss.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Adam optimizer")
print("=" * 60)

for epoch in range(epochs_adam):
    print(f"[Adam] {epoch+1}/{epochs_adam}")

    eval_log = run_evaluation()

    # ── Sample training batch from the train pool ─────────────────────────────
    epoch_perm   = torch.randperm(train_pool_idx.shape[0])
    train_idx    = train_pool_idx[epoch_perm[:Num_points_used_for_training]]
    train_data   = data_full[train_idx]
    train_labels = all_labels[train_idx]

    # All training points need gradients for the physics losses
    train_points = train_data[:, :3].clone().requires_grad_(True)

    # Supervised sub-batch: first rows of the already-shuffled training batch
    sup_cfd = train_data[:Num_training_points_supervised]

    pinn_model.train()
    adam_optimizer.zero_grad()
    adam_weights.zero_grad()

    # ── Single forward pass — reuse train_fields for both physics and supervised ──
    train_fields = pinn_model(train_points)

    # Physics + BC losses (all training points)
    (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    ) = get_values_and_derivatives(train_fields, train_points)

    (
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        loss_outlet_boundary,
        loss_wall_no_slip,
        loss_heat,
        loss_wall_temp,
    ) = get_loss(
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
        train_labels,
        p_outlet,
    )

    # Inlet BC
    inlet_fields    = pinn_model(inlet_points)
    loss_inlet_vx   = torch.mean((inlet_fields[:, 0] - vx_inlet_data) ** 2)
    loss_inlet_vy   = torch.mean((inlet_fields[:, 1] - vy_inlet_data) ** 2)
    loss_inlet_vz   = torch.mean((inlet_fields[:, 2] - vz_inlet_data) ** 2)
    loss_inlet_bc   = loss_inlet_vx + loss_inlet_vy + loss_inlet_vz
    loss_inlet_temp = torch.mean((inlet_fields[:, 4] * 1000 - T_inlet) ** 2) / 1e6

    # Supervised loss — sub-batch, no second forward pass needed
    fields_sup = train_fields[:Num_training_points_supervised]
    sup_vx = torch.mean((fields_sup[:, 0] - sup_cfd[:, 3]) ** 2)
    sup_vy = torch.mean((fields_sup[:, 1] - sup_cfd[:, 4]) ** 2)
    sup_vz = torch.mean((fields_sup[:, 2] - sup_cfd[:, 5]) ** 2)
    sup_p  = torch.mean((fields_sup[:, 3] - sup_cfd[:, 6]) ** 2)
    sup_T  = torch.mean((fields_sup[:, 4] * 1000 - sup_cfd[:, 7]) ** 2) / 1e6

    loss_total = weighted_total_loss([
        loss_divergence, loss_momentum_x, loss_momentum_y, loss_momentum_z, loss_heat,
        loss_inlet_bc, loss_outlet_boundary, loss_wall_no_slip,
        loss_inlet_temp, loss_wall_temp,
        sup_vx, sup_vy, sup_vz, sup_p, sup_T,
    ])

    loss_total.backward()
    adam_optimizer.step()
    adam_weights.step()
    with torch.no_grad():
        loss_weights.clamp_(min=1e-6)
    adam_lr_scheduler.step()

    training_loss_track.append(loss_total.item())
    train_log = {
        "Train/divergence":          np.log10(loss_divergence.item()),
        "Train/momentum_x":          np.log10(loss_momentum_x.item()),
        "Train/momentum_y":          np.log10(loss_momentum_y.item()),
        "Train/momentum_z":          np.log10(loss_momentum_z.item()),
        "Train/heat":                np.log10(loss_heat.item()),
        "Train/inlet_bc":            np.log10(loss_inlet_bc.item()),
        "Train/outlet_p":            np.log10(loss_outlet_boundary.item()),
        "Train/wall_no_slip":        np.log10(loss_wall_no_slip.item()),
        "Train/inlet_temp":          np.log10(loss_inlet_temp.item()),
        "Train/wall_temp":           np.log10(loss_wall_temp.item()),
        "Train/supervised_vx":       np.log10(sup_vx.item()),
        "Train/supervised_vy":       np.log10(sup_vy.item()),
        "Train/supervised_vz":       np.log10(sup_vz.item()),
        "Train/supervised_p":        np.log10(sup_p.item()),
        "Train/supervised_T":        np.log10(sup_T.item()),
        "Train/total":               np.log10(loss_total.item()),
        "LR": adam_lr_scheduler.get_last_lr()[0],
        **weights_log_dict(),
    }
    print(
        f"  [Train] Div: {loss_divergence.item():.3e}  "
        f"Mom-X: {loss_momentum_x.item():.3e}  "
        f"Wall: {loss_wall_no_slip.item():.3e}  "
        f"Sup(vx/vy/vz/p/T): "
        f"{sup_vx.item():.2e}/{sup_vy.item():.2e}/{sup_vz.item():.2e}/"
        f"{sup_p.item():.2e}/{sup_T.item():.2e}  "
        f"Total: {loss_total.item():.3e}"
    )

    if not debug:
        wandb.log({**train_log, **eval_log}, step=global_step)
    global_step += 1


stop_time = time.time()
print(f"Training time: {stop_time - start_time:.1f}s")
torch.save(pinn_model.state_dict(), "../run/pinn_model_v6.pt")
torch.save(loss_weights.detach().cpu(), "../run/loss_weights_v6.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Final test — inference on ALL CSV points
# Computes every loss (physics, BC, supervised) as a final performance summary.
# No gradient updates; results are printed and logged once.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Final test (all CSV points)")
print("=" * 60)

pinn_model.eval()
all_test_points = data_points.clone().requires_grad_(True)
test_losses = compute_all_losses(all_test_points, all_labels, data_full)

for name, val in test_losses.items():
    print(f"  {name}: {val.item():.4e}")

if not debug:
    test_log = {f"Test/{k}": safe_log10(v) for k, v in test_losses.items()}
    test_log = {k: v for k, v in test_log.items() if v is not None}
    wandb.log(test_log, step=global_step)


# ─────────────────────────────────────────────────────────────────────────────
# Inference on CPU + plotting
# ─────────────────────────────────────────────────────────────────────────────
device = "cpu"
torch.set_default_device(device)
base_pinn_inf  = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
pinn_model_inf = NormalizedPINNs(
    base_pinn_inf,
    coord_mean.to(device), coord_std.to(device),
    out_mean.to(device),   out_std.to(device),
)
pinn_model_inf.load_state_dict(
    torch.load("../run/pinn_model_v6.pt", weights_only=True, map_location="cpu")
)

all_points = torch.tensor(coords)
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
    repo.remote(name="origin").push()
    print("Pushed changes to the remote repository.")
else:
    print("No changes to commit.")
