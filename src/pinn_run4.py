# Notes: should be run from the `src/` folder using `python pinn_run4.py`
# Two-stage training: Adam (global exploration) → L-BFGS (fine-tuning).
# L-BFGS uses fixed collocation points to keep the loss landscape consistent
# across line-search evaluations, avoiding the loss spikes seen with resampling.
# Adaptive loss weighting: 11 learnable scalars w_i trained alongside the PINN.
# Total loss = sum(w_i * l_i) - sum(w_i)  (regularization prevents w_i → 0).
import trimesh
import numpy as np
import random
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
from utils import *
from unet import *
from torch.optim import Adam, LBFGS
from pinn2 import *
import time
import wandb
import os
from git import Repo
from utils import *
from constants import *

folder = "dp11"
Project_name = "PINNs-baseline_with_heat_adaptive_weights"
device = "cuda"
debug = False

data_folder = "./preProcessedData/with_T/" + folder + "/"
Full_Project_name = Project_name + "_" + folder

# Hyperparams
epochs_adam  = 50
epochs_lbfgs = 100
run_lbfgs    = False
hidden_dim   = 128
num_layer    = 4
seed         = 42
lr_adam      = 1e-2
lr_weights   = 1e-3
lr_decay_rate  = 0.80
lr_decay_steps = 400

LOSS_NAMES = [
    "w_divergence",
    "w_momentum_x",
    "w_momentum_y",
    "w_momentum_z",
    "w_inlet_boundary",
    "w_outlet_boundary",
    "w_other_boundary",
    "w_supervised",
    "w_heat",
    "w_inlet_temp_boundary",
    "w_t_wall_boundary",
]

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo = Repo(parent_dir)

if not debug:
    run = wandb.init(
        project=Full_Project_name,
        config={
            "optimizer": "Adam -> L-BFGS",
            "architecture": "FFNN",
            "epochs_adam": epochs_adam,
            "epochs_lbfgs": epochs_lbfgs,
            "lr_adam": lr_adam,
            "lr_weights": lr_weights,
            "lr_decay_rate": lr_decay_rate,
            "lr_decay_steps": lr_decay_steps,
            "adam_betas": (0.99, 0.999),
            "seed": seed,
            "hidden_dim": hidden_dim,
            "num_layers": num_layer,
        },
    )

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.set_default_device(device)
print(f"Using device: {device}")

inlet = np.load(data_folder + "vel_x_inlet.npy")
inlet_points   = torch.tensor(inlet[:, 0:3])
vx_inlet_data  = torch.tensor(np.load(data_folder + "vel_x_inlet.npy")[:, 3])
vy_inlet_data  = torch.tensor(np.load(data_folder + "vel_y_inlet.npy")[:, 3])
vz_inlet_data  = torch.tensor(np.load(data_folder + "vel_z_inlet.npy")[:, 3])

data_points = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 0:3])
vx_data     = torch.tensor(np.load(data_folder + "vel_x.npy")[:, 3])
vy_data     = torch.tensor(np.load(data_folder + "vel_y.npy")[:, 3])
vz_data     = torch.tensor(np.load(data_folder + "vel_z.npy")[:, 3])
p_data      = torch.tensor(np.load(data_folder + "press.npy")[:, 3])
temp_data   = torch.tensor(np.load(data_folder + "temp.npy")[:, 3])

num_samples = 5000
n_test      = 100000
perm        = torch.randperm(data_points.shape[0])
train_idx   = perm[n_test:]
test_idx    = perm[:n_test]

train_data_points = data_points[train_idx]
train_vx_data     = vx_data[train_idx]
train_vy_data     = vy_data[train_idx]
train_vz_data     = vz_data[train_idx]
train_p_data      = p_data[train_idx] / 10**5
train_temp_data   = temp_data[train_idx]

test_data_points = data_points[test_idx]
test_vx_data     = vx_data[test_idx]
test_vy_data     = vy_data[test_idx]
test_vz_data     = vz_data[test_idx]
test_p_data      = p_data[test_idx] / 10**5
test_temp_data   = temp_data[test_idx]

obj = trimesh.load("./Baseline_ML4Science.stl")

pinn_model = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
pinn_model.apply(init_weights)
pinn_model = pinn_model.double()

# Learnable adaptive loss weights, one per loss term, initialized to 1.0
loss_weights = nn.Parameter(torch.ones(11, dtype=torch.float64, device=device))

adam_optimizer = Adam(
    [
        {"params": pinn_model.parameters(), "lr": lr_adam},
        {"params": [loss_weights],          "lr": lr_weights},
    ],
    betas=(0.99, 0.999),
)
adam_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    adam_optimizer, step_size=lr_decay_steps, gamma=lr_decay_rate
)

start_time = time.time()
training_loss_track  = []
validation_loss_track = []

validation_points_raw, validation_labels = sample_points(obj, 30000, 3000, 20000)
validation_points = validation_points_raw / 1000
global_step = 0


def weighted_total_loss(losses_list):
    """
    losses_list: list of 11 scalar tensors in the same order as LOSS_NAMES.
    Returns: sum(w_i * l_i) - sum(w_i)
    """
    losses_stack = torch.stack(losses_list)
    return torch.sum(loss_weights * losses_stack) - torch.sum(loss_weights)


def weights_log_dict():
    return {name: loss_weights[i].item() for i, name in enumerate(LOSS_NAMES)}


def run_validation(epoch_label):
    pinn_model.eval()
    adam_optimizer.zero_grad() if epoch_label == "adam" else None

    validation_points.requires_grad_(True)
    validation_fields = pinn_model(validation_points)

    (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    ) = get_values_and_derivatives(validation_fields, validation_points)

    (
        val_loss_divergence,
        val_loss_momentum_x,
        val_loss_momentum_y,
        val_loss_momentum_z,
        val_loss_outlet_boundary,
        val_loss_other_boundary,
        val_loss_heat,
        val_loss_t_wall_boundary,
    ) = get_loss(
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
        validation_labels,
        p_outlet,
    )

    val_loss_total = (
        val_loss_divergence
        + val_loss_momentum_x
        + val_loss_momentum_y
        + val_loss_momentum_z
        + val_loss_outlet_boundary
        + val_loss_other_boundary
        + val_loss_heat
        + val_loss_t_wall_boundary
    )

    with torch.no_grad():
        test_fields = pinn_model(test_data_points)
    test_supervised_loss = (
        torch.mean((test_fields[:, 0] - test_vx_data) ** 2)
        + torch.mean((test_fields[:, 1] - test_vy_data) ** 2)
        + torch.mean((test_fields[:, 2] - test_vz_data) ** 2)
        + torch.mean((test_fields[:, 3] - test_p_data) ** 2)
        + torch.mean((test_fields[:, 4] * 1000 - test_temp_data) ** 2) / 10**6
    )

    val_fields_list = [
        ("vx", validation_fields[:, 0].cpu().detach().numpy()),
        ("vy", validation_fields[:, 1].cpu().detach().numpy()),
        ("vz", validation_fields[:, 2].cpu().detach().numpy()),
        ("p",  validation_fields[:, 3].cpu().detach().numpy()),
        ("T",  validation_fields[:, 4].cpu().detach().numpy() * 1000),
    ]
    plot_fields(val_fields_list, validation_points.cpu().detach().numpy())

    val_log = {
        "Validation Divergence Loss":                   np.log10(val_loss_divergence.item()),
        "Validation X Momentum Loss":                   np.log10(val_loss_momentum_x.item()),
        "Validation Y Momentum Loss":                   np.log10(val_loss_momentum_y.item()),
        "Validation Z Momentum Loss":                   np.log10(val_loss_momentum_z.item()),
        "Validation outlet Boundary Loss":              np.log10(val_loss_outlet_boundary.item()),
        "Validation Other Boundary Loss":               np.log10(val_loss_other_boundary.item()),
        "Validation Heat Loss":                         np.log10(val_loss_heat.item()),
        "Validation Surface Temperature Boundary Loss": np.log10(val_loss_t_wall_boundary.item()),
        "Validation Total Loss":                        np.log10(val_loss_total.item()),
        "Test Supervised Loss":                         np.log10(test_supervised_loss.item()),
    }

    validation_loss_track.append(val_loss_total.item())
    print(
        f"  Val Divergence: {val_loss_divergence.item():.4e}  "
        f"Val X-Mom: {val_loss_momentum_x.item():.4e}  "
        f"Val Y-Mom: {val_loss_momentum_y.item():.4e}  "
        f"Val Z-Mom: {val_loss_momentum_z.item():.4e}  "
        f"Val Outlet: {val_loss_outlet_boundary.item():.4e}  "
        f"Val Wall: {val_loss_other_boundary.item():.4e}  "
        f"Val Total: {val_loss_total.item():.4e}  "
        f"Test Supervised: {test_supervised_loss.item():.4e}"
    )

    return val_log


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Adam  (resample collocation points each epoch)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Stage 1: Adam optimizer")
print("=" * 60)

for epoch in range(epochs_adam):
    print(f"[Adam] {epoch+1}/{epochs_adam}")

    val_log = run_validation("adam")

    train_points, train_labels = sample_points(obj, 30000, 3000, 20000)
    train_points = train_points / 1000
    train_points.requires_grad_(True)

    indices         = torch.randint(0, train_data_points.shape[0], (num_samples,))
    sampled_points  = train_data_points[indices]
    vx_sampled_data = train_vx_data[indices]
    vy_sampled_data = train_vy_data[indices]
    vz_sampled_data = train_vz_data[indices]
    p_sampled_data  = train_p_data[indices]
    temp_sampled_data = train_temp_data[indices]

    pinn_model.train()
    adam_optimizer.zero_grad()

    train_fields = pinn_model(train_points)
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
        loss_other_boundary,
        loss_heat,
        loss_t_wall_boundary,
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

    inlet_fields = pinn_model(inlet_points)
    loss_inlet_boundary = (
        torch.mean((inlet_fields[:, 0] - vx_inlet_data) ** 2)
        + torch.mean((inlet_fields[:, 1] - vy_inlet_data) ** 2)
        + torch.mean((inlet_fields[:, 2] - vz_inlet_data) ** 2)
    )
    loss_inlet_temp_boundary = (
        torch.mean((inlet_fields[:, 4] * 1000 - T_inlet) ** 2) / 10**6
    )

    fields_supervised = pinn_model(sampled_points)
    supervised_loss = (
        torch.mean((fields_supervised[:, 0] - vx_sampled_data) ** 2)
        + torch.mean((fields_supervised[:, 1] - vy_sampled_data) ** 2)
        + torch.mean((fields_supervised[:, 2] - vz_sampled_data) ** 2)
        + torch.mean((fields_supervised[:, 3] - p_sampled_data) ** 2)
        + torch.mean((fields_supervised[:, 4] * 1000 - temp_sampled_data) ** 2) / 10**6
    )

    loss_total = weighted_total_loss([
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        loss_inlet_boundary,
        loss_outlet_boundary,
        loss_other_boundary,
        supervised_loss,
        loss_heat,
        loss_inlet_temp_boundary,
        loss_t_wall_boundary,
    ])

    loss_total.backward()
    adam_optimizer.step()
    adam_lr_scheduler.step()

    training_loss_track.append(loss_total.item())
    train_log = {
        "Divergence Loss":                    np.log10(loss_divergence.item()),
        "X Momentum Loss":                    np.log10(loss_momentum_x.item()),
        "Y Momentum Loss":                    np.log10(loss_momentum_y.item()),
        "Z Momentum Loss":                    np.log10(loss_momentum_z.item()),
        "Inlet Boundary Loss":                np.log10(loss_inlet_boundary.item()),
        "Outlet Boundary Loss":               np.log10(loss_outlet_boundary.item()),
        "Other Boundary Loss":                np.log10(loss_other_boundary.item()),
        "Supervised Loss":                    np.log10(supervised_loss.item()),
        "Heat Loss":                          np.log10(loss_heat.item()),
        "Inlet Temperature Boundary Loss":    np.log10(loss_inlet_temp_boundary.item()),
        "Surface Temperature Boundary Loss":  np.log10(loss_t_wall_boundary.item()),
        "Total Loss":                         np.log10(loss_total.item()),
        "Stage": 1,
        "LR": adam_lr_scheduler.get_last_lr()[0],
        **weights_log_dict(),
    }
    print(
        f"  Div: {loss_divergence.item():.4e}  "
        f"X-Mom: {loss_momentum_x.item():.4e}  "
        f"Wall: {loss_other_boundary.item():.4e}  "
        f"Sup: {supervised_loss.item():.4e}  "
        f"Total: {loss_total.item():.4e}  "
        f"Weights: {[f'{w:.3f}' for w in loss_weights.detach().cpu().tolist()]}"
    )

    if not debug:
        wandb.log({**train_log, **val_log}, step=global_step)
    global_step += 1


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: L-BFGS  (fixed collocation points — keeps loss landscape stable)
# ─────────────────────────────────────────────────────────────────────────────
if run_lbfgs:
    print("=" * 60)
    print("Stage 2: L-BFGS optimizer (fixed collocation points)")
    print("=" * 60)
else:
    print("Stage 2 (L-BFGS) skipped.")

if run_lbfgs:
    lbfgs_optimizer = LBFGS(
        list(pinn_model.parameters()) + [loss_weights],
        line_search_fn="strong_wolfe",
    )

    train_points_fixed, train_labels_fixed = sample_points(obj, 30000, 3000, 20000)
    train_points_fixed = train_points_fixed / 1000
    train_points_fixed.requires_grad_(True)

    indices_fixed        = torch.randint(0, train_data_points.shape[0], (num_samples,))
    sampled_points_fixed = train_data_points[indices_fixed]
    vx_sampled_fixed     = train_vx_data[indices_fixed]
    vy_sampled_fixed     = train_vy_data[indices_fixed]
    vz_sampled_fixed     = train_vz_data[indices_fixed]
    p_sampled_fixed      = train_p_data[indices_fixed]
    temp_sampled_fixed   = train_temp_data[indices_fixed]

    _last_train_losses = {}

    for epoch in range(epochs_lbfgs):
        print(f"[L-BFGS] {epoch+1}/{epochs_lbfgs}")

        val_log = run_validation("lbfgs")

        pinn_model.train()
        lbfgs_optimizer.zero_grad(True)

        def closure():
            lbfgs_optimizer.zero_grad()

            train_fields = pinn_model(train_points_fixed)
            (
                vx, vy, vz, p, T,
                vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
                vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
                vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
                p_x, p_y, p_z,
                T_x, T_y, T_z, T_xx, T_yy, T_zz,
            ) = get_values_and_derivatives(train_fields, train_points_fixed)

            (
                loss_divergence,
                loss_momentum_x,
                loss_momentum_y,
                loss_momentum_z,
                loss_outlet_boundary,
                loss_other_boundary,
                loss_heat,
                loss_t_wall_boundary,
            ) = get_loss(
                vx, vy, vz, p, T,
                vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
                vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
                vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
                p_x, p_y, p_z,
                T_x, T_y, T_z, T_xx, T_yy, T_zz,
                train_labels_fixed,
                p_outlet,
            )

            inlet_fields = pinn_model(inlet_points)
            loss_inlet_boundary = (
                torch.mean((inlet_fields[:, 0] - vx_inlet_data) ** 2)
                + torch.mean((inlet_fields[:, 1] - vy_inlet_data) ** 2)
                + torch.mean((inlet_fields[:, 2] - vz_inlet_data) ** 2)
            )
            loss_inlet_temp_boundary = (
                torch.mean((inlet_fields[:, 4] * 1000 - T_inlet) ** 2) / 10**6
            )

            fields_supervised = pinn_model(sampled_points_fixed)
            supervised_loss = (
                torch.mean((fields_supervised[:, 0] - vx_sampled_fixed) ** 2)
                + torch.mean((fields_supervised[:, 1] - vy_sampled_fixed) ** 2)
                + torch.mean((fields_supervised[:, 2] - vz_sampled_fixed) ** 2)
                + torch.mean((fields_supervised[:, 3] - p_sampled_fixed) ** 2)
                + torch.mean((fields_supervised[:, 4] * 1000 - temp_sampled_fixed) ** 2) / 10**6
            )

            loss_total = weighted_total_loss([
                loss_divergence,
                loss_momentum_x,
                loss_momentum_y,
                loss_momentum_z,
                loss_inlet_boundary,
                loss_outlet_boundary,
                loss_other_boundary,
                supervised_loss,
                loss_heat,
                loss_inlet_temp_boundary,
                loss_t_wall_boundary,
            ])

            _last_train_losses.update({
                "Divergence Loss":                   np.log10(loss_divergence.item()),
                "X Momentum Loss":                   np.log10(loss_momentum_x.item()),
                "Y Momentum Loss":                   np.log10(loss_momentum_y.item()),
                "Z Momentum Loss":                   np.log10(loss_momentum_z.item()),
                "Inlet Boundary Loss":               np.log10(loss_inlet_boundary.item()),
                "Outlet Boundary Loss":              np.log10(loss_outlet_boundary.item()),
                "Other Boundary Loss":               np.log10(loss_other_boundary.item()),
                "Supervised Loss":                   np.log10(supervised_loss.item()),
                "Heat Loss":                         np.log10(loss_heat.item()),
                "Inlet Temperature Boundary Loss":   np.log10(loss_inlet_temp_boundary.item()),
                "Surface Temperature Boundary Loss": np.log10(loss_t_wall_boundary.item()),
                "Total Loss":                        np.log10(loss_total.item()),
                "Stage": 2,
                "_total_raw": loss_total.item(),
                "_div_raw":   loss_divergence.item(),
                "_wall_raw":  loss_other_boundary.item(),
                "_sup_raw":   supervised_loss.item(),
                **weights_log_dict(),
            })

            loss_total.backward()
            return loss_total

        lbfgs_optimizer.step(closure)

        training_loss_track.append(_last_train_losses["_total_raw"])
        print(
            f"  Div: {_last_train_losses['_div_raw']:.4e}  "
            f"Wall: {_last_train_losses['_wall_raw']:.4e}  "
            f"Sup: {_last_train_losses['_sup_raw']:.4e}  "
            f"Total: {_last_train_losses['_total_raw']:.4e}  "
            f"Weights: {[f'{w:.3f}' for w in loss_weights.detach().cpu().tolist()]}"
        )

        if not debug:
            train_log = {k: v for k, v in _last_train_losses.items() if not k.startswith("_")}
            wandb.log({**train_log, **val_log}, step=global_step)
        global_step += 1


stop_time = time.time()
print(f"Time taken for training: {stop_time - start_time:.1f}s")
torch.save(pinn_model.state_dict(), "../run/pinn_model_v4.pt")
torch.save(loss_weights.detach().cpu(), "../run/loss_weights_v4.pt")


## Plotting
pinn_model.eval()
validation_points.requires_grad_(True)
validation_fields = pinn_model(validation_points)
(
    vx, vy, vz, p, T,
    vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
    vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
    vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
    p_x, p_y, p_z,
    T_x, T_y, T_z, T_xx, T_yy, T_zz,
) = get_values_and_derivatives(validation_fields, validation_points)
(
    validation_loss_divergence,
    validation_loss_momentum_x,
    validation_loss_momentum_y,
    validation_loss_momentum_z,
    validation_loss_outlet_boundary,
    validation_loss_other_boundary,
    validation_loss_heat,
    validation_loss_t_wall_boundary,
) = get_loss(
    vx, vy, vz, p, T,
    vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
    vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
    vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
    p_x, p_y, p_z,
    T_x, T_y, T_z, T_xx, T_yy, T_zz,
    validation_labels,
    p_outlet,
)

fields = [
    ("vx", validation_fields[:, 0].cpu().detach().numpy()),
    ("vy", validation_fields[:, 1].cpu().detach().numpy()),
    ("vz", validation_fields[:, 2].cpu().detach().numpy()),
    ("p",  validation_fields[:, 3].cpu().detach().numpy()),
    ("T",  validation_fields[:, 4].cpu().detach().numpy() * 1000),
]
plot_fields(fields, validation_points)


######## Inference

device = "cpu"
torch.set_default_device(device)
pinn_model = PINNs(in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer).to(device)
pinn_model.load_state_dict(
    torch.load("../run/pinn_model_v4.pt", weights_only=True, map_location=torch.device("cpu"))
)

all_points = torch.tensor(
    np.concatenate(
        (
            np.load(os.path.join(data_folder, "vel_x_inlet.npy"))[:, :3],
            np.load(os.path.join(data_folder, "vel_x.npy"))[:, :3],
        )
    )
)

all_fields = pinn_model(all_points)

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
