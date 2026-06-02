# Notes: should be run from the `src/` folder using `python pinn_run4.py`
# Adaptive loss balancing via inverse-EMA weighting.
# Each loss term is re-weighted as w_i = mean_ema / ema_i, so terms that are
# currently large get down-weighted and terms that are small get up-weighted.
# This drives ALL losses toward zero simultaneously without manual tuning.
import trimesh
import numpy as np
import random
import torch

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
Project_name = "PINNs-baseline_with_heat"
device = "cuda"
debug = False

data_folder = "./preProcessedData/with_T/" + folder + "/"
Full_Project_name = Project_name + "_" + folder

# Hyperparams
epochs_adam  = 200
epochs_lbfgs = 100
run_lbfgs    = False
hidden_dim   = 128
num_layer    = 4
seed         = 42
lr_adam      = 1e-3
ema_alpha    = 0.9   # EMA decay for adaptive weighting (higher = slower adaptation)

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
            "seed": seed,
            "hidden_dim": hidden_dim,
            "num_layers": num_layer,
            "loss_balancing": "inverse-EMA adaptive",
            "ema_alpha": ema_alpha,
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

adam_optimizer = Adam(pinn_model.parameters(), lr=lr_adam)

start_time = time.time()
training_loss_track  = []
validation_loss_track = []

validation_points_raw, validation_labels = sample_points(obj, 30000, 3000, 20000)
validation_points = validation_points_raw / 1000
global_step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive loss weighting
# ─────────────────────────────────────────────────────────────────────────────
LOSS_KEYS = [
    "divergence", "momentum_x", "momentum_y", "momentum_z",
    "inlet_boundary", "outlet_boundary", "other_boundary",
    "heat", "inlet_temp_boundary", "t_wall_boundary", "supervised",
]
n_losses = len(LOSS_KEYS)

# EMA of each loss magnitude — initialised to 1 so first-step weights are equal
loss_ema = torch.ones(n_losses, dtype=torch.float64)


def get_adaptive_weights(ema, eps=1e-10):
    """Inverse-EMA weights, normalised so their mean equals 1."""
    inv = 1.0 / (ema + eps)
    return inv / inv.mean()


def update_ema(ema, losses_list, alpha=ema_alpha):
    """Update EMA in-place with the current raw loss values."""
    with torch.no_grad():
        vals = torch.stack([l.detach() for l in losses_list])
        return alpha * ema + (1.0 - alpha) * vals


def weighted_total(losses_list, weights):
    return sum(w * l for w, l in zip(weights, losses_list))


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────
def run_validation(epoch_label):
    pinn_model.eval()
    if epoch_label == "adam":
        adam_optimizer.zero_grad()

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
print("Stage 1: Adam optimizer  [adaptive loss weighting]")
print("=" * 60)

for epoch in range(epochs_adam):
    print(f"[Adam] {epoch+1}/{epochs_adam}")

    val_log = run_validation("adam")

    # Resample collocation points each Adam epoch
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

    all_losses = [
        loss_divergence, loss_momentum_x, loss_momentum_y, loss_momentum_z,
        loss_inlet_boundary, loss_outlet_boundary, loss_other_boundary,
        loss_heat, loss_inlet_temp_boundary, loss_t_wall_boundary, supervised_loss,
    ]

    # Compute weights from EMA, then update EMA for next step
    weights = get_adaptive_weights(loss_ema)
    loss_total = weighted_total(all_losses, weights)
    loss_ema[:] = update_ema(loss_ema, all_losses)

    loss_total.backward()
    adam_optimizer.step()

    training_loss_track.append(loss_total.item())
    weight_log = {f"Weight/{k}": weights[i].item() for i, k in enumerate(LOSS_KEYS)}
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
        **weight_log,
    }
    print(
        f"  Div: {loss_divergence.item():.4e} (w={weights[0].item():.2f})  "
        f"Inlet BC: {loss_inlet_boundary.item():.4e} (w={weights[4].item():.2f})  "
        f"Wall BC: {loss_other_boundary.item():.4e} (w={weights[6].item():.2f})  "
        f"T-wall: {loss_t_wall_boundary.item():.4e} (w={weights[9].item():.2f})  "
        f"Total: {loss_total.item():.4e}"
    )

    if not debug:
        wandb.log({**train_log, **val_log}, step=global_step)
    global_step += 1


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: L-BFGS  (fixed collocation points — keeps loss landscape stable)
# ─────────────────────────────────────────────────────────────────────────────
if run_lbfgs:
    print("=" * 60)
    print("Stage 2: L-BFGS optimizer  [adaptive loss weighting]")
    print("=" * 60)
else:
    print("Stage 2 (L-BFGS) skipped.")

if run_lbfgs:
    lbfgs_optimizer = LBFGS(pinn_model.parameters(), line_search_fn="strong_wolfe")

    # Sample once and keep fixed for the entire L-BFGS stage
    train_points_fixed, train_labels_fixed = sample_points(obj, 30000, 3000, 20000)
    train_points_fixed = train_points_fixed / 1000
    train_points_fixed.requires_grad_(True)

    indices_fixed           = torch.randint(0, train_data_points.shape[0], (num_samples,))
    sampled_points_fixed    = train_data_points[indices_fixed]
    vx_sampled_fixed        = train_vx_data[indices_fixed]
    vy_sampled_fixed        = train_vy_data[indices_fixed]
    vz_sampled_fixed        = train_vz_data[indices_fixed]
    p_sampled_fixed         = train_p_data[indices_fixed]
    temp_sampled_fixed      = train_temp_data[indices_fixed]

    _last_train_losses = {}

    for epoch in range(epochs_lbfgs):
        print(f"[L-BFGS] {epoch+1}/{epochs_lbfgs}")

        val_log = run_validation("lbfgs")

        pinn_model.train()
        lbfgs_optimizer.zero_grad(True)

        # Freeze weights for the duration of this epoch's closure calls
        current_weights = get_adaptive_weights(loss_ema)

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

            all_losses = [
                loss_divergence, loss_momentum_x, loss_momentum_y, loss_momentum_z,
                loss_inlet_boundary, loss_outlet_boundary, loss_other_boundary,
                loss_heat, loss_inlet_temp_boundary, loss_t_wall_boundary, supervised_loss,
            ]

            # Use weights frozen before this epoch's step (stable across line-search calls)
            loss_total = weighted_total(all_losses, current_weights)

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
                "_total_raw":   loss_total.item(),
                "_div_raw":     loss_divergence.item(),
                "_wall_raw":    loss_other_boundary.item(),
                "_sup_raw":     supervised_loss.item(),
                "_losses_list": all_losses,
            })

            loss_total.backward()
            return loss_total

        lbfgs_optimizer.step(closure)

        # Update EMA once per epoch with the last closure evaluation
        if "_losses_list" in _last_train_losses:
            loss_ema[:] = update_ema(loss_ema, _last_train_losses["_losses_list"])

        training_loss_track.append(_last_train_losses["_total_raw"])
        print(
            f"  Div: {_last_train_losses['_div_raw']:.4e}  "
            f"Wall: {_last_train_losses['_wall_raw']:.4e}  "
            f"Sup: {_last_train_losses['_sup_raw']:.4e}  "
            f"Total: {_last_train_losses['_total_raw']:.4e}"
        )

        if not debug:
            weight_log = {f"Weight/{k}": current_weights[i].item() for i, k in enumerate(LOSS_KEYS)}
            train_log = {k: v for k, v in _last_train_losses.items() if not k.startswith("_")}
            wandb.log({**train_log, **val_log, **weight_log}, step=global_step)
        global_step += 1


stop_time = time.time()
print(f"Time taken for training: {stop_time - start_time:.1f}s")
torch.save(pinn_model.state_dict(), "../run/pinn_model_v4.pt")


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
