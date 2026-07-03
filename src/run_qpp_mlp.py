import argparse
import numpy as np
import torch
import os
import sys
import wandb
from torch.optim import Adam

torch.set_default_dtype(torch.float64)

from config import (
    DP_CONFIGS,
    PARAM_NAMES, PARAM_MEANS, PARAM_STDS,
    WALL_Y1, WALL_Y2, WALL_EPS,
    BUFFER_M,
)
from models import QPP_MLP
from utils import _Tee, plot_qpp_single

# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="QPP-MLP: data-driven surrogate for wall heat flux q'' [W/m²].\n"
                "Trains (x, y, z, AR, e/Dh, P/e, alpha, Re) → q''.\n"
                "Combine with PINN-derived Tb to get HTC = q'' / (Tb - Tw).")

parser.add_argument("--data-dir",  default="./preProcessedData/With_T",
                    help="Root folder; q'' data expected at <data-dir>/<folder>/qpp.npy "
                         "(shape [N,4]: x y z q'').  "
                         "CSV alternative: qpp.csv with columns x,y,z,<qpp-col>.")
parser.add_argument("--qpp-col",   default=None,
                    help="Column name for q'' in CSV files (if None, uses qpp.npy).")
parser.add_argument("--run-path",  default=None)
parser.add_argument("--project",   default="QPP_MLP")
parser.add_argument("--device",    default="cuda")
parser.add_argument("--debug",     action="store_true", help="Disable W&B logging")

parser.add_argument("--hidden-dim", type=int,   default=64)
parser.add_argument("--n-layers",   type=int,   default=4)
parser.add_argument("--epochs",     type=int,   default=10_000)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--lr",         type=float, default=3e-3)
parser.add_argument("--lr-end",     type=float, default=1e-4)
parser.add_argument("--batch-size", type=int,   default=8_192)

parser.add_argument("--patience",     type=int,   default=500,
                    help="Stop if test loss does not improve by --min-delta for this many epochs.")
parser.add_argument("--min-delta",    type=float, default=1e-5,
                    help="Minimum absolute improvement in normalised MSE to reset patience counter.")
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 weight decay for Adam optimizer.")

parser.add_argument("--dps", nargs="+", default=None, metavar="DP",
                    help="Explicit subset of DPs, e.g. --dps dp00 dp03 dp49.")
parser.add_argument("--test-split", type=float, default=0.2,
                    help="Fraction of loaded DPs held out for testing (default 0.2). "
                         "Split is done at the DP level, seeded by --seed.")
parser.add_argument("--test-dps", nargs="+", default=None, metavar="DP",
                    help="Explicit test DPs (overrides --test-split), e.g. --test-dps dp05 dp23.")

args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════

DEVICE     = args.device
EPOCHS     = args.epochs
HIDDEN_DIM = args.hidden_dim
N_LAYERS   = args.n_layers
SEED       = args.seed
LR         = args.lr
LR_END     = args.lr_end
GAMMA      = (LR_END / LR) ** (1.0 / EPOCHS)
BATCH_SIZE = args.batch_size

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

N_PARAMS = 5
# Wall heat flux is on the z=0 bottom wall only, so z carries no information.
# Input: (x, y, 5 params, 2 wall sigmoid features) = 9D
IN_DIM   = 2 + N_PARAMS + 2

# ═══════════════════════════════════════════════════════════════
# RUN PATH + LOGGING
# ═══════════════════════════════════════════════════════════════

dp_filter = set(args.dps) if args.dps else None

IS_SWEEP = os.environ.get("WANDB_SWEEP_ID") is not None

if args.run_path:
    RUN_PATH = args.run_path
elif IS_SWEEP:
    _sweep_short = os.environ.get("WANDB_SWEEP_ID", "sweep")[-8:]
    _trial_uid   = os.environ.get("SLURM_ARRAY_TASK_ID", str(os.getpid()))
    RUN_PATH = f"../qpp_mlp_sweep_runs/{_sweep_short}/trial_{_trial_uid}"
else:
    RUN_PATH = (f"../qpp_mlp_runs/"
                f"h{HIDDEN_DIM}_l{N_LAYERS}_e{EPOCHS}_lr{LR:.0e}_lrend{LR_END:.0e}"
                f"_b{BATCH_SIZE}_s{SEED}")
os.makedirs(RUN_PATH, exist_ok=True)

_log_handle = open(os.path.join(RUN_PATH, "training.log"), "w", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_handle)
sys.stderr = _Tee(sys.__stderr__, _log_handle)

print(f"Run path : {RUN_PATH}")
print(f"Device   : {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_qpp_dp(cfg, data_root, qpp_col=None):
    """
    Load wall heat-flux point cloud for one DP.

    Expected file (choose one):
      - <data_root>/<folder>/qpp.npy  — shape (N, 4): [x, y, z, q'']  [W/m²]
      - <data_root>/<folder>/qpp.csv  — CSV with header; x/y/z columns named 'x','y','z';
                                        --qpp-col names the q'' column.

    Returns:
        pts      : float64 tensor (N, 3)  — wall coordinates [m]
        qpp      : float64 tensor (N,)    — heat flux q'' [W/m²]
        params_nd: float64 tensor (5,)    — z-scored design parameters
    """
    folder  = cfg["folder"]
    dp_dir  = os.path.join(data_root, folder)
    npy_path = os.path.join(dp_dir, "qpp.npy")
    csv_path = os.path.join(dp_dir, "qpp.csv")

    if os.path.exists(npy_path):
        arr = np.load(npy_path)              # (N, 4): x y z q''
        pts = torch.tensor(arr[:, :3], dtype=torch.float64)
        qpp = torch.tensor(arr[:,  3], dtype=torch.float64)
    elif os.path.exists(csv_path) and qpp_col is not None:
        raw = np.genfromtxt(csv_path, delimiter=',', names=True)
        pts = torch.tensor(np.column_stack([raw['x'], raw['y'], raw['z']]), dtype=torch.float64)
        qpp = torch.tensor(raw[qpp_col], dtype=torch.float64)
    else:
        raise FileNotFoundError(
            f"No q'' data found for {folder}.\n"
            f"Expected '{npy_path}' (shape [N,4]: x y z q'') "
            f"or '{csv_path}' with --qpp-col set."
        )

    # Trim 420 mm inlet/outlet buffer — keep only the periodic interior zone,
    # matching the STL buffer used in the PINN.
    x    = pts[:, 0]
    x_lo = float(x.min()) + BUFFER_M
    x_hi = float(x.max()) - BUFFER_M
    mask = (x >= x_lo) & (x <= x_hi)
    pts  = pts[mask]
    qpp  = qpp[mask]

    params_raw = torch.tensor(
        [cfg["ar"], cfg["e_dh"], cfg["p_e"], cfg["alpha"], cfg["re"]], dtype=torch.float64)
    params_nd = (params_raw - PARAM_MEANS) / PARAM_STDS

    print(f"  {folder}: {pts.shape[0]:,} wall pts (after buffer trim)  "
          f"x=[{float(pts[:,0].min()):.3f}, {float(pts[:,0].max()):.3f}] m  "
          f"q''=[{qpp.min():.1f}, {qpp.max():.1f}] W/m²")
    return pts, qpp, params_nd


print("\nLoading q'' data…")
all_dps = []
for cfg in DP_CONFIGS:
    if dp_filter and cfg["folder"] not in dp_filter:
        continue
    dp_dir  = os.path.join(args.data_dir, cfg["folder"])
    if not os.path.isdir(dp_dir):
        print(f"  [SKIP] {cfg['folder']} — folder not found")
        continue
    npy_ok = os.path.exists(os.path.join(dp_dir, "qpp.npy"))
    csv_ok = os.path.exists(os.path.join(dp_dir, "qpp.csv")) and args.qpp_col is not None
    if not (npy_ok or csv_ok):
        print(f"  [SKIP] {cfg['folder']} — no qpp.npy / qpp.csv found yet")
        continue
    pts, qpp, params_nd = load_qpp_dp(cfg, args.data_dir, args.qpp_col)
    all_dps.append({"folder": cfg["folder"], "pts": pts, "qpp": qpp, "params_nd": params_nd})

if not all_dps:
    raise RuntimeError(
        "No q'' data found.\n"
        "Place qpp.npy (shape [N,4]: x y z q'') in each "
        "preProcessedData/With_T/<dpXX>/ folder, then rerun."
    )

N_DPS    = len(all_dps)
dp_names = [d["folder"] for d in all_dps]
print(f"\nLoaded {N_DPS} DP(s): {dp_names}")

# ── Train / test split at the DP level ───────────────────────
if args.test_dps:
    test_set   = set(args.test_dps)
    train_dps  = [d for d in all_dps if d["folder"] not in test_set]
    test_dps   = [d for d in all_dps if d["folder"] in test_set]
else:
    n_test    = max(1, round(N_DPS * args.test_split))
    rng       = np.random.default_rng(SEED)
    shuffled  = rng.permutation(N_DPS)
    test_idx  = set(shuffled[:n_test].tolist())
    train_dps = [d for i, d in enumerate(all_dps) if i not in test_idx]
    test_dps  = [d for i, d in enumerate(all_dps) if i in test_idx]

train_names = [d["folder"] for d in train_dps]
test_names  = [d["folder"] for d in test_dps]
print(f"Train DPs ({len(train_dps)}): {train_names}")
print(f"Test  DPs ({len(test_dps)}):  {test_names}")

# ═══════════════════════════════════════════════════════════════
# GLOBAL NORMALISATION  (all DPs)
# ═══════════════════════════════════════════════════════════════

all_pts_train = torch.cat([d["pts"] for d in train_dps], dim=0)
all_qpp_train = torch.cat([d["qpp"] for d in train_dps], dim=0)

# Use only x, y — z is identically 0 (bottom wall surface)
xy_mean = all_pts_train[:, :2].mean(dim=0)
xy_std  = all_pts_train[:, :2].std(dim=0).clamp(min=1e-6)

_y_all = all_pts_train[:, 1]
_s1    = torch.sigmoid((_y_all - WALL_Y1) / WALL_EPS)
_s2    = torch.sigmoid((_y_all - WALL_Y2) / WALL_EPS)
_s1_mean, _s1_std = float(_s1.mean()), float(_s1.std().clamp(min=1e-3))
_s2_mean, _s2_std = float(_s2.mean()), float(_s2.std().clamp(min=1e-3))
del all_pts_train, _y_all, _s1, _s2

coord_mean_net = torch.cat([
    xy_mean,
    torch.zeros(N_PARAMS, dtype=torch.float64),
    torch.tensor([_s1_mean, _s2_mean], dtype=torch.float64),
])
coord_std_net = torch.cat([
    xy_std,
    torch.ones(N_PARAMS, dtype=torch.float64),
    torch.tensor([_s1_std, _s2_std], dtype=torch.float64),
])

qpp_mean = float(all_qpp_train.mean())
qpp_std  = float(all_qpp_train.std().clamp(min=1e-3))
del all_qpp_train

print(f"\nNormalisation  (all DPs)")
print(f"  xy_mean : {xy_mean.tolist()}")
print(f"  xy_std  : {xy_std.tolist()}")
print(f"  q'' mean={qpp_mean:.2f}  std={qpp_std:.2f}  W/m²")
print(f"  Wall sigmoid: s1 mean={_s1_mean:.3f} std={_s1_std:.3f}  "
      f"s2 mean={_s2_mean:.3f} std={_s2_std:.3f}")

# ═══════════════════════════════════════════════════════════════
# INPUT BUILDER + NORM HELPERS
# ═══════════════════════════════════════════════════════════════

def build_input(pts, params_nd):
    """(x, y) + 5 z-scored params + 2 wall sigmoid features → (N, 9)."""
    N  = pts.shape[0]
    xy = pts[:, :2]
    y  = pts[:, 1:2]
    s1 = torch.sigmoid((y - WALL_Y1) / WALL_EPS)
    s2 = torch.sigmoid((y - WALL_Y2) / WALL_EPS)
    return torch.cat([xy, params_nd.unsqueeze(0).expand(N, -1), s1, s2], dim=1)


def norm_input(x_raw):
    return (x_raw - coord_mean_net.to(x_raw.device)) / coord_std_net.to(x_raw.device)


def norm_qpp(qpp):
    return (qpp - qpp_mean) / qpp_std


def denorm_qpp(qpp_nd):
    return qpp_nd * qpp_std + qpp_mean


# ── Build training and test tensors ──────────────────────────
train_X = torch.cat([norm_input(build_input(d["pts"], d["params_nd"])) for d in train_dps])
train_Y = torch.cat([norm_qpp(d["qpp"]) for d in train_dps])
test_X  = torch.cat([norm_input(build_input(d["pts"], d["params_nd"])) for d in test_dps])
test_Y  = torch.cat([norm_qpp(d["qpp"]) for d in test_dps])
print(f"\nTotal training points : {train_X.shape[0]:,}")
print(f"Total test points     : {test_X.shape[0]:,}")

# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

model     = QPP_MLP(IN_DIM, HIDDEN_DIM, N_LAYERS).to(DEVICE).to(torch.float64)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
n_params_model = sum(p.numel() for p in model.parameters())
print(f"\nModel : {IN_DIM}→{HIDDEN_DIM}×{N_LAYERS-1}→1   params={n_params_model:,}")

# ═══════════════════════════════════════════════════════════════
# W&B
# ═══════════════════════════════════════════════════════════════

if not args.debug:
    api_key = "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"
    wandb.login(key=api_key)
    wandb.init(
        project=args.project,
        name=os.path.basename(RUN_PATH),
        config={
            "n_dps": N_DPS, "n_train": len(train_dps), "n_test": len(test_dps),
            "train_dps": train_names, "test_dps": test_names,
            "epochs": EPOCHS, "lr": LR, "lr_end": LR_END, "batch_size": BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "in_dim": IN_DIM, "seed": SEED,
            "weight_decay": args.weight_decay, "patience": args.patience,
        },
    )
    if wandb.run.sweep_id:
        wc = wandb.config
        HIDDEN_DIM        = int(getattr(wc, "hidden_dim",    HIDDEN_DIM))
        N_LAYERS          = int(getattr(wc, "n_layers",      N_LAYERS))
        LR                = float(getattr(wc, "lr",           LR))
        LR_END            = float(getattr(wc, "lr_end",       LR_END))
        args.weight_decay = float(getattr(wc, "weight_decay", args.weight_decay))
        BATCH_SIZE        = int(getattr(wc, "batch_size",     BATCH_SIZE))
        GAMMA             = (LR_END / LR) ** (1.0 / EPOCHS)
        model     = QPP_MLP(IN_DIM, HIDDEN_DIM, N_LAYERS).to(DEVICE).to(torch.float64)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
        print(f"  Sweep override : h={HIDDEN_DIM} l={N_LAYERS} lr={LR:.2e} "
              f"lr_end={LR_END:.2e} wd={args.weight_decay:.2e}  "
              f"params={sum(p.numel() for p in model.parameters()):,}")

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

train_X = train_X.to(DEVICE)
train_Y = train_Y.to(DEVICE)
test_X  = test_X.to(DEVICE)
test_Y  = test_Y.to(DEVICE)
N_train = train_X.shape[0]

def save_checkpoint(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state":    model.state_dict(),
        "coord_mean_net": coord_mean_net,
        "coord_std_net":  coord_std_net,
        "qpp_mean":       qpp_mean,
        "qpp_std":        qpp_std,
        "hidden_dim":     HIDDEN_DIM,
        "n_layers":       N_LAYERS,
        "in_dim":         IN_DIM,
        "train_dps":      train_names,
        "test_dps":       test_names,
    }, path)

PATIENCE  = args.patience
MIN_DELTA = args.min_delta

best_test_loss = float("inf")
no_improve     = 0
stopped_early  = False
best_ckpt_path = os.path.join(RUN_PATH, "qpp_mlp_best.pt")

print(f"\nTraining  (max {EPOCHS} epochs, batch={BATCH_SIZE}, "
      f"patience={PATIENCE}, min_delta={MIN_DELTA:.0e}, weight_decay={args.weight_decay:.0e})")
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm       = torch.randperm(N_train, device=DEVICE)
    epoch_loss = 0.0
    n_batches  = 0
    for i in range(0, N_train, BATCH_SIZE):
        idx  = perm[i : i + BATCH_SIZE]
        pred = model(train_X[idx])
        loss = torch.mean((pred - train_Y[idx]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1
    scheduler.step()
    epoch_loss /= n_batches

    model.eval()
    with torch.no_grad():
        test_loss = torch.mean((model(test_X) - test_Y) ** 2).item()
    model.train()

    # ── Early stopping on test loss + best checkpoint ─────────
    if test_loss < best_test_loss - MIN_DELTA:
        best_test_loss = test_loss
        no_improve     = 0
        save_checkpoint(best_ckpt_path)
    else:
        no_improve += 1

    log = {"epoch": epoch, "train_mse_nd": epoch_loss,
           "test_mse_nd": test_loss, "best_test_mse_nd": best_test_loss,
           "lr": scheduler.get_last_lr()[0]}
    print(f"  epoch {epoch:5d}  train_mse_nd={epoch_loss:.4e}  "
          f"test_mse_nd={test_loss:.4e}  "
          f"no_improve={no_improve}/{PATIENCE}  lr={scheduler.get_last_lr()[0]:.2e}")
    if not args.debug:
        wandb.log(log)

    if no_improve >= PATIENCE:
        print(f"\nEarly stop at epoch {epoch}  (best test_mse_nd={best_test_loss:.4e})")
        stopped_early = True
        break

save_checkpoint(os.path.join(RUN_PATH, "qpp_mlp_final.pt"))
print(f"\nModel saved  ({'early stop' if stopped_early else f'epoch {epoch}'})")

# ═══════════════════════════════════════════════════════════════
# FINAL EVALUATION AND PLOTS
# ═══════════════════════════════════════════════════════════════

model.eval()


def eval_and_plot(dps, tag):
    with torch.no_grad():
        for d in dps:
            x_nd     = norm_input(build_input(d["pts"], d["params_nd"])).to(DEVICE)
            pred_qpp = denorm_qpp(model(x_nd)).cpu()
            rmse = float(torch.sqrt(torch.mean((pred_qpp - d["qpp"]) ** 2)))
            r2   = 1.0 - float(torch.mean((pred_qpp - d["qpp"])**2)) / float(d["qpp"].var())
            print(f"  {d['folder']:6s}  RMSE={rmse:.2f} W/m²  R²={r2:.4f}")
            out_path = os.path.join(RUN_PATH, f"qpp_{tag}_{d['folder']}.png")
            plot_qpp_single(d["folder"], d["pts"], d["qpp"], pred_qpp, out_path)


# Train self-check (sample to keep runtime short)
sample_train = train_dps[::max(1, len(train_dps) // 5)][:5]
print(f"\nGenerating train plots ({len(sample_train)} DPs)…")
eval_and_plot(sample_train, tag="train")

# Test evaluation (all test DPs)
print(f"\nGenerating test plots ({len(test_dps)} DPs)…")
eval_and_plot(test_dps, tag="test")

if not args.debug:
    wandb.finish()

_log_handle.close()
print("\nDone.")
