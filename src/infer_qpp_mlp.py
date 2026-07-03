import argparse
import glob
import numpy as np
import torch
import os
torch.set_default_dtype(torch.float64)

from config import (
    DP_REGISTRY,
    PARAM_MEANS, PARAM_STDS,
    BUFFER_M,
)
from models import QPP_MLP
from utils import build_mlp_input, plot_qpp_single

# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="QPP-MLP inference on a test dataset.\n"
                "Loads qpp_mlp_final.pt, evaluates on DPs with qpp.npy, "
                "prints RMSE/R² per DP and saves plots.")

parser.add_argument("--checkpoint", required=True,
                    help="Path to qpp_mlp_final.pt (or any saved checkpoint).")
parser.add_argument("--data-dir",   default="./preProcessedData/With_T",
                    help="Root folder containing dpXX subfolders with qpp.npy.")
parser.add_argument("--out-dir",    default=None,
                    help="Where to save plots (default: alongside the checkpoint).")
parser.add_argument("--device",     default="cuda")
parser.add_argument("--dps", nargs="+", default=None, metavar="DP",
                    help="Restrict to these DPs, e.g. --dps dp46 dp47. "
                         "Default: all dpXX folders found in --data-dir that have qpp.npy.")

args = parser.parse_args()

DEVICE  = args.device
OUT_DIR = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════

print(f"Loading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

coord_mean_net = ckpt["coord_mean_net"].to(torch.float64)
coord_std_net  = ckpt["coord_std_net"].to(torch.float64)
qpp_mean       = float(ckpt["qpp_mean"])
qpp_std        = float(ckpt["qpp_std"])
HIDDEN_DIM     = int(ckpt["hidden_dim"])
N_LAYERS       = int(ckpt["n_layers"])
IN_DIM         = int(ckpt["in_dim"])
print(f"Architecture : {IN_DIM}→{HIDDEN_DIM}×{N_LAYERS-1}→1")
print(f"Trained on   : {ckpt['train_dps']}")
print(f"q'' norm     : mean={qpp_mean:.2f}  std={qpp_std:.2f}  W/m²")

# Inference-only: load weights without Xavier init (init is overwritten by load_state_dict)
model = QPP_MLP(IN_DIM, HIDDEN_DIM, N_LAYERS).to(torch.float64).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

coord_mean_net = coord_mean_net.to(DEVICE)
coord_std_net  = coord_std_net.to(DEVICE)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def norm_input(x_raw):
    return (x_raw - coord_mean_net) / coord_std_net


def denorm_qpp(qpp_nd):
    return qpp_nd * qpp_std + qpp_mean


def load_and_trim(dp_dir):
    """Load qpp.npy and apply the 420 mm inlet/outlet buffer trim."""
    npy_path = os.path.join(dp_dir, "qpp.npy")
    if not os.path.exists(npy_path):
        return None, None
    arr = np.load(npy_path)
    pts = torch.tensor(arr[:, :3], dtype=torch.float64)
    qpp = torch.tensor(arr[:,  3], dtype=torch.float64)
    x   = pts[:, 0]
    mask = (x >= float(x.min()) + BUFFER_M) & (x <= float(x.max()) - BUFFER_M)
    return pts[mask], qpp[mask]

# ═══════════════════════════════════════════════════════════════
# DISCOVER TEST DPs
# ═══════════════════════════════════════════════════════════════

if args.dps:
    folders = args.dps
else:
    folders = sorted([
        os.path.basename(d)
        for d in glob.glob(os.path.join(args.data_dir, "dp*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "qpp.npy"))
    ])

print(f"\nTest DPs found: {folders}")

# ═══════════════════════════════════════════════════════════════
# INFERENCE + EVALUATION
# ═══════════════════════════════════════════════════════════════

results = []

with torch.no_grad():
    for folder in folders:
        cfg = DP_REGISTRY.get(folder)
        if cfg is None:
            print(f"  [SKIP] {folder} — not in DP_REGISTRY")
            continue

        dp_dir = os.path.join(args.data_dir, folder)
        pts, true_qpp = load_and_trim(dp_dir)
        if pts is None:
            print(f"  [SKIP] {folder} — no qpp.npy")
            continue

        params_raw = torch.tensor(
            [cfg["ar"], cfg["e_dh"], cfg["p_e"], cfg["alpha"], cfg["re"]], dtype=torch.float64)
        params_nd = (params_raw - PARAM_MEANS) / PARAM_STDS

        x_nd     = norm_input(build_mlp_input(pts, params_nd).to(DEVICE))
        pred_qpp = denorm_qpp(model(x_nd)).cpu()

        rmse = float(torch.sqrt(torch.mean((pred_qpp - true_qpp) ** 2)))
        mae  = float(torch.mean(torch.abs(pred_qpp - true_qpp)))
        r2   = 1.0 - float(torch.mean((pred_qpp - true_qpp)**2)) / float(true_qpp.var())

        print(f"  {folder}  n={pts.shape[0]:,}  "
              f"RMSE={rmse:.1f} W/m²  MAE={mae:.1f} W/m²  R²={r2:.4f}")

        out_path = os.path.join(OUT_DIR, f"qpp_test_{folder}.png")
        plot_qpp_single(folder, pts, true_qpp, pred_qpp, out_path)
        results.append({"folder": folder, "rmse": rmse, "mae": mae, "r2": r2})

# ── Summary ──────────────────────────────────────────────────
if results:
    rmses = [r["rmse"] for r in results]
    r2s   = [r["r2"]   for r in results]
    print(f"\n{'─'*55}")
    print(f"  Mean RMSE : {np.mean(rmses):.2f} W/m²")
    print(f"  Mean R²   : {np.mean(r2s):.4f}")
    print(f"  Plots saved to: {OUT_DIR}")
else:
    print("\nNo DPs evaluated.")
