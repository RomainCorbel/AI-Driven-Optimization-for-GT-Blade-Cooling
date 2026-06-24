import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

WALL_Y1  = 0.185
WALL_Y2  = 0.375
WALL_EPS = 0.002
BUFFER_M = 0.420

PARAM_NAMES = ["AR",   "e/Dh",  "P/e",  "alpha",   "Re"    ]
PARAM_MEANS = torch.tensor([ 9.0,   0.127,  10.0,   52.0,  108000.0], dtype=torch.float64)
PARAM_STDS  = torch.tensor([ 3.5,   0.048,   3.1,   14.5,   58000.0], dtype=torch.float64)

DP_CONFIGS = [
    {"folder": "dp00", "ar":  7.5,      "e_dh": 0.074,    "p_e":  8.0,      "alpha": 60.0,     "re": 100000.0 },
    {"folder": "dp01", "ar":  3.644728, "e_dh": 0.049743, "p_e":  4.7724,   "alpha": 25.41772, "re":  16278.05},
    {"folder": "dp02", "ar":  3.890253, "e_dh": 0.046589, "p_e":  4.984645, "alpha": 28.74614, "re":  19457.92},
    {"folder": "dp03", "ar":  9.002429, "e_dh": 0.18974,  "p_e":  6.423034, "alpha": 74.29562, "re":  33203.83},
    {"folder": "dp04", "ar": 14.2659,   "e_dh": 0.094117, "p_e": 11.94046,  "alpha": 54.08848, "re": 171579.3 },
    {"folder": "dp05", "ar":  5.67934,  "e_dh": 0.151972, "p_e": 12.38583,  "alpha": 73.44649, "re": 164877.9 },
    {"folder": "dp06", "ar":  6.027419, "e_dh": 0.18568,  "p_e":  8.921497, "alpha": 59.02505, "re": 121107.7 },
    {"folder": "dp07", "ar": 10.98555,  "e_dh": 0.101665, "p_e":  8.443827, "alpha": 41.87127, "re":  24717.01},
    {"folder": "dp08", "ar": 12.29549,  "e_dh": 0.078886, "p_e":  5.449208, "alpha": 35.0332,  "re": 196382.7 },
    {"folder": "dp09", "ar":  8.515138, "e_dh": 0.069877, "p_e":  8.861573, "alpha": 36.31164, "re":  76870.97},
    {"folder": "dp10", "ar": 14.40566,  "e_dh": 0.15972,  "p_e":  8.679227, "alpha": 65.17492, "re": 189773.0 },
    {"folder": "dp11", "ar":  4.378955, "e_dh": 0.140784, "p_e":  8.114662, "alpha": 37.30411, "re": 112306.8 },
    {"folder": "dp12", "ar":  7.607081, "e_dh": 0.122157, "p_e": 14.1121,   "alpha": 40.9318,  "re":  35772.6 },
    {"folder": "dp13", "ar": 10.89665,  "e_dh": 0.164771, "p_e": 11.72562,  "alpha": 46.07647, "re":  48748.35},
    {"folder": "dp14", "ar":  4.668702, "e_dh": 0.083476, "p_e":  9.196702, "alpha": 44.45944, "re":  95242.31},
    {"folder": "dp15", "ar":  5.398892, "e_dh": 0.097308, "p_e": 11.52129,  "alpha": 51.20543, "re": 124027.6 },
    {"folder": "dp16", "ar":  6.640551, "e_dh": 0.156298, "p_e": 11.27025,  "alpha": 34.82768, "re": 101590.6 },
    {"folder": "dp17", "ar": 12.90792,  "e_dh": 0.125448, "p_e": 14.94032,  "alpha": 38.43865, "re":  21978.75},
    {"folder": "dp18", "ar": 13.75498,  "e_dh": 0.121149, "p_e": 12.54017,  "alpha": 61.05144, "re": 153635.5 },
    {"folder": "dp19", "ar":  4.079941, "e_dh": 0.086112, "p_e": 12.01869,  "alpha": 65.94071, "re":  87782.58},
    {"folder": "dp20", "ar":  7.056191, "e_dh": 0.148406, "p_e": 13.78874,  "alpha": 45.18996, "re": 143472.5 },
    {"folder": "dp21", "ar":  9.87082,  "e_dh": 0.168413, "p_e":  5.105393, "alpha": 43.60288, "re":  61098.39},
    {"folder": "dp22", "ar": 14.77513,  "e_dh": 0.061514, "p_e": 10.96145,  "alpha": 69.06866, "re": 140693.4 },
    {"folder": "dp23", "ar": 14.62171,  "e_dh": 0.073192, "p_e": 14.56414,  "alpha": 63.4404,  "re": 107269.2 },
    {"folder": "dp24", "ar": 10.4212,   "e_dh": 0.131579, "p_e": 10.35821,  "alpha": 47.54174, "re": 148329.5 },
    {"folder": "dp25", "ar": 13.89294,  "e_dh": 0.196634, "p_e":  5.968536, "alpha": 40.4697,  "re":  64305.93},
    {"folder": "dp26", "ar":  8.260275, "e_dh": 0.106006, "p_e": 10.63216,  "alpha": 31.18607, "re": 183860.6 },
    {"folder": "dp27", "ar":  6.610613, "e_dh": 0.090241, "p_e":  7.549495, "alpha": 71.61877, "re":  56800.33},
    {"folder": "dp28", "ar":  9.687974, "e_dh": 0.068103, "p_e":  9.436798, "alpha": 48.67483, "re":  44366.84},
    {"folder": "dp29", "ar":  9.374113, "e_dh": 0.064434, "p_e":  6.303342, "alpha": 59.78739, "re": 158601.6 },
    {"folder": "dp30", "ar":  7.427527, "e_dh": 0.145224, "p_e":  6.949837, "alpha": 56.97142, "re": 185584.6 },
    {"folder": "dp31", "ar": 13.21633,  "e_dh": 0.178371, "p_e": 12.96159,  "alpha": 67.7618,  "re":  80936.64},
    {"folder": "dp32", "ar": 12.7963,   "e_dh": 0.109381, "p_e":  8.007497, "alpha": 39.72069, "re": 132395.1 },
    {"folder": "dp33", "ar": 10.64298,  "e_dh": 0.115023, "p_e":  9.933429, "alpha": 33.21242, "re":  39651.15},
    {"folder": "dp34", "ar": 11.43933,  "e_dh": 0.058513, "p_e": 10.1587,   "alpha": 32.23051, "re":  82745.98},
    {"folder": "dp35", "ar":  7.275716, "e_dh": 0.117108, "p_e":  5.367998, "alpha": 30.2532,  "re": 177585.9 },
    {"folder": "dp36", "ar":  5.135855, "e_dh": 0.191313, "p_e": 13.56869,  "alpha": 61.60822, "re": 150207.4 },
    {"folder": "dp37", "ar":  5.459505, "e_dh": 0.054453, "p_e":  9.665539, "alpha": 53.22351, "re": 115302.3 },
    {"folder": "dp38", "ar": 11.38441,  "e_dh": 0.079722, "p_e": 14.61072,  "alpha": 70.09904, "re":  94011.89},
    {"folder": "dp39", "ar":  7.906148, "e_dh": 0.171919, "p_e": 10.78854,  "alpha": 48.24447, "re":  68803.85},
    {"folder": "dp40", "ar": 10.13908,  "e_dh": 0.051254, "p_e": 13.31478,  "alpha": 66.29723, "re":  71450.48},
    {"folder": "dp41", "ar": 12.53072,  "e_dh": 0.128661, "p_e":  7.684222, "alpha": 62.53379, "re": 163960.9 },
    {"folder": "dp42", "ar": 11.69391,  "e_dh": 0.174673, "p_e":  6.619144, "alpha": 70.79936, "re": 102851.9 },
    {"folder": "dp43", "ar": 11.99886,  "e_dh": 0.197554, "p_e":  6.98893,  "alpha": 55.25445, "re": 175714.0 },
    {"folder": "dp44", "ar":  4.802336, "e_dh": 0.137666, "p_e":  5.77712,  "alpha": 50.38406, "re": 127704.6 },
    {"folder": "dp45", "ar":  6.189208, "e_dh": 0.18303,  "p_e":  7.266435, "alpha": 56.3389,  "re":  54043.76},
    {"folder": "dp46", "ar":  8.640326, "e_dh": 0.144186, "p_e": 13.09386,  "alpha": 57.94209, "re":  30389.25},
    {"folder": "dp47", "ar": 13.48607,  "e_dh": 0.161387, "p_e": 12.7088,   "alpha": 52.23429, "re": 192455.8 },
    {"folder": "dp48", "ar":  9.210315, "e_dh": 0.104191, "p_e": 14.13385,  "alpha": 72.81153, "re": 137273.4 },
    {"folder": "dp49", "ar": 15.06195,  "e_dh": 0.202031, "p_e": 15.16555,  "alpha": 77.84987, "re": 204946.7 },
    {"folder": "dp50", "ar": 15.45474,  "e_dh": 0.204572, "p_e": 15.07501,  "alpha": 75.50823, "re": 202176.8 },
    {"folder": "dp102", "ar":  9.008373, "e_dh": 0.195552, "p_e":  9.576425, "alpha": 42.49032, "re": 187014.7 },
    {"folder": "dp103", "ar":  4.255637, "e_dh": 0.134505, "p_e": 10.87458,  "alpha": 64.28583, "re":  25922.21},
    {"folder": "dp104", "ar": 13.79748,  "e_dh": 0.094424, "p_e":  5.842396, "alpha": 59.20589, "re":  98254.38},
    {"folder": "dp105", "ar": 10.43976,  "e_dh": 0.079381, "p_e": 14.35755,  "alpha": 30.3749,  "re": 142570.6 },
]
DP_REGISTRY = {c["folder"]: c for c in DP_CONFIGS}

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


class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)


class QPP_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), Sin()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = QPP_MLP(IN_DIM, HIDDEN_DIM, N_LAYERS).to(torch.float64).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

coord_mean_net = coord_mean_net.to(DEVICE)
coord_std_net  = coord_std_net.to(DEVICE)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def build_input(pts, params_nd):
    N  = pts.shape[0]
    xy = pts[:, :2]
    y  = pts[:, 1:2]
    s1 = torch.sigmoid((y - WALL_Y1) / WALL_EPS)
    s2 = torch.sigmoid((y - WALL_Y2) / WALL_EPS)
    return torch.cat([xy, params_nd.unsqueeze(0).expand(N, -1), s1, s2], dim=1)


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


def plot_qpp(folder, pts, true_qpp, pred_qpp, out_dir):
    true_np = true_qpp.numpy()
    pred_np = pred_qpp.numpy()
    pts_np  = pts.numpy()
    rmse    = np.sqrt(np.mean((pred_np - true_np) ** 2))
    vmin, vmax = true_np.min(), true_np.max()

    err_np   = pred_np - true_np
    mape     = float(np.mean(np.abs(err_np / true_np)) * 100)
    err_abs  = np.abs(err_np)
    err_vmax = np.percentile(err_abs, 99)  # clip colorbar at 99th pct to avoid outlier dominance

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f"{folder}  RMSE={rmse:.1f} W/m²  MAPE={mape:.2f}%")

    ax = axes[0]
    ax.scatter(true_np, pred_np, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1)
    ax.set_xlabel("CFD q'' [W/m²]"); ax.set_ylabel("Pred q'' [W/m²]"); ax.set_title("Parity")

    for ax, vals, title in [(axes[1], true_np, "CFD"), (axes[2], pred_np, "MLP")]:
        sc = ax.scatter(pts_np[:,0], pts_np[:,1], c=vals,
                        cmap="hot", s=1, rasterized=True, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="q'' [W/m²]")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)

    ax = axes[3]
    sc = ax.scatter(pts_np[:,0], pts_np[:,1], c=err_abs,
                    cmap="OrRd", s=1, rasterized=True, vmin=0, vmax=err_vmax)
    plt.colorbar(sc, ax=ax, label="|error| [W/m²]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Error map\nMAPE = {mape:.2f}%")

    fig.tight_layout()
    path = os.path.join(out_dir, f"qpp_test_{folder}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path

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

        x_nd     = norm_input(build_input(pts, params_nd).to(DEVICE))
        pred_qpp = denorm_qpp(model(x_nd)).cpu()

        rmse = float(torch.sqrt(torch.mean((pred_qpp - true_qpp) ** 2)))
        mae  = float(torch.mean(torch.abs(pred_qpp - true_qpp)))
        r2   = 1.0 - float(torch.mean((pred_qpp - true_qpp)**2)) / float(true_qpp.var())

        print(f"  {folder}  n={pts.shape[0]:,}  "
              f"RMSE={rmse:.1f} W/m²  MAE={mae:.1f} W/m²  R²={r2:.4f}")

        path = plot_qpp(folder, pts, true_qpp, pred_qpp, OUT_DIR)
        results.append({"folder": folder, "rmse": rmse, "mae": mae, "r2": r2, "plot": path})

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
