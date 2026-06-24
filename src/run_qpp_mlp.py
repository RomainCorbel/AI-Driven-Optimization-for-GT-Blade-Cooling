import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from torch.optim import Adam

torch.set_default_dtype(torch.float64)

# ═══════════════════════════════════════════════════════════════
# SHARED CONSTANTS  (identical to run_pinn27.py)
# ═══════════════════════════════════════════════════════════════

PARAM_NAMES = ["AR",   "e/Dh",  "P/e",  "alpha",   "Re"    ]
PARAM_MEANS = torch.tensor([ 9.0,   0.127,  10.0,   52.0,  108000.0], dtype=torch.float64)
PARAM_STDS  = torch.tensor([ 3.5,   0.048,   3.1,   14.5,   58000.0], dtype=torch.float64)

WALL_Y1  = 0.185
WALL_Y2  = 0.375
WALL_EPS = 0.002

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
]

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

class _Tee(io.TextIOBase):
    def __init__(self, stream, logfile):
        self._stream = stream; self._logfile = logfile
    def write(self, s):
        self._stream.write(s); self._logfile.write(s); return len(s)
    def flush(self):
        self._stream.flush(); self._logfile.flush()

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
      • <data_root>/<folder>/qpp.npy  — shape (N, 4): [x, y, z, q'']  [W/m²]
      • <data_root>/<folder>/qpp.csv  — CSV with header; x/y/z columns named 'x','y','z';
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
    BUFFER_M = 0.420
    x = pts[:, 0]
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
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (N,)


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


def plot_qpp(dp, pred_qpp, output_dir, tag):
    true_np = dp["qpp"].numpy()
    pred_np = pred_qpp.numpy()
    pts_np  = dp["pts"].numpy()
    rmse    = np.sqrt(np.mean((pred_np - true_np) ** 2))
    vmin, vmax = true_np.min(), true_np.max()

    err_np   = pred_np - true_np
    nrmse    = float(rmse / (true_np.std() + 1e-8) * 100)   # RMSE / std(q''), in %
    err_abs  = np.abs(err_np)
    err_vmax = np.percentile(err_abs, 99)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f"{dp['folder']}  RMSE={rmse:.1f} W/m²  NRMSE={nrmse:.1f}%")

    ax = axes[0]
    ax.scatter(true_np, pred_np, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1)
    ax.set_xlabel("CFD q'' [W/m²]"); ax.set_ylabel("Predicted q'' [W/m²]")
    ax.set_title("Parity")

    for ax, vals, title in [(axes[1], true_np, "CFD — true"), (axes[2], pred_np, "MLP — pred")]:
        sc = ax.scatter(pts_np[:,0], pts_np[:,1], c=vals,
                        cmap="hot", s=1, rasterized=True, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="q'' [W/m²]")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)

    ax = axes[3]
    sc = ax.scatter(pts_np[:,0], pts_np[:,1], c=err_abs,
                    cmap="OrRd", s=1, rasterized=True, vmin=0, vmax=err_vmax)
    plt.colorbar(sc, ax=ax, label="|error| [W/m²]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Error map\nNRMSE = {nrmse:.1f}%")

    fig.tight_layout()
    path = os.path.join(output_dir, f"qpp_{tag}_{dp['folder']}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def eval_and_plot(dps, tag):
    with torch.no_grad():
        for d in dps:
            x_nd     = norm_input(build_input(d["pts"], d["params_nd"])).to(DEVICE)
            pred_qpp = denorm_qpp(model(x_nd)).cpu()
            rmse = float(torch.sqrt(torch.mean((pred_qpp - d["qpp"]) ** 2)))
            r2   = 1.0 - float(torch.mean((pred_qpp - d["qpp"])**2)) / float(d["qpp"].var())
            print(f"  {d['folder']:6s}  RMSE={rmse:.2f} W/m²  R²={r2:.4f}")
            plot_qpp(d, pred_qpp, RUN_PATH, tag=tag)


# Train self-check (sample to keep runtime short)
sample_train = train_dps[::max(1, len(train_dps) // 5)][:5]
print(f"\nGenerating train plots ({len(sample_train)} DPs)…")
eval_and_plot(sample_train, tag="train")

# Test evaluation (all test DPs)
print(f"\nGenerating test plots ({len(test_dps)} DPs)…")
eval_and_plot(test_dps, tag="test")

if not args.debug:
    wandb.finish()

print("\nDone.")
