"""
infer_pinn27.py — standalone inference for a pretrained PINN27 model.

Physical scales are computed from CFD data, exactly as in load_dp during training:
  V_IN    = mean |inlet vx|
  P_REF   = mean(cfd_p[outlet face])
  P_SCALE = std(cfd_p)
  T_WALL  = min(cfd_T)
  DELTA_T = mean(cfd_T[inlet face]) - T_WALL

The model output (pred[:, i]) is already dimensionless — CFD data is
non-dimensionalized to match, so RMSE and plots are in the same space.
This works for any DP whether seen during training or not.

Usage:
  python infer_pinn27.py \\
      --run-path  ../pinn27_runs/my_run \\
      --dps       dp00 dp05 dp12 \\
      --data-root ./preProcessedData/With_T
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# ═══════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════

M          = 28.96e-3
R          = 8.314
K          = 2.61e-2
CP         = 1.00e3
T_REF_SUTH = 278.15
MU_REF     = 1.716e-5
S_SUTH     = 110.4

# ═══════════════════════════════════════════════════════════════
# GEOMETRY CONSTANTS  (SolidWorks, fixed across all DPs)
# ═══════════════════════════════════════════════════════════════

BETA1     = np.deg2rad(-0.15)   # DS_beta1 in degrees
BETA2     = np.deg2rad(-0.40)   # DS_beta2 in degrees
L_HALF_MM = 567.5               # mm (= 1135/2)
W1MID_MM  = 177.0 + np.tan(BETA1) * L_HALF_MM   # ≈ 175.51 mm
W3MID_MM  = 157.0 - np.tan(BETA2) * L_HALF_MM   # ≈ 160.96 mm
W2MID_MM  = (544.75 - W1MID_MM - W3MID_MM
             - np.cos(BETA1)*20 - np.cos(BETA2)*20)  # ≈ 168.27 mm
W_TOT_MM  = W1MID_MM + W2MID_MM + W3MID_MM          # ≈ 504.75 mm

T_STD         = 293.15    # K  — standard conditions for Re definition
P_STD         = 101325.0  # Pa
T_WALL_FIXED  = 293.15    # K  — fixed thermal BC (wall)
T_INLET_FIXED = 329.0     # K  — fixed thermal BC (inlet)
DELTA_T_FIXED = T_INLET_FIXED - T_WALL_FIXED  # K


def analytical_scales(ar, re):
    """Return (V_IN, P_DYN, Dh1) from AR and Re, purely analytically."""
    H_mm   = W_TOT_MM / (3.0 * ar)
    dh1_mm = (4 * (W1MID_MM*H_mm - H_mm**2 + np.pi*H_mm**2/4)
                / (2*W1MID_MM - 2*H_mm + np.pi*H_mm))
    dh1    = dh1_mm / 1000.0   # m
    mu_std  = MU_REF * (T_STD/T_REF_SUTH)**1.5 * (T_REF_SUTH + S_SUTH)/(T_STD + S_SUTH)
    rho_std = P_STD * M / (R * T_STD)
    nu_std  = mu_std / rho_std
    V_IN    = re * nu_std / dh1
    P_DYN   = rho_std * V_IN**2
    return V_IN, P_DYN, dh1

# ═══════════════════════════════════════════════════════════════
# PARAMETRIC INPUT NORMALIZATION
# ═══════════════════════════════════════════════════════════════

PARAM_NAMES = ["AR", "e/Dh", "P/e", "alpha", "Re"]
PARAM_MEANS = torch.tensor([ 9.0,   0.127,  10.0,   52.0,  108000.0], dtype=torch.float64)
PARAM_STDS  = torch.tensor([ 3.5,   0.048,   3.1,   14.5,   58000.0], dtype=torch.float64)

WALL_Y1  = 0.185
WALL_Y2  = 0.375
WALL_EPS = 0.002

# ═══════════════════════════════════════════════════════════════
# DESIGN-POINT REGISTRY
# ═══════════════════════════════════════════════════════════════

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

_DP_BY_FOLDER = {c["folder"]: c for c in DP_CONFIGS}

# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (must match training)
# ═══════════════════════════════════════════════════════════════

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FFNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), Sin()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        for lin in [m for m in self.net if isinstance(m, nn.Linear)]:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return self.net(x)


class NormalizedPINN(nn.Module):
    def __init__(self, net, coord_mean, coord_std, out_mean, out_std):
        super().__init__()
        self.net = net
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std",  coord_std)
        self.register_buffer("out_mean",   out_mean)
        self.register_buffer("out_std",    out_std)

    def forward(self, x):
        y   = x[:, 1:2]
        s1  = torch.sigmoid((y - WALL_Y1) / WALL_EPS)
        s2  = torch.sigmoid((y - WALL_Y2) / WALL_EPS)
        x   = torch.cat([x, s1, s2], dim=1)
        x_n = (x - self.coord_mean) / self.coord_std
        y_n = self.net(x_n)
        safe_std = torch.where(self.out_std == 0, torch.ones_like(self.out_std), self.out_std)
        return y_n * safe_std + self.out_mean


def with_params(pts_xyz, params_nd):
    row = params_nd.detach().unsqueeze(0).expand(pts_xyz.shape[0], -1)
    return torch.cat([pts_xyz, row], dim=1)

# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_fields(pts, fields, output_dir, slice_frac=0.10):
    os.makedirs(output_dir, exist_ok=True)

    def to_np(x):
        if x is None: return None
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pts_np = to_np(pts)
    ranges = pts_np.max(axis=0) - pts_np.min(axis=0)
    box_aspect = (ranges / ranges.max()).tolist()
    z_vals = pts_np[:, 2]
    z_mid  = 0.5 * (z_vals.min() + z_vals.max())
    z_tol  = slice_frac * (z_vals.max() - z_vals.min())
    cut_mask  = np.abs(z_vals - z_mid) < z_tol
    p_cut     = pts_np[cut_mask]
    cut_label = f"x-y  z={z_mid*1000:.1f}mm  (n={cut_mask.sum()})"

    for name, data_raw, pred_raw in fields:
        data, pred = to_np(data_raw), to_np(pred_raw)
        has_data   = data is not None
        vmin = float(min(data.min(), pred.min()) if has_data else pred.min())
        vmax = float(max(data.max(), pred.max()) if has_data else pred.max())

        subplots = []
        if has_data: subplots.append((f"CFD – {name}",  data,        vmin, vmax))
        subplots.append(             (f"PINN – {name}", pred,        vmin, vmax))
        if has_data: subplots.append((f"Diff – {name}", data - pred, None, None))

        n_sub = len(subplots)
        fig   = plt.figure(figsize=(6 * n_sub, 10))
        for i, (title, color, cmin, cmax) in enumerate(subplots):
            ax = fig.add_subplot(2, n_sub, i + 1, projection="3d")
            sc = ax.scatter(pts_np[:,0], pts_np[:,1], pts_np[:,2],
                            c=color, cmap="viridis", vmin=cmin, vmax=cmax, s=1, rasterized=True)
            ax.set_box_aspect(box_aspect); ax.set_title(title, fontsize=10)
            ax.set_xlabel("X"); ax.set_yticklabels([]); ax.set_zticklabels([])
            plt.colorbar(sc, ax=ax, shrink=0.5, label=name)
        for i, (title, color, cmin, cmax) in enumerate(subplots):
            ax  = fig.add_subplot(2, n_sub, n_sub + i + 1)
            sc  = ax.scatter(p_cut[:,0], p_cut[:,1], c=color[cut_mask],
                             cmap="viridis", vmin=cmin, vmax=cmax, s=4, rasterized=True)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
            ax.set_title(f"{title}\n{cut_label}", fontsize=9)
            plt.colorbar(sc, ax=ax, label=name, shrink=0.5, aspect=15)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="PINN27 standalone inference with analytical physical scales")
parser.add_argument("--run-path",  required=True,
                    help="Run directory containing pinn_model.pt and normalization.pt")
parser.add_argument("--dps", nargs="+", required=True, metavar="DP",
                    help="DP folders to infer on, e.g. --dps dp00 dp05 dp12")
parser.add_argument("--data-root", default="./preProcessedData/With_T",
                    help="Root directory for CFD data (ground truth, optional per DP)")
parser.add_argument("--out-dir",   default=None,
                    help="Output directory for plots (default: <run-path>/inference)")
parser.add_argument("--n-plot",    type=int, default=100_000,
                    help="Max points to plot (subsample if larger)")
parser.add_argument("--outlet-y-min-frac", type=float, default=0.70,
                    help="Outlet y-filter fraction (same default as training)")
parser.add_argument("--inlet-y-max-frac",  type=float, default=0.33,
                    help="Inlet y-filter fraction (same default as training)")
args = parser.parse_args()

out_root = args.out_dir or os.path.join(args.run_path, "inference")

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════

norm_path  = os.path.join(args.run_path, "normalization.pt")
model_path = os.path.join(args.run_path, "pinn_model.pt")
for p in [norm_path, model_path]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required file not found: {p}")

norm = torch.load(norm_path, weights_only=True, map_location="cpu")
HIDDEN_DIM = int(norm["hidden_dim"])
N_LAYERS   = int(norm["n_layers"])
IN_DIM     = int(norm["in_dim"])
# Use param stats from checkpoint (consistent with training)
_pm = norm.get("param_means", PARAM_MEANS)
_ps = norm.get("param_stds",  PARAM_STDS)

net   = FFNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS)
model = NormalizedPINN(net,
                       norm["coord_mean_net"], norm["coord_std_net"],
                       norm["out_mean"],       norm["out_std"])
model.load_state_dict(
    torch.load(model_path, weights_only=True, map_location="cpu"))
model.eval()

print(f"Model loaded: h={HIDDEN_DIM}  l={N_LAYERS}  in_dim={IN_DIM}")
print(f"Run path: {args.run_path}")

rmse = lambda a, b: float(np.sqrt(np.mean((a - b)**2)))

# ═══════════════════════════════════════════════════════════════
# INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════

for dp_name in args.dps:
    cfg = _DP_BY_FOLDER.get(dp_name)
    if cfg is None:
        print(f"\n[SKIP] {dp_name} — not found in DP_CONFIGS")
        continue

    print(f"\n{'─'*60}")
    print(f"Inference — {dp_name}"
          f"  (AR={cfg['ar']}  e/Dh={cfg['e_dh']}  P/e={cfg['p_e']}"
          f"  α={cfg['alpha']}°  Re={cfg['re']:.0f})")

    # ── Parametric input (normalised) ─────────────────────────
    params_raw = torch.tensor(
        [cfg["ar"], cfg["e_dh"], cfg["p_e"], cfg["alpha"], cfg["re"]],
        dtype=torch.float64)
    params_nd = (params_raw - _pm) / _ps
    print(f"  params_nd = {[f'{v:.3f}' for v in params_nd.tolist()]}")

    # ── CFD data ───────────────────────────────────────────────
    data_dir = os.path.join(args.data_root, dp_name)
    if not (os.path.isdir(data_dir) and os.path.isfile(os.path.join(data_dir, "vel_x.npy"))):
        print(f"  [SKIP] CFD data not found in {data_dir}")
        continue

    _vx_raw = np.load(os.path.join(data_dir, "vel_x.npy"))
    pts     = torch.tensor(_vx_raw[:, :3], dtype=torch.float64)
    cfd_vx  = _vx_raw[:, 3];  del _vx_raw
    cfd_vy  = np.load(os.path.join(data_dir, "vel_y.npy"))[:, 3]
    cfd_vz  = np.load(os.path.join(data_dir, "vel_z.npy"))[:, 3]
    cfd_p   = np.load(os.path.join(data_dir, "press.npy"))[:, 3]
    _temp   = np.load(os.path.join(data_dir, "temp.npy"))
    cfd_T   = _temp[:, 3]
    print(f"  CFD: {len(pts):,} pts")

    # ── Physical scales from CFD (identical to load_dp in training) ──
    # Inlet velocity → V_IN
    _in_vx = np.load(os.path.join(data_dir, "vel_x_inlet.npy"))
    if args.inlet_y_max_frac < 1.0:
        _y_max = float(_in_vx[:, 1].max())
        _in_vx = _in_vx[_in_vx[:, 1] <= args.inlet_y_max_frac * _y_max]
    V_IN = float(np.abs(_in_vx[:, 3]).mean())

    # Outlet mask → P_REF
    pts_np_all = pts.numpy()
    x_max_m    = float(pts_np_all[:, 0].max())
    out_mask   = pts_np_all[:, 0] > (x_max_m - 0.005)
    if args.outlet_y_min_frac > 0.0:
        _y_max_o = float(pts_np_all[:, 1].max())
        out_mask = out_mask & (pts_np_all[:, 1] >= args.outlet_y_min_frac * _y_max_o)
    P_REF  = float(cfd_p[out_mask].mean())
    P_SCALE = float(max(np.std(cfd_p), 1.0))

    # Temperatures → T_WALL, DELTA_T
    T_WALL  = float(cfd_T.min())
    _x_min  = float(_temp[:, 0].min())
    _face   = _temp[_temp[:, 0] == _x_min]
    if args.inlet_y_max_frac < 1.0:
        _y_max_t = float(_temp[:, 1].max())
        _face    = _face[_face[:, 1] <= args.inlet_y_max_frac * _y_max_t]
    t_inlet = float(_face[:, 3].mean())
    DELTA_T = t_inlet - T_WALL
    del _temp, _in_vx, _face

    print(f"  V_IN={V_IN:.3f} m/s  P_REF={P_REF:.0f} Pa  P_SCALE={P_SCALE:.1f} Pa"
          f"  T_WALL={T_WALL:.2f} K  ΔT={DELTA_T:.2f} K")

    # ── Non-dimensionalize CFD (same convention as training) ───
    cfd_vx_nd = cfd_vx / V_IN
    cfd_vy_nd = cfd_vy / V_IN
    cfd_vz_nd = cfd_vz / V_IN
    cfd_p_nd  = (cfd_p - P_REF) / P_SCALE
    cfd_T_nd  = (cfd_T - T_WALL) / DELTA_T

    # ── Model prediction (already dimensionless) ───────────────
    with torch.no_grad():
        pred = model(with_params(pts, params_nd))

    # ── RMSE in dimensionless space ────────────────────────────
    print(f"\n  {'field':<6}  {'RMSE (nd)':>12}  {'MSE (nd)':>12}")
    print(f"  {'─'*36}")
    for name, true_nd, pred_nd in [
        ("vx", cfd_vx_nd, pred[:, 0].numpy()),
        ("vy", cfd_vy_nd, pred[:, 1].numpy()),
        ("vz", cfd_vz_nd, pred[:, 2].numpy()),
        ("p",  cfd_p_nd,  pred[:, 3].numpy()),
        ("T",  cfd_T_nd,  pred[:, 4].numpy()),
    ]:
        r = rmse(true_nd, pred_nd)
        m = float(np.mean((true_nd - pred_nd)**2))
        print(f"  {name:<6}  {r:>12.4e}  {m:>12.4e}")

    # ── Plots in dimensionless space ───────────────────────────
    inf_dir  = os.path.join(out_root, dp_name)
    pts_np   = pts.numpy()
    n_pts    = len(pts_np)
    idx_plot = (np.random.choice(n_pts, min(args.n_plot, n_pts), replace=False)
                if n_pts > args.n_plot else np.arange(n_pts))

    plot_fields(
        pts_np[idx_plot],
        [
            ("vx_nd", cfd_vx_nd[idx_plot], pred[:, 0].numpy()[idx_plot]),
            ("vy_nd", cfd_vy_nd[idx_plot], pred[:, 1].numpy()[idx_plot]),
            ("vz_nd", cfd_vz_nd[idx_plot], pred[:, 2].numpy()[idx_plot]),
            ("p_nd",  cfd_p_nd[idx_plot],  pred[:, 3].numpy()[idx_plot]),
            ("T_nd",  cfd_T_nd[idx_plot],  pred[:, 4].numpy()[idx_plot]),
        ],
        output_dir=inf_dir,
    )
    print(f"  Plots saved → {inf_dir}/")

print(f"\nDone.")
