import argparse
import numpy as np
import torch
import os
torch.set_default_dtype(torch.float64)

from config import (
    DP_REGISTRY,
    PARAM_NAMES, PARAM_MEANS, PARAM_STDS,
    WALL_Y1, WALL_Y2, WALL_EPS,
)
from models import FFNN, NormalizedPINN
from utils import plot_fields

# ── Parametric input helper ────────────────────────────────────

def with_params(pts_xyz, params_nd):
    row = params_nd.detach().unsqueeze(0).expand(pts_xyz.shape[0], -1)
    return torch.cat([pts_xyz, row], dim=1)

# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="PINN standalone inference with CFD-derived physical scales")
parser.add_argument("--run-path",  required=True,
                    help="Run directory containing pinn_model.pt and normalization.pt")
parser.add_argument("--dps", nargs="+", required=True, metavar="DP",
                    help="DP folders to infer on, e.g. --dps dp00 dp05 dp12")
parser.add_argument("--data-root", default="./preProcessedData/With_T",
                    help="Root directory for CFD data (ground truth)")
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
    cfg = DP_REGISTRY.get(dp_name)
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
