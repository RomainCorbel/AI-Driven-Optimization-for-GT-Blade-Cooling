"""
compute_htc.py — HTC = q'' / (Tw - Tb) from PINN volume T + QPP-MLP wall q''

  - 6 cross-sectional T-planes from PINN predictions
  - Linear interpolation in the 3 passages
  - asin-based angular interpolation in the 2 bends

Usage:
  python compute_htc.py \\
      --pinn-path  ../pinn27_sweep_runs/bovcpaue/trial_5 \\
      --mlp-ckpt   ../qpp_mlp_sweep_runs/enim31rb/trial_4/qpp_mlp_final.pt \\
      --data-dir   ./preProcessedData/Inference \\
      --dps        dp102 dp103 dp105 \\
      --out-dir    ../HTC_pred
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from config import (
    DP_REGISTRY, DP_CONFIGS,
    PARAM_MEANS, PARAM_STDS,
    WALL_Y1, WALL_Y2, WALL_EPS,
    BUFFER_M,
    K_FLUID, PR_FLUID,
    DH_REGISTRY,
    T_PLANES, PLANE_NAMES,
    X_TOP_BEND, X_BOT_BEND, Y_P12, Y_P23, CY_TOP, CY_BOT, X_TOUT, L_PASS3,
    PLANE_TOL,
    SECTION_LABELS, SEC_TPLANE,
)
from models import FFNN, NormalizedPINN, QPP_MLP
from utils import (
    _Tee,
    plot_qpp_single,
    plot_field, plot_htc_section, plot_t_planes,
    plot_nu_field, plot_qpp_full, plot_htc_full,
    save_epsilon_table, save_nu_section_table,
)

# ═══════════════════════════════════════════════════════════════
# TB COMPUTATION  (supervisor's formulas)
# ═══════════════════════════════════════════════════════════════

def compute_t_planes(x_vol, y_vol, T_vol, tol=PLANE_TOL):
    """Average PINN temperature at each of the 6 key cross-sections."""
    T_pl = np.full(6, np.nan)
    for i, (xp, ymin, ymax) in enumerate(T_PLANES):
        mask = (np.abs(x_vol - xp) <= tol) & (y_vol >= ymin) & (y_vol <= ymax)
        n = int(mask.sum())
        if n >= 3:
            T_pl[i] = float(T_vol[mask].mean())
        else:
            print(f"  [warn] plane {i} ({PLANE_NAMES[i]}) only {n} pts – widening tol")
            mask2 = (np.abs(x_vol - xp) <= 2 * tol) & (y_vol >= ymin) & (y_vol <= ymax)
            if mask2.sum() >= 3:
                T_pl[i] = float(T_vol[mask2].mean())
    return T_pl


def compute_tb(x_w, y_w, T_pl):
    """Bulk temperature at wall points using supervisor's 5 Tb formulas."""
    Tb = np.full(len(x_w), np.nan)

    # --- Pass 1 ---
    m = (x_w <= X_TOP_BEND) & (y_w <= Y_P12)
    Tb[m] = T_pl[0] + (T_pl[1] - T_pl[0]) / X_TOP_BEND * x_w[m]

    # --- Pass 2 ---
    m = (x_w >= X_BOT_BEND) & (x_w <= X_TOP_BEND) & (y_w > Y_P12) & (y_w <= Y_P23)
    Tb[m] = T_pl[2] + (T_pl[3] - T_pl[2]) / (X_TOP_BEND - X_BOT_BEND) * (X_TOP_BEND - x_w[m])

    # --- Pass 3 ---
    m = (x_w >= X_BOT_BEND) & (y_w > Y_P23)
    Tb[m] = T_pl[4] + (T_pl[5] - T_pl[4]) / L_PASS3 * (x_w[m] - X_BOT_BEND)

    # --- Top bend (overwrites pass-1 edge) ---
    m = (x_w >= X_TOP_BEND) & (y_w <= Y_P23)
    dy = y_w[m] - CY_TOP
    dx = x_w[m] - X_TOP_BEND
    r  = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
    Tb[m] = ((T_pl[1] + T_pl[2]) / 2
             + (T_pl[2] - T_pl[1]) / np.pi * np.arcsin(np.clip(dy / r, -1.0, 1.0)))

    # --- Bottom bend (overwrites pass-2 edge) ---
    m = (x_w <= X_BOT_BEND) & (y_w > Y_P12)
    dy = y_w[m] - CY_BOT
    dx = X_BOT_BEND - x_w[m]
    r  = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
    Tb[m] = ((T_pl[3] + T_pl[4]) / 2
             + (T_pl[4] - T_pl[3]) / np.pi * np.arcsin(np.clip(dy / r, -1.0, 1.0)))

    return Tb


# ═══════════════════════════════════════════════════════════════
# GT HTC LOADER
# ═══════════════════════════════════════════════════════════════

def load_htc_gt(data_dir, dp_name):
    """Return dict {1: array(N,4), ..., 5: array(N,4)} where cols = x,y,z,htc."""
    results = {}
    for i in range(1, 6):
        path = os.path.join(data_dir, dp_name, f"HTC{i}_V_{dp_name}.csv")
        if not os.path.exists(path):
            continue
        rows = []
        in_data = False
        with open(path) as f:
            for line in f:
                if "[Data]" in line:
                    next(f)    # skip column header line
                    in_data = True
                    continue
                if not in_data:
                    continue
                parts = line.strip().split(",")
                if len(parts) == 4:
                    try:
                        rows.append([float(v) for v in parts])
                    except ValueError:
                        pass
        if rows:
            results[i] = np.array(rows, dtype=np.float64)
    return results


# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--pinn-path", required=True)
parser.add_argument("--mlp-ckpt",  required=True)
parser.add_argument("--data-dir",  default="./preProcessedData/Inference")
parser.add_argument("--dps",  nargs="+", default=["dp102", "dp103", "dp105"])
parser.add_argument("--out-dir",   default="../HTC_pred")
parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--pinn-batch", type=int, default=500_000,
                    help="PINN inference batch size (volume pts)")
parser.add_argument("--inlet-y-frac", type=float, default=0.33)
args = parser.parse_args()

DEVICE = args.device
os.makedirs(args.out_dir, exist_ok=True)

_log_path   = os.path.join(args.out_dir, "compute_htc.log")
_log_handle = open(_log_path, "w", buffering=1, encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__,  _log_handle)
sys.stderr  = _Tee(sys.__stderr__,  _log_handle)
print(f"Log → {_log_path}")

# ═══════════════════════════════════════════════════════════════
# LOAD PINN
# ═══════════════════════════════════════════════════════════════

norm_path  = os.path.join(args.pinn_path, "normalization.pt")
model_path = os.path.join(args.pinn_path, "pinn_model.pt")
norm = torch.load(norm_path, weights_only=True, map_location="cpu")
PINN_HIDDEN = int(norm["hidden_dim"])
PINN_LAYERS = int(norm["n_layers"])
PINN_INDIM  = int(norm["in_dim"])
_pm = norm.get("param_means", PARAM_MEANS)
_ps = norm.get("param_stds",  PARAM_STDS)

pinn_net   = FFNN(PINN_INDIM, PINN_HIDDEN, 5, PINN_LAYERS)
pinn_model = NormalizedPINN(pinn_net,
                            norm["coord_mean_net"], norm["coord_std_net"],
                            norm["out_mean"],       norm["out_std"])
pinn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
pinn_model.eval()   # keep on CPU — float64 CUDA causes overflow on IZAR GPU
print(f"PINN loaded  h={PINN_HIDDEN}  l={PINN_LAYERS}  in={PINN_INDIM}")

# ═══════════════════════════════════════════════════════════════
# LOAD MLP
# ═══════════════════════════════════════════════════════════════

mlp_ckpt = torch.load(args.mlp_ckpt, map_location="cpu", weights_only=True)
MLP_HIDDEN = int(mlp_ckpt["hidden_dim"])
MLP_LAYERS = int(mlp_ckpt["n_layers"])
MLP_INDIM  = int(mlp_ckpt["in_dim"])
QPP_MEAN   = float(mlp_ckpt["qpp_mean"])
QPP_STD    = float(mlp_ckpt["qpp_std"])
mlp_coord_mean = mlp_ckpt["coord_mean_net"].to(torch.float64).to(DEVICE)
mlp_coord_std  = mlp_ckpt["coord_std_net"].to(torch.float64).to(DEVICE)

mlp_model = QPP_MLP(MLP_INDIM, MLP_HIDDEN, MLP_LAYERS).to(torch.float64).to(DEVICE)
mlp_model.load_state_dict(mlp_ckpt["model_state"])
mlp_model.eval()
print(f"MLP  loaded  h={MLP_HIDDEN}  l={MLP_LAYERS}  in={MLP_INDIM}"
      f"  q'' mean={QPP_MEAN:.1f}  std={QPP_STD:.1f} W/m²")


def mlp_predict_qpp(x_np, y_np, params_nd_t):
    """Run MLP on numpy (x,y) wall points → q'' in W/m²."""
    pts = torch.tensor(np.stack([x_np, y_np], axis=1), dtype=torch.float64)
    N   = pts.shape[0]
    y_t = pts[:, 1:2]
    s1  = torch.sigmoid((y_t - WALL_Y1) / WALL_EPS)
    s2  = torch.sigmoid((y_t - WALL_Y2) / WALL_EPS)
    inp = torch.cat([pts[:, :2],
                     params_nd_t.unsqueeze(0).expand(N, -1), s1, s2], dim=1)
    inp_n = (inp.to(DEVICE) - mlp_coord_mean) / mlp_coord_std
    with torch.no_grad():
        q_nd = mlp_model(inp_n).cpu().numpy()
    return q_nd * QPP_STD + QPP_MEAN


def pinn_predict_all(pts_np, params_nd_t):
    """Run PINN on volume points → raw dimensionless output (N, 5): vx,vy,vz,p,T."""
    N          = len(pts_np)
    out        = np.empty((N, 5), dtype=np.float64)
    p_np       = params_nd_t.numpy()
    params_row = np.tile(p_np, (args.pinn_batch, 1))

    for start in range(0, N, args.pinn_batch):
        end   = min(start + args.pinn_batch, N)
        m     = end - start
        inp   = np.concatenate([pts_np[start:end], params_row[:m]], axis=1)
        inp_t = torch.tensor(inp, dtype=torch.float64)
        with torch.no_grad():
            out[start:end] = pinn_model(inp_t).numpy()

    return out


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

t_total_start = time.time()
dp_records    = []   # per-DP midplane data for field figures
qpp_records   = []   # per-DP wall data for combined q'' figure
nu_records    = []   # per-DP full-channel wall data for Nu_norm figure
htc_records   = []   # per-DP full-channel wall data for combined HTC figure
nu_sec_rows   = []   # per-section Nu_norm stats for Table 5
eps_T_list    = []   # per-DP ε_θ  scalars for the table
eps_p_list    = []   # per-DP ε_p  scalars
eps_vx_list   = []   # per-DP ε_vx scalars
eps_nu_list   = []   # per-DP ε_Nu scalars

for dp_name in args.dps:
    cfg = DP_REGISTRY.get(dp_name)
    if cfg is None:
        print(f"\n[SKIP] {dp_name} — not in DP_REGISTRY")
        continue

    t_dp_start = time.time()
    t_inf      = 0.0   # accumulates pure inference time (no plots)
    print(f"\n{'─'*60}\n{dp_name}  AR={cfg['ar']}  Re={cfg['re']:.0f}")
    out_dp = os.path.join(args.out_dir, dp_name)
    os.makedirs(out_dp, exist_ok=True)

    data_dp = os.path.join(args.data_dir, dp_name)
    if not os.path.isdir(data_dp):
        print(f"  [SKIP] data not found: {data_dp}")
        continue

    # ── PINN params ────────────────────────────────────────
    params_raw = torch.tensor(
        [cfg["ar"], cfg["e_dh"], cfg["p_e"], cfg["alpha"], cfg["re"]],
        dtype=torch.float64)
    params_nd = (params_raw - _pm) / _ps

    # ── Load volume data ────────────────────────────────────
    temp_data = np.load(os.path.join(data_dp, "temp.npy"))   # (N, 4): x,y,z,T
    x_vol = temp_data[:, 0]
    y_vol = temp_data[:, 1]
    z_vol = temp_data[:, 2]
    T_cfd = temp_data[:, 3]
    del temp_data

    # Derive T_WALL and DELTA_T from CFD boundary values (min T = wall BC,
    # inlet face mean = inlet BC), exactly as done in load_dp during PINN training.
    # Using fixed constants (293.15 / 35.85 K) instead causes a systematic
    # T_nd → T_physical rescaling error: the actual per-DP DELTA_T in the training
    # data varies from ~33 to ~38 K, so the fixed value is wrong by up to ±8%.
    T_WALL  = float(T_cfd.min())
    _x_min  = float(x_vol.min())
    _face_mask = (x_vol == _x_min)
    if args.inlet_y_frac < 1.0:
        _face_mask = _face_mask & (y_vol <= args.inlet_y_frac * float(y_vol.max()))
    T_INLET = float(T_cfd[_face_mask].mean())
    DELTA_T = T_INLET - T_WALL
    print(f"  T_WALL={T_WALL:.2f} K  T_inlet={T_INLET:.2f} K  ΔT={DELTA_T:.2f} K"
          f"  (from CFD BCs, same as training)")

    # ── Load pressure and velocity CFD data ────────────────
    _in_vx = np.load(os.path.join(data_dp, "vel_x_inlet.npy"))
    if args.inlet_y_frac < 1.0:
        _in_vx = _in_vx[_in_vx[:, 1] <= args.inlet_y_frac * float(_in_vx[:, 1].max())]
    V_IN = float(np.abs(_in_vx[:, 3]).mean()); del _in_vx

    _p_raw  = np.load(os.path.join(data_dp, "press.npy"))   # (N, 4): x,y,z,p
    _out_m  = (_p_raw[:, 0] > float(_p_raw[:, 0].max()) - 0.005) & \
              (_p_raw[:, 1] >= 0.7 * float(_p_raw[:, 1].max()))
    P_REF   = float(_p_raw[_out_m, 3].mean())
    P_SCALE = float(max(float(_p_raw[:, 3].std()), 1.0))
    p_cfd   = (_p_raw[:, 3] - P_REF) / P_SCALE; del _p_raw, _out_m

    _vx_raw = np.load(os.path.join(data_dp, "vel_x.npy"))   # (N, 4): x,y,z,vx
    vx_cfd  = _vx_raw[:, 3] / V_IN; del _vx_raw

    print(f"  V_IN={V_IN:.3f} m/s  P_REF={P_REF:.0f} Pa  P_SCALE={P_SCALE:.1f} Pa")

    # ── Run PINN once on ALL volume pts ────────────────────
    pts_vol_np = np.stack([x_vol, y_vol, z_vol], axis=1)
    t0 = time.time()
    pred_all   = pinn_predict_all(pts_vol_np, params_nd)   # (N, 5): vx,vy,vz,p,T
    del pts_vol_np
    _dt = time.time() - t0; t_inf += _dt
    print(f"  PINN inference done  ({len(x_vol):,} pts)  [{_dt:.1f}s]")

    # ── θ_CFD and ε_θ ──────────────────────────────────────
    theta_cfd = (T_cfd - T_WALL) / DELTA_T
    theta_bar = float(np.mean(np.abs(theta_cfd)))
    eps_theta = float(np.mean(np.abs(pred_all[:, 4] - theta_cfd)) / (theta_bar if theta_bar > 0 else 1.0) * 100)
    eps_T_list.append(eps_theta)
    print(f"  ε_θ  = {eps_theta:.2f} %")

    p_bar  = float(np.mean(np.abs(p_cfd)))
    eps_p  = float(np.mean(np.abs(pred_all[:, 3] - p_cfd)) / (p_bar if p_bar > 0 else 1.0) * 100)
    eps_p_list.append(eps_p)
    print(f"  ε_p  = {eps_p:.2f} %")

    vx_bar = float(np.mean(np.abs(vx_cfd)))
    eps_vx = float(np.mean(np.abs(pred_all[:, 0] - vx_cfd)) / (vx_bar if vx_bar > 0 else 1.0) * 100)
    eps_vx_list.append(eps_vx)
    print(f"  ε_vx = {eps_vx:.2f} %")

    # ── z-midplane slice for field figures ─────────────────
    z_mid    = 0.5 * (float(z_vol.min()) + float(z_vol.max()))
    z_tol    = 0.05 * (float(z_vol.max()) - float(z_vol.min()))
    mid_mask = np.abs(z_vol - z_mid) < z_tol
    x_max_v  = float(x_vol.max())
    y_max_v  = float(y_vol.max())
    _xm, _ym = x_vol[mid_mask], y_vol[mid_mask]
    _tc, _tp = theta_cfd[mid_mask], pred_all[mid_mask, 4]
    _pc, _pp = p_cfd[mid_mask],     pred_all[mid_mask, 3]
    _vc, _vp = vx_cfd[mid_mask],    pred_all[mid_mask, 0]
    if _xm.size > 100_000:
        _idx = np.random.choice(_xm.size, 100_000, replace=False)
        _xm, _ym = _xm[_idx], _ym[_idx]
        _tc, _tp = _tc[_idx], _tp[_idx]
        _pc, _pp = _pc[_idx], _pp[_idx]
        _vc, _vp = _vc[_idx], _vp[_idx]
    dp_records.append({
        "name":        dp_name,
        "x_nd":        _xm / x_max_v,
        "y_nd":        _ym / y_max_v,
        "theta_cfd":   _tc,  "theta_pinn": _tp,  "theta_bar": theta_bar,
        "p_cfd":       _pc,  "p_pinn":     _pp,
        "vx_cfd":      _vc,  "vx_pinn":    _vp,
        "wall_y1_nd":  WALL_Y1 / y_max_v,
        "wall_y2_nd":  WALL_Y2 / y_max_v,
    })
    del _xm, _ym, _tc, _tp, _pc, _pp, _vc, _vp

    # ── T-planes from full-volume PINN result ───────────────
    plane_mask = np.zeros(len(x_vol), dtype=bool)
    for xp, ymin, ymax in T_PLANES:
        plane_mask |= ((np.abs(x_vol - xp) <= PLANE_TOL)
                       & (y_vol >= ymin) & (y_vol <= ymax))
    x_pl = x_vol[plane_mask]
    y_pl = y_vol[plane_mask]
    print(f"  {int(plane_mask.sum()):,} pts selected near T-planes")

    # CFD T_pl — for comparison bar chart only, not used in HTC prediction
    T_pl_cfd = compute_t_planes(x_pl, y_pl, T_cfd[plane_mask])
    print("  T_pl_cfd (K):", " | ".join(f"{PLANE_NAMES[i]}={T_pl_cfd[i]:.2f}" for i in range(6)))

    # PINN T_pl from full-volume inference
    T_pl_pinn_K = pred_all[plane_mask, 4] * DELTA_T + T_WALL
    T_pl = compute_t_planes(x_pl, y_pl, T_pl_pinn_K)
    del pred_all, theta_cfd, p_cfd, vx_cfd
    del x_pl, y_pl, T_pl_pinn_K, T_cfd, x_vol, y_vol, z_vol

    print("  T_pl_pinn(K):", " | ".join(f"{PLANE_NAMES[i]}={T_pl[i]:.2f}" for i in range(6)))

    if np.any(np.isnan(T_pl)):
        print("  [warn] some T_pl are NaN — check plane selections")

    # ── T_pl bar chart ─────────────────────────────────────
    plot_t_planes(dp_name, T_pl, T_pl_cfd,
                  os.path.join(out_dp, "T_planes.png"), t_wall=T_WALL)

    # ── q'' plot: CFD vs MLP at qpp.npy wall points ────────
    qpp_path = os.path.join(data_dp, "qpp.npy")
    mlp_params_nd = (params_raw - PARAM_MEANS) / PARAM_STDS
    if os.path.exists(qpp_path):
        qpp_data = np.load(qpp_path)            # (N, 4): x, y, z, q''
        x_qpp = qpp_data[:, 0]
        y_qpp = qpp_data[:, 1]
        q_cfd = qpp_data[:, 3]
        # apply inlet/outlet buffer trim (same as MLP training)
        x_min_q, x_max_q = float(x_qpp.min()), float(x_qpp.max())
        trim = (x_qpp >= x_min_q + BUFFER_M) & (x_qpp <= x_max_q - BUFFER_M)
        x_qpp, y_qpp, q_cfd = x_qpp[trim], y_qpp[trim], q_cfd[trim]
        t0 = time.time()
        q_pred_wall = mlp_predict_qpp(x_qpp, y_qpp, mlp_params_nd)
        _dt = time.time() - t0; t_inf += _dt
        print(f"  MLP q'' inference  ({len(x_qpp):,} pts)  [{_dt*1000:.0f}ms]")
        _dh_reg = DH_REGISTRY.get(dp_name, {})
        dh_avg = float(np.mean(list(_dh_reg.values()))) if _dh_reg else None
        _pts_t = torch.tensor(np.stack([x_qpp, y_qpp, np.zeros_like(x_qpp)], axis=1),
                              dtype=torch.float64)
        _true_t = torch.tensor(q_cfd, dtype=torch.float64)
        _pred_t = torch.tensor(q_pred_wall, dtype=torch.float64)
        plot_qpp_single(dp_name, _pts_t, _true_t, _pred_t,
                        os.path.join(out_dp, "qpp.png"),
                        dh=dh_avg, delta_T=DELTA_T)
        qpp_rmse = float(np.sqrt(np.mean((q_pred_wall - q_cfd) ** 2)))
        print(f"  q'' RMSE={qpp_rmse:.1f} W/m²  (n={len(q_cfd):,})")
        _q_star_scale = (dh_avg / (K_FLUID * DELTA_T)
                         if (dh_avg is not None and DELTA_T > 0) else None)
        qpp_records.append({
            "name":         dp_name,
            "x_w_nd":       x_qpp / x_max_v,
            "y_w_nd":       y_qpp / y_max_v,
            "qpp_cfd":      q_cfd,
            "qpp_pred":     q_pred_wall,
            "q_star_scale": _q_star_scale,
            "wall_y1_nd":   WALL_Y1 / y_max_v,
            "wall_y2_nd":   WALL_Y2 / y_max_v,
        })
    else:
        print(f"  [warn] qpp.npy not found — skipping q'' plot")

    # ── Load GT HTC ────────────────────────────────────────
    htc_gt_all = load_htc_gt(args.data_dir, dp_name)
    if not htc_gt_all:
        print(f"  [warn] no GT HTC files found in {data_dp}")
        eps_nu_list.append(float("nan"))
        continue

    # ── Nu_norm scaling factors ─────────────────────────────
    dh_map = DH_REGISTRY.get(dp_name, {})
    Nu0    = 0.023 * cfg["re"]**0.8 * PR_FLUID**0.3

    # ── Per-section HTC prediction + Nu_norm ───────────────
    print(f"\n  {'sec':<12} {'n':>6}  {'RMSE':>9}  {'R²':>8}")
    print(f"  {'─'*40}")
    _htc_pred_all, _htc_gt_all = [], []
    _x_htc_all,    _y_htc_all  = [], []
    _x_wall_all,   _y_wall_all = [], []
    _nu_cfd_all,   _nu_pinn_all = [], []

    for sec_idx, gt_arr in sorted(htc_gt_all.items()):
        x_gt     = gt_arr[:, 0]
        y_gt     = gt_arr[:, 1]
        htc_gt   = gt_arr[:, 3]
        t0 = time.time()
        q_pred   = mlp_predict_qpp(x_gt, y_gt, mlp_params_nd)
        t_inf   += time.time() - t0
        Tb       = compute_tb(x_gt, y_gt, T_pl)
        htc_pred = q_pred / (T_WALL - Tb)

        _htc_pred_all.append(htc_pred)
        _htc_gt_all.append(htc_gt)
        _x_htc_all.append(x_gt)
        _y_htc_all.append(y_gt)

        rmse   = float(np.sqrt(np.mean((htc_pred - htc_gt)**2)))
        ss_res = np.sum((htc_pred - htc_gt)**2)
        ss_tot = np.sum((htc_gt - htc_gt.mean())**2)
        r2     = 1.0 - ss_res / (ss_tot + 1e-30)
        print(f"  HTC{sec_idx} {SECTION_LABELS[sec_idx]:<8} "
              f"{len(x_gt):>6}  {rmse:>9.1f}  {r2:>8.4f}")

        fname = f"HTC{sec_idx}_{SECTION_LABELS[sec_idx].replace(' ', '_')}.png"
        plot_htc_section(dp_name, sec_idx,
                         x_gt, y_gt, htc_gt, htc_pred,
                         os.path.join(out_dp, fname),
                         rmse, r2)

        dh_sec = dh_map.get(sec_idx)
        if dh_sec is not None:
            scale        = dh_sec / (K_FLUID * Nu0)
            nu_c         = htc_gt   * scale
            nu_p         = htc_pred * scale
            nu_cfd_mean  = float(nu_c.mean())
            nu_pinn_mean = float(nu_p.mean())
            eps_nu_sec   = float(np.mean(np.abs(nu_p - nu_c)) / abs(nu_cfd_mean) * 100)
            _x_wall_all.append(x_gt);  _y_wall_all.append(y_gt)
            _nu_cfd_all.append(nu_c);  _nu_pinn_all.append(nu_p)

            # q'' stats (|q''| = htc * |T_wall - Tb|)
            qpp_cfd_mean  = float(np.mean(np.abs(htc_gt   * (Tb - T_WALL))))
            qpp_pred_mean = float(np.mean(np.abs(q_pred)))
            eps_qpp_sec   = float(abs(qpp_pred_mean - qpp_cfd_mean) / qpp_cfd_mean * 100)

            # T_pl at inlet/outlet of this section
            i_in, i_out  = SEC_TPLANE[sec_idx]
            T_in_cfd     = float(T_pl_cfd[i_in]);   T_in_pinn  = float(T_pl[i_in])
            T_out_cfd    = float(T_pl_cfd[i_out]);  T_out_pinn = float(T_pl[i_out])
            err_T_in     = abs(T_in_pinn  - T_in_cfd) / abs(T_in_cfd)  * 100
            err_T_out    = abs(T_out_pinn - T_out_cfd) / abs(T_out_cfd) * 100

            nu_sec_rows.append(
                (dp_name, SECTION_LABELS[sec_idx],
                 nu_cfd_mean, nu_pinn_mean, eps_nu_sec,
                 qpp_cfd_mean, qpp_pred_mean, eps_qpp_sec,
                 T_in_cfd, T_in_pinn, err_T_in,
                 T_out_cfd, T_out_pinn, err_T_out))
            print(f"         Nu_norm  CFD={nu_cfd_mean:.3f}  "
                  f"PINN={nu_pinn_mean:.3f}  ε={eps_nu_sec:.1f}%")

    if _htc_pred_all:
        _hp      = np.concatenate(_htc_pred_all)
        _hg      = np.concatenate(_htc_gt_all)
        _htc_bar = float(np.mean(np.abs(_hg)))
        eps_nu   = float(np.mean(np.abs(_hp - _hg)) / _htc_bar * 100)
    else:
        eps_nu = float("nan")
    eps_nu_list.append(eps_nu)
    print(f"  ε_Nu = {eps_nu:.2f} %")

    if _htc_pred_all:
        htc_records.append({
            "name":       dp_name,
            "x_w_nd":     np.concatenate(_x_htc_all) / x_max_v,
            "y_w_nd":     np.concatenate(_y_htc_all) / y_max_v,
            "htc_cfd":    np.concatenate(_htc_gt_all),
            "htc_pred":   np.concatenate(_htc_pred_all),
            "wall_y1_nd": WALL_Y1 / y_max_v,
            "wall_y2_nd": WALL_Y2 / y_max_v,
        })

    if _nu_cfd_all:
        _xw = np.concatenate(_x_wall_all)
        _yw = np.concatenate(_y_wall_all)
        _nc = np.concatenate(_nu_cfd_all)
        _np_arr = np.concatenate(_nu_pinn_all)
        nu_records.append({
            "name":       dp_name,
            "x_w_nd":     _xw / x_max_v,
            "y_w_nd":     _yw / y_max_v,
            "nu_cfd":     _nc,
            "nu_pinn":    _np_arr,
            "wall_y1_nd": WALL_Y1 / y_max_v,
            "wall_y2_nd": WALL_Y2 / y_max_v,
        })

    print(f"\n  Saved → {out_dp}/  [inference: {t_inf:.1f}s  total: {time.time()-t_dp_start:.1f}s]")

# ── Field reconstruction figures + ε table ─────────────────
if dp_records:
    _names = [r["name"] for r in dp_records]
    plot_field(dp_records, "theta_pinn", "theta_cfd",
               (r"CFD $\theta = (T-T_w)\,/\,(T_\mathrm{in}-T_w)$", r"PINN $\theta$"),
               os.path.join(args.out_dir, "T_field.png"),
               err_title=(r"$|\theta_\mathrm{PINN}-\theta_\mathrm{CFD}|\,"
                          r"/\,\overline{|\theta_\mathrm{CFD}|}\times100\,(\%)$"))
    plot_field(dp_records, "p_pinn", "p_cfd",
               (r"CFD $p_\mathrm{nd} = (p-p_\mathrm{ref})\,/\,\sigma_p$",
                r"PINN $p_\mathrm{nd}$"),
               os.path.join(args.out_dir, "p_field.png"),
               err_title=(r"$|p^\mathrm{nd}_\mathrm{PINN}-p^\mathrm{nd}_\mathrm{CFD}|\,"
                          r"/\,\overline{|p^\mathrm{nd}_\mathrm{CFD}|}\times100\,(\%)$"))
    plot_field(dp_records, "vx_pinn", "vx_cfd",
               (r"CFD $v_{x,\mathrm{nd}} = v_x\,/\,V_\mathrm{in}$",
                r"PINN $v_{x,\mathrm{nd}}$"),
               os.path.join(args.out_dir, "vx_field.png"),
               err_title=(r"$|v^\mathrm{nd}_{x,\mathrm{PINN}}-v^\mathrm{nd}_{x,\mathrm{CFD}}|\,"
                          r"/\,\overline{|v^\mathrm{nd}_{x,\mathrm{CFD}}|}\times100\,(\%)$"))
    save_epsilon_table(
        _names,
        [
            ("Temperature", eps_T_list),
            ("Pressure",    eps_p_list),
            ("Vx",          eps_vx_list),
            ("Nu_norm",     eps_nu_list),
        ],
        os.path.join(args.out_dir, "epsilon_table.csv"))

if qpp_records:
    plot_qpp_full(qpp_records,
                  os.path.join(args.out_dir, "qpp_field.png"))

if htc_records:
    plot_htc_full(htc_records,
                  os.path.join(args.out_dir, "HTC_field.png"))

if nu_records:
    plot_nu_field(nu_records,
                  os.path.join(args.out_dir, "Nu_field.png"))

if nu_sec_rows:
    _dp_names  = list(dict.fromkeys(r[0] for r in nu_sec_rows))
    _sec_order = {"Pass 1": 0, "Pass 2": 1, "Pass 3": 2, "Top Bend": 3, "Bottom Bend": 4}
    nu_sec_rows.sort(key=lambda r: (_dp_names.index(r[0]), _sec_order.get(r[1], 9)))
    save_nu_section_table(nu_sec_rows, _dp_names,
                          os.path.join(args.out_dir, "Nu_section_table.csv"))

_log_handle.close()
print(f"\n{'═'*60}\nDone. Results in {args.out_dir}  [total: {time.time()-t_total_start:.1f}s]")
