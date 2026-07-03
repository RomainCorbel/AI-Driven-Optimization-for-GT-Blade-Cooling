"""
utils.py — shared utilities and plot/table functions for the PINN/QPP-MLP pipeline.
"""

import csv
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    WALL_Y1, WALL_Y2, WALL_EPS,
    K_FLUID,
    PLANE_NAMES, SECTION_LABELS,
)

# ═══════════════════════════════════════════════════════════════
# TEE — stdout+file duplicator
# ═══════════════════════════════════════════════════════════════

class _Tee(io.TextIOBase):
    def __init__(self, stream, logfile):
        self._stream = stream; self._logfile = logfile
    def write(self, s):
        self._stream.write(s); self._logfile.write(s); return len(s)
    def flush(self):
        self._stream.flush(); self._logfile.flush()


# ═══════════════════════════════════════════════════════════════
# MLP INPUT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_mlp_input(pts, params_nd,
                    wall_y1=WALL_Y1, wall_y2=WALL_Y2, wall_eps=WALL_EPS):
    """Build (x, y, 5 params, s1, s2) input tensor for the QPP-MLP.

    z is dropped: the bottom wall surface has z≈0 everywhere, so z carries
    no information for q'' prediction.

    Args:
        pts:       float64 tensor (N, 3) — wall coordinates [m]
        params_nd: float64 tensor (5,)   — z-scored design parameters
    Returns:
        float64 tensor (N, 9)
    """
    N  = pts.shape[0]
    xy = pts[:, :2]
    y  = pts[:, 1:2]
    s1 = torch.sigmoid((y - wall_y1) / wall_eps)
    s2 = torch.sigmoid((y - wall_y2) / wall_eps)
    return torch.cat([xy, params_nd.unsqueeze(0).expand(N, -1), s1, s2], dim=1)


# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════

def _to_np(x):
    if x is None:
        return None
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ═══════════════════════════════════════════════════════════════
# plot_qpp_single — 4-panel single-DP q'' / q* plot
# ═══════════════════════════════════════════════════════════════

def plot_qpp_single(folder, pts, true_qpp, pred_qpp, out_path, dh=None, delta_T=None):
    """4-panel q'' comparison for one DP.

    When dh and delta_T are both provided the panels show dimensionless
    q* = |q''|·Dh / (k_f·|ΔT|) (paper eq. 25).  Otherwise raw W/m² are shown.

    Panels: parity scatter | CFD spatial | pred spatial | error spatial
    """
    true_np = _to_np(true_qpp)
    pred_np = _to_np(pred_qpp)
    pts_np  = _to_np(pts)
    rmse    = float(np.sqrt(np.mean((pred_np - true_np) ** 2)))

    use_qstar = (dh is not None) and (delta_T is not None) and (abs(delta_T) > 0)
    if use_qstar:
        scale     = dh / (K_FLUID * abs(delta_T))
        disp_true = np.abs(true_np) * scale
        disp_pred = np.abs(pred_np) * scale
        q_label   = r"$q^* = |q''_w|\,D_h\,/\,(k_f\,|T_\mathrm{in}-T_w|)$"
        # Error metric: ε_q (%)
        q_bar     = float(np.mean(np.abs(true_np)))
        eps_q_map = np.abs(pred_np - true_np) / (q_bar if q_bar > 0 else 1.0) * 100
        eps_q     = float(np.mean(eps_q_map))
        err_vmax  = float(np.percentile(eps_q_map, 99))
        err_label = r"$\varepsilon_q$ (%)"
        suptitle  = f"{folder}  RMSE={rmse:.1f} W/m²  ε_q={eps_q:.2f}%"
    else:
        disp_true = true_np
        disp_pred = pred_np
        q_label   = "q'' [W/m²]"
        err_np    = np.abs(pred_np - true_np)
        err_vmax  = float(np.percentile(err_np, 99))
        nrmse     = float(rmse / (true_np.std() + 1e-8) * 100)
        eps_q_map = err_np
        err_label = "|error| [W/m²]"
        suptitle  = f"{folder}  RMSE={rmse:.1f} W/m²  NRMSE={nrmse:.1f}%"

    vmin = float(np.percentile(disp_true, 1))
    vmax = float(np.percentile(disp_true, 99))

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(suptitle)

    ax = axes[0]
    ax.scatter(disp_true, disp_pred, s=0.5, alpha=0.3, rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1)
    ax.set_xlabel(f"CFD {q_label}"); ax.set_ylabel(f"Pred {q_label}")
    ax.set_title("Parity")

    for ax, vals, title in [(axes[1], disp_true, "CFD"), (axes[2], disp_pred, "MLP")]:
        sc = ax.scatter(pts_np[:, 0], pts_np[:, 1], c=vals,
                        cmap="hot" if not use_qstar else "plasma",
                        s=1, rasterized=True, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=q_label)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)

    ax = axes[3]
    sc = ax.scatter(pts_np[:, 0], pts_np[:, 1], c=eps_q_map,
                    cmap="OrRd", s=1, rasterized=True, vmin=0, vmax=err_vmax)
    plt.colorbar(sc, ax=ax, label=err_label)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Error map")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# plot_fields — 3D scatter + 2D midplane cut per field
# ═══════════════════════════════════════════════════════════════

def plot_fields(pts, fields, output_dir, slice_frac=0.10):
    """3D scatter + z-midplane 2D cut for each field in *fields*.

    Args:
        pts:        (N, 3) array or tensor
        fields:     list of (name, data_or_None, pred) tuples — numpy or tensor
        output_dir: directory where PNGs are written
        slice_frac: fraction of z-range used for midplane slice thickness
    """
    os.makedirs(output_dir, exist_ok=True)

    pts_np = _to_np(pts)
    ranges = pts_np.max(axis=0) - pts_np.min(axis=0)
    box_aspect = (ranges / ranges.max()).tolist()
    z_vals = pts_np[:, 2]
    z_mid  = 0.5 * (z_vals.min() + z_vals.max())
    z_tol  = slice_frac * (z_vals.max() - z_vals.min())
    cut_mask  = np.abs(z_vals - z_mid) < z_tol
    p_cut     = pts_np[cut_mask]
    cut_label = f"x-y  z={z_mid*1000:.1f}mm  (n={cut_mask.sum()})"

    for name, data_raw, pred_raw in fields:
        data, pred = _to_np(data_raw), _to_np(pred_raw)
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
            ax.set_box_aspect(box_aspect); ax.set_title(title, fontsize=13)
            ax.set_xlabel("X"); ax.set_yticklabels([]); ax.set_zticklabels([])
            plt.colorbar(sc, ax=ax, shrink=0.5, label=name)
        for i, (title, color, cmin, cmax) in enumerate(subplots):
            ax  = fig.add_subplot(2, n_sub, n_sub + i + 1)
            sc  = ax.scatter(p_cut[:,0], p_cut[:,1], c=color[cut_mask],
                             cmap="viridis", vmin=cmin, vmax=cmax, s=4, rasterized=True)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
            ax.set_title(f"{title}\n{cut_label}", fontsize=13)
            plt.colorbar(sc, ax=ax, label=name, shrink=0.5, aspect=15)
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# plot_field — N_dp × 3 field figure (CFD | PINN | error %)
# ═══════════════════════════════════════════════════════════════

_FIELD_LABELS = {
    "theta_cfd": r"$\theta = (T - T_w)\,/\,(T_\mathrm{in} - T_w)$",
    "p_cfd":     r"$p_\mathrm{nd} = (p - p_\mathrm{ref})\,/\,\sigma_p$",
    "vx_cfd":    r"$v_{x,\mathrm{nd}} = v_x\,/\,V_\mathrm{in}$",
}


def plot_field(dp_records, pinn_key, cfd_key, col_titles, out_path, err_title=None):
    """N_dp × 3 figure: CFD field | PINN field | normalised |error| (%).
    Axes normalised: x/L_x, y/L_y ∈ [0, 1].  White dashed lines at pass walls."""
    N   = len(dp_records)
    fig, axes = plt.subplots(N, 3, figsize=(15, 2.5 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    all_err = []
    for r in dp_records:
        ref = float(np.mean(np.abs(r[cfd_key])))
        if ref > 0:
            all_err.append(np.abs(r[pinn_key] - r[cfd_key]) / ref * 100)
    err_vmax = float(np.percentile(np.concatenate(all_err), 99)) if all_err else 20.0

    fld_label = _FIELD_LABELS.get(cfd_key, cfd_key)

    for row, rec in enumerate(dp_records):
        cfd_v  = rec[cfd_key]
        pinn_v = rec[pinn_key]
        ref    = float(np.mean(np.abs(cfd_v)))
        err    = np.abs(pinn_v - cfd_v) / (ref if ref > 0 else 1.0) * 100
        x_nd, y_nd = rec["x_nd"], rec["y_nd"]
        ms = 4 if rec["name"] == "dp103" else 1

        vlo = float(np.percentile(cfd_v, 1))
        vhi = float(np.percentile(cfd_v, 99))
        mean_err = float(np.mean(err))
        for col, (vals, cmap, vmin_, vmax_) in enumerate([
            (cfd_v,  "plasma", vlo,  vhi),
            (pinn_v, "plasma", vlo,  vhi),
            (err,    "plasma", 0.0,  err_vmax),
        ]):
            ax = axes[row, col]
            ax.scatter(x_nd, y_nd, c=vals, cmap=cmap,
                       vmin=vmin_, vmax=vmax_, s=ms, rasterized=True)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if col == 2:
                ax.set_xlabel(r"$x/L_x$" + f"\n$\\bar{{\\varepsilon}} = {mean_err:.1f}\\%$",
                              fontsize=13)
            else:
                ax.set_xlabel(r"$x/L_x$", fontsize=13)
            if col == 0:
                ax.set_ylabel(r"$y/L_y$", fontsize=13)
                ax.text(-0.25, 0.5, rec["name"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row == 0:
                _err_t = (err_title if err_title is not None else
                          r"$|\phi_\mathrm{PINN}-\phi_\mathrm{CFD}|\,"
                          r"/\,\overline{|\phi_\mathrm{CFD}|}\times100\,(\%)$")
                ax.set_title([col_titles[0], col_titles[1], _err_t][col], fontsize=13)

        sm_fld = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vlo, vmax=vhi), cmap="plasma")
        fig.colorbar(sm_fld, ax=axes[row, 1], shrink=0.85, label=fld_label)
        sm_err = plt.cm.ScalarMappable(
            norm=plt.Normalize(0, err_vmax), cmap="plasma")
        fig.colorbar(sm_err, ax=axes[row, 2], shrink=0.85, label="Error (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════
# plot_htc_section
# ═══════════════════════════════════════════════════════════════

def plot_htc_section(dp_name, sec_idx, x, y, htc_gt, htc_pred, out_path, rmse, r2):
    label = SECTION_LABELS[sec_idx]
    diff  = htc_pred - htc_gt

    vmin  = float(np.nanpercentile(htc_gt, 1))
    vmax  = float(np.nanpercentile(htc_gt, 99))
    dlim  = float(np.nanpercentile(np.abs(diff), 99))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dp_name}  —  HTC {label}   RMSE={rmse:.1f} W/m²K   R²={r2:.4f}",
                 fontsize=13)

    for ax, vals, title, cmap, vlo, vhi in [
        (axes[0], htc_gt,   "CFD (data)",       "plasma", vmin,  vmax),
        (axes[1], htc_pred, "Pred (PINN+MLP)",  "plasma", vmin,  vmax),
        (axes[2], diff,     "Diff (pred−data)", "plasma", 0,     dlim),
    ]:
        sc = ax.scatter(x, y, c=vals, cmap=cmap, vmin=vlo, vmax=vhi,
                        s=2, rasterized=True)
        plt.colorbar(sc, ax=ax, label="HTC [W/m²K]" if "Diff" not in title
                     else "ΔHTC [W/m²K]", shrink=0.85)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(title); ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# plot_t_planes
# ═══════════════════════════════════════════════════════════════

def plot_t_planes(dp_name, T_pl, T_pl_cfd, out_path, t_wall=293.15):
    """Side-by-side bar chart: PINN predicted vs CFD mean at each of the 6 planes."""
    x     = np.arange(6)
    w     = 0.35
    delta     = np.where(np.isfinite(T_pl),     T_pl     - t_wall, np.nan)
    delta_cfd = np.where(np.isfinite(T_pl_cfd), T_pl_cfd - t_wall, np.nan)

    d_plot    = np.nan_to_num(delta,     nan=0.0)
    dcfd_plot = np.nan_to_num(delta_cfd, nan=0.0)

    fig, ax = plt.subplots(figsize=(11, 4))
    b1 = ax.bar(x - w/2, dcfd_plot,  w, label="CFD (data)",      color="#4C72B0", edgecolor="k", linewidth=0.7)
    b2 = ax.bar(x + w/2, d_plot,     w, label="PINN prediction", color="#DD8452", edgecolor="k", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(PLANE_NAMES, rotation=20, ha="right", fontsize=13)
    ax.set_ylabel("T – T_wall  [K]")
    ax.set_title(f"{dp_name}  —  Cross-section bulk temperatures: CFD vs PINN")
    ax.legend(fontsize=13)

    for bar, val in zip(b1, T_pl_cfd):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#4C72B0")
    for bar, val in zip(b2, T_pl):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#DD8452")

    all_vals = np.concatenate([d_plot[np.isfinite(delta)], dcfd_plot[np.isfinite(delta_cfd)]])
    if len(all_vals) > 0:
        ax.set_ylim(max(0, all_vals.min() - 2), all_vals.max() + 5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# plot_nu_field
# ═══════════════════════════════════════════════════════════════

def plot_nu_field(nu_records, out_path):
    """N_dp × 3 figure: CFD Nu_norm | PINN Nu_norm | |error| (%)  — full channel."""
    N   = len(nu_records)
    fig, axes = plt.subplots(N, 3, figsize=(15, 2.5 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    all_err = []
    for r in nu_records:
        ref = float(np.mean(np.abs(r["nu_cfd"])))
        if ref > 0:
            all_err.append(np.abs(r["nu_pinn"] - r["nu_cfd"]) / ref * 100)
    err_vmax = float(np.percentile(np.concatenate(all_err), 99)) if all_err else 20.0

    for row, rec in enumerate(nu_records):
        nu_c = rec["nu_cfd"];  nu_p = rec["nu_pinn"]
        x_nd = rec["x_w_nd"]; y_nd = rec["y_w_nd"]
        ms   = 4 if rec["name"] == "dp103" else 1
        ref  = float(np.mean(np.abs(nu_c)))
        err  = np.abs(nu_p - nu_c) / (ref if ref > 0 else 1.0) * 100
        vlo  = float(np.percentile(nu_c, 1))
        vhi  = float(np.percentile(nu_c, 99))

        mean_err = float(np.mean(err))
        for col, (vals, cmap, vmin_, vmax_) in enumerate([
            (nu_c, "plasma", vlo,  vhi),
            (nu_p, "plasma", vlo,  vhi),
            (err,  "plasma", 0.0,  err_vmax),
        ]):
            ax = axes[row, col]
            ax.scatter(x_nd, y_nd, c=vals, cmap=cmap,
                       vmin=vmin_, vmax=vmax_, s=ms, rasterized=True)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if col == 2:
                ax.set_xlabel(r"$x/L_x$" + f"\n$\\bar{{\\varepsilon}} = {mean_err:.1f}\\%$",
                              fontsize=13)
            else:
                ax.set_xlabel(r"$x/L_x$", fontsize=13)
            if col == 0:
                ax.set_ylabel(r"$y/L_y$", fontsize=13)
                ax.text(-0.25, 0.5, rec["name"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(
                    [r"CFD $Nu_\mathrm{norm} = \mathrm{HTC}\cdot D_h\,/\,(k_f\,Nu_0)$",
                     r"PINN $Nu_\mathrm{norm}$",
                     r"$|Nu_\mathrm{PINN}-Nu_\mathrm{CFD}|\,"
                     r"/\,\overline{|Nu_\mathrm{CFD}|}\times100\,(\%)$"][col],
                    fontsize=13)

        sm_nu = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vlo, vmax=vhi), cmap="plasma")
        fig.colorbar(sm_nu, ax=axes[row, 1], shrink=0.85,
                     label=r"$Nu_\mathrm{norm} = \mathrm{HTC}\cdot D_h\,/\,(k_f\,Nu_0)$")
        sm_err = plt.cm.ScalarMappable(
            norm=plt.Normalize(0, err_vmax), cmap="plasma")
        fig.colorbar(sm_err, ax=axes[row, 2], shrink=0.85, label="Error (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════
# plot_qpp_full
# ═══════════════════════════════════════════════════════════════

def plot_qpp_full(qpp_records, out_path):
    """N_dp × 3 figure: CFD q'' | MLP q'' | ε_q (%).
    Error = |pred − cfd| / mean(|cfd|) × 100, consistent with all other field figures."""
    N   = len(qpp_records)
    fig, axes = plt.subplots(N, 3, figsize=(15, 2.5 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    all_err = []
    for r in qpp_records:
        ref = float(np.mean(np.abs(r["qpp_cfd"])))
        if ref > 0:
            all_err.append(np.abs(r["qpp_pred"] - r["qpp_cfd"]) / ref * 100)
    err_vmax = float(np.percentile(np.concatenate(all_err), 99)) if all_err else 20.0

    for row, rec in enumerate(qpp_records):
        q_c  = rec["qpp_cfd"];  q_p = rec["qpp_pred"]
        x_nd = rec["x_w_nd"];   y_nd = rec["y_w_nd"]

        sc     = rec["q_star_scale"]
        disp_c = np.abs(q_c) * sc
        disp_p = np.abs(q_p) * sc
        q_label    = r"$q^* = |q''_w|\,D_h\,/\,(k_f\,|T_\mathrm{in}-T_w|)$"
        col0_title = r"CFD $q^* = |q''_w|\,D_h\,/\,(k_f\,|T_\mathrm{in}-T_w|)$"
        col1_title = r"MLP $q^*$"

        ms   = 7 if rec["name"] == "dp103" else 1
        ref  = float(np.mean(np.abs(q_c)))
        err  = np.abs(q_p - q_c) / (ref if ref > 0 else 1.0) * 100
        vlo  = float(np.percentile(disp_c, 1))
        vhi  = float(np.percentile(disp_c, 99))

        mean_err = float(np.mean(err))
        for col, (vals, cmap, vmin_, vmax_) in enumerate([
            (disp_c, "plasma", vlo,  vhi),
            (disp_p, "plasma", vlo,  vhi),
            (err,    "plasma", 0.0,  err_vmax),
        ]):
            ax = axes[row, col]
            ax.scatter(x_nd, y_nd, c=vals, cmap=cmap,
                       vmin=vmin_, vmax=vmax_, s=ms, rasterized=True)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if col == 2:
                ax.set_xlabel(r"$x/L_x$" + f"\n$\\bar{{\\varepsilon}} = {mean_err:.1f}\\%$",
                              fontsize=13)
            else:
                ax.set_xlabel(r"$x/L_x$", fontsize=13)
            if col == 0:
                ax.set_ylabel(r"$y/L_y$", fontsize=13)
                ax.text(-0.25, 0.5, rec["name"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(
                    [col0_title, col1_title,
                     r"$|q^*_\mathrm{MLP}-q^*_\mathrm{CFD}|\,"
                     r"/\,\overline{q^*_\mathrm{CFD}}\times100\,(\%)$"][col],
                    fontsize=13)

        sm_q = plt.cm.ScalarMappable(
            norm=plt.Normalize(vlo, vhi), cmap="plasma")
        fig.colorbar(sm_q, ax=axes[row, 1], shrink=0.85, label=q_label)
        sm_err = plt.cm.ScalarMappable(
            norm=plt.Normalize(0, err_vmax), cmap="plasma")
        fig.colorbar(sm_err, ax=axes[row, 2], shrink=0.85, label="Error (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════
# plot_htc_full
# ═══════════════════════════════════════════════════════════════

def plot_htc_full(htc_records, out_path):
    """N_dp × 3 figure: CFD HTC | Predicted HTC | |error| (%).
    Error = |pred − cfd| / mean(|cfd|) × 100.
    All sections combined; axes normalised x/L_x, y/L_y."""
    N   = len(htc_records)
    fig, axes = plt.subplots(N, 3, figsize=(15, 2.5 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    all_err = []
    for r in htc_records:
        ref = float(np.mean(np.abs(r["htc_cfd"])))
        if ref > 0:
            all_err.append(np.abs(r["htc_pred"] - r["htc_cfd"]) / ref * 100)
    err_vmax = float(np.percentile(np.concatenate(all_err), 99)) if all_err else 20.0

    for row, rec in enumerate(htc_records):
        htc_c = rec["htc_cfd"];  htc_p = rec["htc_pred"]
        x_nd  = rec["x_w_nd"];   y_nd  = rec["y_w_nd"]
        ms    = 4 if rec["name"] == "dp103" else 1
        ref   = float(np.mean(np.abs(htc_c)))
        err   = np.abs(htc_p - htc_c) / (ref if ref > 0 else 1.0) * 100
        vlo   = float(np.percentile(htc_c, 1))
        vhi   = float(np.percentile(htc_c, 99))

        mean_err = float(np.mean(err))
        for col, (vals, cmap, vmin_, vmax_) in enumerate([
            (htc_c, "plasma", vlo,  vhi),
            (htc_p, "plasma", vlo,  vhi),
            (err,   "plasma", 0.0,  err_vmax),
        ]):
            ax = axes[row, col]
            ax.scatter(x_nd, y_nd, c=vals, cmap=cmap,
                       vmin=vmin_, vmax=vmax_, s=ms, rasterized=True)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if col == 2:
                ax.set_xlabel(r"$x/L_x$" + f"\n$\\bar{{\\varepsilon}} = {mean_err:.1f}\\%$",
                              fontsize=13)
            else:
                ax.set_xlabel(r"$x/L_x$", fontsize=13)
            if col == 0:
                ax.set_ylabel(r"$y/L_y$", fontsize=13)
                ax.text(-0.25, 0.5, rec["name"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(
                    [r"CFD $\mathrm{HTC} = q''_w\,/\,(T_w-T_b)$  [W/m²·K]",
                     r"Predicted HTC  [W/m²·K]",
                     r"$|\mathrm{HTC}_\mathrm{pred}-\mathrm{HTC}_\mathrm{CFD}|\,"
                     r"/\,\overline{|\mathrm{HTC}_\mathrm{CFD}|}\times100\,(\%)$"][col],
                    fontsize=13)

        sm_htc = plt.cm.ScalarMappable(
            norm=plt.Normalize(vlo, vhi), cmap="plasma")
        fig.colorbar(sm_htc, ax=axes[row, 1], shrink=0.85,
                     label=r"$\mathrm{HTC} = q''_w\,/\,(T_w - T_b)$  [W/m²·K]")
        sm_err = plt.cm.ScalarMappable(
            norm=plt.Normalize(0, err_vmax), cmap="plasma")
        fig.colorbar(sm_err, ax=axes[row, 2], shrink=0.85, label="Error (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════
# save_epsilon_table
# ═══════════════════════════════════════════════════════════════

def save_epsilon_table(dp_names, rows, out_path):
    """Print multi-row ε table to stdout and save as CSV.
    rows: list of (label, [eps_per_dp]) pairs — one per quantity.
    """
    col_w  = 9
    lbl_w  = max(14, max(len(lbl) for lbl, _ in rows) + 2)
    header = f"{'Quantity':<{lbl_w}}" + "".join(f"{n:>{col_w}}" for n in dp_names)
    sep    = "─" * len(header)
    lines  = [f"\n{sep}", header, sep]
    for lbl, vals in rows:
        lines.append(f"{lbl:<{lbl_w}}" + "".join(
            f"{'—':>{col_w}}" if np.isnan(v) else f"{v:>{col_w}.2f}"
            for v in vals))
    lines.append(sep + "\n")
    print("\n".join(lines))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Quantity"] + list(dp_names))
        for lbl, vals in rows:
            w.writerow([lbl] + ["—" if np.isnan(v) else f"{v:.2f}" for v in vals])
    print(f"  ε table → {out_path}")


# ═══════════════════════════════════════════════════════════════
# save_nu_section_table
# ═══════════════════════════════════════════════════════════════

def save_nu_section_table(section_rows, dp_names, out_path):
    """Print and save section-wise Nu_norm table with q'' and T_pl columns."""
    col_w = 10
    lbl_w = 14
    _h = ["CFD Nu", "PINN Nu", "e_Nu (%)",
          "CFD q''", "PINN q''", "e_q'' (%)",
          "T_in CFD", "T_in PINN", "e_T_in (%)",
          "T_out CFD", "T_out PINN", "e_T_out (%)"]
    header = (f"{'DP':<10}{'Section':<{lbl_w}}"
              + "".join(f"{h:>{col_w}}" for h in _h))
    sep = "─" * len(header)
    prev_dp = None
    lines = [f"\n{sep}", header, sep]
    for row in section_rows:
        dp, sec = row[0], row[1]
        nu_cfd, nu_pinn, eps_nu = row[2], row[3], row[4]
        q_cfd, q_pinn, eps_q   = row[5], row[6], row[7]
        Ti_c, Ti_p, dTi        = row[8], row[9], row[10]
        To_c, To_p, dTo        = row[11], row[12], row[13]
        if dp != prev_dp and prev_dp is not None:
            lines.append(sep)
        lines.append(
            f"{dp:<10}{sec:<{lbl_w}}"
            f"{nu_cfd:>{col_w}.3f}{nu_pinn:>{col_w}.3f}{eps_nu:>{col_w}.2f}"
            f"{q_cfd:>{col_w}.1f}{q_pinn:>{col_w}.1f}{eps_q:>{col_w}.2f}"
            f"{Ti_c:>{col_w}.2f}{Ti_p:>{col_w}.2f}{dTi:>{col_w}.2f}"
            f"{To_c:>{col_w}.2f}{To_p:>{col_w}.2f}{dTo:>{col_w}.2f}")
        prev_dp = dp
    lines.append(sep + "\n")
    print("\n".join(lines))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DP", "Section",
                    "CFD Nu_norm", "PINN Nu_norm", "eps_Nu (%)",
                    "CFD q'' (W/m2)", "PINN q'' (W/m2)", "eps_q'' (%)",
                    "T_in CFD (K)", "T_in PINN (K)", "e_T_in (%)",
                    "T_out CFD (K)", "T_out PINN (K)", "e_T_out (%)"])
        for row in section_rows:
            w.writerow([row[0], row[1],
                        f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.2f}",
                        f"{row[5]:.1f}", f"{row[6]:.1f}", f"{row[7]:.2f}",
                        f"{row[8]:.2f}", f"{row[9]:.2f}", f"{row[10]:.2f}",
                        f"{row[11]:.2f}", f"{row[12]:.2f}", f"{row[13]:.2f}"])
    print(f"  Nu_norm section table → {out_path}")
