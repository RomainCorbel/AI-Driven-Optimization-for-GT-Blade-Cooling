import argparse
import glob
import trimesh
import numpy as np
import random
import torch
import torch.nn as nn
import os
import sys
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import wandb
from torch.optim import Adam

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

# Any DP with DELTA_T below this is rejected — pinn25 is non-isothermal only.
ISOTHERMAL_DELTA_T_THRESHOLD = 5.0   # [K]

# ═══════════════════════════════════════════════════════════════
# PARAMETRIC INPUT NORMALIZATION  (anchors from the full 51-DP space)
# ═══════════════════════════════════════════════════════════════

PARAM_NAMES = ["AR",   "e/Dh",  "P/e",  "alpha",   "Re"    ]
PARAM_MEANS = torch.tensor([ 9.0,   0.127,  10.0,   52.0,  108000.0], dtype=torch.float64)
PARAM_STDS  = torch.tensor([ 3.5,   0.048,   3.1,   14.5,   58000.0], dtype=torch.float64)

# Internal-wall sigmoid features — same y positions for ALL 50 DPs.
WALL_Y1  = 0.185   # [m]
WALL_Y2  = 0.375   # [m]
WALL_EPS = 0.002    # [m] sigmoid width

# ═══════════════════════════════════════════════════════════════
# DESIGN-POINT REGISTRY  (DP0–DP50, all inline)
# ═══════════════════════════════════════════════════════════════
# outlet_y_min_frac / inlet_y_max_frac: None = use CLI defaults.
# Set explicitly for geometries where you know the internal wall position.

DP_CONFIGS = [
    # DP0  (reference case)
    {"folder": "dp00", "ar":  7.5,      "e_dh": 0.074,    "p_e":  8.0,      "alpha": 60.0,     "re": 100000.0,
     "outlet_y_min_frac": 0.70, "inlet_y_max_frac": 0.33},
    # DP1–DP50  (LHS design space — outlet/inlet fracs TBD once data arrives)
    {"folder": "dp01", "ar":  3.644728, "e_dh": 0.049743, "p_e":  4.7724,   "alpha": 25.41772, "re":  16278.05, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp02", "ar":  3.890253, "e_dh": 0.046589, "p_e":  4.984645, "alpha": 28.74614, "re":  19457.92, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp03", "ar":  9.002429, "e_dh": 0.18974,  "p_e":  6.423034, "alpha": 74.29562, "re":  33203.83, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp04", "ar": 14.2659,   "e_dh": 0.094117, "p_e": 11.94046,  "alpha": 54.08848, "re": 171579.3,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp05", "ar":  5.67934,  "e_dh": 0.151972, "p_e": 12.38583,  "alpha": 73.44649, "re": 164877.9,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp06", "ar":  6.027419, "e_dh": 0.18568,  "p_e":  8.921497, "alpha": 59.02505, "re": 121107.7,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp07", "ar": 10.98555,  "e_dh": 0.101665, "p_e":  8.443827, "alpha": 41.87127, "re":  24717.01, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp08", "ar": 12.29549,  "e_dh": 0.078886, "p_e":  5.449208, "alpha": 35.0332,  "re": 196382.7,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp09", "ar":  8.515138, "e_dh": 0.069877, "p_e":  8.861573, "alpha": 36.31164, "re":  76870.97, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp10", "ar": 14.40566,  "e_dh": 0.15972,  "p_e":  8.679227, "alpha": 65.17492, "re": 189773.0,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp11", "ar":  4.378955, "e_dh": 0.140784, "p_e":  8.114662, "alpha": 37.30411, "re": 112306.8,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp12", "ar":  7.607081, "e_dh": 0.122157, "p_e": 14.1121,   "alpha": 40.9318,  "re":  35772.6,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp13", "ar": 10.89665,  "e_dh": 0.164771, "p_e": 11.72562,  "alpha": 46.07647, "re":  48748.35, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp14", "ar":  4.668702, "e_dh": 0.083476, "p_e":  9.196702, "alpha": 44.45944, "re":  95242.31, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp15", "ar":  5.398892, "e_dh": 0.097308, "p_e": 11.52129,  "alpha": 51.20543, "re": 124027.6,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp16", "ar":  6.640551, "e_dh": 0.156298, "p_e": 11.27025,  "alpha": 34.82768, "re": 101590.6,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp17", "ar": 12.90792,  "e_dh": 0.125448, "p_e": 14.94032,  "alpha": 38.43865, "re":  21978.75, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp18", "ar": 13.75498,  "e_dh": 0.121149, "p_e": 12.54017,  "alpha": 61.05144, "re": 153635.5,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp19", "ar":  4.079941, "e_dh": 0.086112, "p_e": 12.01869,  "alpha": 65.94071, "re":  87782.58, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp20", "ar":  7.056191, "e_dh": 0.148406, "p_e": 13.78874,  "alpha": 45.18996, "re": 143472.5,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp21", "ar":  9.87082,  "e_dh": 0.168413, "p_e":  5.105393, "alpha": 43.60288, "re":  61098.39, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp22", "ar": 14.77513,  "e_dh": 0.061514, "p_e": 10.96145,  "alpha": 69.06866, "re": 140693.4,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp23", "ar": 14.62171,  "e_dh": 0.073192, "p_e": 14.56414,  "alpha": 63.4404,  "re": 107269.2,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp24", "ar": 10.4212,   "e_dh": 0.131579, "p_e": 10.35821,  "alpha": 47.54174, "re": 148329.5,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp25", "ar": 13.89294,  "e_dh": 0.196634, "p_e":  5.968536, "alpha": 40.4697,  "re":  64305.93, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp26", "ar":  8.260275, "e_dh": 0.106006, "p_e": 10.63216,  "alpha": 31.18607, "re": 183860.6,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp27", "ar":  6.610613, "e_dh": 0.090241, "p_e":  7.549495, "alpha": 71.61877, "re":  56800.33, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp28", "ar":  9.687974, "e_dh": 0.068103, "p_e":  9.436798, "alpha": 48.67483, "re":  44366.84, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp29", "ar":  9.374113, "e_dh": 0.064434, "p_e":  6.303342, "alpha": 59.78739, "re": 158601.6,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp30", "ar":  7.427527, "e_dh": 0.145224, "p_e":  6.949837, "alpha": 56.97142, "re": 185584.6,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp31", "ar": 13.21633,  "e_dh": 0.178371, "p_e": 12.96159,  "alpha": 67.7618,  "re":  80936.64, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp32", "ar": 12.7963,   "e_dh": 0.109381, "p_e":  8.007497, "alpha": 39.72069, "re": 132395.1,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp33", "ar": 10.64298,  "e_dh": 0.115023, "p_e":  9.933429, "alpha": 33.21242, "re":  39651.15, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp34", "ar": 11.43933,  "e_dh": 0.058513, "p_e": 10.1587,   "alpha": 32.23051, "re":  82745.98, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp35", "ar":  7.275716, "e_dh": 0.117108, "p_e":  5.367998, "alpha": 30.2532,  "re": 177585.9,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp36", "ar":  5.135855, "e_dh": 0.191313, "p_e": 13.56869,  "alpha": 61.60822, "re": 150207.4,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp37", "ar":  5.459505, "e_dh": 0.054453, "p_e":  9.665539, "alpha": 53.22351, "re": 115302.3,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp38", "ar": 11.38441,  "e_dh": 0.079722, "p_e": 14.61072,  "alpha": 70.09904, "re":  94011.89, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp39", "ar":  7.906148, "e_dh": 0.171919, "p_e": 10.78854,  "alpha": 48.24447, "re":  68803.85, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp40", "ar": 10.13908,  "e_dh": 0.051254, "p_e": 13.31478,  "alpha": 66.29723, "re":  71450.48, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp41", "ar": 12.53072,  "e_dh": 0.128661, "p_e":  7.684222, "alpha": 62.53379, "re": 163960.9,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp42", "ar": 11.69391,  "e_dh": 0.174673, "p_e":  6.619144, "alpha": 70.79936, "re": 102851.9,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp43", "ar": 11.99886,  "e_dh": 0.197554, "p_e":  6.98893,  "alpha": 55.25445, "re": 175714.0,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp44", "ar":  4.802336, "e_dh": 0.137666, "p_e":  5.77712,  "alpha": 50.38406, "re": 127704.6,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp45", "ar":  6.189208, "e_dh": 0.18303,  "p_e":  7.266435, "alpha": 56.3389,  "re":  54043.76, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp46", "ar":  8.640326, "e_dh": 0.144186, "p_e": 13.09386,  "alpha": 57.94209, "re":  30389.25, "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp47", "ar": 13.48607,  "e_dh": 0.161387, "p_e": 12.7088,   "alpha": 52.23429, "re": 192455.8,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp48", "ar":  9.210315, "e_dh": 0.104191, "p_e": 14.13385,  "alpha": 72.81153, "re": 137273.4,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp49", "ar": 15.06195,  "e_dh": 0.202031, "p_e": 15.16555,  "alpha": 77.84987, "re": 204946.7,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
    {"folder": "dp50", "ar": 15.45474,  "e_dh": 0.204572, "p_e": 15.07501,  "alpha": 75.50823, "re": 202176.8,  "outlet_y_min_frac": None, "inlet_y_max_frac": None},
]

# ═══════════════════════════════════════════════════════════════
# GEOMETRY SAMPLING  (labels: 0=interior, 1=inlet, 2=outlet, 3=wall)
# ═══════════════════════════════════════════════════════════════

def build_wall_pool(mesh, x_stl_min_mm, x_stl_max_mm, n_pool=200_000):
    pts, _ = trimesh.sample.sample_surface(mesh, n_pool * 3)
    x    = pts[:, 0]
    mask = (
        (np.abs(x - x_stl_min_mm) > 0.1) &
        (np.abs(x - x_stl_max_mm) > 0.1) &
        (x >= x_stl_min_mm) &
        (x <= x_stl_max_mm)
    )
    wall_pts = torch.tensor(pts[mask], dtype=torch.float64)
    if len(wall_pts) > n_pool:
        wall_pts = wall_pts[torch.randperm(len(wall_pts))[:n_pool]]
    return wall_pts


def sample_wall(wall_pool, n):
    idx = torch.randperm(len(wall_pool))[:n]
    return wall_pool[idx], 3 * torch.ones(n, dtype=torch.int64)


def build_volume_pool(mesh, x_stl_min_mm, x_stl_max_mm,
                      n_pool=800_000, wall_fraction=0.5,
                      delta_horiz_mm=1.0, delta_vert_mm=1.0, vert_frac=0.7):
    n_near    = int(n_pool * wall_fraction)
    n_uniform = n_pool - n_near

    raw  = trimesh.sample.volume_mesh(mesh, n_uniform * 3)
    mask = (raw[:, 0] >= x_stl_min_mm) & (raw[:, 0] <= x_stl_max_mm)
    raw  = raw[mask]
    if len(raw) > n_uniform:
        raw = raw[np.random.choice(len(raw), n_uniform, replace=False)]

    def _near(face_ids, n, delta_mm):
        sub = mesh.submesh([face_ids], only_watertight=False)
        if not sub:
            return np.zeros((0, 3))
        s_pts, s_fids = trimesh.sample.sample_surface(sub[0], n * 2)
        eps  = np.random.uniform(0, delta_mm, size=(len(s_pts), 1))
        near = s_pts - eps * sub[0].face_normals[s_fids]
        keep = (near[:, 0] >= x_stl_min_mm) & (near[:, 0] <= x_stl_max_mm)
        near = near[keep]
        if len(near) > n:
            near = near[np.random.choice(len(near), n, replace=False)]
        return near

    horiz_fids  = np.where(np.abs(mesh.face_normals[:, 2]) >= 0.7)[0]
    vert_fids   = np.where(np.abs(mesh.face_normals[:, 2]) <  0.7)[0]
    near_horiz  = _near(horiz_fids, int(n_near * (1 - vert_frac)), delta_horiz_mm)
    near_vert   = _near(vert_fids,  int(n_near * vert_frac),       delta_vert_mm)
    all_pts     = np.vstack([raw, near_horiz, near_vert])
    print(f"    uniform={len(raw)}  near-horiz={len(near_horiz)}  near-vert={len(near_vert)}")
    return torch.tensor(all_pts, dtype=torch.float64), len(raw), len(near_horiz)


def sample_volume(vol_pool, n):
    idx = torch.randperm(len(vol_pool))[:n]
    return vol_pool[idx], torch.zeros(n, dtype=torch.int64)


def sample_collocation(vol_pool, wall_pool, n_vol, n_wall):
    p, l_vol  = sample_volume(vol_pool, n_vol)
    w, l_wall = sample_wall(wall_pool, n_wall)
    pts  = torch.cat([p, w])
    lbls = torch.cat([l_vol, l_wall])
    perm = torch.randperm(pts.size(0))
    return pts[perm], lbls[perm]


# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def dynamic_viscosity_tilde(T_tilde, delta_t, t_wall):
    """mu/MU_REF given dimensionless T̃ and per-DP DELTA_T, T_WALL."""
    T_K = T_tilde * delta_t + t_wall
    mu  = MU_REF * (T_K.abs() / T_REF_SUTH) ** 1.5 * (T_REF_SUTH + S_SUTH) / (T_K.abs() + S_SUTH)
    return mu / MU_REF


def _grad(f, pts):
    return torch.autograd.grad(f, pts, torch.ones_like(f),
                                retain_graph=True, create_graph=True)[0]


def _grad2(f, pts, dim):
    g = _grad(f, pts)[:, dim:dim+1]
    return torch.autograd.grad(g, pts, torch.ones_like(g),
                                retain_graph=True, create_graph=True)[0][:, dim:dim+1]


def compute_derivatives(fields, pts_xyz):
    """All spatial derivatives. pts_xyz [N,3] must have requires_grad=True."""
    vx = fields[:, 0:1]; vy = fields[:, 1:2]; vz = fields[:, 2:3]
    p  = fields[:, 3:4]; T  = fields[:, 4:5]

    gvx = _grad(vx, pts_xyz); vx_x, vx_y, vx_z = gvx[:,0:1], gvx[:,1:2], gvx[:,2:3]
    vx_xx = _grad2(vx, pts_xyz, 0); vx_yy = _grad2(vx, pts_xyz, 1); vx_zz = _grad2(vx, pts_xyz, 2)

    gvy = _grad(vy, pts_xyz); vy_x, vy_y, vy_z = gvy[:,0:1], gvy[:,1:2], gvy[:,2:3]
    vy_xx = _grad2(vy, pts_xyz, 0); vy_yy = _grad2(vy, pts_xyz, 1); vy_zz = _grad2(vy, pts_xyz, 2)

    gvz = _grad(vz, pts_xyz); vz_x, vz_y, vz_z = gvz[:,0:1], gvz[:,1:2], gvz[:,2:3]
    vz_xx = _grad2(vz, pts_xyz, 0); vz_yy = _grad2(vz, pts_xyz, 1); vz_zz = _grad2(vz, pts_xyz, 2)

    gp  = _grad(p,  pts_xyz); p_x,  p_y,  p_z  = gp[:,0:1],  gp[:,1:2],  gp[:,2:3]
    gT  = _grad(T,  pts_xyz); T_x,  T_y,  T_z  = gT[:,0:1],  gT[:,1:2],  gT[:,2:3]
    T_xx = _grad2(T, pts_xyz, 0); T_yy = _grad2(T, pts_xyz, 1); T_zz = _grad2(T, pts_xyz, 2)

    return (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    )


def compute_losses(
    vx, vy, vz, p, T,
    vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
    vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
    vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
    p_x, p_y, p_z,
    T_x, T_y, T_z, T_xx, T_yy, T_zz,
    labels,
    vx_in, vy_in, vz_in,         # per-batch inlet BC velocities (non-dim)
    dp_ctx,                       # per-DP scalings dict
    div_scale, mom_scale_x, mom_scale_y, mom_scale_z,  # global
):
    interior = labels == 0
    inlet    = labels == 1
    outlet   = labels == 2
    wall     = labels == 3

    mu_tilde    = dynamic_viscosity_tilde(T, dp_ctx["delta_t"], dp_ctx["t_wall"])
    visc_coeff  = dp_ctx["visc_coeff"]   # ν/V_IN [m]
    therm_coeff = dp_ctx["therm_coeff"]  # α_th/V_IN [m]

    vx_var = dp_ctx["vx_var"]; vy_var = dp_ctx["vy_var"]; vz_var = dp_ctx["vz_var"]
    p_var  = dp_ctx["p_var"];  t_var  = dp_ctx["t_var"]

    # ── PDE losses ────────────────────────────────────────────
    l_div = torch.mean(((vx_x + vy_y + vz_z) / div_scale) ** 2)

    def navier_stokes(u_x, u_y, u_z, u_xx, u_yy, u_zz, p_grad, mom_scale):
        advec = (vx[interior]*u_x[interior] + vy[interior]*u_y[interior] + vz[interior]*u_z[interior])
        lap   = u_xx[interior] + u_yy[interior] + u_zz[interior]
        res   = advec + p_grad[interior] - visc_coeff * mu_tilde[interior] * lap
        return torch.mean((res / mom_scale) ** 2)

    l_mom_x = navier_stokes(vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz, p_x, mom_scale_x)
    l_mom_y = navier_stokes(vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz, p_y, mom_scale_y)
    l_mom_z = navier_stokes(vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz, p_z, mom_scale_z)

    l_heat = torch.mean((
        vx[interior]*T_x[interior] + vy[interior]*T_y[interior] + vz[interior]*T_z[interior]
        - therm_coeff * (T_xx[interior] + T_yy[interior] + T_zz[interior])
    ) ** 2)

    # ── BC losses ─────────────────────────────────────────────
    l_inlet_vx = torch.mean((vx[inlet].squeeze(1) - vx_in) ** 2) / vx_var
    l_inlet_vy = torch.mean((vy[inlet].squeeze(1) - vy_in) ** 2) / vy_var
    l_inlet_vz = torch.mean((vz[inlet].squeeze(1) - vz_in) ** 2) / vz_var
    l_inlet_T  = torch.mean((T[inlet]              - 1.0)  ** 2) / t_var   # T̃_inlet = 1 always
    l_outlet_p = torch.mean((p[outlet]             - 0.0)  ** 2) / p_var   # p̃_outlet = 0 by P_REF
    l_wall_vx  = torch.mean( vx[wall] ** 2)                      / vx_var
    l_wall_vy  = torch.mean( vy[wall] ** 2)                      / vy_var
    l_wall_vz  = torch.mean( vz[wall] ** 2)                      / vz_var
    l_wall_T   = torch.mean((T[wall]  - 0.0) ** 2)               / t_var   # T̃_wall = 0 always

    return (
        l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
        l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
        l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
    )


# ═══════════════════════════════════════════════════════════════
# MODEL
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
    """Z-scores inputs; un-standardizes outputs.
    Input layout: [x, y, z, AR_nd, EDH_nd, PE_nd, ALPHA_nd, RE_nd, s1, s2]  (10 dims).
    s1, s2 are sigmoid wall-distance features computed from x[:,1] (physical y).
    coord_mean/coord_std cover all 10 dims.
    """
    def __init__(self, net, coord_mean, coord_std, out_mean, out_std):
        super().__init__()
        self.net = net
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std",  coord_std)
        self.register_buffer("out_mean",   out_mean)
        self.register_buffer("out_std",    out_std)

    def forward(self, x):
        # x[:,1] is physical y — append wall sigmoid features before z-scoring
        y   = x[:, 1:2]
        s1  = torch.sigmoid((y - WALL_Y1) / WALL_EPS)
        s2  = torch.sigmoid((y - WALL_Y2) / WALL_EPS)
        x   = torch.cat([x, s1, s2], dim=1)
        x_norm   = (x - self.coord_mean) / self.coord_std
        y_norm   = self.net(x_norm)
        safe_std = torch.where(self.out_std == 0, torch.ones_like(self.out_std), self.out_std)
        return y_norm * safe_std + self.out_mean


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_fields(pts, fields, output_dir, tag="", slice_frac=0.10):
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
        if has_data: subplots.append((f"Data – {name}", data,        vmin, vmax))
        subplots.append(             (f"Pred – {name}", pred,        vmin, vmax))
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
        fig.savefig(os.path.join(output_dir, f"{name}{'_'+tag if tag else ''}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_point_clouds(title, path, datasets, wall_ys=None, unit="m"):
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(title, fontsize=10)

    ax3d = fig.add_subplot(141, projection="3d")
    for pts, color, s, alpha, label in datasets:
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, alpha=alpha, c=color, label=label)
    ax3d.set_xlabel(f"x [{unit}]"); ax3d.set_ylabel(f"y [{unit}]"); ax3d.set_zlabel(f"z [{unit}]")
    ax3d.set_title("3D view"); ax3d.legend(markerscale=8, fontsize=7)
    if wall_ys:
        _xr = [min(d[0][:, 0].min() for d in datasets), max(d[0][:, 0].max() for d in datasets)]
        _zr = [min(d[0][:, 2].min() for d in datasets), max(d[0][:, 2].max() for d in datasets)]
        for wy in wall_ys:
            ax3d.plot(_xr, [wy, wy], [_zr[0], _zr[0]], c="orange", lw=1.0, ls="--")
            ax3d.plot(_xr, [wy, wy], [_zr[1], _zr[1]], c="orange", lw=1.0, ls="--")

    for i, (xl, yl, xi, yi) in enumerate(
            [(f"x [{unit}]", f"y [{unit}]", 0, 1),
             (f"x [{unit}]", f"z [{unit}]", 0, 2),
             (f"y [{unit}]", f"z [{unit}]", 1, 2)], start=2):
        ax = fig.add_subplot(1, 4, i)
        for pts, color, s, alpha, _ in datasets:
            ax.scatter(pts[:, xi], pts[:, yi], s=s, alpha=alpha, c=color)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}")
        if wall_ys:
            for wy in wall_ys:
                if yi == 1:
                    ax.axhline(wy, color="orange", lw=1.2, ls="--")
                elif xi == 1:
                    ax.axvline(wy, color="orange", lw=1.2, ls="--")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{os.path.basename(path)}  → {path}")


# ═══════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="PINN25 — multi-DP parametric training on all available non-isothermal cases")

parser.add_argument("--run-path",   default=None,     help="Output dir (auto-derived if omitted)")
parser.add_argument("--project",    default="PINN25", help="W&B project name")
parser.add_argument("--device",     default="cuda")
parser.add_argument("--debug",      action="store_false", help="Skip W&B logging")

parser.add_argument("--hidden-dim", type=int,   default=20)
parser.add_argument("--n-layers",   type=int,   default=4)

parser.add_argument("--epochs",     type=int,   default=6_000)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--lr",         type=float, default=3e-3)
parser.add_argument("--lr-end",     type=float, default=1e-4)

parser.add_argument("--n-train",       type=int,   default=5_000)
parser.add_argument("--n-test",        type=int,   default=10_000)
parser.add_argument("--n-sup",         type=int,   default=500,
                    help="Supervised CFD points per DP (0 = physics-only for that DP)")
parser.add_argument("--n-snapshots",   type=int,   default=20_000)
parser.add_argument("--wall-fraction", type=float, default=0.5)
parser.add_argument("--n-pool",        type=int,   default=500_000)
parser.add_argument("--pool-frac-vol", type=float, default=0.5)
parser.add_argument("--pool-frac-wall",type=float, default=0.5)
parser.add_argument("--delta-mm-horiz",type=float, default=1.0)
parser.add_argument("--delta-mm-vert", type=float, default=20.0)
parser.add_argument("--w-pde",   type=float, default=1.0)
parser.add_argument("--w-bc",    type=float, default=1.0)
parser.add_argument("--w-sup",   type=float, default=1.0)
parser.add_argument("--w-sup-p", type=float, default=1.0)
# Default outlet/inlet fracs applied to any DP whose config entry is None
parser.add_argument("--outlet-y-min-frac", type=float, default=0.0)
parser.add_argument("--inlet-y-max-frac",  type=float, default=1.0)

args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

DEVICE     = args.device
PROJECT    = args.project
DEBUG      = args.debug
EPOCHS     = args.epochs
HIDDEN_DIM = args.hidden_dim
N_LAYERS   = args.n_layers
SEED       = args.seed
LR         = args.lr
LR_END     = args.lr_end
GAMMA      = (LR_END / LR) ** (1.0 / EPOCHS)
N_TEST              = args.n_test
N_TOTAL_TRAIN       = args.n_train
N_SUP               = args.n_sup
N_POINT_SNAPSHOTS   = args.n_snapshots
WALL_FRACTION       = args.wall_fraction
N_TOTAL_POOL        = args.n_pool
POOL_FRAC_VOL       = args.pool_frac_vol
POOL_FRAC_WALL      = args.pool_frac_wall
DELTA_HORIZ_MM      = args.delta_mm_horiz
DELTA_VERT_MM       = args.delta_mm_vert
W_PDE   = args.w_pde
W_BC    = args.w_bc
W_SUP   = args.w_sup
W_SUP_P = args.w_sup_p

N_PARAMS = 5
IN_DIM   = 3 + N_PARAMS + 2   # 10: (x, y, z, AR_nd, EDH_nd, PE_nd, ALPHA_nd, RE_nd, s1, s2)

FRAC_VOL    = 0.20
FRAC_WALL   = 0.70
FRAC_INLET  = 0.05
FRAC_OUTLET = 0.05
n_vol_train    = int(FRAC_VOL    * N_TOTAL_TRAIN)
n_wall_train   = int(FRAC_WALL   * N_TOTAL_TRAIN)
n_inlet_train  = int(FRAC_INLET  * N_TOTAL_TRAIN)
n_outlet_train = int(FRAC_OUTLET * N_TOTAL_TRAIN)

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_default_device(DEVICE)
print(f"Device : {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# DP LOADING
# ═══════════════════════════════════════════════════════════════

def _var(raw_var, floor, name):
    v = raw_var.clamp(min=floor)
    if raw_var.item() < floor:
        print(f"    [variance floor] {name}: raw={raw_var.item():.2e} → {floor:.2e}")
    return v


def load_dp(cfg, args):
    """Load one design point. Returns a dict with all data and per-DP scalings.
    Raises RuntimeError for isothermal DPs (DELTA_T < ISOTHERMAL_DELTA_T_THRESHOLD).
    """
    folder   = cfg["folder"]
    data_dir = f"./preProcessedData/with_T/{folder}/"
    print(f"\n{'─'*60}")
    print(f"  Loading {folder}  (AR={cfg['ar']}  e/Dh={cfg['e_dh']}  P/e={cfg['p_e']}"
          f"  α={cfg['alpha']}°  Re={cfg['re']:.0f})")

    outlet_frac = cfg["outlet_y_min_frac"] if cfg["outlet_y_min_frac"] is not None else args.outlet_y_min_frac
    inlet_frac  = cfg["inlet_y_max_frac"]  if cfg["inlet_y_max_frac"]  is not None else args.inlet_y_max_frac

    # ── Inlet velocity data ────────────────────────────────────
    _in_vx = np.load(data_dir + "vel_x_inlet.npy")
    _in_vy = np.load(data_dir + "vel_y_inlet.npy")
    _in_vz = np.load(data_dir + "vel_z_inlet.npy")
    if inlet_frac < 1.0:
        _y_max  = float(_in_vx[:, 1].max())
        _y_cut  = inlet_frac * _y_max
        _mask   = _in_vx[:, 1] <= _y_cut
        _in_vx  = _in_vx[_mask]; _in_vy = _in_vy[_mask]; _in_vz = _in_vz[_mask]
        print(f"    Inlet y-filter: y ≤ {_y_cut:.4f} m  (kept {_mask.sum()})")

    # ── CFD volume data ────────────────────────────────────────
    _vx_raw  = np.load(data_dir + "vel_x.npy")
    _temp    = np.load(data_dir + "temp.npy")
    cfd_pts  = torch.tensor(_vx_raw[:, :3])
    cfd_vx   = torch.tensor(_vx_raw[:, 3])
    cfd_vy   = torch.tensor(np.load(data_dir + "vel_y.npy")[:, 3])
    cfd_vz   = torch.tensor(np.load(data_dir + "vel_z.npy")[:, 3])
    cfd_p    = torch.tensor(np.load(data_dir + "press.npy")[:, 3])
    cfd_T    = torch.tensor(_temp[:, 3])

    # ── T_WALL from CFD minimum ────────────────────────────────
    t_wall = float(cfd_T.min())

    # ── T_inlet from CFD inlet face ────────────────────────────
    _x_min_temp = float(_temp[:, 0].min())
    _temp_face  = _temp[_temp[:, 0] == _x_min_temp]
    if inlet_frac < 1.0:
        _y_max_t    = float(_temp[:, 1].max())
        _temp_face  = _temp_face[_temp_face[:, 1] <= inlet_frac * _y_max_t]
    t_inlet = float(_temp_face[:, 3].mean())
    print(f"    T_WALL={t_wall:.2f} K   T_inlet={t_inlet:.2f} K")

    # ── Isothermal guard ───────────────────────────────────────
    delta_t = t_inlet - t_wall
    if delta_t < ISOTHERMAL_DELTA_T_THRESHOLD:
        raise RuntimeError(
            f"DP '{folder}' is isothermal or near-isothermal: "
            f"DELTA_T = {delta_t:.3f} K < {ISOTHERMAL_DELTA_T_THRESHOLD} K. "
            f"PINN25 is for non-isothermal cases only. "
            f"Remove or replace this DP folder before running."
        )

    # ── Domain bounds ──────────────────────────────────────────
    x_min_mm = float(cfd_pts[:, 0].min()) * 1000.0
    x_max_mm = float(cfd_pts[:, 0].max()) * 1000.0

    # ── Outlet mask ────────────────────────────────────────────
    out_mask = cfd_pts[:, 0] > (x_max_mm / 1000.0 - 0.005)
    if outlet_frac > 0.0:
        _y_max_o  = float(cfd_pts[:, 1].max())
        _y_min_o  = outlet_frac * _y_max_o
        out_mask  = out_mask & (cfd_pts[:, 1] >= _y_min_o)
        print(f"    Outlet y-filter: y ≥ {_y_min_o:.4f} m  (kept {out_mask.sum().item()})")

    # ── Physical scales ────────────────────────────────────────
    V_IN   = float(torch.tensor(_in_vx[:, 3]).abs().mean())
    P_REF  = float(cfd_p[out_mask].mean())
    P_DYN  = float(cfd_p[out_mask].std().clamp(min=1.0))   # dynamic scale from outlet spread
    # Use rho-based P_DYN for consistent NS non-dim:
    # Reference state at inlet temperature (Re is defined at inlet conditions)
    rho       = P_REF * M / (R * t_inlet)
    P_DYN     = rho * V_IN ** 2
    nu        = MU_REF / rho
    alpha_th  = K / (rho * CP)
    visc_coeff  = nu       / V_IN
    therm_coeff = alpha_th / V_IN
    print(f"    V_IN={V_IN:.3f} m/s  P_REF={P_REF:.0f} Pa  P_DYN={P_DYN:.1f} Pa"
          f"  DELTA_T={delta_t:.2f} K  ν/V_IN={visc_coeff:.2e} m")

    # ── Non-dim fields ─────────────────────────────────────────
    cfd_vx_nd = cfd_vx / V_IN
    cfd_vy_nd = cfd_vy / V_IN
    cfd_vz_nd = cfd_vz / V_IN
    cfd_p_nd  = (cfd_p - P_REF) / P_DYN
    cfd_T_nd  = (cfd_T - t_wall) / delta_t   # in [0, 1]: 0 at wall, 1 at inlet

    # ── Pools (inlet uses physical metres, vol/wall in STL mm) ─
    inlet_pool = (
        torch.tensor(_in_vx[:, :3]),
        torch.tensor(_in_vx[:, 3]) / V_IN,
        torch.tensor(_in_vy[:, 3]) / V_IN,
        torch.tensor(_in_vz[:, 3]) / V_IN,
    )
    outlet_pool = cfd_pts[out_mask]

    # ── STL mesh and pools ─────────────────────────────────────
    stl_files = glob.glob(data_dir + "*.stl") + glob.glob(data_dir + "*.STL")
    if not stl_files:
        raise FileNotFoundError(f"No STL found in {data_dir}")
    print(f"    STL: {os.path.basename(stl_files[0])}")
    mesh = trimesh.load(stl_files[0])

    stl_buffer_mm = 420.0
    x_stl_min_mm = x_min_mm + stl_buffer_mm
    x_stl_max_mm = x_max_mm + stl_buffer_mm
    print(f"    STL buffer={stl_buffer_mm:.1f} mm  "
          f"physical x=[{x_min_mm:.1f}, {x_max_mm:.1f}] mm")

    print(f"    Building volume pool…")
    vol_pool, _n_uni, _n_horiz = build_volume_pool(
        mesh, x_stl_min_mm, x_stl_max_mm,
        n_pool=int(POOL_FRAC_VOL * N_TOTAL_POOL),
        wall_fraction=WALL_FRACTION,
        delta_horiz_mm=DELTA_HORIZ_MM, delta_vert_mm=DELTA_VERT_MM)
    print(f"    Building wall pool…")
    wall_pool = build_wall_pool(
        mesh, x_stl_min_mm, x_stl_max_mm,
        n_pool=int(POOL_FRAC_WALL * N_TOTAL_POOL))

    # ── Train / test split ─────────────────────────────────────
    n_total = cfd_pts.shape[0]
    n_test  = min(N_TEST, n_total - 1)
    perm    = torch.randperm(n_total)
    t_idx   = perm[:n_test]
    tr_idx  = perm[n_test:]

    test_pts  = cfd_pts[t_idx];    train_pts = cfd_pts[tr_idx]
    test_vx   = cfd_vx_nd[t_idx];  train_vx  = cfd_vx_nd[tr_idx]
    test_vy   = cfd_vy_nd[t_idx];  train_vy  = cfd_vy_nd[tr_idx]
    test_vz   = cfd_vz_nd[t_idx];  train_vz  = cfd_vz_nd[tr_idx]
    test_p    = cfd_p_nd[t_idx];   train_p   = cfd_p_nd[tr_idx]
    test_T    = cfd_T_nd[t_idx];   train_T   = cfd_T_nd[tr_idx]

    # ── Supervised points ─────────────────────────────────────
    n_sup = min(N_SUP, train_pts.shape[0])
    if n_sup > 0:
        _sup_idx = torch.randint(0, train_pts.shape[0], (n_sup,))
        sup_pts  = train_pts[_sup_idx]
        sup_vx   = train_vx[_sup_idx]; sup_vy = train_vy[_sup_idx]; sup_vz = train_vz[_sup_idx]
        sup_p    = train_p[_sup_idx];  sup_T  = train_T[_sup_idx]
    else:
        sup_pts = sup_vx = sup_vy = sup_vz = sup_p = sup_T = None

    # ── Loss normalisation variances ──────────────────────────
    vx_raw = torch.maximum(train_vx.var(), inlet_pool[1].var())
    vy_raw = torch.maximum(train_vy.var(), inlet_pool[2].var())
    vz_raw = torch.maximum(train_vz.var(), inlet_pool[3].var())
    vx_var = _var(vx_raw, 1e-6, "vx")
    vy_var = _var(vy_raw, 1e-6, "vy")
    vz_var = _var(vz_raw, 1e-6, "vz")
    p_var  = _var(train_p.var(), 1e-6, "p")
    t_var  = _var(train_T.var(), 1e-2, "T")

    # ── Snap points ───────────────────────────────────────────
    n_snap   = min(N_POINT_SNAPSHOTS, n_total)
    snap_idx = np.random.choice(n_total, n_snap, replace=False)
    snap_pts = cfd_pts[snap_idx]
    snap_vx  = cfd_vx_nd[snap_idx]; snap_vy = cfd_vy_nd[snap_idx]; snap_vz = cfd_vz_nd[snap_idx]
    snap_p   = cfd_p_nd[snap_idx];  snap_T  = cfd_T_nd[snap_idx]

    # ── Pre-normalised parametric input ───────────────────────
    params_raw = torch.tensor([cfg["ar"], cfg["e_dh"], cfg["p_e"], cfg["alpha"], cfg["re"]],
                               dtype=torch.float64)
    params_nd  = (params_raw - PARAM_MEANS.to(params_raw.device)) / PARAM_STDS.to(params_raw.device)

    print(f"    params_nd = {[f'{v:.3f}' for v in params_nd.tolist()]}")
    print(f"    vx_var={vx_var.item():.3e}  vy_var={vy_var.item():.3e}"
          f"  vz_var={vz_var.item():.3e}  p_var={p_var.item():.3e}  T_var={t_var.item():.3e}")
    print(f"    CFD pts={n_total:,}  train={len(tr_idx):,}  test={n_test:,}  sup={n_sup}")

    return {
        "folder": folder,
        "params_nd": params_nd,
        # pools
        "vol_pool":    vol_pool,
        "vol_pool_n_uni":   _n_uni,
        "vol_pool_n_horiz": _n_horiz,
        "wall_pool":   wall_pool,
        "inlet_pool":  inlet_pool,
        "outlet_pool": outlet_pool,
        "stl_buffer_mm":  stl_buffer_mm,
        "x_min_mm":       x_min_mm,
        "x_max_mm":       x_max_mm,
        "x_stl_min_mm":   x_stl_min_mm,
        "x_stl_max_mm":   x_stl_max_mm,
        # per-DP physics context (passed to compute_losses)
        "dp_ctx": {
            "delta_t":    delta_t,
            "t_wall":     t_wall,
            "visc_coeff": visc_coeff,
            "therm_coeff":therm_coeff,
            "vx_var": vx_var, "vy_var": vy_var, "vz_var": vz_var,
            "p_var":  p_var,  "t_var":  t_var,
        },
        # scales needed for plotting (physical units)
        "V_IN": V_IN, "P_DYN": P_DYN, "P_REF": P_REF,
        "DELTA_T": delta_t, "T_WALL": t_wall,
        # CFD data (non-dim, for global stats + eval)
        "cfd_pts_nd":    cfd_pts,
        "cfd_fields_nd": torch.stack([cfd_vx_nd, cfd_vy_nd, cfd_vz_nd, cfd_p_nd, cfd_T_nd], dim=1),
        # split
        "train_idx": tr_idx, "test_idx": t_idx,
        "train_pts": train_pts, "train_vx": train_vx, "train_vy": train_vy,
        "train_vz": train_vz,  "train_p":  train_p,  "train_T":  train_T,
        "test_pts":  test_pts,  "test_vx":  test_vx,  "test_vy":  test_vy,
        "test_vz":  test_vz,   "test_p":   test_p,   "test_T":   test_T,
        # supervised
        "n_sup": n_sup,
        "sup_pts": sup_pts, "sup_vx": sup_vx, "sup_vy": sup_vy,
        "sup_vz":  sup_vz,  "sup_p":  sup_p,  "sup_T":  sup_T,
        # snap
        "snap_pts": snap_pts, "snap_vx": snap_vx, "snap_vy": snap_vy,
        "snap_vz":  snap_vz,  "snap_p":  snap_p,  "snap_T":  snap_T,
    }


# ═══════════════════════════════════════════════════════════════
# DISCOVER AND LOAD AVAILABLE DPs
# ═══════════════════════════════════════════════════════════════

print("\nScanning DP_CONFIGS for available folders…")
dps = []
for cfg in DP_CONFIGS:
    dp_dir = f"./preProcessedData/with_T/{cfg['folder']}/"
    if not os.path.isdir(dp_dir):
        print(f"  [SKIP] {cfg['folder']} — folder not found")
        continue
    dp = load_dp(cfg, args)
    dps.append(dp)

if not dps:
    raise RuntimeError("No DP folders found in preProcessedData/with_T/. "
                       "Expected at least one of: " + ", ".join(c["folder"] for c in DP_CONFIGS))

N_DPS = len(dps)
dp_names = [dp["folder"] for dp in dps]
print(f"\nLoaded {N_DPS} DP(s): {dp_names}")

if args.run_path:
    RUN_PATH = args.run_path
else:
    RUN_PATH = (f"../pinn25_runs/"
                f"n{N_DPS}_h{HIDDEN_DIM}_l{N_LAYERS}"
                f"_e{EPOCHS}_lr{LR:.0e}_lrend{LR_END:.0e}"
                f"_ntrain{N_TOTAL_TRAIN}_sup{N_SUP}_s{SEED}")
os.makedirs(RUN_PATH, exist_ok=True)

# ── Visualize pools/BCs for the first DP only ──────────────────
_dp0         = dps[0]
_buf         = _dp0["stl_buffer_mm"]
_x_min_mm    = _dp0["x_min_mm"]
_x_max_mm    = _dp0["x_max_mm"]
_x_stl_min   = _dp0["x_stl_min_mm"]
_x_stl_max   = _dp0["x_stl_max_mm"]
print(f"[viz] {_dp0['folder']}  physical x=[{_x_min_mm:.1f}, {_x_max_mm:.1f}] mm"
      f"  STL x=[{_x_stl_min:.1f}, {_x_stl_max:.1f}] mm  (buffer={_buf:.1f} mm)")
_sub = 20_000

_pool_np   = _dp0["vol_pool"].cpu().numpy()
_n_uni     = _dp0["vol_pool_n_uni"]
_n_horiz   = _dp0["vol_pool_n_horiz"]
_uni       = _pool_np[:_n_uni]
_near_horiz= _pool_np[_n_uni : _n_uni + _n_horiz]
_near_vert = _pool_np[_n_uni + _n_horiz:]
_ui  = np.random.choice(len(_uni),         min(_sub, len(_uni)),         replace=False)
_hi  = np.random.choice(len(_near_horiz),  min(_sub, len(_near_horiz)),  replace=False)
_vi  = np.random.choice(len(_near_vert),   min(_sub, len(_near_vert)),   replace=False)
plot_point_clouds(
    title=(f"[{_dp0['folder']}] Volume pool — "
           f"uniform (blue, n={len(_uni)})  near-horiz (red, n={len(_near_horiz)})  "
           f"near-vert (green, n={len(_near_vert)})"),
    path=os.path.join(RUN_PATH, f"viz_volume_pool_{_dp0['folder']}.png"),
    datasets=[
        (_uni[_ui],         "steelblue", 0.5, 0.2, "uniform"),
        (_near_horiz[_hi],  "crimson",   0.5, 0.3, "near-horiz"),
        (_near_vert[_vi],   "limegreen", 0.5, 0.3, "near-vert"),
    ],
    unit="mm",
)

_pool_cfd_m = _pool_np.copy().astype(float)
_pool_cfd_m[:, 0] -= _buf
_pool_cfd_m /= 1000.0
_bg_i = np.random.choice(len(_pool_cfd_m), min(_sub, len(_pool_cfd_m)), replace=False)
_bg   = _pool_cfd_m[_bg_i]
_in_np  = _dp0["inlet_pool"][0].cpu().numpy()
_out_np = _dp0["outlet_pool"].cpu().numpy()
plot_point_clouds(
    title=(f"[{_dp0['folder']}] BC geometry — "
           f"vol pool (grey)  inlet (green, n={len(_in_np)})  outlet (red, n={len(_out_np)})"
           f"  |  walls y={WALL_Y1:.4f} / {WALL_Y2:.4f} m"),
    path=os.path.join(RUN_PATH, f"viz_geometry_bc_{_dp0['folder']}.png"),
    datasets=[
        (_bg,     "grey",      0.5, 0.15, "vol pool"),
        (_in_np,  "limegreen", 4,   1.0,  "inlet"),
        (_out_np, "crimson",   4,   1.0,  "outlet"),
    ],
    wall_ys=[WALL_Y1, WALL_Y2],
)

_sup_np = _dp0["sup_pts"].cpu().numpy() if _dp0["n_sup"] > 0 else None
_sv_datasets = [(_bg, "grey", 0.5, 0.15, "vol pool")]
if _sup_np is not None:
    _sv_datasets.append((_sup_np, "red", 1.5, 1.0, "supervised"))
plot_point_clouds(
    title=(f"[{_dp0['folder']}] Supervised pts (red, n={_dp0['n_sup']})  vol pool (grey)"
           if _dp0["n_sup"] > 0 else f"[{_dp0['folder']}] Vol pool — no supervised pts"),
    path=os.path.join(RUN_PATH, f"viz_supervised_points_{_dp0['folder']}.png"),
    datasets=_sv_datasets,
)
del _dp0, _buf, _x_min_mm, _x_max_mm, _x_stl_min, _x_stl_max
del _pool_np, _n_uni, _n_horiz, _uni, _near_horiz, _near_vert
del _ui, _hi, _vi, _pool_cfd_m, _bg_i, _bg, _in_np, _out_np, _sv_datasets

# ═══════════════════════════════════════════════════════════════
# GLOBAL NORMALIZATION
# ═══════════════════════════════════════════════════════════════

all_pts_nd    = torch.cat([dp["cfd_pts_nd"]    for dp in dps], dim=0)
all_fields_nd = torch.cat([dp["cfd_fields_nd"] for dp in dps], dim=0)

coord_mean = all_pts_nd.mean(dim=0)
coord_std  = all_pts_nd.std(dim=0).clamp(min=1e-6)
out_mean   = all_fields_nd.mean(dim=0)
out_std    = all_fields_nd.std(dim=0).clamp(min=1e-3)

# Extend coord stats: xyz (data-derived) + 5 params (fixed 0/1) + 2 wall features (from data)
_param_mean_ext = torch.zeros(N_PARAMS,  dtype=torch.float64)
_param_std_ext  = torch.ones( N_PARAMS,  dtype=torch.float64)

_y_all = all_pts_nd[:, 1]
_s1    = torch.sigmoid((_y_all - WALL_Y1) / WALL_EPS)
_s2    = torch.sigmoid((_y_all - WALL_Y2) / WALL_EPS)
_s1_mean = float(_s1.mean()); _s1_std = float(_s1.std().clamp(min=1e-3))
_s2_mean = float(_s2.mean()); _s2_std = float(_s2.std().clamp(min=1e-3))
print(f"  Wall sigmoid features: "
      f"s1 mean={_s1_mean:.3f} std={_s1_std:.3f}  "
      f"s2 mean={_s2_mean:.3f} std={_s2_std:.3f}")

coord_mean_net = torch.cat([coord_mean, _param_mean_ext,
                             torch.tensor([_s1_mean, _s2_mean], dtype=torch.float64)])
coord_std_net  = torch.cat([coord_std,  _param_std_ext,
                             torch.tensor([_s1_std,  _s2_std],  dtype=torch.float64)])

# Global PDE residual scales (depend only on global coord_std and out_std)
MOM_SCALE_X = 1.0 / float(coord_std[0])
MOM_SCALE_Y = 1.0 / float(coord_std[1])
MOM_SCALE_Z = 1.0 / float(coord_std[2])
DIV_SCALE   = float(max(out_std[0] / coord_std[0],
                        out_std[1] / coord_std[1],
                        out_std[2] / coord_std[2]))

print(f"\nGlobal stats  (from {N_DPS} DPs, {len(all_pts_nd):,} pts total)")
print(f"  coord_mean : {coord_mean.tolist()}")
print(f"  coord_std  : {coord_std.tolist()}")
print(f"  out_mean   : {out_mean.tolist()}")
print(f"  out_std    : {out_std.tolist()}")
print(f"  MOM_SCALE  : X={MOM_SCALE_X:.1f}  Y={MOM_SCALE_Y:.1f}  Z={MOM_SCALE_Z:.1f}")
print(f"  DIV_SCALE  : {DIV_SCALE:.1f}")

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

_log_path = os.path.join(RUN_PATH, "training.log")

class _Tee(io.TextIOBase):
    def __init__(self, stream, logfile):
        self._stream = stream; self._logfile = logfile
    def write(self, s):
        self._stream.write(s); self._logfile.write(s); return len(s)
    def flush(self):
        self._stream.flush(); self._logfile.flush()

_log_handle = open(_log_path, "w", buffering=1, encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__,  _log_handle)
sys.stderr  = _Tee(sys.__stderr__,  _log_handle)

print(f"Run path : {RUN_PATH}")
print(f"DPs      : {dp_names}")
print(f"Config   : h={HIDDEN_DIM}  l={N_LAYERS}  epochs={EPOCHS}")
print(f"           lr={LR}  lr_end={LR_END}  gamma={GAMMA:.6f}")
print(f"           n_train={N_TOTAL_TRAIN}/DP  n_sup={N_SUP}/DP  n_test={N_TEST}/DP")
print(f"           w_pde={W_PDE}  w_bc={W_BC}  w_sup={W_SUP}  w_sup_p={W_SUP_P}")
print(f"           optimizer steps/epoch = {N_DPS}  (one per DP, shuffled)")

if not DEBUG:
    api_key = "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"
    wandb.login(key=api_key)
    run = wandb.init(
        project=PROJECT,
        name=os.path.basename(RUN_PATH),
        config={
            "n_dps": N_DPS, "dp_names": dp_names,
            "epochs": EPOCHS, "lr": LR, "lr_end": LR_END,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "in_dim": IN_DIM,
            "seed": SEED, "n_train": N_TOTAL_TRAIN, "n_sup": N_SUP, "n_test": N_TEST,
            "w_pde": W_PDE, "w_bc": W_BC, "w_sup": W_SUP, "w_sup_p": W_SUP_P,
        },
    )

# ═══════════════════════════════════════════════════════════════
# MODEL + OPTIMIZER
# ═══════════════════════════════════════════════════════════════

net   = FFNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS).to(DEVICE).double()
model = NormalizedPINN(net, coord_mean_net, coord_std_net, out_mean, out_std)

opt_model = Adam(model.parameters(), lr=LR, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt_model, gamma=GAMMA)


def with_params(pts_xyz, params_nd_dp):
    """[N,3] → [N,8]: append pre-normalised param row. Detached so grad only flows through xyz."""
    row = params_nd_dp.detach().to(pts_xyz.device).unsqueeze(0).expand(pts_xyz.shape[0], -1)
    return torch.cat([pts_xyz, row], dim=1)


def safe_log10(x):
    v = x.item() if isinstance(x, torch.Tensor) else float(x)
    return float(np.log10(v)) if v > 0 else float("nan")


plot_epochs = set(np.linspace(0, EPOCHS - 1, 10, dtype=int).tolist())

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

start = time.time()

for epoch in range(EPOCHS):
    random.shuffle(dps)

    # Accumulate per-epoch averages across all DPs
    ep_losses = {k: 0.0 for k in [
        "div","mom_x","mom_y","mom_z","heat",
        "in_vx","in_vy","in_vz","in_T","out_p",
        "wall_vx","wall_vy","wall_vz","wall_T",
        "sup_vx","sup_vy","sup_vz","sup_p","sup_T",
        "pde","bc","sup","total"]}
    ep_mse = {dp["folder"]: {} for dp in dps}

    for dp in dps:
        # ── Sample collocation points ──────────────────────────
        coll_pts, coll_lbls = sample_collocation(
            dp["vol_pool"], dp["wall_pool"], n_vol_train, n_wall_train)
        coll_pts = coll_pts.clone()
        coll_pts[:, 0] -= dp["stl_buffer_mm"]
        coll_pts /= 1000.0   # mm → m

        _in_idx     = torch.randperm(len(dp["inlet_pool"][0]))[:n_inlet_train]
        inlet_pts   = dp["inlet_pool"][0][_in_idx]
        vx_inlet_bc = dp["inlet_pool"][1][_in_idx]
        vy_inlet_bc = dp["inlet_pool"][2][_in_idx]
        vz_inlet_bc = dp["inlet_pool"][3][_in_idx]
        _out_idx    = torch.randperm(len(dp["outlet_pool"]))[:n_outlet_train]
        outlet_pts  = dp["outlet_pool"][_out_idx]

        pts_xyz = torch.cat([coll_pts, inlet_pts, outlet_pts])
        pts_xyz.requires_grad_(True)
        lbls = torch.cat([
            coll_lbls,
            torch.ones( inlet_pts.shape[0],  dtype=torch.long),
            2 * torch.ones(outlet_pts.shape[0], dtype=torch.long),
        ])

        model.train()
        opt_model.zero_grad()

        fields = model(with_params(pts_xyz, dp["params_nd"]))
        derivs = compute_derivatives(fields, pts_xyz)

        (l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
         l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
         l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
        ) = compute_losses(
            *derivs, lbls,
            vx_inlet_bc, vy_inlet_bc, vz_inlet_bc,
            dp["dp_ctx"],
            DIV_SCALE, MOM_SCALE_X, MOM_SCALE_Y, MOM_SCALE_Z,
        )

        if dp["n_sup"] > 0:
            sup_f    = model(with_params(dp["sup_pts"], dp["params_nd"]))
            ctx      = dp["dp_ctx"]
            l_sup_vx = torch.mean((sup_f[:, 0] - dp["sup_vx"]) ** 2) / ctx["vx_var"]
            l_sup_vy = torch.mean((sup_f[:, 1] - dp["sup_vy"]) ** 2) / ctx["vy_var"]
            l_sup_vz = torch.mean((sup_f[:, 2] - dp["sup_vz"]) ** 2) / ctx["vz_var"]
            l_sup_p  = torch.mean((sup_f[:, 3] - dp["sup_p"])  ** 2) / ctx["p_var"]
            l_sup_T  = torch.mean((sup_f[:, 4] - dp["sup_T"])  ** 2) / ctx["t_var"]
        else:
            z = torch.zeros(1, dtype=torch.float64, device=DEVICE).squeeze()
            l_sup_vx = l_sup_vy = l_sup_vz = l_sup_p = l_sup_T = z

        l_pde   = l_div + l_mom_x + l_mom_y + l_mom_z + l_heat
        l_bc    = (l_inlet_vx + l_inlet_vy + l_inlet_vz + l_inlet_T + l_outlet_p
                   + l_wall_vx + l_wall_vy + l_wall_vz + l_wall_T)
        l_sup   = l_sup_vx + l_sup_vy + l_sup_vz + l_sup_p + l_sup_T
        l_total = W_PDE * l_pde + W_BC * l_bc + W_SUP * (l_sup - l_sup_p) + W_SUP_P * l_sup_p

        l_total.backward()
        opt_model.step()

        # ── Per-DP eval ────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred = model(with_params(dp["test_pts"], dp["params_nd"]))
        mse_vx = torch.mean((pred[:, 0] - dp["test_vx"]) ** 2)
        mse_vy = torch.mean((pred[:, 1] - dp["test_vy"]) ** 2)
        mse_vz = torch.mean((pred[:, 2] - dp["test_vz"]) ** 2)
        mse_p  = torch.mean((pred[:, 3] - dp["test_p"])  ** 2)
        mse_T  = torch.mean((pred[:, 4] - dp["test_T"])  ** 2)
        ep_mse[dp["folder"]] = {
            "vx": mse_vx.item(), "vy": mse_vy.item(), "vz": mse_vz.item(),
            "p":  mse_p.item(),  "T":  mse_T.item(),
        }

        # Accumulate epoch averages
        for k, v in zip(
            ["div","mom_x","mom_y","mom_z","heat",
             "in_vx","in_vy","in_vz","in_T","out_p",
             "wall_vx","wall_vy","wall_vz","wall_T",
             "sup_vx","sup_vy","sup_vz","sup_p","sup_T",
             "pde","bc","sup","total"],
            [l_div, l_mom_x, l_mom_y, l_mom_z, l_heat,
             l_inlet_vx, l_inlet_vy, l_inlet_vz, l_inlet_T, l_outlet_p,
             l_wall_vx, l_wall_vy, l_wall_vz, l_wall_T,
             l_sup_vx, l_sup_vy, l_sup_vz, l_sup_p, l_sup_T,
             l_pde, l_bc, l_sup, l_total],
        ):
            ep_losses[k] += v.item() / N_DPS

    scheduler.step()

    # ── Snap plots — all DPs at final epoch, min(1, N_DPS) otherwise ──
    if epoch in plot_epochs:
        snap_dps = dps
        for dp in snap_dps:
            snap_dir = os.path.join(RUN_PATH, f"snap_{epoch+1}_of_{EPOCHS}", dp["folder"])
            model.eval()
            with torch.no_grad():
                pred_snap = model(with_params(dp["snap_pts"], dp["params_nd"]))
            plot_fields(
                dp["snap_pts"].cpu().numpy(),
                [
                    ("vx", dp["snap_vx"].cpu().numpy() * dp["V_IN"],
                           pred_snap[:, 0].cpu().numpy() * dp["V_IN"]),
                    ("vy", dp["snap_vy"].cpu().numpy() * dp["V_IN"],
                           pred_snap[:, 1].cpu().numpy() * dp["V_IN"]),
                    ("vz", dp["snap_vz"].cpu().numpy() * dp["V_IN"],
                           pred_snap[:, 2].cpu().numpy() * dp["V_IN"]),
                    ("p",  (dp["snap_p"].cpu().numpy() * dp["P_DYN"] + dp["P_REF"]) / 1e5,
                           (pred_snap[:, 3].cpu().numpy() * dp["P_DYN"] + dp["P_REF"]) / 1e5),
                    ("T",  dp["snap_T"].cpu().numpy() * dp["DELTA_T"] + dp["T_WALL"],
                           pred_snap[:, 4].cpu().numpy() * dp["DELTA_T"] + dp["T_WALL"]),
                ],
                output_dir=snap_dir,
            )

    # ── Logging ────────────────────────────────────────────────
    avg_mse_total = np.mean([sum(m.values()) for m in ep_mse.values()])

    log = {
        "pde/divergence":  safe_log10(ep_losses["div"]),
        "pde/momentum_x":  safe_log10(ep_losses["mom_x"]),
        "pde/momentum_y":  safe_log10(ep_losses["mom_y"]),
        "pde/momentum_z":  safe_log10(ep_losses["mom_z"]),
        "pde/heat":        safe_log10(ep_losses["heat"]),
        "bc/inlet_vx":     safe_log10(ep_losses["in_vx"]),
        "bc/inlet_vy":     safe_log10(ep_losses["in_vy"]),
        "bc/inlet_vz":     safe_log10(ep_losses["in_vz"]),
        "bc/inlet_T":      safe_log10(ep_losses["in_T"]),
        "bc/outlet_p":     safe_log10(ep_losses["out_p"]),
        "bc/wall_vx":      safe_log10(ep_losses["wall_vx"]),
        "bc/wall_vy":      safe_log10(ep_losses["wall_vy"]),
        "bc/wall_vz":      safe_log10(ep_losses["wall_vz"]),
        "bc/wall_T":       safe_log10(ep_losses["wall_T"]),
        "sup/vx":          safe_log10(ep_losses["sup_vx"]),
        "sup/vy":          safe_log10(ep_losses["sup_vy"]),
        "sup/vz":          safe_log10(ep_losses["sup_vz"]),
        "sup/p":           safe_log10(ep_losses["sup_p"]),
        "sup/T":           safe_log10(ep_losses["sup_T"]),
        "loss/pde":        safe_log10(ep_losses["pde"]),
        "loss/bc":         safe_log10(ep_losses["bc"]),
        "loss/sup":        safe_log10(ep_losses["sup"]),
        "loss/total":      safe_log10(ep_losses["total"]),
        "eval/mse_total":  safe_log10(avg_mse_total),
        "train/lr":        scheduler.get_last_lr()[0],
    }
    for dp_name, mses in ep_mse.items():
        for field, val in mses.items():
            log[f"eval/{dp_name}/mse_{field}"] = safe_log10(val)
    log = {k: v for k, v in log.items() if not (isinstance(v, float) and np.isnan(v))}

    mse_str = "  ".join(
        f"{n}[vx={m['vx']:.2e} T={m['T']:.2e}]" for n, m in ep_mse.items()
    )
    print(
        f"\n\n[{epoch+1:>5}/{EPOCHS}]\n"
        f"PDE   div={ep_losses['div']:.3e}  mom_x={ep_losses['mom_x']:.3e}"
        f"  mom_y={ep_losses['mom_y']:.3e}  mom_z={ep_losses['mom_z']:.3e}"
        f"  heat={ep_losses['heat']:.3e}  | total={ep_losses['pde']:.3e}\n"
        f"BC    in_vx={ep_losses['in_vx']:.3e}  in_T={ep_losses['in_T']:.3e}"
        f"  out_p={ep_losses['out_p']:.3e}  wall_T={ep_losses['wall_T']:.3e}"
        f"  | total={ep_losses['bc']:.3e}\n"
        f"SUP   vx={ep_losses['sup_vx']:.3e}  p={ep_losses['sup_p']:.3e}"
        f"  T={ep_losses['sup_T']:.3e}  | total={ep_losses['sup']:.3e}\n"
        f"LOSS  total={ep_losses['total']:.3e}  lr={scheduler.get_last_lr()[0]:.3e}\n"
        f"MSE   {mse_str}"
    )

    if not DEBUG:
        wandb.log(log, step=epoch)

print(f"\nTraining done in {time.time() - start:.1f}s")
torch.save(model.state_dict(), os.path.join(RUN_PATH, "pinn_model.pt"))
# Save global normalization stats for inference without reloading data
torch.save({
    "coord_mean_net": coord_mean_net,
    "coord_std_net":  coord_std_net,
    "out_mean":       out_mean,
    "out_std":        out_std,
    "param_means":    PARAM_MEANS,
    "param_stds":     PARAM_STDS,
    "hidden_dim":     HIDDEN_DIM,
    "n_layers":       N_LAYERS,
    "in_dim":         IN_DIM,
}, os.path.join(RUN_PATH, "normalization.pt"))

# ═══════════════════════════════════════════════════════════════
# INFERENCE  (CPU, full domain, per DP)
# ═══════════════════════════════════════════════════════════════

torch.set_default_device("cpu")
net_inf   = FFNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=5, n_layers=N_LAYERS)
model_inf = NormalizedPINN(net_inf,
                            coord_mean_net.cpu(), coord_std_net.cpu(),
                            out_mean.cpu(),       out_std.cpu())
model_inf.load_state_dict(
    torch.load(os.path.join(RUN_PATH, "pinn_model.pt"),
               weights_only=True, map_location="cpu"))
model_inf.eval()

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

for dp in dps:
    print(f"\n{'─'*60}\nInference — {dp['folder']}")
    inf_dir = os.path.join(RUN_PATH, "inference", dp["folder"])
    os.makedirs(inf_dir, exist_ok=True)

    params_cpu = dp["params_nd"].cpu()
    cfd_pts_cpu = dp["cfd_pts_nd"].cpu()
    _pts_tensor = cfd_pts_cpu

    with torch.no_grad():
        pred_all = model_inf(with_params(_pts_tensor, params_cpu))

    V_IN   = dp["V_IN"];   P_DYN  = dp["P_DYN"];  P_REF  = dp["P_REF"]
    DT     = dp["DELTA_T"]; TW    = dp["T_WALL"]

    vx_pred = pred_all[:, 0].numpy() * V_IN
    vy_pred = pred_all[:, 1].numpy() * V_IN
    vz_pred = pred_all[:, 2].numpy() * V_IN
    p_pred  = (pred_all[:, 3].numpy() * P_DYN + P_REF) / 1e5
    T_pred  = pred_all[:, 4].numpy() * DT + TW

    # Recover physical CFD data for RMSE
    cfd_fields = dp["cfd_fields_nd"].cpu()
    vx_true = cfd_fields[:, 0].numpy() * V_IN
    vy_true = cfd_fields[:, 1].numpy() * V_IN
    vz_true = cfd_fields[:, 2].numpy() * V_IN
    p_true  = (cfd_fields[:, 3].numpy() * P_DYN + P_REF) / 1e5
    T_true  = cfd_fields[:, 4].numpy() * DT + TW

    tr_np = dp["train_idx"].cpu().numpy()
    te_np = dp["test_idx"].cpu().numpy()
    for true_arr, pred_arr, name in [
        (vx_true, vx_pred, "vx"), (vy_true, vy_pred, "vy"), (vz_true, vz_pred, "vz"),
        (p_true,  p_pred,  "p"),  (T_true,  T_pred,  "T"),
    ]:
        print(f"  {name}  train RMSE={rmse(true_arr[tr_np], pred_arr[tr_np]):.4e}"
              f"  test RMSE={rmse(true_arr[te_np], pred_arr[te_np]):.4e}")

    pts_np   = cfd_pts_cpu.numpy()
    idx_plot = np.random.choice(len(pts_np), min(100_000, len(pts_np)), replace=False)
    plot_fields(
        pts_np[idx_plot],
        [
            ("vx", vx_true[idx_plot], vx_pred[idx_plot]),
            ("vy", vy_true[idx_plot], vy_pred[idx_plot]),
            ("vz", vz_true[idx_plot], vz_pred[idx_plot]),
            ("p",  p_true[idx_plot],  p_pred[idx_plot]),
            ("T",  T_true[idx_plot],  T_pred[idx_plot]),
        ],
        output_dir=inf_dir,
    )

_log_handle.close()
