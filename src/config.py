import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════
# PHYSICAL FLUID CONSTANTS  (air)
# ═══════════════════════════════════════════════════════════════

M          = 28.96e-3    # molar mass [kg/mol]
R          = 8.314       # universal gas constant [J/(mol·K)]
K          = 2.61e-2     # thermal conductivity [W/(m·K)]
CP         = 1.00e3      # specific heat [J/(kg·K)]
T_REF_SUTH = 278.15      # Sutherland reference temperature [K]
MU_REF     = 1.716e-5   # reference dynamic viscosity [Pa·s]
S_SUTH     = 110.4       # Sutherland constant [K]

# ── Fluid inference constants ────────────────────────────────────
K_FLUID  = 2.61e-2   # W/(m·K)
PR_FLUID = 0.71       # Prandtl number (fixed, air)

# ═══════════════════════════════════════════════════════════════
# PARAMETRIC INPUT NORMALIZATION  (anchors from the full 51-DP space)
# ═══════════════════════════════════════════════════════════════

PARAM_NAMES = ["AR",   "e/Dh",  "P/e",  "alpha",   "Re"    ]
PARAM_MEANS = torch.tensor([ 9.0,   0.127,  10.0,   52.0,  108000.0], dtype=torch.float64)
PARAM_STDS  = torch.tensor([ 3.5,   0.048,   3.1,   14.5,   58000.0], dtype=torch.float64)

# ═══════════════════════════════════════════════════════════════
# WALL SIGMOID FEATURES  (same y positions for ALL DPs)
# ═══════════════════════════════════════════════════════════════

WALL_Y1  = 0.185   # [m] — internal wall at y=0.185 m
WALL_Y2  = 0.375   # [m] — internal wall at y=0.375 m
WALL_EPS = 0.002   # [m] sigmoid width

# ═══════════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════════

# Inlet/outlet buffer: 420 mm of the STL is non-physical inlet/outlet extension.
# All q'' and volume data outside [x_min+BUFFER_M, x_max-BUFFER_M] is trimmed.
BUFFER_M = 0.420   # [m]

# ── SolidWorks geometry (fixed across all DPs) ──────────────────
BETA1     = np.deg2rad(-0.15)   # taper angle pass 1
BETA2     = np.deg2rad(-0.40)   # taper angle pass 3
L_HALF_MM = 567.5               # half-length [mm] (= 1135/2)
W1MID_MM  = 177.0 + np.tan(BETA1) * L_HALF_MM   # ≈ 175.51 mm
W3MID_MM  = 157.0 - np.tan(BETA2) * L_HALF_MM   # ≈ 160.96 mm
W2MID_MM  = (544.75 - W1MID_MM - W3MID_MM
             - np.cos(BETA1)*20 - np.cos(BETA2)*20)   # ≈ 168.27 mm
W_TOT_MM  = W1MID_MM + W2MID_MM + W3MID_MM            # ≈ 504.75 mm

# ── Fixed thermal BCs (used in analytical_scales only) ──────────
T_STD         = 293.15    # K  — standard conditions for Re definition
P_STD         = 101325.0  # Pa
T_WALL_FIXED  = 293.15    # K
T_INLET_FIXED = 329.0     # K
DELTA_T_FIXED = T_INLET_FIXED - T_WALL_FIXED   # K

# ═══════════════════════════════════════════════════════════════
# HTC T-PLANE CONSTANTS
# ═══════════════════════════════════════════════════════════════

# 6 cross-sectional T-planes: (x_pos, y_min, y_max)
# T_pl[0]=Tin, [1]=end pass1, [2]=start pass2, [3]=end pass2,
# [4]=start pass3, [5]=Tout
T_PLANES = [
    (0.0000, 0.00, 0.18),   # 0  Tin
    (0.9395, 0.00, 0.18),   # 1  end pass 1
    (0.9395, 0.18, 0.38),   # 2  start pass 2
    (0.1870, 0.18, 0.38),   # 3  end pass 2
    (0.1870, 0.38, 0.55),   # 4  start pass 3
    (1.1320, 0.38, 0.55),   # 5  Tout
]
PLANE_NAMES = ["Tin", "T5 (end P1)", "T6 (start P2)", "T10 (end P2)", "T11 (start P3)", "Tout"]

X_TOP_BEND = 0.9395
X_BOT_BEND = 0.187
Y_P12      = 0.18
Y_P23      = 0.38
CY_TOP     = 0.183   # arc centre y for top bend
CY_BOT     = 0.37    # arc centre y for bottom bend
X_TOUT     = 1.132
L_PASS3    = X_TOUT - X_BOT_BEND   # 0.945 m

PLANE_TOL = 6e-3   # ±6 mm tolerance for x-slice selection

SECTION_LABELS = {
    1: "Pass 1",
    2: "Top Bend",
    3: "Pass 2",
    4: "Bottom Bend",
    5: "Pass 3",
}

# inlet / outlet T_pl indices for each section (Pass1→[0,1], TopBend→[1,2], etc.)
SEC_TPLANE = {1: (0, 1), 2: (1, 2), 3: (2, 3), 4: (3, 4), 5: (4, 5)}

# ═══════════════════════════════════════════════════════════════
# DESIGN-POINT REGISTRY  (dp00–dp50 + dp102–dp105)
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

DP_REGISTRY = {c["folder"]: c for c in DP_CONFIGS}

# ═══════════════════════════════════════════════════════════════
# HYDRAULIC DIAMETERS [m]  (from CFD geometry, per DP and section)
# ═══════════════════════════════════════════════════════════════
# Keys: 1=Pass1, 2=TopBend, 3=Pass2, 4=BotBend, 5=Pass3

DH_REGISTRY = {
    "dp102": {1: 0.034410999, 2: 0.034351457, 3: 0.034291914,
              4: 0.034226677, 5: 0.034161440},
    "dp103": {1: 0.066676338, 2: 0.066441216, 3: 0.066206093,
              4: 0.065949791, 5: 0.065693489},
    "dp104": {1: 0.023108490, 2: 0.023082047, 3: 0.023055604,
              4: 0.023026582, 5: 0.022997560},
    "dp105": {1: 0.030023788, 2: 0.029978735, 3: 0.029933682,
              4: 0.029884287, 5: 0.029834892},
}

# ═══════════════════════════════════════════════════════════════
# ANALYTICAL SCALES  (for inference without CFD data)
# ═══════════════════════════════════════════════════════════════

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
