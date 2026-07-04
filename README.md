# AI-Driven Optimization for GT Blade Cooling

## Introduction

Gas turbine blades operate under extreme thermal conditions, requiring efficient cooling strategies to prevent failure. One approach is using turbulated internal cooling channels, which enhance heat transfer but also increase pressure losses. This project focuses on optimizing these geometries using AI-driven learning to maximize cooling while minimizing pressure drop. Traditional Computational Fluid Dynamics (CFD) simulations are used for design evaluation, but they require weeks of computation. This project proposes a Physics-Informed Neural Network (PINN)-based surrogate model to significantly accelerate optimization, reducing computation time from weeks to minutes while maintaining accuracy.

Full write-up and slides: [`Report_LAMD_Corbel.pdf`](Report_LAMD_Corbel.pdf) · [`Slide_LAMD_Corbel.pdf`](Slide_LAMD_Corbel.pdf)

---

## Design Space

The surrogate models are trained over a parametric space of **55 design points** (dp00–dp50 for training and dp102–dp105 for testing), each defined by 5 geometric and flow parameters:

| Parameter | Symbol | Range |
|-----------|--------|-------|
| Aspect ratio | AR | 4 – 15 |
| Rib height-to-hydraulic diameter | e/Dh [%] | 5 – 20 |
| Rib pitch-to-height | P/e | 5 – 15 |
| Rib attack angle | α [°] | 30 – 75 |
| Reynolds number | Re | 20 000 – 200 000 |

The channel is a three-pass serpentine geometry (two 180° U-bends). 

---

## Repository Structure

```
.
├── src/
│   ├── config.py               # shared constants: DP_CONFIGS, physics, geometry, HTC planes
│   ├── models.py               # neural architectures: FFNN, NormalizedPINN, QPP_MLP
│   ├── utils.py                # _Tee logger, build_mlp_input, all plot/table functions
│   ├── run_pinn.py             # PINN training (multi-DP, W&B sweep compatible)
│   ├── run_qpp_mlp.py          # QPP-MLP training (multi-DP, W&B sweep compatible)
│   ├── run_pinn_isothermal.py  # single-DP PINN for the isothermal dp11 benchmark (separate pipeline, see below)
│   ├── infer_pinn.py           # PINN standalone inference with CFD ground-truth comparison
│   ├── infer_qpp_mlp.py        # QPP-MLP standalone inference with CFD comparison
│   ├── compute_htc.py          # HTC = q'' / (Tw - Tb) from pretrained PINN + MLP
│   ├── preProcessedData/
│   │   ├── With_T/dpXX/         # main 55-DP serpentine dataset (see Data Format)
│   │   ├── Inference/dpXX/      # held-out DPs (dp102-dp105) used for inference/HTC
│   │   └── isothermal/dp11/     # single straight-pass geometry, T constant everywhere (see below)
│   ├── pools/                  # cached geometry sampling pools (auto-generated, see below)
│   └── batch/                  # SLURM batch scripts and W&B sweep configs
│
├── best_PINN/                    # best trained PINN weights + normalization
├── best_MLP/                     # best trained QPP-MLP weights
├── isothermal_PINN/               # isothermal benchmark, trained with supervised + physics loss
├── isothermal_PINN_no_sup/        # isothermal benchmark, trained with physics loss only (no CFD supervision)
├── HTC_pred/               # HTC prediction outputs (plots, CSVs)
├── logs/                   # SLURM stdout logs
└── pinn27_sweep_runs/      # W&B sweep trial outputs
    qpp_mlp_sweep_runs/
```

---

## Models

### PINN : Physics-Informed Neural Network

Predicts the **5 volumetric flow fields** simultaneously from spatial coordinates and design parameters:

- Output: (vx, vy, vz, p, T) : all non-dimensionalized
- Input: (x, y, z) + 5 z-scored design params + 2 wall sigmoid features → 10D
- Architecture: fully connected net with Sin activations (FFNN), wrapped in `NormalizedPINN` for automatic input/output normalization
- Best config: h=128, 6 layers, trained on 10 000 collocation pts/DP

The PDE residuals (incompressible NS + energy), inlet/outlet BCs, and wall BCs are all enforced as soft loss terms alongside supervised data from CFD snapshots.

> **GPU note:** the PINN is float64 throughout. Training runs fine on CUDA, but running the trained PINN in float64 on the IZAR GPU at **inference** time causes numerical overflow (outputs blow up to values like `1e330` / NaN). Because of this, `run_pinn.py`'s post-training inference pass, `infer_pinn.py`, and the PINN forward pass in `compute_htc.py` all force the model onto CPU regardless of `--device`. The QPP-MLP is unaffected and runs on CUDA normally.

### QPP-MLP : Wall Heat Flux Surrogate

Predicts **q'' [W/m²]** at wall surface points from 2D wall coordinates and design parameters:

- Output: q'' (scalar, normalized)
- Input: (x, y) + 5 z-scored design params + 2 wall sigmoid features → 9D
  - z is dropped: the bottom wall is nearly flat, so z carries no information
- Architecture: fully connected net with Sin activations (QPP_MLP)
- Best config: h=128, 8 layers, lr=5.94e-4, trained across all 51 DPs

### HTC Computation

The Heat Transfer Coefficient is assembled as:

```
HTC(x,y) = q''(x,y) / (T_wall - T_bulk(x,y))
```

where:
- `q''` comes from the QPP-MLP
- `T_bulk` is interpolated from PINN-predicted temperatures at 6 cross-sectional planes using linear (straight passes) and arcsin (bends) interpolation

Normalised Nusselt number: `Nu_norm = HTC · Dh / (k_f · Nu_0)` with `Nu_0 = 0.023 Re^0.8 Pr^0.3`.

---

## Isothermal Single-DP Benchmark (`run_pinn_isothermal.py`)

Before tackling the full 55-point parametric PINN, we first validated the PINN approach on a single, much simpler geometry: a plain straight rib pass instead of the full 3-pass serpentine, to check that the PDE/BC loss formulation converges at all before adding the design-parameter conditioning and multi-DP complexity on top. `preProcessedData/isothermal/dp11/` is that benchmark; it is **not** part of the 55-point `DP_CONFIGS` family above, despite the folder name, it's a different, much smaller geometry:

| | Main dataset (`With_T/dp11`) | Isothermal benchmark (`isothermal/dp11`) |
|---|---|---|
| Geometry | 3-pass serpentine, two 180° bends | single straight rib pass |
| Domain | x∈[0, 1.135] y∈[0, 0.545] z∈[0, 0.038] m | x∈[0, 0.953] y∈[0, 0.177] z∈[0, 0.022] m |
| Mesh | `dp11_AR4.379_...stl` (347 KB) | `Baseline_ML4Science.stl` (15 KB) |
| Temperature | T_inlet=329.0 K, T_wall=293.15 K | T ≡ 298.15 K everywhere (std ~1e-13 K,float noise) |

Because T carries no signal at all, `run_pinn.py`'s θ = (T − T_wall)/(T_inlet − T_wall) non-dimensionalization divides by ~0. `run_pinn_isothermal.py` is a dedicated sibling script that instead z-scores T directly from data (mean/std, with a 1 K floor),the same way vx/vy/vz/p already are,and reconstructs real Kelvin from that normalized output before evaluating Sutherland's law, so the momentum equation's viscosity term stays physically correct even though T is effectively constant.

It also drops what doesn't apply to this single, non-parametric geometry:
- no 5 design-param + 2 wall-sigmoid input features (`in_dim=3`: x, y, z only),this domain isn't part of the DP_CONFIGS sweep and has no internal partition walls to distinguish
- no ±420 mm STL inlet/outlet buffer trim,this mesh's extent already matches the CFD data exactly
- no collocation-pool disk cache,the mesh is tiny, so volume/wall points are resampled fresh in memory every run instead of being cached under `src/pools/`

Everything else (pool-based collocation sampling, PDE/BC loss structure, `_Tee`/`plot_fields` from `utils.py`, `FFNN` from `models.py`, CLI conventions, W&B sweep compatibility) mirrors `run_pinn.py`.

### With vs. without supervised data

Since this benchmark exists to probe the PINN formulation itself, we run it two ways to see how much the physics/BC losses alone can do versus adding CFD-supervised points on top:

| | Supervised + physics | Physics only |
|---|---|---|
| Batch script | `run_pinn_isothermal.batch` | `run_pinn_isothermal_no_sup.batch` |
| Key flag | `--n-sup 500` | `--n-sup 0` (supervised loss terms are exactly 0) |
| W&B project | `PINN_isothermal` | `PINN_isothermal_no_sup` |
| Output dir | `isothermal_PINN/` | `isothermal_PINN_no_sup/` |


```bash
# Supervised + physics
python run_pinn_isothermal.py \
    --project    PINN_isothermal \
    --hidden-dim 20 \
    --n-layers   4 \
    --epochs     6000 \
    --lr         3e-3 \
    --lr-end     1e-4 \
    --n-train    5000 \
    --n-sup      500 \
    --run-path   ../isothermal_PINN

# Physics only (no CFD supervision)
python run_pinn_isothermal.py \
    --project    PINN_isothermal_no_sup \
    --hidden-dim 20 \
    --n-layers   4 \
    --epochs     6000 \
    --lr         3e-3 \
    --lr-end     1e-4 \
    --n-train    5000 \
    --n-sup      0 \
    --run-path   ../isothermal_PINN_no_sup
```

Or on SLURM:

```bash
sbatch src/batch/run_pinn_isothermal.batch          # saves to isothermal_PINN/
sbatch src/batch/run_pinn_isothermal_no_sup.batch   # saves to isothermal_PINN_no_sup/
```


---

## Workflow

```
CFD data (npy)
      │
      ├──► run_pinn.py      → best_PINN/pinn_model.pt
      │                              normalization.pt
      │
      ├──► run_qpp_mlp.py   → best_MLP/qpp_mlp_best.pt
      │                              qpp_mlp_final.pt
      │
      └──► compute_htc.py ◄─── both models
                   │
                   └──► HTC_pred/  (plots, epsilon_table.csv,
                                    Nu_section_table.csv)
```

---

## Geometry Pools (Collocation Point Cache)

For each DP, `run_pinn.py` samples volume and wall collocation points from the STL mesh and saves them to `src/pools/<dpXX>/pool_np<N>_fv<fv>_fw<fw>_wf<wf>_dh<dh>_dv<dv>.pt`. On subsequent runs : including all sweep trials : the file is loaded directly if it exists, skipping the expensive trimesh sampling step.

The filename encodes all sampling parameters (`--n-pool`, `--pool-frac-vol`, `--pool-frac-wall`, wall fraction, horizontal/vertical deltas), so changing any of those automatically triggers a rebuild. This was key to making hyperparameter sweeps fast.

---

## Data Format

Each DP folder (`preProcessedData/With_T/dpXX/`) contains:

| File | Shape | Description |
|------|-------|-------------|
| `vel_x.npy` | (N, 4) | x, y, z, vx [m/s] |
| `vel_y.npy` | (N, 4) | x, y, z, vy [m/s] |
| `vel_z.npy` | (N, 4) | x, y, z, vz [m/s] |
| `press.npy` | (N, 4) | x, y, z, p [Pa] |
| `temp.npy` | (N, 4) | x, y, z, T [K] |
| `vel_x_inlet.npy` | (M, 4) | inlet face vx (for V_IN) |
| `qpp.npy` | (W, 4) | x, y, z, q'' [W/m²] : wall surface |
| `*.stl` | : | geometry mesh (for collocation sampling) |

The original CFD exports (`Vel_x_V_ribs_*.csv`, `Abs_p_V_ribs_*.csv`, `heat_flux_V_ribs_*.csv`, `Temp_V_ribs_*.csv`, `vel_x_inlet.csv`, …) are also present in each folder but are **not read by any script** : they were the source files used to generate the `.npy` files (via `preprocess.ipynb` / `preprocess_inference.py`) and are kept for reference only.

The one exception: inference DPs (`preProcessedData/Inference/dpXX/`) contain `HTC{1-5}_V_dpXX.csv` which **are** read by `compute_htc.py` for ground-truth HTC comparison : these have no `.npy` equivalent.

---

## Running

All scripts run from `src/` with `python <script>.py --args`.

### Train PINN

```bash
python run_pinn.py \
    --project    PINN27 \
    --hidden-dim 128 \
    --n-layers   6 \
    --epochs     5000 \
    --lr         3.6774e-3 \
    --lr-end     1e-4 \
    --n-train    10000 \
    --n-sup      500 \
    --run-path   ../best_PINN
```

### Train QPP-MLP

```bash
python run_qpp_mlp.py \
    --project      QPP_MLP \
    --hidden-dim   128 \
    --n-layers     8 \
    --lr           5.94e-4 \
    --lr-end       3.46e-4 \
    --weight-decay 1.32e-4 \
    --epochs       1000 \
    --patience     300 \
    --run-path     ../best_MLP
```

### PINN Inference

```bash
python infer_pinn.py \
    --run-path  ../best_PINN \
    --dps       dp102 dp103 dp104 dp105 \
    --data-root ./preProcessedData/Inference
```

### QPP-MLP Inference

```bash
python infer_qpp_mlp.py \
    --checkpoint ../best_MLP/qpp_mlp_final.pt \
    --data-dir   ./preProcessedData/Inference \
    --dps        dp102 dp103 dp104 dp105
```

### Compute HTC

```bash
python compute_htc.py \
    --pinn-path  ../best_PINN \
    --mlp-ckpt   ../best_MLP/qpp_mlp_final.pt \
    --data-dir   ./preProcessedData/Inference \
    --dps        dp102 dp103 dp104 dp105 \
    --out-dir    ../HTC_pred
```

---

## SLURM (IZAR)

All batch scripts are in `src/batch/`. Submit from the repo root or `src/`:

```bash
# Train PINN (best hyperparameters hardcoded)
sbatch src/batch/run_pinn.batch

# Train QPP-MLP
sbatch src/batch/run_qpp_mlp.batch

# Train PINN on the isothermal dp11 benchmark (separate single-DP pipeline, see above)
sbatch src/batch/run_pinn_isothermal.batch          # supervised + physics
sbatch src/batch/run_pinn_isothermal_no_sup.batch   # physics only, no CFD supervision

# Run PINN inference on test DPs
sbatch src/batch/slurm_infer_pinn.batch

# Run QPP-MLP inference
sbatch src/batch/slurm_infer_qpp_mlp.batch

# Compute HTC
sbatch src/batch/slurm_compute_htc.batch
```

SLURM logs go to `logs/slurm_<jobid>.log`. Scripts that write their own log (training, HTC) also save a `training.log` / `compute_htc.log` in their output directory.

### W&B Hyperparameter Sweeps

```bash
# PINN : create a sweep (once), then submit N parallel agents
cd src
wandb sweep batch/sweep_config.yaml                            # prints <SWEEP_ID>
sbatch --export=ALL,SWEEP_ID=<SWEEP_ID> batch/run_pinn_sweep.batch

# QPP-MLP : same pattern
wandb sweep batch/sweep_config_qpp_mlp.yaml                    # prints <SWEEP_ID>
sbatch --export=ALL,SWEEP_ID=<SWEEP_ID> batch/run_qpp_mlp_sweep.batch
```

Both sweep batch scripts are SLURM array jobs (`--array=1-20` by default); each array task runs one `wandb agent --count 1` trial. The PINN sweep (`sweep_config.yaml`) uses Bayesian optimization over hidden_dim, n_layers, n_train, epochs, lr, lr_decay_ratio, minimizing `loss/total`. The QPP-MLP sweep (`sweep_config_qpp_mlp.yaml`) sweeps hidden_dim, n_layers, lr, lr_end, weight_decay (epochs/patience fixed at 1000/300), minimizing `best_test_mse_nd`. Trial outputs land in `pinn27_sweep_runs/<sweep_id>/` and `qpp_mlp_sweep_runs/<sweep_id>/`.

---

## Environment

Conda environment: `blade-cooling`

| Package | Version |
|---------|---------|
| PyTorch | 2.5.1+cu118 |
| NumPy | 2.0.2 |
| trimesh | 4.5.3 |
| wandb | 0.27.1 |
| matplotlib | 3.9.3 |
| scipy | 1.17.1 |

Create it from [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate blade-cooling
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `pinn_model.pt` | `best_PINN/` | PINN weights |
| `normalization.pt` | `best_PINN/` | coord/output normalization stats |
| `training.log` | `best_PINN/` or `best_MLP/` | full stdout log of training |
| `qpp_mlp_best.pt` | `best_MLP/` | best checkpoint (early stopping) |
| `qpp_mlp_final.pt` | `best_MLP/` | final epoch checkpoint |
| `pinn_model.pt` | `isothermal_PINN/` or `isothermal_PINN_no_sup/` | isothermal dp11 benchmark PINN weights (with / without CFD supervision) |
| `normalization.pt` | `isothermal_PINN/` or `isothermal_PINN_no_sup/` | coord/output normalization stats (incl. T z-score mean/std) |
| `epsilon_table.csv` | `HTC_pred/` | ε_θ, ε_p, ε_vx, ε_Nu per DP |
| `Nu_section_table.csv` | `HTC_pred/` | Nu_norm CFD vs PINN per section |
| `compute_htc.log` | `HTC_pred/` | full stdout log of HTC computation |
