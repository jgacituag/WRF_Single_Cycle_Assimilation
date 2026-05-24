# WRF Single-Cycle Assimilation

Radar data assimilation experiments with a real cases WRF ensemble,
using the Local Ensemble Transform Kalman Filter (LETKF) and likelihood
tempering (TEnKF).
---

## Repository layout

```
.
├── src/
│   ├── da/
│   │   ├── metrics.py
│   │   └── core.py                  # All DA methods (LETKF, TEnKF, AOEI)
│   ├── runners/
│   │   └── run_experiment.py        # Unified runner for all experiment modes
│   ├── extract_3d_subset.py         # Extract WRF ensemble subsets to .npz
│   ├── build_fortran.sh             # Compiles the Fortran LETKF module
│   ├── fortran/                     # Fortran LETKF source and compiled .so
│   ├── queue_ws.sh                  # PBS script for sweep / single_obs mode
│   ├── queue_ws2.sh                 # PBS script for legacy strided experiment
│   └── queue_multiobs.sh            # PBS script for multi_obs mode
├── configs/
│   ├── template.yaml                # Full reference template — start here
│   ├── build_3D_section_wrfout.yaml # Data extraction from raw wrfout files
│   ├── build_3D_section_post.yaml   # Data extraction from post-processed files
│   ├── ws_sweep_test.yaml           # Sweep experiment (stride 20, Nt 1–10)
│   ├── ws_multiobs_<HHMM>.yaml      # Per-sweep multi_obs configs
│   └── ws_sweep_<HHMM>.yaml         # Per-sweep sweep configs
├── Notebooks/
│   ├── S0_Test.ipynb
│   ├── S1_Explore_and_extract_3d_sections_WRF.ipynb
│   ├── S2_obs_explorer_ws2.ipynb
│   ├── S3_sweep_diagnostics_Final.ipynb       # Main diagnostic notebook
│   ├── S3_Explore_output_single_obs.ipynb
│   ├── S3_Explore_output_Multy_Obs.ipynb
│   ├── Plot_Evaluate_output_multiple_obs.ipynb
│   ├── Plot_Evaluate_3D_multiple_obs.ipynb
│   └── collector.ipynb
├── test/
│   ├── run_sanity_check.py
│   └── test_da_core.py
├── data/                            # Input subsets and output results (git-ignored)
├── logs/                            # Runtime logs from PBS jobs
├── REPO_STRUCTURE.md
└── environment.yml
```

---

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate intermediate_exp
```

### 2. Build the Fortran LETKF module

```bash
bash src/build_fortran.sh
```

This compiles `cletkf_wloc` via `f2py` and places the `.so` in `src/fortran/`.
All runners add that path to `sys.path` automatically.
The PBS queue scripts re-run this step on each compute node before launching Python.

---

## Data preparation

Before running any experiment you need to extract the 3D WRF ensemble subset
from the raw `wrfout` files or post-processed output.

**Interactive** — open `Notebooks/S1_Explore_and_extract_3d_sections_WRF.ipynb`
and follow the four steps: choose region → visualise → extract → sanity check.

**Command line** — once the appropriate config is configured:

```bash
# From raw wrfout files:
python src/extract_3d_subset.py --config configs/build_3D_section_wrfout.yaml

# From post-processed files:
python src/extract_3d_subset.py --config configs/build_3D_section_post.yaml
```

The output is a compressed `.npz` file with the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `state_ensemble` | `(nx, ny, nz, Ne, 8)` | All members |
| `pos_km` | `(nx, ny, nz, 3)` | Position [x_km, y_km, z_km] from domain corner |

Variable order in the last axis of `state_ensemble`:

| Index | Variable | Units |
|-------|----------|-------|
| 0 | QGRAUP | kg/kg |
| 1 | QRAIN  | kg/kg |
| 2 | QSNOW  | kg/kg |
| 3 | T (temperature) | K |
| 4 | P (pressure) | Pa |
| 5 | UA (u-wind) | m/s |
| 6 | VA (v-wind) | m/s |
| 7 | WA (w-wind) | m/s |

---

## DA methods

All methods live in `src/da/core.py`.

| Method | Config name | Description |
|--------|-------------|-------------|
| LETKF | `TEnKF` (Nt=1) | Standard single-step LETKF — backward-compat alias |
| TEnKF | `TEnKF` | Tempered LETKF — H(x) recomputed at each tempering step |
| AOEI | `AOEI` | LETKF + Adaptive Observation Error Inflation (single step) |

### Tempering schedule

Weights follow:

```
alpha_i = exp(-(Nt+1)*alpha_s / i) / sum_j exp(-(Nt+1)*alpha_s / j)
```

`sum(alpha_i) = 1` guarantees total information across all steps equals
`R0` (information-preserving property). Larger `alpha_s` back-loads weight
toward later iterations; `alpha_s = 0` gives equal weights.

### Localization

R-localization (Greybush et al. 2011). The Fortran inflates observation error
by `exp(0.5*(d/L)^2)` at distance `d` from a grid point with scale `L`.
Distances and scales are in km, computed from `pos_km`.
Set `loc_x/y/z: null` to disable localization on an axis.

---

## Running experiments

### 1. Sanity check first

Before any full run, verify all methods work correctly on a single point:

```bash
python test/run_sanity_check.py --config configs/test_sanity.yaml \
    --truth 0 --x 10 --y 0 --z 15
```

### 2. Observation modes

The mode is set in `sweep.obs_points.mode` in the config. Three modes are available:

| Mode | Description | Output |
|------|-------------|--------|
| `single_obs` | One fixed obs point, all method combos | One `.npz` per run |
| `sweep` | Every QC-passing stride point, each as an independent single-obs | One `.npz` per truth member |
| `multi_obs` | All QC-passing stride points assimilated jointly (one Fortran call per combo) | One `.npz` per combo + one ref file per truth member |

### 3. Run locally

```bash
# Single truth member, sequential:
python src/runners/run_experiment.py --config configs/ws_sweep_test.yaml --tm 0

# With parallel workers (sweep mode, Linux only):
python src/runners/run_experiment.py --config configs/ws_sweep_test.yaml --tm 0 --workers 46

# multi_obs mode — set OMP_NUM_THREADS instead of --workers:
export OMP_NUM_THREADS=46
python src/runners/run_experiment.py --config configs/ws_multiobs_1800.yaml --tm 0
```

### 4. Submit to cluster (PBS)

**Sweep / single_obs** — one job per truth member, parallelises with `--workers`:

```bash
# Single truth member:
qsub -v CONFIG=configs/ws_sweep_test.yaml,TM=0 src/queue_ws.sh

# All truth members:
for tm in $(seq 0 59); do
    qsub -v CONFIG=configs/ws_sweep_test.yaml,TM=$tm src/queue_ws.sh
done
```

**Multi-obs** — one job per truth member, parallelises via OpenMP inside Fortran:

```bash
for tm in $(seq 0 59); do
    qsub -v CONFIG=configs/ws_multiobs_1800.yaml,TM=$tm src/queue_multiobs.sh
done
```

Optional override variables: `CONFIG` (path to yaml), `TM` (truth member index, required), `WORKERS` (sweep only, default = N_CORES − 2).

---

## Output files

A copy of the config yaml is written to `outdir` before any results, so
every output folder is self-contained.

### Sweep mode (one file per truth member)

```
{tag}_sweep_Ne{ne:03d}_tm{tm:02d}.npz
```

Loaded with `np.load(..., allow_pickle=True)['arr_0']` → a NumPy array of
row dicts, one per QC-passing observation point. Each row contains:

**Obs-space point metrics:**

| Key | Description |
|-----|-------------|
| `yo` | Noisy observation [dBZ] |
| `yo_clean` | Noise-free truth H(x) [dBZ] |
| `dep_b` | Prior innovation `yo − H(x̄ᶠ)` [dBZ] |
| `dep_a` | Posterior residual `yo − H(x̄ᵃ)` [dBZ] |
| `hxf_mean_obs` | Prior ensemble mean at obs point [dBZ] |
| `hxa_mean_obs` | Posterior ensemble mean at obs point [dBZ] |
| `inc_obs` | Analysis increment at obs point [dBZ] |
| `spread_f_obs` | Prior ensemble spread at obs point [dBZ] |
| `spread_a_obs` | Posterior ensemble spread at obs point [dBZ] |
| `rmse_f/a_point_obs` | Abs error at obs point (prior/posterior) |
| `rmse_f/a_w_obs` | Weighted RMSE in obs space over localization volume |
| `rmse_f/a_u_obs` | Unweighted RMSE in obs space over localization volume |
| `loc_weights_sum` | Sum of localization weights |
| `n_updated` | Number of grid points updated (rloc > 0) |
| `precip_fraction_f` | Fraction of updated points with H(x̄ᶠ) > 0 |
| `hx_dbz_local_mean_w/u` | Weighted/unweighted mean reflectivity in localization volume |

**Per state variable** (for each var in `{qg, qr, qs, T, P, u, v, w}`):

| Key pattern | Description |
|-------------|-------------|
| `rmse_f/a_point_{var}` | Abs error at obs point |
| `rmse_f/a_w_{var}` | Weighted RMSE over localization volume |
| `rmse_f/a_u_{var}` | Unweighted RMSE over localization volume |
| `spread_f/a_point_{var}` | Ensemble spread at obs point |
| `spread_f/a_w_{var}` | Weighted ensemble spread over localization volume |
| `spread_f/a_u_{var}` | Unweighted ensemble spread over localization volume |

### Multi-obs mode

**Analysis file** (one per method combo):
```
{tag}_multi_obs_{method}_Nt{nt:02d}_as{alpha_s}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne:03d}_tm{tm:02d}.npz
```

| Key | Description |
|-----|-------------|
| `xa` | Posterior ensemble `(nx, ny, nz, Ne, nvar)` |
| `hxf_mean_field` | Prior ensemble-mean reflectivity field |
| `hxa_mean_field` | Posterior ensemble-mean reflectivity field |
| `residual_field` | `hxa_mean − truth_hx` |
| `abs_err_f/a_field` | Absolute state error (prior/posterior) |
| `bias_f/a_field` | Signed state bias |
| `spread_f/a_field` | Ensemble spread field |
| `rmse_f/a_global_{var}` | Domain-wide RMSE per state variable |
| `bias_f/a_global_{var}` | Domain-wide bias per state variable |
| `spread_f/a_global_{var}` | Domain-wide spread per state variable |

**Reference file** (shared across all combos for one truth member):
```
{tag}_multi_obs_ref_Ne{ne:03d}_tm{tm:02d}.npz
```

| Key | Description |
|-----|-------------|
| `truth_hx_field` | Truth H(x) reflectivity field |
| `xf_mean` | Prior ensemble mean state |

**QC codes** (appear in filenames when relevant):

| Code | Meaning |
|------|---------|
| `none` | no filtering |
| `E` | ensemble filter only |
| `T` | truth filter only |
| `ET_and` | both filters, AND logic |
| `ET_or` | both filters, OR logic |

---

## Config reference

See `configs/template.yaml` for the full documented reference with all
options, accepted formats, and defaults. Key points:

- `obs_error_var` is variance (dBZ²), not std
- `obs.add_noise: true` adds N(0, √obs_error_var) noise to synthetic observations
- `prior_size: null` uses all remaining members (default); set to integer for ensemble size sensitivity
- Sweep parameters accept scalar, list, or `{start, stop, num}` (stop inclusive)
- `loc_x/y/z: null` disables localization (equivalent to L=99999)
- `qc.clamp_obs: true` clamps observations to the ensemble H(x) range
- `qc.filter_variance: true` enables variance-based obs point selection
- `skip_existing: true` resumes a partial run without recomputing finished files
- `verbose: 1` is recommended for cluster runs (one line per truth member)
- `store_fields: true` (single_obs only) saves `xf_sub`, `truth_sub`, and `xa_sub`

---

## Known limitations

- LETKF nonlinearity concentrates at high reflectivity (> 40 dBZ)
- AOEI ≈ TEnKF(Nt=1) in the linear regime
- Tempering shows diminishing returns beyond Nt = 5
