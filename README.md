# WRF Single-Cycle Assimilation

Radar data assimilation experiments with a 30-member WRF supercell ensemble,
using the Local Ensemble Transform Kalman Filter (LETKF) and likelihood
tempering (TEnKF).
---

## Repository layout

```
.
├── src/
│   ├── da/
│   │   └── core.py                        # All DA methods (LETKF, TEnKF, AOEI, ATEnKF, TAOEI)
│   ├── runners/
│   │   ├── run_single_obs_exps.py         # WS-1, WS-2
│   │   └── run_full3d_multicycle_exps.py  # WS-3, WS-4, WS-5
│   ├── extract_3d_subset.py               # Extract WRF ensemble subsets to .npz
│   ├── selectors.py                       # Observation grid selectors
│   └── fortran/                           # Fortran LETKF source + Makefile
├── configs/
│   ├── build_3D_section.yaml    # Config for data extraction (Notebook 1)
│   ├── ws1.yaml                 # WS-1: Ntemp sweep
│   ├── ws2.yaml                 # WS-2: obs position comparison
│   ├── ws3.yaml                 # WS-3: multiple obs, no QC filter
│   ├── ws4.yaml                 # WS-4: multiple obs, with QC filter
│   └── ws5.yaml                 # WS-5: localization/inflation tuning
├── Notebooks/
│   ├── S1_Explore_and_extract_3d_sections_WRF.ipynb
│   └── S2_obs_explorer_ws2.ipynb
└── tests/
    └── test_da_core.py
```

---

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate wrf_python_assimilation
```

### 2. Build the Fortran LETKF module

```bash
cd src/fortran && bash ../build_fortran.sh && cd ../..
```

This compiles `cletkf_wloc` via `f2py` and places the `.so` in `src/fortran/`.
All runners add that path to `sys.path` automatically.

---

## Data preparation

Before running any experiment you need to extract the 3D WRF ensemble subset
from the raw `wrfout` files.

**Interactive** — open `Notebooks/1__Explore_and_extract_3d_sections_WRF.ipynb`
and follow the four steps: choose region → visualise → extract → sanity check.

**Command line** — once `configs/build_3D_section.yaml` is configured:

```bash
python src/extract_3d_subset.py --config configs/build_3D_section.yaml
```

The output is a compressed `.npz` file with the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `state_ensemble` | `(nx, ny, nz, Ne, 8)` | All members |
| `lats` | `(ny, nx)` | Latitude [°] |
| `lons` | `(ny, nx)` | Longitude [°] |
| `z_heights` | `(nz, ny, nx)` | Height above sea level [m] |

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

| Method | Function | Description |
|--------|----------|-------------|
| LETKF | `letkf_update` | Standard single-step LETKF |
| TEnKF | `tenkf_update` | Tempered LETKF — fixed Ntemp, H(x) recomputed at each step |
| AOEI | `aoei_update` | LETKF + Adaptive Observation Error Inflation (single step) |
| ATEnKF | `atenkf_update` | Locally adaptive tempering — Ntemp determined per observation from AOEI inflation ratio |
| TAOEI | `taoei_update` | TEnKF with AOEI recomputed at every tempering step |

### Tempering schedule

Weights follow :

```
alpha_i = exp(-(Nt+1)*alpha_s / i) / sum_j exp(-(Nt+1)*alpha_s / j)
```

`sum(alpha_i) = 1` guarantees that total information across all steps equals
`R0` (information-preserving property). Larger `alpha_s` back-loads weight
toward later iterations; `alpha_s = 0` gives equal weights.

---

## Experiments

All experiments use a **single-cycle** (no cycling) setup. The full ensemble
of 30 members is used: at each run one member is withheld as truth and the
remaining 29 form the prior ensemble. All 30 combinations are looped over.

### WS-1 — Ntemp sweep

**Question:** how many tempering iterations are needed in 3D WRF before
performance saturates?

- Single observation at a fixed grid location
- Methods: LETKF (baseline) and TEnKF for Ntemp = 1…10
- Fixed `alpha_s = 2`, all 30 truth members

```bash
# Set paths.prepared and paths.outdir in configs/ws1.yaml first
python src/runners/run_single_obs_exps.py --config configs/ws1.yaml
```

---

### WS-2 — Observation position comparison

**Question:** how do methods respond to observations at different positions
relative to the prior ensemble — near the mean, above it, or below it?

- Three fixed observation locations (A: near mean, B: above mean, C: below mean)
- Methods: LETKF, TEnKF, AOEI, ATEnKF
- Ntemp fixed from WS-1 results, all 30 truth members

```bash
# Set paths.prepared, paths.outdir, and obs_positions in configs/ws2.yaml
python src/runners/run_single_obs_exps.py --config configs/ws2.yaml
```

Observation positions are selected using the exploration notebook
(`obs_explorer_ws2.ipynb`) or by inspecting the extracted `.npz` directly.
Placeholder values `(x:0, y:0, z:0)` are set in the config until final
positions are chosen.

---

### WS-3 — Multiple observations, no QC filter

**Question:** how do methods scale to a dense radar-like observation network?

- Uniform `::2` stride grid over the full 3D domain
- Methods: LETKF, TEnKF, AOEI, ATEnKF
- No near-zero departure filtering, all 30 truth members

```bash
python src/runners/run_full3d_multicycle_exps.py --config configs/ws3.yaml
```

---

### WS-4 — Multiple observations, with QC filter

Same as WS-3 but with near-zero departure filtering enabled: grid points
where both truth and ensemble mean are below `dbz_min = 5 dBZ` are dropped,
mimicking operational clear-air quality control.

```bash
python src/runners/run_full3d_multicycle_exps.py --config configs/ws4.yaml
```

---

### WS-5 — Localization and inflation tuning

**Question:** what are the optimal localization scale and inflation for the
best methods from WS-2/3/4?

- 2–3 best-performing methods from previous experiments
- Sweeps `loc_scales` over `[3, 5, 7, 10]` grid points
- Fixed Ntemp from WS-1, filtered obs (WS-4 setup)

```bash
python src/runners/run_full3d_multicycle_exps.py --config configs/ws5.yaml
```

---

## Running the tests

```bash
python tests/test_da_core.py
```

Covers: tempering schedule properties, AOEI floor guarantee, ATEnKF
per-observation Ntemp logic, and the information-preserving property.

---

## Config reference

```yaml
# ── ws1.yaml / ws2.yaml  (run_single_obs_exps.py) ─────────────────────────
experiment: WS-1   # or WS-2

paths:
  prepared:  /path/to/state_ensemble.npz
  outdir:    /path/to/output/

state:
  var_idx: {qg: 0, qr: 1, qs: 2, T: 3, P: 4, u: 5, v: 6, w: 7}

obs:
  sigma_dbz: 5.0              # obs error std [dBZ] — squared to variance internally
  loc: {x: 0, y: 0, z: 0}    # single location (WS-1)
  # obs_positions:            # three locations (WS-2)
  #   A_near_mean:  {x: ?, y: ?, z: ?}
  #   B_above_mean: {x: ?, y: ?, z: ?}
  #   C_below_mean: {x: ?, y: ?, z: ?}

da:
  ntemp_sweep: [1,2,3,4,5,6,7,8,9,10]   # WS-1: Ntemp values to test
  ntemp:   3                             # WS-2: fixed Ntemp (from WS-1 results)
  alpha_s: 2.0
  loc_scales: [5, 5, 5]                  # localization [x, y, z] grid points

# ── ws3.yaml – ws5.yaml  (run_full3d_multicycle_exps.py) ──────────────────
paths:
  prepared:  /path/to/state_ensemble.npz
  outdir:    /path/to/output/

obs:
  sigma_dbz: 5.0
  stride: 2                    # obs every N grid points in each dimension
  filter_near_zero: false      # true for WS-4 and WS-5
  dbz_min: 5.0

da:
  ntemps:  [1, 3]              # list of Ntemp values to run
  alphas:  [2.0]               # list of alpha_s values to run
  loc_scales_list:             # list of [lx, ly, lz] combinations
    - [5, 5, 5]
  methods: [LETKF, TEnKF, AOEI, ATEnKF]
  n_members: 30
```
