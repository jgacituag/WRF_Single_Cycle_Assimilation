# WRF Single-Cycle Assimilation

Radar data assimilation experiments with a real cases WRF ensemble,
using the Local Ensemble Transform Kalman Filter (LETKF) and likelihood
tempering (TEnKF).
---

## Repository layout

```
.
├── src/                             # everything needed to RUN an experiment
│   ├── da/
│   │   ├── __init__.py              # package marker (empty, but required)
│   │   ├── core.py                  # the DA methods: LETKF, TEnKF, AOEI
│   │   └── metrics.py               # every diagnostic (see "Output files")
│   ├── runners/
│   │   └── run_experiment.py        # the entry point; all three modes
│   ├── fortran/
│   │   ├── common_tools.f90         # precision kinds, constants
│   │   ├── common_mtx.f90           # symmetric eigendecomposition (LAPACK)
│   │   ├── common_letkf.f90         # letkf_core, one grid point
│   │   ├── common_da_wloc.f90       # grid loop + R-localization + calc_ref
│   │   └── f2py_f2cmap.txt          # maps r_size->double, r_sngl->float
│   ├── build_fortran.sh             # compiles the four .f90 into cletkf_wloc
│   ├── naming.py                    # THE naming scheme + tag/output validation
│   ├── extract_3d_subset.py         # WRF output -> subset .npz  (data prep)
│   ├── queue_ws.sh                  # PBS: sweep / single_obs
│   └── queue_multiobs.sh            # PBS: multi_obs
├── configs/
│   ├── template.yaml                # full documented reference — start here
│   ├── build_3D_section_Dataset_*.yaml  # data extraction, one per dataset
│   ├── build_3D_section_wrfout.yaml # data extraction from raw wrfout
│   └── ws_*.yaml                    # experiment configs (git-ignored)
├── Notebooks/                       # the analysis layer
│   ├── nbcommon.py                  # shared helpers + THE sign convention
│   ├── N1_Prepare_Data.ipynb        # explore WRF output, extract subsets
│   ├── N2_Prior_Conditions.ipynb    # the prior ensemble (no DA output)
│   ├── N3_Pointwise_Skill.ipynb     # where assimilation helps and hurts
│   ├── N4_Method_Comparison.ipynb   # AOEI vs LETKF vs TEnKF Nt=1..5
│   └── figures/                     # PDF + PNG output of the notebooks
├── test/
│   ├── test_install.py              # can this checkout run an experiment?
│   └── test_ensemble_stats.py       # skew / kurtosis / CRPS correctness
├── data/                            # subsets and results (git-ignored)
│   ├── 3D_subsets_{A,B,C,D}/        # prepared prior ensembles, one dir per dataset
│   ├── WS_*/                        # experiment output, one dir per run
│   └── derived/                     # notebook caches and published tables
├── logs/                            # PBS job logs
├── REPO_STRUCTURE.md
└── environment.yml
```

The compiled `cletkf_wloc*.so` is **git-ignored**, so a fresh clone has only the
`.f90` sources — `build_fortran.sh` is not optional, and the PBS scripts re-run it
on every compute node. The experiment configs (`configs/ws*`) are git-ignored too;
the runner copies whichever one it used into the output directory, so each result
folder is self-describing.

---

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate wrf_python_assimilation
```

> The working environment on this cluster is named `intermediate_exp` (Python 3.8),
> which is what the PBS scripts and the shipped `.so` were built against.
> `environment.yml` declares `name: wrf_python_assimilation` and pins no Python
> version — a fresh env may get a Python that cannot load a prebuilt extension, so
> always rebuild it (step 2) rather than copying a `.so` between machines.

### 2. Build the Fortran LETKF module

```bash
bash src/build_fortran.sh
```

Compiles `cletkf_wloc` via `f2py` from the four `.f90` sources and places the `.so`
in `src/fortran/`. All runners add that path to `sys.path` automatically. Requires
`CONDA_PREFIX` to be set and `$CONDA_PREFIX/lib/liblapack.so` to exist; the script
exits with a clear message if not. On this cluster the queue scripts first run
`source /opt/load-libs.sh 3`.

### 3. Check the installation

```bash
python test/test_install.py
```

Data-free and config-free: it builds a tiny synthetic ensemble in memory and runs
LETKF, TEnKF and AOEI end to end. If this passes, the checkout can run experiments.

---

## Naming scheme

Everything under `data/` is named dataset-first, so all the files for one dataset sort
together. The scheme lives in **`src/naming.py`**, which the runner, the extractor and
the migration all import — there is one definition, not three.

```
WS_{DATASET}_{mode}_{HHMM}_LOC{L}[_{FLAGS}]
3D_subsets_{DATASET}/subset_{DATASET}_{YYYYMMDDHHMMSS}.npz
```

| element | values |
|---|---|
| `DATASET` | `A` `B` `C` `D` |
| `mode` | `sweep` `multiobs` `point` (the runner's `single_obs`) |
| `HHMM` | `1800` `1900` `2000` `2100` |
| `L` | the isotropic localization scale in km, e.g. `0.1`, `2.0`, `4.0` |

Flags, combinable, in this order. **No field flag means light mode** — per-domain
scalars only.

| flag | meaning | checked against |
|---|---|---|
| `QC<code>` | departure-band rejection active | `qc.dep_band`, **and the code matches it** |
| `WIN` | subset cut to the analysis window | — |
| `STEPS` | intermediate tempering steps stored | any `output.steps_*` switch |
| `REF` | reflectivity metric fields only | `output.store_ref_fields` |
| `FULL` | state metric fields too | `output.store_state_fields` |
| `ALLMEM` | every truth member, not just `tm=0` | not checkable — see below |
| `AS<x>` | tempering slope other than the default | — |

The `QC` flag carries the band, because otherwise a run with a band and a run without
one collide on disk under one name — and so do two runs with different bands.
`naming.qc_code` generates it: `[2, 8]` → `QC2t8`, `[-1, 3]` → `QCm1t3` (`m` is the
minus sign). Bare `QC` is the legacy spelling and still validates; it claims a band
without naming it.

`ALLMEM` and `AS<x>` extend the five documented flags because the runs on disk need
them: without `ALLMEM`, `WS_D_sweep_1800_LOC0.1` and its all-member sibling collide on
one name, and they are different runs (different stride, tempering ladder and
truth-member set). It cannot be cross-checked against the config, because
`sweep.truth_members` is only a fallback for when `--tm` is absent and an all-member run
is a campaign of 60 one-member jobs.

### The datasets, and the trap

| dataset | grid | DA cycle | physics | was |
|---|---|---|---|---|
| **A** | 4 km | 5 min | multi | *a rebuild; built 2026-08, sweep complete* |
| **B** | 4 km | 1 h | multi | `..._SINGLECONF_{HH}00_LOC*_1HR` |
| **C** | 2 km | 5 min | multi | `..._{HH}00_LOC*_hires` |
| **D** | 4 km | 5 min | **single** | `..._SINGLECONF_{HH}00_LOC*` |

**`SINGLECONF` maps to D, not to A.** The chapter's old dataset A was the single-physics
run; the new A is a multi-physics rebuild. Anything that assumes `SINGLECONF == A` is
wrong. The `_1HR` suffix on a tag that also said `SINGLECONF` — B's
old name — is exactly the collision that caused the confusion.

Dataset **C is not homogeneous**: four of its eight runs were built from `DAFCST`
subsets and four from `GUES`. The letter alone does not identify the source, so every
file also carries `upstream`.

### The identity is in the data, not the name

Filenames drift; this is the part that outlasts a rename. Every subset and every
experiment output carries:

| key | |
|---|---|
| `dataset_id` | `A`/`B`/`C`/`D`, written from an explicit config key, never parsed from the filename |
| `da_cycle_min` | upstream DA cycle length, minutes |
| `dx_km` | measured from `pos_km` at extraction time, not configured |
| `physics` | `multi` or `single` |
| `upstream` | `GUES` or `DAFCST` |
| `config_index` | per member — **all `-1` for every pre-migration dataset** |

`config_index` is `-1` because the member-to-physics mapping is recorded nowhere: not in
the subset npz, the run configs, the extraction configs, or the upstream `POST` trees,
which carry no namelists. `-1` means *not recorded*, never *configuration -1*, and the
files carry a `config_index_note` saying so. It is never invented. This absence is the
reason the naming discussion happened at all.

`nbcommon.assert_dataset` reads `dataset_id` back at load and refuses a file that is not
the dataset the notebook believed, so a mis-migrated file fails loudly instead of
quietly being analysed as the wrong one.

### Enforcement

`run_experiment.py` validates `experiment_tag` **at startup**, before anything is
loaded — a run that takes six hours should not discover at write time that its name is
wrong. A tag that disagrees with its own config is worse than an ugly one, because it is
believed, so every field is cross-checked rather than trusted: `LOC` against
`sweep.loc_x` (and `loc_y`/`loc_z`, since the tag claims isotropy), `mode` against
`sweep.obs_points.mode`, `QC<code>` against `qc.dep_band` (both that it is set and that
the code names the same band), `STEPS` against the `output.steps_*` switches, and
`REF`/`FULL` against `output.store_ref_fields` / `output.store_state_fields`. The
`output` block itself is validated at the same moment: unknown keys, non-boolean values
and mixing it with the deprecated `store_fields` / `storage_level` all fail there.
`extract_3d_subset.py` does the same for the subset output path and its `dataset_id`.

### Migrating

`src/migrate_naming.py` performed the one-time rename. It is dry-run by default, writes
`data/derived/rename_manifest.csv` first, and only moves anything on a second invocation
with `--apply`; `--revert` undoes it from the same manifest and `--verify` re-checks that
every move landed and every `paths.prepared` resolves. `src/backfill_identity.py` then
appended the identity keys to the existing files — appended to the npz zip rather than
rewritten through `np.savez`, which would have meant recompressing ~250 GB.

## Data preparation

Before running any experiment you need to extract the 3D WRF ensemble subset
from the raw `wrfout` files or post-processed output.

**Interactive** — open `Notebooks/N1_Prepare_Data.ipynb`: inspect a source file →
draw the region on a reflectivity map → get the grid indices for the YAML → extract
→ verify → health-sweep every subset on disk.

**Command line** — once the appropriate config is configured:

```bash
# From raw wrfout files:
python src/extract_3d_subset.py --config configs/build_3D_section_wrfout.yaml

# From post-processed files:
python src/extract_3d_subset.py --config configs/build_3D_section_post.yaml
```

The output is a compressed `.npz` file:

| Key | Shape | Description |
|-----|-------|-------------|
| `state_ensemble` | `(nx, ny, nz, Ne, 8)` | All members |
| `lats` / `lons` | `(ny, nx)` | **Transposed relative to the state** |
| `z_heights` | `(nz, ny, nx)` | Height ASL [m] |
| `pos_km` | `(nx, ny, nz, 3)` | Position [x_km, y_km, z_km] from domain corner |
| `valid_mask` | `(nx, ny, nz)` `bool` | `False` where **any** member was non-finite |
| `members_read` | `(Ne,)` `bool` | which member files were found |
| `n_masked_cells` | scalar | how many cells the mask removed |

**`valid_mask` and the union.** A handful of cells arrive non-finite: in dataset A,
41–56 per hour, identical across all eight variables, never a whole column and never at
a domain edge. 32 of 60 members carry 1–4 cells each, and those cells are masked in that
member's own source netCDF, which `np.ma.filled(..., np.nan)` propagates one cell at a
time. A cell valid in 28 members and not in the other 32 gives that column a different
ensemble size than its neighbours, which biases every shape statistic and every
covariance computed there. So the extractor masks the **union** across all members and
sets those cells to NaN in every member and variable. It costs ~50 cells out of 1.5 M,
and the mask is written into the subset so downstream code can assert on it rather than
rediscover it. Applied to A, B, C and D alike.

Two traps worth stating once. **`lats`/`lons` are `(ny, nx)`** while everything else
is `(nx, ny, …)` — `nbcommon.map_field` applies the transpose internally so no
notebook does it by hand. And **`pos_km` is relative to each subset's own corner**,
so `x_km` from a 3.95 km grid and a 1.98 km grid are not comparable.

**Reflectivity is not stored** and must be derived from `qr/qs/qg/T/P`
(`nbcommon.hx`, or the Fortran `calc_ref_ens` the assimilation itself uses).

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

### The hydrometeor floor

After every LETKF update the three hydrometeors are floored at zero:

```python
for q in ("qr", "qs", "qg"):
    np.maximum(x[..., var_idx[q]], 0.0, out=...)
```

This is a **nonlinear projection applied after a linear update**, so the analysis is no
longer the state the covariances that produced it describe. `assimilation.clamp_hydro`
selects when it happens:

| Setting | When the floor is applied |
|---------|---------------------------|
| `per_step` | after every tempering step — **the production default, and it stays it** |
| `final` | once, after the last step, so an intermediate state keeps its negative mass |
| `never` | not at all — unphysical as a product, and the control for what the floor does |

The forward operator guards each species with `IF (q > 0)` (`calc_ref`, [common_da_wloc.f90](src/fortran/common_da_wloc.f90)), so **H(x) cannot see a
negative mixing ratio**. The LETKF weights depend on H(x), the departure and R alone,
which means the floor is invisible to the analysis of every other variable at `Nt = 1`
and `Nt = 2`, and can only enter from `Nt = 3` — through the hydrometeor perturbations
of a state that has since been through two more operator evaluations.

Every run records what the floor touched, per tempering step, at every storage level:

| Key | Description |
|-----|-------------|
| `clamp_mode` | the setting the run used |
| `clamp_applied` | `(n_steps,)` — whether the floor was applied at that step |
| `clamp_n_pairs` | `(n_steps,)` — (cell, member) pairs below zero, the union over the three species |
| `clamp_n_{qr,qs,qg}` | the same, per species |
| `clamp_mass_total`, `clamp_mass_{qr,qs,qg}` | the mass raising them to zero adds, kg/kg, **summed over members** |
| `mass_{f,a}_{q}_{all,in,out}` | domain hydrometeor mass, prior and posterior, split by the 5 dBZ truth contour |
| `mass_{fneg,aneg}_{q}_{...}` | the part of it below zero — identically 0 under `per_step` |
| `mass_t_{q}_{...}` | the truth's mass, which is what "a clamp toward zero is a push toward the truth" is measured against |

Under `never`, and on `final`'s early steps, the counters are a **counterfactual**:
they measure that setting's own unclamped trajectory, not the production run's.

Mind the two normalisations: `clamp_mass_*` is summed over the ensemble, `mass_*` is
the ensemble **mean** summed over cells, so the two differ by `Ne`. They are computed by
different code paths — the counter inside the tempering loop, the budget after it — and
on a single step they agree to 2e-16 relative, which is the cross-check that the loop
counter measures what it claims.

### Localization

R-localization (Greybush et al. 2011). The Fortran inflates observation error
by `exp(0.5*(d/L)^2)` at distance `d` from a grid point with scale `L`.
Distances and scales are in km, computed from `pos_km`.

The compact-support cutoff is **hardcoded** at `d² ≤ (2√(10/3))² ≈ 13.33`, i.e. a
normalized radius of ≈ 3.65 L, in both `common_da_wloc.f90` and `_compute_rho`.
It is *not* `cutoff_factor` — that is a separate Python-only subdomain box used by
sweep and single_obs to keep memory down (see the config reference below).

`loc_x/y/z: null` **does not work** — it raises `TypeError` in `_build_combos`.
Setting an axis to `0.0` drops it from the distance calculation in Fortran, but in
sweep mode the Python subdomain box also collapses to one cell on that axis, so the
two modes then disagree. Use a small positive scale instead.

---

## Running experiments

### 1. Sanity check first

```bash
python test/test_install.py          # methods run, AOEI inflates, bounds hold
python test/test_ensemble_stats.py   # skew / kurtosis / CRPS against closed forms
```

### 2. Observation modes

The mode is set in `sweep.obs_points.mode` in the config. Three modes are available:

| Mode | Description | Output |
|------|-------------|--------|
| `single_obs` | One fixed obs point, all method combos | One `.npz` per run |
| `point` | `single_obs` at a **list** of points, keeping everything | One `.npz` per point + one ref file |
| `sweep` | Every QC-passing stride point, each as an independent single-obs | One `.npz` per truth member |
| `multi_obs` | All QC-passing stride points assimilated jointly (one Fortran call per combo) | One `.npz` per combo + one ref file per truth member |

`point` is `single_obs` with `obs_points.points: [[i,j,k], ...]` — the same code path,
not a fourth one. At each point, each combo and **every tempering step** it stores the
full ensemble in state space (before *and* after the hydrometeor clamp) and in
observation space, plus the LETKF weight matrix `trans` / `transm`. Sixty members ×
nine fields × sixteen steps × three points is kilobytes, so it stores all of it.

Two things that cannot be recovered any other way:

- **The weights.** The increment is `X_f w`, so `transm` says whether one member
  dominates the update and whether tempering concentrates weight as the iterations
  grow — collapse by weight *degeneracy* rather than by the transform. The Fortran
  returns `xa` and a counter, so `da.core.letkf_weights` recomputes them from the same
  inputs; `test/test_install.py` checks it reproduces the Fortran analysis exactly.
- **The clamp pair.** If the posterior clamps `qg`/`qr`/`qs` at zero, that is a
  nonlinear operation applied after a linear update, breaking consistency between the
  state and its covariances, and it would accumulate over iterations. If the two arrays
  coincide, the hypothesis dies in one figure.

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

Loaded with `np.load(...)` → **flat 1-D columns**, one element per
(observation point × method combo) row, so `d["rmse_a_w_w"][n]` pairs with
`d["i"][n]`, `d["method"][n]`, etc. All metric columns are `float32`.

Every metric family comes in three spatial reductions, encoded as the infix:

| Infix | Meaning |
|-------|---------|
| `_point_` | value at the observation grid point |
| `_w_` | localization-weighted mean over the subdomain (weights = `rho`) |
| `_u_` | unweighted mean over the cutoff zone (`rho > 0`) |

> Note the `w`/`u` infixes collide visually with the state variables also named
> `w` and `u`. The variable suffix is always **last**: `crps_f_w_w` is the
> weighted CRPS of vertical wind, `rmse_f_u_u` the unweighted RMSE of `u`.

**Row metadata:** `i, j, k` (grid index), `x_km, y_km, z_km`, `yo`, `yo_clean`,
`method`, `ntemp`, `alpha_s`, `lx_km, ly_km, lz_km`, plus a `var_names` array
`['qg','qr','qs','T','P','u','v','w','ref']`.

**Obs-space (reflectivity) metrics** — suffix `_obs`:

| Key | Description |
|-----|-------------|
| `yo` | Noisy observation [dBZ] |
| `yo_clean` | Noise-free truth H(x) [dBZ] |
| `dep_b` / `dep_a` | Prior innovation / posterior residual `yo − H(x̄)` [dBZ] |
| `hxf_mean_obs` / `hxa_mean_obs` | Prior/posterior ensemble mean at obs point [dBZ] |
| `inc_obs` | Analysis increment at obs point [dBZ] |
| `rmse_f/a_{point,w,u}_obs` | Abs error / RMSE vs `truth_hx` |
| `bias_f/a_{point,w,u}_obs` | Signed error `H(x̄) − H(truth)`; unlike `dep_*` this carries no obs noise |
| `spread_f/a_{point,w,u}_obs` | Ensemble spread [dBZ] |
| `skew_f/a_{point,w,u}_obs` | Ensemble skewness (Nerger 2022 Eq. 25); `_w`/`_u` average `\|skew\|` |
| `kurt_f/a_{point,w,u}_obs` | Excess kurtosis (Nerger 2022 Eq. 26); `_w`/`_u` average `\|kurt\|` |
| `crps_f/a_{point,w,u}_obs` | Ensemble CRPS vs `truth_hx` [dBZ] |
| `n_active_f_{point,w,u}` | Members with signal (`H(x) > dbz_min`), of `Ne` |
| `loc_weights_sum` | Sum of localization weights |
| `n_updated` | Number of grid points updated (`rloc > 0`) |
| `precip_fraction_f` | Fraction of updated points with `H(x̄ᶠ) > dbz_min` |
| `hx_dbz_local_mean_w/u` | Weighted/unweighted mean reflectivity in localization volume |
| `spread_f/a_obs` | Legacy aliases of `spread_f/a_point_obs`, kept for older notebooks |

**Per state variable** (for each var in `{qg, qr, qs, T, P, u, v, w}`):

| Key pattern | Description |
|-------------|-------------|
| `mean_f/a_point_{var}` | Signed ensemble-mean state value at obs point |
| `truth_point_{var}` | Truth value at obs point |
| `rmse_f/a_{point,w,u}_{var}` | Abs error / RMSE |
| `bias_f/a_{w,u}_{var}` | Signed error `x̄ − truth`. No `_point_` variant: it is `mean_*_point_{var} − truth_point_{var}` |
| `spread_f/a_{point,w,u}_{var}` | Ensemble spread |
| `skew_f/a_{point,w,u}_{var}` | Ensemble skewness; `_w`/`_u` average `\|skew\|` |
| `kurt_f/a_{point,w,u}_{var}` | Excess kurtosis; `_w`/`_u` average `\|kurt\|` |
| `crps_f/a_{point,w,u}_{var}` | Ensemble CRPS vs truth |

### Multi-obs mode

**Analysis file** (one per method combo):
```
{tag}_multi_obs_{method}_Nt{nt:02d}_as{alpha_s}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne:03d}_tm{tm:02d}.npz
```

State-variable fields carry a trailing variable axis of size 8, ordered
`[qg, qr, qs, T, P, u, v, w]` — the `var_names` array in the file. Reflectivity is
**not** a 9th slot on that axis; it uses separate `_ref`-suffixed keys with no
variable axis, so state-field shapes are the same as they have always been.

| Key | Shape | Description |
|-----|-------|-------------|
| `hxf_mean_field` / `hxa_mean_field` | `(nx,ny,nz)` | Prior/posterior ensemble-mean reflectivity |
| `truth_hx_field` | `(nx,ny,nz)` | Truth reflectivity |
| `err_hxf_field` / `residual_field` | `(nx,ny,nz)` | `hx{f,a}_mean − truth_hx` |
| `abs_err_f/a_field` | `(nx,ny,nz,8)` | Absolute state error |
| `bias_f/a_field` | `(nx,ny,nz,8)` | Signed state bias |
| `spread_f/a_field` | `(nx,ny,nz,8)` | Ensemble spread |
| `skew_f/a_field` | `(nx,ny,nz,8)` | Ensemble skewness (Nerger 2022 Eq. 25) |
| `kurt_f/a_field` | `(nx,ny,nz,8)` | Excess kurtosis (Nerger 2022 Eq. 26) |
| `crps_f/a_field` | `(nx,ny,nz,8)` | Ensemble CRPS vs truth |
| `spread_f/a_ref_field` | `(nx,ny,nz)` | Reflectivity ensemble spread [dBZ] |
| `skew_f/a_ref_field` | `(nx,ny,nz)` | Reflectivity ensemble skewness |
| `kurt_f/a_ref_field` | `(nx,ny,nz)` | Reflectivity excess kurtosis |
| `crps_f/a_ref_field` | `(nx,ny,nz)` | Reflectivity ensemble CRPS [dBZ] |
| `n_active_f_field` | `(nx,ny,nz)` `int16` | Prior members with signal (`H(x) > dbz_min`), of `Ne` |
| `xf`, `xa`, `truth_state` | `(nx,ny,nz,Ne,8)` | Only when `output.store_ensemble: true` (off by default) |

#### Domain-restricted scalars

Emitted at **every** storage level, for each `domain` in `{global, storm, obs}`, each
var in `{qg, qr, qs, T, P, u, v, w}` **and** `ref`:

| Key pattern | Description |
|-------------|-------------|
| `rmse_f/a_{domain}_{var\|ref}` | `sqrt(mean(err²))` over the surviving cells |
| `bias_f/a_{domain}_{var\|ref}` | Mean signed error |
| `spread_f/a_{domain}_{var\|ref}` | Ensemble spread |
| `crps_f/a_{domain}_{var\|ref}` | Mean CRPS |
| `skew_f/a_{domain}_{var\|ref}` | Mean of `\|skew\|` |
| `kurt_f/a_{domain}_{var\|ref}` | Mean of `\|kurt\|` |
| `n_{metric}_f/a_{domain}_{var\|ref}` | **The denominator that entered** each scalar above |
| `n_cells_{domain}` | The domain size |
| `n_active_f_{domain}` | Mean count of members with signal |
| `storm_thresh_dbz` | The threshold `storm` used |

The three domains:

| domain | cells |
|---|---|
| `global` | every cell |
| `storm` | **columns** whose truth column-max reflectivity ≥ 20 dBZ, broadcast to all levels |
| `obs` | the cells carrying an assimilated observation |

`storm` is *columns with storm*, not *cells with echo* — the clear air above and below a
storm column is inside it, which is the point: that is where the increment lands without
an observation to justify it. `da.metrics.domain_masks` is the single definition;
`Notebooks/nbcommon.py` imports it rather than keeping a copy.

The chapter's argument rests on these three disagreeing — single-step and AOEI degrade
reflectivity globally and improve it inside storm columns, and the sign disagrees
between domains in 24 of 60 experiment-scheme pairs. In light mode the fields are gone,
so without these the comparison could not be made at all.

**Every scalar carries its denominator.** `n_rmse_f_global_w` is the count that actually
entered `rmse_f_global_w`; the dropped count is `n_cells_global` minus it. A mean over
1,523,027 cells and one over 1,522,832 are not the same number, and nothing else on disk
would say which happened. Every aggregate — RMSE, bias, spread and CRPS included, not
only the shape metrics — drops non-finite cells first, and RMSE is `sqrt(mean(err²))`,
never `mean(|err|)`.

> This is a **behaviour change**, and a visible one. In the multi-obs files written
> before this, every `rmse_*_global_{state var}` is literally `nan`: a plain `.mean()`
> over a field with a handful of non-finite cells returns NaN, and the notebooks worked
> around it by recomputing from the fields. In light mode there are no fields to
> recompute from, so the scalar had to become trustworthy. `rmse_*_global_ref` was
> always finite — `calc_ref` floors its output, so reflectivity has no NaN cells.

#### What the update actually reached

| Key | Description |
|-----|-------------|
| `frac_analysis_eq_prior_{domain}` | Fraction of cells where `xa == xf` exactly, in every member and variable |
| `n_touched` / `n_untouched` | The same, as counts over the whole domain |
| `frac_touched_no_obs` | Fraction of **updated** cells carrying no observation |
| `n_obs_cells` | Cells carrying an assimilated observation |

`frac_touched_no_obs` is the reach of the localization, and it is what separates §4.7
from the sweep: in a sweep every updated cell *is* the observation cell. The notebooks
recompute both of these from the ensembles today, and cannot in light mode.

#### QC bookkeeping and dataset identity

Written into **every** output file — sweep, point and multi-obs alike:

| Key | Description |
|-----|-------------|
| `qc_dep_band` / `qc_dep_band_active` | The active rejection band [dBZ], or `[nan, nan]` |
| `qc_dep_band_per_step` | Always `False`; see the config reference |
| `n_obs_candidates` | Points before any filter |
| `n_obs_after_first_filter` | After the variance / dbz filter |
| `n_obs_rejected_by_band` | **How many observations the band removed** |
| `n_obs_final` | What was actually assimilated |
| `qc_first_filter` | `filter_variance`, `none`, `dbz_min:{E\|T\|ET_and\|ET_or}`, or `obs_mask_from:{file}` |
| `qc_dbz_min`, `qc_clamp_obs`, `qc_stride` | The rest of the QC configuration |
| `dataset_id`, `da_cycle_min`, `dx_km`, `physics`, `upstream`, `source_run`, `config_index` | Copied from the subset |

A notebook asserts which rule was active instead of inferring it from a filename. The
identity block is copied from the subset the run read; a subset that carries none gets
an explicit empty `dataset_id` and an all `-1` `config_index` with a note saying it is
unrecoverable, never a guess from the path.

`abs_err`/`bias` have no `_ref_field` counterpart: they are exactly
`abs(err_hxf_field)` and `err_hxf_field`, already in the file. The
`{rmse,bias}_f/a_global_ref` scalars are still emitted so the scalar naming stays
uniform across `var_names + ["ref"]`.

**Reference file** (shared across all combos for one truth member):
```
{tag}_multi_obs_ref_Ne{ne:03d}_tm{tm:02d}.npz
{tag}_point_ref_Ne{ne:03d}_tm{tm:02d}.npz
```

**Written unconditionally, at every storage level**, and the runner asserts it exists
before writing any scheme file. It is not a storage level and not optional: at `light`
it is the only place the truth field exists at all, and it is where the domain masks and
the Jensen gap come from. Without it §4.7.6 loses its central predictor and the domain
restriction becomes impossible.

| Key | Description |
|-----|-------------|
| `truth_hx_field` | Truth H(x) reflectivity field (`truth_hx` is a kept alias) |
| `xf_mean` | Prior ensemble mean state |
| `truth` | Truth state, all 8 variables |
| `hxf_mean_field` | `mean(h(x_f))` |
| `hx_of_xfmean_field` | `h(mean(x_f))` |
| `yo`, `yo_clean`, `ix`, `iy`, `iz` | The assimilated observation set |

The Jensen gap is `|hx_of_xfmean_field − hxf_mean_field|`. Both halves are stored side
by side rather than their difference, so a reader can see which one moved.

**QC codes** — `qc_first_filter` in every output file, and the filename token for the
dbz filter:

| Code | Meaning |
|------|---------|
| `none` | no filtering |
| `E` | ensemble filter only |
| `T` | truth filter only |
| `ET_and` | both filters, AND logic |
| `ET_or` | both filters, OR logic |

The **departure band** is separate, and it is the one that reaches the filename: see the
`QC<code>` flag in the naming scheme above.

---

## Analysis

Four notebooks, meant to be read in order. They are configuration plus narrative —
everything mechanical lives in `Notebooks/nbcommon.py`.

| Notebook | Reads | Question |
|---|---|---|
| **N1** `Prepare_Data` | raw WRF output | What is in the source files, and which subset do we extract? |
| **N2** `Prior_Conditions` | `3D_subsets_*` **only** | What does the forecast ensemble look like before any observation is used? |
| **N3** `Pointwise_Skill` | **sweep** output, one truth member × 3 localizations | Where does assimilation improve, where does it degrade, and can the prior predict which? |
| **N3-1D** `Pointwise_Skill_1D` | **sweep** output, 4 datasets × 4 hours × 60 truth members | Same question at one localization, pooled across datasets and analysis times, plus the state variables. |
| **N4** `Method_Comparison` | **multi-obs** output | Which method wins — AOEI, LETKF, or TEnKF at Nt = 1…5, with observations interacting? |

N4 reads **multi-obs**, not sweep. It is the only place in this repository where
observation–observation interaction can appear at all; N3 assimilates every observation
on its own, so no observation there ever sees another.

N2 deliberately touches **no** assimilation output, so it can be run before the
experiments and is what selects the hour and sub-box they target.

### How the notebooks map onto chapter §4.6

The chapter document is not in this repository, so the intended order is recorded here
and in the notebook headers. **The localization subsection moved to the end of §4.6.**
It is exploratory — one truth member, the localization-weighted reduction — and it
previously sat between the method comparison and the predictors, interrupting the
argument where the comparison of schemes hands over to the rejection rule built on it.

| thesis | notebook | what |
|---|---|---|
| §4.6.1 | N3-1D §1 | the headline: schemes × datasets, fraction improved / median / mean |
| §4.6.2 | N3-1D §13 | what the 60 truth members are, and how many the multi-obs step needs |
| §4.6.3 | N3-1D §10, §12 | where tempering helps most; step count vs weight distribution |
| §4.6.5 | N3-1D §6–§9 | prior conditions, the band criterion, the common rule |
| §4.6.6 | N3-1D §7a, §7c | the rule the chapter quotes |
| §4.6.7 | N3-1D §14 | the eight state variables |
| §4.6.8 | N3-1D §15 | is the state-variable damage selectable from the prior? |
| §4.6.9 | N3-1D §16 | A against D: does tempering need physics diversity? |
| §4.6.10 | **N3** | localization — **moved here from §4.6.4** |
| §4.7 | N4 | the multi-obs experiment |

`N3_Pointwise_Skill.ipynb` has not been re-run against the rebuilt dataset A: its `A`
is the ensemble now called **D**. Its header says so.

### Verifying that a run is on the subset its name claims

A tag names a dataset and a run's config records the file it was given, but both are
names, and names survive a migration that moves data. `src/verify_run_subset.py`
checks the values instead: the sweep stores `truth_point_{var}`, which is
`state_ensemble[i, j, k, tm, var]` read straight out of the subset, so a run made on
that file is bit-identical to it.

```bash
python src/verify_run_subset.py WS_C_sweep_1900_LOC0.1_ALLMEM
python src/verify_run_subset.py --all-sweeps
```

Four of dataset C's runs record a path under a `..._DAFCST_...` directory while the
file the migration consolidated into `3D_subsets_C/` came from a `..._GUES_...` one.
The value check settles it: they are the same data under a stale name. N3-1D's §0
refuses any other name mismatch rather than assuming the same.

### The sign convention

`nbcommon` owns one rule, and enforces it:

> Every quantity named **skill** is `prior − analysis`.
> **Positive means the analysis is better.**

There is no other difference in the module — no `drmse`, no `delta`. `skill()` is
the only function that subtracts, `skill_label()` generates the axis text from the
same call chain as the numbers, `CMAP_SKILL` is fixed blue-positive, and
`assert_convention()` runs **at import**, so flipping the sign breaks every notebook
at once rather than producing a plausible wrong figure.

`spread` is not a skill metric and `skill(metric='spread')` raises.
`bias`, `skew` and `kurt` are compared as magnitudes.

### Why N3 and N4 cannot disagree

Both notebooks call exactly one chain:

```
nb.load_runs(...) -> nb.align(df, combos) -> nb.skill_summary(aligned, metric, var, red)
```

`align()` inner-joins every method combo on a stable point key, so both notebooks
necessarily aggregate over the *same* intersected point set, and it reports the NaN
count per combo — divergent NaN rates are the likeliest real cause of two analyses
of the same data reaching opposite conclusions. N3 ends by publishing its headline
table; N4 cell 5 re-derives it and compares against both the published CSV and the
digest, raising `ConsistencyError` with a row-aligned diff if they differ. N4's
final cell re-derives every prose conclusion as a PASS/FAIL audit.

### Pointing N3 and N4 at a run

**N3** takes its run list from a `RUNS` literal in cell 1:

```python
RUNS = [dict(tag="WS_D_sweep_1900_LOC0.1", loc=0.1, hour="19", ds="D"),
        dict(tag="WS_D_sweep_1900_LOC2.0", loc=2.0, hour="19", ds="D")]
```

`tag` is the output directory name under `data/`. `ds` is optional but worth setting:
`load_runs` then reads `dataset_id` out of every file it opens and refuses to pool one
that disagrees, so a mis-migrated run fails at load instead of being analysed as the
wrong dataset.

**N4 does not use a literal.** Its §0 discovers the runs under `data/` with the scheme's
own regex, reads `dataset_id` from inside each file, and asserts that the twelve
(dataset, hour) cells at `L = 2.0` all resolve — so a renamed or missing directory fails
loudly rather than silently dropping a dataset. Run N3 first regardless: it publishes
`data/derived/N3_headline.csv`, which N4's cross-check compares against.

`nbcommon.load_sweep` warns if a file does not have 362 columns, which means it
predates the reflectivity-metrics change and will be missing the `_ref`,
`n_active` and `bias_*` columns N3 and N4 depend on.

---

## Config reference

See `configs/template.yaml` for the full documented reference with all
options, accepted formats, and defaults. Key points:

- `obs_error_var` is variance (dBZ²), not std
- `obs.add_noise: true` adds N(0, √obs_error_var) noise to synthetic observations
- `prior_size: null` uses all remaining members (default); set to integer for ensemble size sensitivity
- Sweep parameters accept scalar, list, or `{start, stop, num}` (stop inclusive)
- `loc_x/y/z: null` **raises** — see the localization note above
- `qc.clamp_obs: true` clamps observations to the ensemble H(x) range
- `qc.filter_variance: true` enables variance-based obs point selection
- `skip_existing: true` resumes a partial run without recomputing finished files
- `verbose: 1` is recommended for cluster runs (one line per truth member)
- `qc.dep_band: [lo, hi]` **rejects** observations whose signed innovation
  `d = yo − mean(H(xf))` satisfies `lo ≤ d ≤ hi`, in dBZ. See below.
- `experiment_tag` is validated against the scheme **at startup** — see
  [Naming scheme](#naming-scheme)
- `cutoff_factor` (default 4.0) is a **sweep / single_obs memory optimization only**.
  It sizes the Python subdomain box handed to Fortran; `_run_multi_obs` never reads
  it. Values below ≈ 3.65 would silently truncate both the analysis and the metric
  masks, so leave it at 4.0.

### `assimilation:` — the hydrometeor floor

```yaml
assimilation:
  clamp_hydro: per_step   # per_step (default) | final | never
```

A misspelt mode is refused at startup, beside the tag check, rather than silently
falling back to the default. See [The hydrometeor floor](#the-hydrometeor-floor).

### `output:` — what reaches disk

This is what makes the §4.7/§4.8 batch affordable. The metric fields are the bulk:
~0.7 GB per multi-obs file on A and B, ~2.4 GB on C. Five schemes × five truth members ×
twelve (dataset, hour) combinations is ~380 GB, and §4.8 doubles it.

```yaml
output:
  store_scalars:      true    # always; domain aggregates only, a few kB
  store_ref_fields:   false   # 13 (nx,ny,nz) reflectivity fields
  store_state_fields: false   # 12 (nx,ny,nz,8) state fields
  store_ensemble:     false   # xf / xa / truth_state
  steps_departure:    false   # departure VECTOR per tempering step, float32
  steps_cells:        []      # [[i,j,k], ...] full ensemble at these cells, per step
  steps_fields:       false   # full metric fields per step. Expensive; rarely wanted
```

None of these change what is **computed** — every scalar is a reduction of a field, so
the fields are built either way. They change what lands on disk. `store_scalars: false`
is refused: it would write an empty file.

Measured on dataset A at 19 UTC (307 × 451 × 11, Ne = 59, one TEnKF Nt = 3 combo,
565 observations, 16 OpenMP threads), one scheme file plus the shared reference file:

| level | scheme file | wall (scheme) | wall (incl. setup) | peak RSS |
|---|---|---|---|---|
| `light` | **144 kB** | 98 s | 140 s | 11.8 GB |
| `ref` | 39.5 MB | 101 s | 143 s | 11.8 GB |
| `full` | 579.6 MB | 120 s | 162 s | 11.8 GB |
| `full` + `store_ensemble` | 3537.2 MB | 554 s | 596 s | 13.7 GB |

The reference file is 73.5 MB and is written once per truth member at every level.

Between the three field levels wall time barely moves — the metric fields are computed
either way, so the difference is compression and I/O — and **disk is the whole saving:
`light` is 4,016× smaller than `full` per scheme file.** Peak RSS is set by the prior
ensemble and the CRPS temporaries, not by the storage level.

`store_ensemble` is the exception and is why it is a separate switch: it costs 6.1× the
disk of `full` **and 4.6× the wall time**, because 5.9 GB of `xf`/`xa` go through zlib
single-threaded. Turn it on for one run, never for a batch.

The step switches are separate from each other because their costs differ by four orders
of magnitude. `steps_departure` stores the departure as a **vector, never a histogram** —
fixed bin edges are irreversible, and re-binning would mean re-running the experiment,
which is the one thing storing intermediate steps is meant to avoid. It also stores the
departure over every candidate point **before any filtering**, with one mask per filter,
so the effect of a different band stays reconstructable without another run. Alongside
each step it stores `alpha_i` and the effective `R/alpha_i`.

`steps_fields` is the one switch that is expensive on disk on its own: at Nt = 3 it adds
257 MB to a `light` file, because it writes two full state fields per step.

Deprecated spellings, still accepted so old configs reproduce, and **refused if mixed**
with `output:` because they describe the same bytes:

| deprecated | equivalent |
|---|---|
| `store_fields: true` | `store_ref_fields` + `store_state_fields` + `store_ensemble` |
| `storage_level: full` | `store_ref_fields` + `store_state_fields` |
| `storage_level: ref` | `store_ref_fields` |
| `storage_level: light` | neither |

With none of them present the defaults are what the code has always done: `multi_obs`
wrote every metric field, `sweep` and `single_obs` wrote scalar rows.

### `qc.dep_band` — the §4.8 rejection rule

```yaml
qc:
  dbz_min: 0.0
  clamp_obs: true
  filter_variance: true
  dep_band: [2.0, 8.0]      # reject when dep_min <= d <= dep_max; absent disables
  dep_band_per_step: false
```

Four things are asserted rather than assumed:

- **Signed, not `|d|`.** The damaging band sits just above zero; a rule on `|d|` would
  also reject departures near −5 dBZ, which are not damaging.
- **Rejection, not retention.** A point satisfying the condition is dropped.
- **Evaluated after `clamp_obs` and `filter_variance`, before the assimilation call.**
  The band was fitted on the sweep's stored `dep_b`, which N3-1D §0a confirmed is
  post-clamp, so evaluating it anywhere else would apply a different rule than the one
  the chapter derived.
- **Evaluated once on the original prior**, with the surviving set held fixed across
  every tempering step. `dep_band_per_step: true` is *not implemented* and raises: it is
  a different method, closer to an iterative robust filter, and an observation rejected
  at step 2 has already influenced step 1. The key exists so the choice this run made is
  visible in the config rather than being a hidden assumption in the code.

The band is written into the output as `qc_dep_band`, and how many observations it
removed as `n_obs_rejected_by_band`, so a notebook asserts which rule was active instead
of inferring it from a filename. The tag must carry the matching `QC<code>`, which is
what keeps a banded run and an unbanded one from colliding on disk.

> **The key was renamed.** `qc.departure_band` meant **retention** — keep the points
> inside the band. `dep_band` is **rejection** on the same numbers, i.e. the opposite
> observation set. The old key is therefore refused with an error rather than silently
> aliased. No config on disk used it.

### Two traps when writing a sweep matrix

`_build_combos` takes the **Cartesian product** of `loc_x × loc_y × loc_z`. Writing

```yaml
loc_x: [0.1, 2.0]
loc_y: [0.1, 2.0]
loc_z: [0.1, 2.0]
```

gives **8** combinations with mixed `lx ≠ ly ≠ lz`, not the 2 isotropic ones
intended. Run each localization scale as its own config into its own output
directory.

Keep `alpha_s` **scalar**. `LETKF` and `AOEI` de-duplicate on
`(method, alpha_s, lx, ly, lz)` and have `ntemp` forced to 1, so a *list* of
`alpha_s` produces duplicate rows differing only in a field that is not part of the
point key — which `nbcommon.align()` refuses to align, deliberately.

---

## Testing

| Command | Needs data? | Checks |
|---|---|---|
| `python test/test_install.py` | no | Fortran extension imports; `calc_ref` / `calc_ref_ens` agree; LETKF / TEnKF / AOEI run, stay finite, keep hydrometeors ≥ 0 and move toward the observation; AOEI inflates and respects its floor; tempering schedule sums to 1; **`letkf_weights` rebuilds the Fortran analysis from the weights alone**; **light-mode domain scalars equal the same quantities recomputed from full-mode fields**; the three domains can disagree and every scalar carries its denominator; a NaN cell does not poison a protected reduction |
| `python test/test_ensemble_stats.py` | no | Skewness and excess kurtosis against exact closed forms (Nerger 2022); the O(Ne log Ne) CRPS against the O(Ne²) pairwise reference; both metric entry points for shape, NaN handling and the Ne=1 degenerate case |

`nbcommon` self-tests on import: `assert_convention()` checks the sign of `skill()`
and the two column-naming hazards, so a broken convention fails every notebook
immediately rather than producing a plausible wrong figure.

---

## Known limitations

- LETKF nonlinearity concentrates at high reflectivity (> 40 dBZ)
- AOEI ≈ TEnKF(Nt=1) in the linear regime
- Tempering shows diminishing returns beyond Nt = 5

These are prior findings from earlier experiment rounds, not claims re-derived by
the current notebooks. N4's closing cell audits its own conclusions against whatever
output it was run on; treat that audit, not this list, as the current state.
