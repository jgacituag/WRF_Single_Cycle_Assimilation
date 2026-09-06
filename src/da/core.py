import os
import sys
import time
import numpy as np

# Verbosity: 0=silent, 1=method start/finish, 2=per-step, 3=debug
_VERBOSE = 1

def set_verbose(level: int):
    """Set verbosity for all DA methods in this process."""
    global _VERBOSE
    _VERBOSE = int(level)

def _log(level: int, msg: str):
    if _VERBOSE >= level:
        print(msg, flush=True)

_cda = None

def _get_cda():
    global _cda
    if _cda is not None:
        return _cda
    for attempt in range(2):
        try:
            from cletkf_wloc import common_da as cda
            _cda = cda
            return _cda
        except ImportError:
            if attempt == 0:
                here = os.path.dirname(os.path.abspath(__file__))
                fort_dir = os.path.normpath(os.path.join(here, "..", "fortran"))
                if fort_dir not in sys.path:
                    sys.path.insert(0, fort_dir)
    raise RuntimeError(
        "Fortran backend (cletkf_wloc) not found. "
        "Run src/build_fortran.sh from the repo root first."
    )

def _parse_omp_stacksize(s):
    """OMP_STACKSIZE grammar: digits, optional whitespace, optional B|K|M|G. Default unit K."""
    s = s.strip()
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0:
        return None
    n = int(s[:i])
    unit = s[i:].strip().upper() or "K"
    return n * {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}.get(unit, 1024)

_stack_checked = set()

def _check_stack_for(nobs, Ne):
    """simple_letkf_wloc holds nobs-sized automatic arrays on *two* different stacks:

      * the outer routine's hxfpert(nobs,nbv), hxfmean/oerr_dp/dep_dp(nobs) live on the
        master thread's stack, bounded by RLIMIT_STACK (`ulimit -s`);
      * the OpenMP-PRIVATE hxfpert_loc(nobs,nbv) and rloc_loc/dep_loc(nobs) get one copy
        per worker thread, bounded by OMP_STACKSIZE.

    Overflowing either is a bare SIGSEGV with no diagnostic, and in multi_obs mode it lands
    *after* the reference file has been written -- which is what stranded an orphaned
    ref_Ne059 in data/WS_multiobs_1800/ (nobs=3,007,528, Ne=59 needs 1354 MB/thread against
    queue_multiobs.sh's OMP_STACKSIZE=512M). Fail loudly instead.
    """
    import resource

    need = int(nobs * (Ne + 2) * 8 * 1.15)      # +2 covers rloc_loc/dep_loc; 15% slack
    if need < 4 * 1024**2 or (nobs, Ne) in _stack_checked:
        return
    _stack_checked.add((nobs, Ne))
    mb = need / 1024**2

    soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
    if soft != resource.RLIM_INFINITY and soft < need:
        raise RuntimeError(
            f"master-thread stack too small for the Fortran LETKF: nobs={nobs}, Ne={Ne} "
            f"needs ~{mb:.0f} MB but `ulimit -s` is {soft/1024**2:.0f} MB. "
            f"Run `ulimit -s unlimited` before invoking the runner."
        )

    raw = os.environ.get("OMP_STACKSIZE")
    have = _parse_omp_stacksize(raw) if raw else None
    if have is None:
        _log(0, f"[WARN] OMP_STACKSIZE is unset; each OpenMP thread needs ~{mb:.0f} MB of "
                f"stack for nobs={nobs}, Ne={Ne}. Export OMP_STACKSIZE={max(1, int(mb/1024)+1)}G "
                f"or the Fortran call will segfault.")
    elif have < need:
        raise RuntimeError(
            f"OMP_STACKSIZE={raw} is too small for the Fortran LETKF: nobs={nobs}, Ne={Ne} "
            f"needs ~{mb:.0f} MB per thread. Export OMP_STACKSIZE="
            f"{max(1, int(mb/1024)+1)}G (and `ulimit -s unlimited`)."
        )

def tempering_schedule(ntemp: int, alpha_s: float) -> np.ndarray:
    """
    Back-loaded exponential weights.
    At step i the obs error is inflated to R / alpha_i.
    """
    if ntemp == 1:
        return np.array([1.0], dtype=np.float32)
    i = np.arange(1, ntemp + 1, dtype=np.float64)
    w = np.exp(-(ntemp + 1) * float(alpha_s) / i)
    w /= w.sum()
    return w.astype(np.float32)

# The compact-support cutoff, squared, in normalised distance. Mirrors
# simple_letkf_wloc's `max_dist = (2*sqrt(10/3))**2`, which is the value the Fortran
# actually compares d^2 against.
LOC_CUTOFF_D2 = (2.0 * np.sqrt(10.0 / 3.0)) ** 2

# The floor simple_letkf_wloc puts under the localization weight before dividing.
LOC_WEIGHT_FLOOR = 1.0e-6


def letkf_weights(hdxb, rloc, dep, parm_infl=1.0):
    """The LETKF weight matrices at one grid point: (T, w_a).

    A faithful NumPy transcription of common_letkf.f90's `letkf_core`, for diagnostics
    only -- nothing here feeds an analysis, and the Fortran remains the only thing that
    computes one. It exists because `simple_letkf_wloc` returns xa and a counter, so
    the weights are not recoverable from a run's output, and the weights are what say
    whether one member dominates the update and whether tempering concentrates weight
    as the iterations grow. That distinction -- collapse by weight degeneracy rather
    than by the transform -- cannot be made from xa alone.

      hdxb (nobsl, ne)  H applied to the prior perturbations, at this point's obs
      rloc (nobsl,)     localised observation error, oerr / weight
      dep  (nobsl,)     departures yo - H(xf_mean)

    The state update the Fortran applies is
        xa[:, im] = xfmean + sum_im2 xfpert[im2] * (T[im2, im] + w_a[im2])
    so `T + w_a[:, None]` is the full weight matrix, column im per member.

    mtx_eigen is a symmetric eigensolver; np.linalg.eigh is the same decomposition, and
    both Pa and T are invariant to eigenvector sign and eigenvalue order, so the result
    matches regardless of which one is used. Everything runs in float64, as the Fortran
    does (r_size).
    """
    hdxb = np.asarray(hdxb, np.float64)
    rloc = np.asarray(rloc, np.float64)
    dep = np.asarray(dep, np.float64)
    ne = hdxb.shape[1]
    if hdxb.shape[0] == 0:
        # letkf_core's nobsl == 0 branch: an inflated identity, no mean shift.
        return np.eye(ne) * np.sqrt(parm_infl), np.zeros(ne)

    hdxb_rinv = hdxb / rloc[:, None]
    c = hdxb_rinv.T @ hdxb
    c[np.diag_indices(ne)] += (ne - 1) / parm_infl
    eival, eivec = np.linalg.eigh(c)
    pa = (eivec / eival) @ eivec.T
    transm = pa @ (hdxb_rinv.T @ dep)
    trans = (eivec * np.sqrt((ne - 1) / eival)) @ eivec.T
    return trans, transm


def local_obs_at(i, j, k, ox, oy, oz, oerr, loc_scales_km, pos_km):
    """Which observations reach grid point (i,j,k), and their localised error.

    Reproduces the selection inside simple_letkf_wloc's grid-point loop exactly: the
    same normalised distance, the same `d2 <= (2*sqrt(10/3))**2` cutoff, and the same
    1e-6 floor under the Gaussian weight before dividing into the observation error.

    Returns (sel, rloc) where `sel` indexes the observation arrays.
    """
    lx, ly, lz = (float(v) for v in loc_scales_km)
    px, py, pz = (float(v) for v in pos_km[i, j, k, :3])
    d2 = np.zeros(len(ox), np.float64)
    if lx > 1e-6:
        d2 += ((px - np.asarray(ox, np.float64)) / lx) ** 2
    if ly > 1e-6:
        d2 += ((py - np.asarray(oy, np.float64)) / ly) ** 2
    if lz > 1e-6:
        d2 += ((pz - np.asarray(oz, np.float64)) / lz) ** 2
    sel = np.where(d2 <= LOC_CUTOFF_D2)[0]
    w = np.maximum(np.exp(-0.5 * d2[sel]), LOC_WEIGHT_FLOOR)
    return sel, np.asarray(oerr, np.float64)[sel] / w


def compute_hxf(xf_grid: np.ndarray,
                ox: np.ndarray,
                oy: np.ndarray,
                oz: np.ndarray,
                var_idx: dict,
                dbz_min: float = 0.0) -> np.ndarray:
    """
    Apply the nonlinear reflectivity operator H to every ensemble member
    at every observation location, floored securely at dbz_min.
    """
    cda  = _get_cda()
    nobs = len(ox)
    Ne   = xf_grid.shape[3]
    hxf  = np.empty((nobs, Ne), dtype=np.float32, order="F")
    
    _log(3, f"Computing H(xf) for {nobs} obs and {Ne} ensemble members...")
    for ii in range(nobs):
        i, j, k = int(ox[ii]), int(oy[ii]), int(oz[ii])
        for m in range(Ne):
            hxf[ii, m] = cda.calc_ref(
                xf_grid[i, j, k, m, var_idx["qr"]],
                xf_grid[i, j, k, m, var_idx["qs"]],
                xf_grid[i, j, k, m, var_idx["qg"]],
                xf_grid[i, j, k, m, var_idx["T"]],
                xf_grid[i, j, k, m, var_idx["P"]],
                min_dbz=dbz_min,
            )
            
    # Enforce clear-air consistency matching the QC threshold
    return np.maximum(hxf, dbz_min).astype(np.float32)

def aoei(yo: np.ndarray,
         hxf: np.ndarray,
         R0: np.ndarray) -> np.ndarray:
    """Adaptive Observation Error Inflation."""
    yo_  = np.asarray(yo,  np.float64)
    hxf_ = np.asarray(hxf, np.float64)
    R0_  = np.asarray(R0,  np.float64)
    d        = yo_ - hxf_.mean(axis=1)
    sigma2_f = hxf_.var(axis=1, ddof=1)
    return np.maximum(R0_, d**2 - sigma2_f).astype(np.float32)

def _letkf_step(xf_grid, hxf, yo, obs_error_var,
                ox_km, oy_km, oz_km, loc_scales_km, pos_km):
    """One LETKF analysis via Fortran."""
    cda = _get_cda()
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)

    ox_f   = np.asarray(ox_km,         np.float32)
    oy_f   = np.asarray(oy_km,         np.float32)
    oz_f   = np.asarray(oz_km,         np.float32)
    dep    = (yo - hxf.mean(axis=1)).astype(np.float32)
    oerr_f = np.asarray(obs_error_var, np.float32)
    locs_f = np.asarray(loc_scales_km, np.float32)

    _log(3, f"Running LETKF step: nobs={nobs}  "
            f"oerr_mean={oerr_f.mean():.2f}  locs={locs_f} km")

    _check_stack_for(nobs, Ne)

    t0 = time.time()
    xa_out, n_updated = cda.simple_letkf_wloc(
        nx=nx, ny=ny, nz=nz,
        nbv=Ne, nvar=nvar, nobs=nobs,
        hxf=np.asfortranarray(hxf),
        xf=np.asfortranarray(xf_grid),
        dep=dep,
        ox=ox_f, oy=oy_f, oz=oz_f,
        locs=locs_f, oerr=oerr_f,
        pos_km=np.asfortranarray(pos_km),
    )
    dt = time.time() - t0
    total_pts = nx * ny * nz
    pct_skipped = ((total_pts - n_updated) / total_pts) * 100
    _log(3, f"      [Fortran LETKF] {dt:.3f}s | "
            f"Updated {int(n_updated)}/{total_pts} pts "
            f"(Skipped {pct_skipped:.1f}%)")
    return xa_out.astype(np.float32)

CLAMP_MODES = ("per_step", "final", "never")
HYDRO_VARS = ("qr", "qs", "qg")


def _clamp_hydro(x, var_idx, diag=None, apply=True):
    """Floor qr/qs/qg at zero, in place. Extracted verbatim from the three update
    functions that each had their own copy; the arithmetic is unchanged.

    It is also the operation §6's `point` mode brackets: a nonlinear projection applied
    after a linear update, which breaks consistency between the state and the
    covariances that produced it, and would accumulate over tempering iterations. If
    the before and after arrays coincide at the probed cells, that hypothesis dies.

    `diag`, when given, is filled with what the floor touches -- the number of
    (cell, member) pairs below zero and the mass raising them to zero adds -- and it is
    filled BEFORE the floor is applied, which is the only order that can see anything.

    `apply=False` measures without changing, so `clamp_hydro: never` still reports what
    a clamp WOULD have removed along its own trajectory. Those counts are a
    counterfactual on a run the clamp never touched, not the counts of the production
    run, and they are labelled that way on disk.
    """
    n_any = None
    for q in HYDRO_VARS:
        v = x[:, :, :, :, var_idx[q]]
        if diag is not None:
            # NaN < 0 is False, so a non-finite cell is neither counted nor massed;
            # np.maximum propagates it, so it is not clamped either. The two agree.
            neg = v < 0.0
            diag["n_" + q] = int(np.count_nonzero(neg))
            diag["mass_" + q] = float(-v[neg].sum(dtype=np.float64))
            if n_any is None:
                n_any = neg
            else:
                np.logical_or(n_any, neg, out=n_any)   # in place: no third 91 MB array
        if apply:
            np.maximum(v, 0.0, out=v)
    if diag is not None:
        # A pair negative in two hydrometeors is one pair, not two: `n_pairs` is the
        # union and is what "how often does the clamp fire" means.
        diag["n_pairs"] = int(np.count_nonzero(n_any))
        diag["mass_total"] = sum(diag["mass_" + q] for q in HYDRO_VARS)
        diag["applied"] = bool(apply)
    return x


def clamp_mode_of(cfg):
    """`assimilation.clamp_hydro` from a config, defaulted and validated at startup.

    The default is `per_step`, which is the production behaviour and stays it. The two
    other settings exist to answer whether the projection is a mechanism or a side
    effect; `never` is not a physical product.
    """
    a = cfg.get("assimilation") or {}
    return _check_clamp_mode(str(a.get("clamp_hydro", "per_step")))


def _check_clamp_mode(mode):
    if mode not in CLAMP_MODES:
        raise ValueError(f"clamp_hydro must be one of {CLAMP_MODES}, got {mode!r}")
    return mode


def _clamp_at_step(x, var_idx, mode, it, n_steps, log):
    """Apply the floor if this mode wants it at this step; measure it either way.

    per_step  every step (production)
    final     once, after the last tempering step
    never     not at all -- unphysical as a product, and the control for how much the
              clamp is doing at all
    """
    apply = (mode == "per_step") or (mode == "final" and it == n_steps - 1)
    d = {}
    _clamp_hydro(x, var_idx, diag=d, apply=apply)
    if log and d["n_pairs"]:
        _log(2, f"    [clamp {mode}] step {it+1}/{n_steps} "
                f"{'applied' if apply else 'measured only'}: "
                f"{d['n_pairs']:,} (cell,member) pairs, "
                f"{d['mass_total']:.4g} kg/kg summed")
    return d


def _clamp_arrays(diags, mode):
    """Per-step clamp bookkeeping, as arrays a scheme file can carry unconditionally.

    Twelve numbers per step. They are written at every storage level because the whole
    point is that the effect be quantifiable in an ordinary run rather than only in a
    special one.
    """
    out = {"clamp_mode": np.array(mode),
           "clamp_applied": np.array([d["applied"] for d in diags], np.bool_),
           "clamp_n_pairs": np.array([d["n_pairs"] for d in diags], np.int64),
           "clamp_mass_total": np.array([d["mass_total"] for d in diags], np.float64)}
    for q in HYDRO_VARS:
        out["clamp_n_" + q] = np.array([d["n_" + q] for d in diags], np.int64)
        out["clamp_mass_" + q] = np.array([d["mass_" + q] for d in diags], np.float64)
    return out


def _call_hook(step_hook, **kw):
    if step_hook is not None:
        step_hook(**kw)


def letkf_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                 loc_scales_km, var_idx, pos_km,
                 dbz_min: float = 0.0, step_hook=None,
                 clamp_mode: str = "per_step"):
    """Standard LETKF: single step, no AOEI, no tempering.

    `step_hook`, when given, is called around the hydrometeor clamp of the single step
    with the same keywords tenkf_update uses, so a caller storing intermediate steps
    does not need a second code path for the single-step methods.
    """
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx, dbz_min=dbz_min)
    idx = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    
    dep = (yo - hxf.mean(axis=1)).astype(np.float32)
    xa  = _letkf_step(xf_grid, hxf, yo, R0,
                      pos_km[idx[0], idx[1], idx[2], 0],
                      pos_km[idx[0], idx[1], idx[2], 1],
                      pos_km[idx[0], idx[1], idx[2], 2],
                      loc_scales_km, pos_km)

    hook_kw = dict(step=0, n_steps=1, alpha=1.0, R_eff=R0, hxf=hxf, dep=dep)
    _call_hook(step_hook, stage="pre_clamp", x=xa, **hook_kw)
    # Strictly enforce non-negative physical bounds on hydrometeors. One step, so
    # `per_step` and `final` are the same operation here and only `never` differs.
    diag = _clamp_at_step(xa, var_idx, _check_clamp_mode(clamp_mode), 0, 1, log=True)
    _call_hook(step_hook, stage="post_clamp", x=xa, **hook_kw)

    return dict(
        xa=xa,
        hxf=hxf,
        dep=dep,
        obs_error=R0,
        clamp=_clamp_arrays([diag], clamp_mode),
    )

def tenkf_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                 loc_scales_km, var_idx, ntemp, alpha_s, pos_km,
                 dbz_min: float = 0.0, step_hook=None, keep_hxfs: bool = False,
                 clamp_mode: str = "per_step"):
    """TEnKF (LETKF-T): Ntemp sequential steps with back-loaded inflation.

    `step_hook`, when given, is called twice per step -- `stage="pre_clamp"` on the raw
    LETKF output and `stage="post_clamp"` after the hydrometeor floor -- with the step
    index, the schedule weight alpha_i, the effective R/alpha_i, and the step's H(x)
    ensemble and departure vector. It is how §5 and §6 capture intermediate state
    without the runner holding every step's ensemble: the hook extracts what it wants
    and the array is reused. It never sees a return value, so it cannot alter the run.
    """
    t_start = time.time()
    steps = tempering_schedule(ntemp, alpha_s)
    R0    = np.asarray(obs_error_var, np.float32)
    nobs  = len(yo)
    Nt    = len(steps)

    idx   = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    ox_km = pos_km[idx[0], idx[1], idx[2], 0].astype(np.float32)
    oy_km = pos_km[idx[0], idx[1], idx[2], 1].astype(np.float32)
    oz_km = pos_km[idx[0], idx[1], idx[2], 2].astype(np.float32)

    _check_clamp_mode(clamp_mode)
    clamp_diags = []

    x_af = xf_grid.copy(order="F")
    hxfs = np.empty((Nt, nobs, xf_grid.shape[3]), np.float32) if keep_hxfs else None
    deps = np.empty((Nt, nobs),                    dtype=np.float32)

    _log(3, f"  [TEnKF] Starting {Nt} tempering steps (alpha_s={alpha_s:.2f})")
    for it in range(Nt):
        t_step = time.time()
        
        # Calculate intermediate observations safely matching the clear-air floor
        hxf = compute_hxf(x_af, ox, oy, oz, var_idx, dbz_min=dbz_min)
        dep = (yo - hxf.mean(axis=1)).astype(np.float32)
        if keep_hxfs:
            hxfs[it] = hxf
        deps[it]  = dep
        oerr = R0 / steps[it]

        _log(2, f"  [TEnKF]  step {it+1}/{Nt}  alpha={steps[it]:.4f}  "
                f"R/alpha={oerr.mean():.2f}  |dep|={np.abs(dep).mean():.3f}")

        x_af = _letkf_step(x_af, hxf, yo, oerr,
                            ox_km, oy_km, oz_km, loc_scales_km, pos_km)

        hook_kw = dict(step=it, n_steps=Nt, alpha=float(steps[it]), R_eff=oerr,
                       hxf=hxf, dep=dep)
        _call_hook(step_hook, stage="pre_clamp", x=x_af, **hook_kw)
        # STRICT PHYSICAL BOUNDS: Protect cross-covariances and stop moisture deficits.
        # This is the step the mode changes: under `final` the projection is deferred to
        # the last iteration, so the intermediate states the next update relinearises
        # around are the raw LETKF output, negative hydrometeors and all.
        clamp_diags.append(
            _clamp_at_step(x_af, var_idx, clamp_mode, it, Nt, log=True))
        _call_hook(step_hook, stage="post_clamp", x=x_af, **hook_kw)

        _log(3, f"    Step {it+1} complete in {time.time()-t_step:.3f}s")

    _log(3, f"  [TEnKF Total] {Nt} steps in {time.time()-t_start:.3f}s")
    out = dict(xa=x_af, deps=deps, alpha_weights=steps, obs_error=R0,
               clamp=_clamp_arrays(clamp_diags, clamp_mode))
    if keep_hxfs:
        out["hxfs"] = hxfs
    return out

def aoei_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                loc_scales_km, var_idx, pos_km,
                dbz_min: float = 0.0, step_hook=None,
                clamp_mode: str = "per_step"):
    """LETKF + AOEI: inflate once from the prior, then one LETKF step.

    `step_hook` matches tenkf_update's, with R_eff the inflated R_tilde rather than R0.
    """
    t_start = time.time()
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx, dbz_min=dbz_min)
    _log(3, f"    [AOEI] H(x) calculated in {time.time()-t_start:.3f}s")

    R_t    = aoei(yo, hxf, R0)
    n_inf  = int((R_t > R0).sum())
    _log(3, f"    [AOEI] Inflated {n_inf}/{len(yo)} obs | "
            f"R_tilde={R_t.mean():.2f} | R0={R0.mean():.2f}")

    idx   = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    dep   = (yo - hxf.mean(axis=1)).astype(np.float32)
    xa    = _letkf_step(xf_grid, hxf, yo, R_t,
                        pos_km[idx[0], idx[1], idx[2], 0],
                        pos_km[idx[0], idx[1], idx[2], 1],
                        pos_km[idx[0], idx[1], idx[2], 2],
                        loc_scales_km, pos_km)

    hook_kw = dict(step=0, n_steps=1, alpha=1.0, R_eff=R_t, hxf=hxf, dep=dep)
    _call_hook(step_hook, stage="pre_clamp", x=xa, **hook_kw)
    # Strictly enforce non-negative physical bounds on hydrometeors
    diag = _clamp_at_step(xa, var_idx, _check_clamp_mode(clamp_mode), 0, 1, log=True)
    _call_hook(step_hook, stage="post_clamp", x=xa, **hook_kw)

    _log(3, f"  [AOEI Total] cycle finished in {time.time()-t_start:.3f}s")
    return dict(
        xa=xa,
        hxf=hxf,
        dep=dep,
        obs_error_raw=R0,
        obs_error=R_t,
        clamp=_clamp_arrays([diag], clamp_mode),
    )