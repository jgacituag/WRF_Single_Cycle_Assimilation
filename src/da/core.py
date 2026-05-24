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

def letkf_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                 loc_scales_km, var_idx, pos_km,
                 dbz_min: float = 0.0):
    """Standard LETKF: single step, no AOEI, no tempering."""
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx, dbz_min=dbz_min)
    idx = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    
    xa  = _letkf_step(xf_grid, hxf, yo, R0,
                      pos_km[idx[0], idx[1], idx[2], 0],
                      pos_km[idx[0], idx[1], idx[2], 1],
                      pos_km[idx[0], idx[1], idx[2], 2],
                      loc_scales_km, pos_km)
                      
    # Strictly enforce non-negative physical bounds on hydrometeors
    xa[:, :, :, :, var_idx["qr"]] = np.maximum(xa[:, :, :, :, var_idx["qr"]], 0.0)
    xa[:, :, :, :, var_idx["qs"]] = np.maximum(xa[:, :, :, :, var_idx["qs"]], 0.0)
    xa[:, :, :, :, var_idx["qg"]] = np.maximum(xa[:, :, :, :, var_idx["qg"]], 0.0)
    
    return dict(
        xa=xa,
        hxf=hxf,
        dep=(yo - hxf.mean(axis=1)).astype(np.float32),
        obs_error=R0,
    )

def tenkf_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                 loc_scales_km, var_idx, ntemp, alpha_s, pos_km,
                 dbz_min: float = 0.0):
    """TEnKF (LETKF-T): Ntemp sequential steps with back-loaded inflation."""
    t_start = time.time()
    steps = tempering_schedule(ntemp, alpha_s)
    R0    = np.asarray(obs_error_var, np.float32)
    nobs  = len(yo)
    Nt    = len(steps)

    idx   = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    ox_km = pos_km[idx[0], idx[1], idx[2], 0].astype(np.float32)
    oy_km = pos_km[idx[0], idx[1], idx[2], 1].astype(np.float32)
    oz_km = pos_km[idx[0], idx[1], idx[2], 2].astype(np.float32)

    x_af = xf_grid.copy(order="F")
    hxfs = np.empty((Nt, nobs, xf_grid.shape[3]), dtype=np.float32)
    deps = np.empty((Nt, nobs),                    dtype=np.float32)

    _log(3, f"  [TEnKF] Starting {Nt} tempering steps (alpha_s={alpha_s:.2f})")
    for it in range(Nt):
        t_step = time.time()
        
        # Calculate intermediate observations safely matching the clear-air floor
        hxf = compute_hxf(x_af, ox, oy, oz, var_idx, dbz_min=dbz_min)
        dep = (yo - hxf.mean(axis=1)).astype(np.float32)
        hxfs[it] = hxf
        deps[it]  = dep
        oerr = R0 / steps[it]

        _log(2, f"  [TEnKF]  step {it+1}/{Nt}  alpha={steps[it]:.4f}  "
                f"R/alpha={oerr.mean():.2f}  |dep|={np.abs(dep).mean():.3f}")

        x_af = _letkf_step(x_af, hxf, yo, oerr,
                            ox_km, oy_km, oz_km, loc_scales_km, pos_km)

        # STRICT PHYSICAL BOUNDS: Protect cross-covariances and stop moisture deficits
        x_af[:, :, :, :, var_idx["qr"]] = np.maximum(x_af[:, :, :, :, var_idx["qr"]], 0.0)
        x_af[:, :, :, :, var_idx["qs"]] = np.maximum(x_af[:, :, :, :, var_idx["qs"]], 0.0)
        x_af[:, :, :, :, var_idx["qg"]] = np.maximum(x_af[:, :, :, :, var_idx["qg"]], 0.0)

        _log(3, f"    Step {it+1} complete in {time.time()-t_step:.3f}s")

    _log(3, f"  [TEnKF Total] {Nt} steps in {time.time()-t_start:.3f}s")
    return dict(xa=x_af, hxfs=hxfs, deps=deps,
                alpha_weights=steps, obs_error=R0)

def aoei_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                loc_scales_km, var_idx, pos_km,
                dbz_min: float = 0.0):
    """LETKF + AOEI: inflate once from the prior, then one LETKF step."""
    t_start = time.time()
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx, dbz_min=dbz_min)
    _log(3, f"    [AOEI] H(x) calculated in {time.time()-t_start:.3f}s")

    R_t    = aoei(yo, hxf, R0)
    n_inf  = int((R_t > R0).sum())
    _log(3, f"    [AOEI] Inflated {n_inf}/{len(yo)} obs | "
            f"R_tilde={R_t.mean():.2f} | R0={R0.mean():.2f}")

    idx   = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    xa    = _letkf_step(xf_grid, hxf, yo, R_t,
                        pos_km[idx[0], idx[1], idx[2], 0],
                        pos_km[idx[0], idx[1], idx[2], 1],
                        pos_km[idx[0], idx[1], idx[2], 2],
                        loc_scales_km, pos_km)

    # Strictly enforce non-negative physical bounds on hydrometeors
    xa[:, :, :, :, var_idx["qr"]] = np.maximum(xa[:, :, :, :, var_idx["qr"]], 0.0)
    xa[:, :, :, :, var_idx["qs"]] = np.maximum(xa[:, :, :, :, var_idx["qs"]], 0.0)
    xa[:, :, :, :, var_idx["qg"]] = np.maximum(xa[:, :, :, :, var_idx["qg"]], 0.0)

    _log(3, f"  [AOEI Total] cycle finished in {time.time()-t_start:.3f}s")
    return dict(
        xa=xa,
        hxf=hxf,
        dep=(yo - hxf.mean(axis=1)).astype(np.float32),
        obs_error_raw=R0,
        obs_error=R_t,
    )