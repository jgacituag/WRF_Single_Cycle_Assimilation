"""
metrics.py
----------
compute_single_obs_metrics(...)
    Returns a flat dict of scalars for one single-observation experiment.
    All weighted metrics use the Gaussian localization weight field rloc.
    Both weighted (rloc-weighted) and unweighted (uniform over updated
    points, i.e. where rloc > 0) versions are provided.

compute_multi_obs_metrics(...)
    Returns a dict of scalars and fields for a multiple-observation
    experiment. Includes global scalars and (nx,ny,nz) fields.

Variable naming convention
--------------------------
  _f   : forecast (prior)
  _a   : analysis (posterior)
  _obs : in observation space (reflectivity)
  _w   : weighted by rloc
  _u   : unweighted, uniform over updated points (rloc > 0)
  _var : per state variable
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensemble_mean_std(arr):
    """arr: (..., Ne) -> mean (...), std (...)"""
    return arr.mean(axis=-1), arr.std(axis=-1, ddof=1)


def _weighted_rmse(field, weights):
    """
    Weighted RMSE:  sqrt( sum(w * field^2) / sum(w) )
    field, weights: same shape (nx, ny, nz).
    NaN weights are excluded.
    """
    w = np.where(np.isnan(weights), 0.0, weights)
    w_sum = w.sum()
    if w_sum == 0.0:
        return np.nan
    return float(np.sqrt((w * field ** 2).sum() / w_sum))


def _weighted_bias(field, weights):
    """
    Weighted bias:  sum(w * field) / sum(w)
    field = (mean - truth), same sign convention throughout.
    """
    w = np.where(np.isnan(weights), 0.0, weights)
    w_sum = w.sum()
    if w_sum == 0.0:
        return np.nan
    return float((w * field).sum() / w_sum)


def _weighted_spread(std_field, weights):
    """
    Weighted spread:  sqrt( sum(w * std^2) / sum(w) )
    std_field: ensemble std at each grid point.
    """
    w = np.where(np.isnan(weights), 0.0, weights)
    w_sum = w.sum()
    if w_sum == 0.0:
        return np.nan
    return float(np.sqrt((w * std_field ** 2).sum() / w_sum))


def _unweighted_mask(rloc):
    """Boolean mask: True where rloc is not NaN (inside cutoff)."""
    return ~np.isnan(rloc)


def _unweighted_rmse(field, mask):
    n = mask.sum()
    if n == 0:
        return np.nan
    return float(np.sqrt((field[mask] ** 2).mean()))


def _unweighted_bias(field, mask):
    n = mask.sum()
    if n == 0:
        return np.nan
    return float(field[mask].mean())


def _unweighted_spread(std_field, mask):
    n = mask.sum()
    if n == 0:
        return np.nan
    return float(np.sqrt((std_field[mask] ** 2).mean()))


def _hx_domain(xa, var_idx):
    """
    Compute H(x) = reflectivity at every grid point for the ensemble mean.
    xa     : (nx, ny, nz, Ne, nvar)
    returns: (nx, ny, nz) float32
    """
    from .core import _get_cda
    cda = _get_cda()
    nx, ny, nz, Ne, _ = xa.shape
    xa_mean = xa.mean(axis=3)          # (nx, ny, nz, nvar)
    out = np.empty((nx, ny, nz), dtype=np.float32)
    vi = var_idx
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = cda.calc_ref(
                    xa_mean[i, j, k, vi["qr"]],
                    xa_mean[i, j, k, vi["qs"]],
                    xa_mean[i, j, k, vi["qg"]],
                    xa_mean[i, j, k, vi["T"]],
                    xa_mean[i, j, k, vi["P"]],
                )
    return out


def _hx_domain_truth(truth, var_idx):
    """
    Compute H(x) = reflectivity at every grid point for a single state.
    truth  : (nx, ny, nz, nvar)
    returns: (nx, ny, nz) float32
    """
    from .core import _get_cda
    cda = _get_cda()
    nx, ny, nz, _ = truth.shape
    out = np.empty((nx, ny, nz), dtype=np.float32)
    vi = var_idx
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = cda.calc_ref(
                    truth[i, j, k, vi["qr"]],
                    truth[i, j, k, vi["qs"]],
                    truth[i, j, k, vi["qg"]],
                    truth[i, j, k, vi["T"]],
                    truth[i, j, k, vi["P"]],
                )
    return out

def compute_single_obs_metrics(
        xf,              # (nx, ny, nz, Ne, nvar)  prior ensemble
        xa,              # (nx, ny, nz, Ne, nvar)  posterior ensemble
        truth,           # (nx, ny, nz, nvar)       truth state (single member)
        rloc,            # (nx, ny, nz)             localization weights (NaN outside cutoff)
        hxf_at_obs,      # (Ne,)                    H(xf) ensemble at obs point
        yo,              # scalar                   observed value
        var_idx,         # dict                     variable index mapping
        var_names,       # list[str]                variable names in index order
        hxf_mean_field=None,   # (nx,ny,nz) precomputed H(xf_mean) — optional
        truth_hx_field=None,   # (nx,ny,nz) precomputed H(truth)   — optional
) -> dict:
    """
    Compute all metrics for one single-observation experiment.

    hxf_mean_field and truth_hx_field are the same for every obs point
    sharing the same prior/truth pair. Pass them in as precomputed arrays
    to avoid recomputing nx*ny*nz H(x) calls per obs point.

    Returns a flat dict of scalars. For per-variable metrics the key
    includes the variable name, e.g. "rmse_f_w_qr", "bias_a_u_T", etc.

    Metrics computed
    ----------------
    Observation space (at obs point only):
      yo                   observed value
      hxf_mean_obs         prior ensemble mean H(x) at obs point
      hxa_mean_obs         analysis ensemble mean H(x) at obs point
      dep_b                innovation  yo - H(xf_mean)
      dep_a                residual    yo - H(xa_mean)
      spread_f_obs         prior ensemble spread in obs space at obs point
      spread_a_obs         analysis ensemble spread in obs space at obs point
      inc_obs              analysis increment H(xa_mean) - H(xf_mean)

    Localization region summary:
      loc_weights_sum      sum of rloc over domain (effective support size)
      n_updated            number of grid points inside cutoff (rloc > 0)

    Per state variable, weighted (suffix _w) and unweighted (suffix _u):
      rmse_f_w_{v}, rmse_a_w_{v}    weighted RMSE
      bias_f_w_{v}, bias_a_w_{v}    weighted bias
      spread_f_w_{v}, spread_a_w_{v} weighted spread
      rmse_f_u_{v}, rmse_a_u_{v}    unweighted RMSE  (over updated points)
      bias_f_u_{v}, bias_a_u_{v}    unweighted bias
      spread_f_u_{v}, spread_a_u_{v} unweighted spread
      xf_mean_{v}, xa_mean_{v}, truth_{v}  point values at obs location

    Obs space weighted and unweighted over localization region:
      rmse_f_obs_w, rmse_a_obs_w, bias_f_obs_w, bias_a_obs_w
      rmse_f_obs_u, rmse_a_obs_u, bias_f_obs_u, bias_a_obs_u
    """
    from .core import _get_cda
    cda = _get_cda()
    vi  = var_idx

    # ---- obs point location (centre of rloc — always weight == 1.0) --------
    i0, j0, k0 = np.unravel_index(np.nanargmax(rloc), rloc.shape)

    # ---- localization region -----------------------------------------------
    mask      = _unweighted_mask(rloc)   # (nx, ny, nz) bool
    loc_wsum  = float(np.nansum(rloc))
    n_updated = int(mask.sum())

    # ---- obs-space at the obs point ----------------------------------------
    hxf_mean_obs   = float(hxf_at_obs.mean())
    hxf_spread_obs = float(hxf_at_obs.std(ddof=1))

    # H(xa) at obs point — Ne calls to calc_ref, cheap
    Ne = xa.shape[3]
    hxa_at_obs = np.array([
        cda.calc_ref(
            xa[i0, j0, k0, m, vi["qr"]], xa[i0, j0, k0, m, vi["qs"]],
            xa[i0, j0, k0, m, vi["qg"]], xa[i0, j0, k0, m, vi["T"]],
            xa[i0, j0, k0, m, vi["P"]],
        ) for m in range(Ne)
    ], dtype=np.float32)
    hxa_mean_obs   = float(hxa_at_obs.mean())
    hxa_spread_obs = float(hxa_at_obs.std(ddof=1))

    dep_b   = float(yo) - hxf_mean_obs
    dep_a   = float(yo) - hxa_mean_obs
    inc_obs = hxa_mean_obs - hxf_mean_obs

    # ---- precomputed fields (compute here only if not provided) ------------
    if hxf_mean_field is None:
        hxf_mean_field = _hx_domain(xf, var_idx)
    if truth_hx_field is None:
        truth_hx_field = _hx_domain_truth(truth, var_idx)

    # ---- H(xa_mean) restricted to updated points only ---------------------
    # Only compute H(xa) where rloc > 0 — avoids nx*ny*nz calc_ref calls
    xa_mean_full = xa.mean(axis=3)   # (nx, ny, nz, nvar)
    pts = np.argwhere(mask)          # (n_updated, 3)
    hxa_mean_field = np.full(rloc.shape, np.nan, dtype=np.float32)
    for (i, j, k) in pts:
        hxa_mean_field[i, j, k] = cda.calc_ref(
            xa_mean_full[i, j, k, vi["qr"]], xa_mean_full[i, j, k, vi["qs"]],
            xa_mean_full[i, j, k, vi["qg"]], xa_mean_full[i, j, k, vi["T"]],
            xa_mean_full[i, j, k, vi["P"]],
        )

    # ---- obs-space error fields (NaN outside mask — harmless for metrics) --
    err_f_obs = hxf_mean_field - truth_hx_field   # (nx, ny, nz)
    err_a_obs = hxa_mean_field - truth_hx_field

    # ---- per-variable fields -----------------------------------------------
    xf_mean = xf.mean(axis=3)        # (nx, ny, nz, nvar)
    xa_mean = xa_mean_full
    xf_std  = xf.std(axis=3, ddof=1)
    xa_std  = xa.std(axis=3, ddof=1)

    out = dict(
        yo=float(yo),
        hxf_mean_obs=hxf_mean_obs,
        hxa_mean_obs=hxa_mean_obs,
        dep_b=dep_b,
        dep_a=dep_a,
        spread_f_obs=hxf_spread_obs,
        spread_a_obs=hxa_spread_obs,
        inc_obs=inc_obs,
        loc_weights_sum=loc_wsum,
        n_updated=n_updated,
        # obs-space weighted
        rmse_f_obs_w=_weighted_rmse(err_f_obs,  rloc),
        rmse_a_obs_w=_weighted_rmse(err_a_obs,  rloc),
        bias_f_obs_w=_weighted_bias(err_f_obs,  rloc),
        bias_a_obs_w=_weighted_bias(err_a_obs,  rloc),
        # obs-space unweighted
        rmse_f_obs_u=_unweighted_rmse(err_f_obs, mask),
        rmse_a_obs_u=_unweighted_rmse(err_a_obs, mask),
        bias_f_obs_u=_unweighted_bias(err_f_obs, mask),
        bias_a_obs_u=_unweighted_bias(err_a_obs, mask),
    )

    for iv, vname in enumerate(var_names):
        err_f = xf_mean[..., iv] - truth[..., iv]
        err_a = xa_mean[..., iv] - truth[..., iv]
        std_f = xf_std[..., iv]
        std_a = xa_std[..., iv]

        out[f"rmse_f_w_{vname}"]    = _weighted_rmse(err_f,  rloc)
        out[f"rmse_a_w_{vname}"]    = _weighted_rmse(err_a,  rloc)
        out[f"bias_f_w_{vname}"]    = _weighted_bias(err_f,  rloc)
        out[f"bias_a_w_{vname}"]    = _weighted_bias(err_a,  rloc)
        out[f"spread_f_w_{vname}"]  = _weighted_spread(std_f, rloc)
        out[f"spread_a_w_{vname}"]  = _weighted_spread(std_a, rloc)

        out[f"rmse_f_u_{vname}"]    = _unweighted_rmse(err_f,  mask)
        out[f"rmse_a_u_{vname}"]    = _unweighted_rmse(err_a,  mask)
        out[f"bias_f_u_{vname}"]    = _unweighted_bias(err_f,  mask)
        out[f"bias_a_u_{vname}"]    = _unweighted_bias(err_a,  mask)
        out[f"spread_f_u_{vname}"]  = _unweighted_spread(std_f, mask)
        out[f"spread_a_u_{vname}"]  = _unweighted_spread(std_a, mask)

        out[f"xf_mean_{vname}"] = float(xf_mean[i0, j0, k0, iv])
        out[f"xa_mean_{vname}"] = float(xa_mean[i0, j0, k0, iv])
        out[f"truth_{vname}"]   = float(truth[i0, j0, k0, iv])

    return out


def compute_multi_obs_metrics(
        xf,        # (nx, ny, nz, Ne, nvar)
        xa,        # (nx, ny, nz, Ne, nvar)
        truth,     # (nx, ny, nz, nvar)
        yo,        # (nobs,)
        ox, oy, oz,# (nobs,) int, 0-based
        var_idx,
        var_names,
) -> dict:
    """
    Compute metrics for a multiple-observation experiment.

    Returns a dict with:
      - full matrices xf, xa, truth_state
      - (nvar,) global scalar RMSE, bias, spread for forecast and analysis
      - (nx,ny,nz) pointwise RMSE, bias, spread fields
      - H(xf) and H(xa) mean fields (nx,ny,nz)
      - innovation and residual fields (nx,ny,nz)
      - obs arrays yo, ox, oy, oz
    """
    xf_mean = xf.mean(axis=3)   # (nx, ny, nz, nvar)
    xa_mean = xa.mean(axis=3)
    xf_std  = xf.std(axis=3, ddof=1)
    xa_std  = xa.std(axis=3, ddof=1)

    hxf_mean_field = _hx_domain(xf, var_idx)   # (nx,ny,nz)
    hxa_mean_field = _hx_domain(xa, var_idx)
    truth_hx_field = _hx_domain_truth(truth, var_idx)

    innovation_field = hxf_mean_field - truth_hx_field   # (nx,ny,nz)
    residual_field   = hxa_mean_field - truth_hx_field

    # pointwise RMSE/bias/spread fields (nx,ny,nz) — mean over variables
    err_f_field = np.sqrt(((xf_mean - truth) ** 2).mean(axis=-1))  # (nx,ny,nz)
    err_a_field = np.sqrt(((xa_mean  - truth) ** 2).mean(axis=-1))
    bias_f_field = (xf_mean - truth).mean(axis=-1)
    bias_a_field = (xa_mean  - truth).mean(axis=-1)
    spread_f_field = xf_std.mean(axis=-1)
    spread_a_field = xa_std.mean(axis=-1)

    out = dict(
        xf=xf.astype(np.float32),
        xa=xa.astype(np.float32),
        truth_state=truth.astype(np.float32),
        yo=np.asarray(yo, np.float32),
        ox=np.asarray(ox, np.int32),
        oy=np.asarray(oy, np.int32),
        oz=np.asarray(oz, np.int32),
        hxf_mean_field=hxf_mean_field,
        hxa_mean_field=hxa_mean_field,
        truth_hx_field=truth_hx_field,
        innovation_field=innovation_field,
        residual_field=residual_field,
        rmse_f_field=err_f_field.astype(np.float32),
        rmse_a_field=err_a_field.astype(np.float32),
        bias_f_field=bias_f_field.astype(np.float32),
        bias_a_field=bias_a_field.astype(np.float32),
        spread_f_field=spread_f_field.astype(np.float32),
        spread_a_field=spread_a_field.astype(np.float32),
    )

    # global scalars per variable
    for iv, vname in enumerate(var_names):
        err_f = xf_mean[..., iv] - truth[..., iv]
        err_a = xa_mean[..., iv] - truth[..., iv]
        out[f"rmse_f_global_{vname}"]   = float(np.sqrt((err_f ** 2).mean()))
        out[f"rmse_a_global_{vname}"]   = float(np.sqrt((err_a ** 2).mean()))
        out[f"bias_f_global_{vname}"]   = float(err_f.mean())
        out[f"bias_a_global_{vname}"]   = float(err_a.mean())
        out[f"spread_f_global_{vname}"] = float(xf_std[..., iv].mean())
        out[f"spread_a_global_{vname}"] = float(xa_std[..., iv].mean())

    return out
