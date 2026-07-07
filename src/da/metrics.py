import warnings
import numpy as np

def _nanmean_quiet(field):
    """Mean of `field` dropping non-finite values (NaN/Inf — e.g. skew/kurt undefined or
    ill-conditioned at a near-zero-variance ensemble point), staying silent on an all-invalid
    slice (expected for a degenerate, e.g. single-member, ensemble) rather than warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(np.where(np.isfinite(field), field, np.nan)))

def _weighted_rmse(field, weights):
    w = np.where(np.isnan(weights), 0.0, weights)
    w_sum = w.sum()
    if w_sum == 0.0: return np.nan
    return float(np.sqrt((w * field ** 2).sum() / w_sum))

def _weighted_mean(field, weights):
    w = np.where(np.isnan(weights), 0.0, weights)
    w_sum = w.sum()
    if w_sum == 0.0: return np.nan
    return float((w * field).sum() / w_sum)

def _weighted_spread(std_field, weights, Ne):
    w = np.where(np.isnan(weights), 0.0, weights)
    w_sum = w.sum()
    if w_sum == 0.0: return np.nan
    return float(np.sqrt((Ne + 1) / Ne * (w * std_field ** 2).sum() / w_sum))

def _unweighted_mask(rloc):
    return ~np.isnan(rloc) & (rloc > 0.0)

def _unweighted_rmse(field, mask):
    if mask.sum() == 0: return np.nan
    return float(np.sqrt((field[mask] ** 2).mean()))

def _unweighted_mean(field, mask):
    if mask.sum() == 0: return np.nan
    return float(field[mask].mean())

def _unweighted_spread(std_field, mask, Ne):
    if mask.sum() == 0: return np.nan
    return float(np.sqrt((Ne + 1) / Ne * (std_field[mask] ** 2).mean()))

def _weighted_mean_safe(field, weights):
    """Like _weighted_mean, but drops non-finite field values (NaN/Inf — e.g. skew/kurt
    undefined or ill-conditioned at a near-zero-variance ensemble point) instead of
    letting them poison the average."""
    w = np.where(np.isnan(weights) | ~np.isfinite(field), 0.0, weights)
    w_sum = w.sum()
    if w_sum == 0.0: return np.nan
    f = np.where(np.isfinite(field), field, 0.0)
    return float((w * f).sum() / w_sum)

def _unweighted_mean_safe(field, mask):
    """Like _unweighted_mean, but drops non-finite field values (NaN/Inf — e.g. skew/kurt
    undefined or ill-conditioned at a near-zero-variance ensemble point) instead of
    letting them poison the average."""
    m = mask & np.isfinite(field)
    if m.sum() == 0: return np.nan
    return float(field[m].mean())

def ensemble_skew(ens, axis=3):
    """Nerger (2022) Eq. 25 sample skewness along `axis` (NOT scipy's default normalization:
    numerator uses 1/Ne, denominator std uses 1/(Ne-1)). NaN wherever the result isn't finite
    — covers both an exactly-zero variance (0/0 = NaN already) and a numerically tiny-but-
    nonzero float32 variance, which can overflow the ratio to +/-Inf rather than NaN; guarding
    on output finiteness catches both instead of only the exact-zero-denominator case."""
    Ne = ens.shape[axis]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        dev = ens - ens.mean(axis=axis, keepdims=True)
        m3 = (dev ** 3).mean(axis=axis)
        var_unbiased = (dev ** 2).sum(axis=axis) / (Ne - 1)
        result = m3 / var_unbiased ** 1.5
    return np.where(np.isfinite(result), result, np.nan)

def ensemble_kurt(ens, axis=3):
    """Nerger (2022) Eq. 26 sample excess kurtosis along `axis` (NOT scipy's default
    normalization: both moments use 1/Ne). NaN wherever the result isn't finite — covers
    both an exactly-zero variance (0/0 = NaN already) and a numerically tiny-but-nonzero
    float32 variance, which can overflow the ratio to +/-Inf rather than NaN; guarding on
    output finiteness catches both instead of only the exact-zero-denominator case."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        dev = ens - ens.mean(axis=axis, keepdims=True)
        m4 = (dev ** 4).mean(axis=axis)
        m2 = (dev ** 2).mean(axis=axis)
        result = m4 / m2 ** 2 - 3.0
    return np.where(np.isfinite(result), result, np.nan)

def crps_ensemble(ens, truth, axis=3):
    """Empirical ensemble CRPS (Gneiting & Raftery 2007): mean|x_i-y| - 0.5*mean|x_i-x_j|.
    O(Ne^2) pairwise term — fine at the single-obs subdomain scale (small grids, Ne<=~60)."""
    ens = np.moveaxis(ens, axis, -1)
    term1 = np.abs(ens - truth[..., np.newaxis]).mean(axis=-1)
    term2 = np.abs(ens[..., :, None] - ens[..., None, :]).mean(axis=(-2, -1))
    return term1 - 0.5 * term2

def crps_ensemble_sorted(ens, truth, axis=3):
    """Memory-efficient O(Ne log Ne) ensemble CRPS via sorted order statistics — numerically
    equivalent to crps_ensemble but avoids the O(Ne^2) pairwise array; use for full-domain
    multi-obs fields where Ne^2 * nx * ny * nz * nvar would be too large to hold in memory."""
    ens = np.moveaxis(ens, axis, -1)
    Ne = ens.shape[-1]
    sorted_ens = np.sort(ens, axis=-1)
    term1 = np.abs(ens - truth[..., np.newaxis]).mean(axis=-1)
    weight = (2 * np.arange(1, Ne + 1) - Ne - 1)
    term2 = (weight * sorted_ens).sum(axis=-1) / (Ne ** 2)
    return term1 - term2

def compute_single_obs_metrics(
        xf_sub, xa_sub, truth_sub,
        ens_hx_sub, hxa_sub, truth_hx_sub,
        rloc, ox_s, oy_s, oz_s, yo, yo_clean, var_names, Ne
) -> dict:
    i0, j0, k0 = int(ox_s), int(oy_s), int(oz_s)  
    mask = _unweighted_mask(rloc)                   
    weights = rloc                                  
    loc_wsum = float(np.nansum(weights))
    n_updated = int(mask.sum())

    hxf_mean_sub = ens_hx_sub.mean(axis=3)
    hxa_mean_sub = hxa_sub.mean(axis=3)

    xf_mean = xf_sub.mean(axis=3)
    xa_mean = xa_sub.mean(axis=3)
    xf_std  = xf_sub.std(axis=3, ddof=1)
    xa_std  = xa_sub.std(axis=3, ddof=1)

    err_f_obs = hxf_mean_sub - truth_hx_sub
    err_a_obs = hxa_mean_sub - truth_hx_sub

    # Diagnostic fractional precipitation metric to map storm-edge conditions
    precip_fraction_f = float((hxf_mean_sub[mask] > 0.0).mean())

    # Nerger (2022)-style non-Gaussianity/skill diagnostics, obs (reflectivity) space
    skew_f_obs_field = ensemble_skew(ens_hx_sub, axis=3)
    skew_a_obs_field = ensemble_skew(hxa_sub, axis=3)
    kurt_f_obs_field = ensemble_kurt(ens_hx_sub, axis=3)
    kurt_a_obs_field = ensemble_kurt(hxa_sub, axis=3)
    crps_f_obs_field = crps_ensemble_sorted(ens_hx_sub, truth_hx_sub, axis=3)
    crps_a_obs_field = crps_ensemble_sorted(hxa_sub, truth_hx_sub, axis=3)

    out = dict(
        yo=float(yo),
        yo_clean=float(yo_clean),
        loc_weights_sum=loc_wsum,
        n_updated=n_updated,

        hxf_mean_obs=float(hxf_mean_sub[i0, j0, k0]),
        hxa_mean_obs=float(hxa_mean_sub[i0, j0, k0]),
        dep_b=float(yo) - float(hxf_mean_sub[i0, j0, k0]),
        dep_a=float(yo) - float(hxa_mean_sub[i0, j0, k0]),
        inc_obs=float(hxa_mean_sub[i0, j0, k0]) - float(hxf_mean_sub[i0, j0, k0]),
        spread_f_obs=float(ens_hx_sub[i0, j0, k0, :].std(ddof=1)),
        spread_a_obs=float(hxa_sub[i0, j0, k0, :].std(ddof=1)),

        rmse_f_point_obs=float(np.abs(err_f_obs[i0, j0, k0])),
        rmse_a_point_obs=float(np.abs(err_a_obs[i0, j0, k0])),
        rmse_f_w_obs=_weighted_rmse(err_f_obs, weights),
        rmse_a_w_obs=_weighted_rmse(err_a_obs, weights),
        rmse_f_u_obs=_unweighted_rmse(err_f_obs, mask),
        rmse_a_u_obs=_unweighted_rmse(err_a_obs, mask),

        hx_dbz_local_mean_w=_weighted_mean(hxf_mean_sub, weights),
        hx_dbz_local_mean_u=_unweighted_mean(hxf_mean_sub, mask),
        precip_fraction_f=precip_fraction_f,

        skew_f_point_obs=float(skew_f_obs_field[i0, j0, k0]),
        skew_a_point_obs=float(skew_a_obs_field[i0, j0, k0]),
        kurt_f_point_obs=float(kurt_f_obs_field[i0, j0, k0]),
        kurt_a_point_obs=float(kurt_a_obs_field[i0, j0, k0]),
        crps_f_point_obs=float(crps_f_obs_field[i0, j0, k0]),
        crps_a_point_obs=float(crps_a_obs_field[i0, j0, k0]),

        skew_f_w_obs=_weighted_mean_safe(np.abs(skew_f_obs_field), weights),
        skew_a_w_obs=_weighted_mean_safe(np.abs(skew_a_obs_field), weights),
        kurt_f_w_obs=_weighted_mean_safe(np.abs(kurt_f_obs_field), weights),
        kurt_a_w_obs=_weighted_mean_safe(np.abs(kurt_a_obs_field), weights),
        crps_f_w_obs=_weighted_mean(crps_f_obs_field, weights),
        crps_a_w_obs=_weighted_mean(crps_a_obs_field, weights),

        skew_f_u_obs=_unweighted_mean_safe(np.abs(skew_f_obs_field), mask),
        skew_a_u_obs=_unweighted_mean_safe(np.abs(skew_a_obs_field), mask),
        kurt_f_u_obs=_unweighted_mean_safe(np.abs(kurt_f_obs_field), mask),
        kurt_a_u_obs=_unweighted_mean_safe(np.abs(kurt_a_obs_field), mask),
        crps_f_u_obs=_unweighted_mean(crps_f_obs_field, mask),
        crps_a_u_obs=_unweighted_mean(crps_a_obs_field, mask),
    )

    for iv, vname in enumerate(var_names):
        err_f = xf_mean[..., iv] - truth_sub[..., iv]
        err_a = xa_mean[..., iv] - truth_sub[..., iv]
        std_f = xf_std[..., iv]
        std_a = xa_std[..., iv]

        out[f"rmse_f_point_{vname}"] = float(np.abs(err_f[i0, j0, k0]))
        out[f"rmse_a_point_{vname}"] = float(np.abs(err_a[i0, j0, k0]))
        out[f"spread_f_point_{vname}"] = float(std_f[i0, j0, k0])
        out[f"spread_a_point_{vname}"] = float(std_a[i0, j0, k0])

        out[f"rmse_f_w_{vname}"] = _weighted_rmse(err_f, weights)
        out[f"rmse_a_w_{vname}"] = _weighted_rmse(err_a, weights)
        out[f"spread_f_w_{vname}"] = _weighted_spread(std_f, weights, Ne)
        out[f"spread_a_w_{vname}"] = _weighted_spread(std_a, weights, Ne)

        out[f"rmse_f_u_{vname}"] = _unweighted_rmse(err_f, mask)
        out[f"rmse_a_u_{vname}"] = _unweighted_rmse(err_a, mask)
        out[f"spread_f_u_{vname}"] = _unweighted_spread(std_f, mask, Ne)
        out[f"spread_a_u_{vname}"] = _unweighted_spread(std_a, mask, Ne)

        skew_f = ensemble_skew(xf_sub[..., iv], axis=3)
        skew_a = ensemble_skew(xa_sub[..., iv], axis=3)
        kurt_f = ensemble_kurt(xf_sub[..., iv], axis=3)
        kurt_a = ensemble_kurt(xa_sub[..., iv], axis=3)
        crps_f = crps_ensemble_sorted(xf_sub[..., iv], truth_sub[..., iv], axis=3)
        crps_a = crps_ensemble_sorted(xa_sub[..., iv], truth_sub[..., iv], axis=3)

        out[f"skew_f_point_{vname}"] = float(skew_f[i0, j0, k0])
        out[f"skew_a_point_{vname}"] = float(skew_a[i0, j0, k0])
        out[f"kurt_f_point_{vname}"] = float(kurt_f[i0, j0, k0])
        out[f"kurt_a_point_{vname}"] = float(kurt_a[i0, j0, k0])
        out[f"crps_f_point_{vname}"] = float(crps_f[i0, j0, k0])
        out[f"crps_a_point_{vname}"] = float(crps_a[i0, j0, k0])

        out[f"skew_f_w_{vname}"] = _weighted_mean_safe(np.abs(skew_f), weights)
        out[f"skew_a_w_{vname}"] = _weighted_mean_safe(np.abs(skew_a), weights)
        out[f"kurt_f_w_{vname}"] = _weighted_mean_safe(np.abs(kurt_f), weights)
        out[f"kurt_a_w_{vname}"] = _weighted_mean_safe(np.abs(kurt_a), weights)
        out[f"crps_f_w_{vname}"] = _weighted_mean(crps_f, weights)
        out[f"crps_a_w_{vname}"] = _weighted_mean(crps_a, weights)

        out[f"skew_f_u_{vname}"] = _unweighted_mean_safe(np.abs(skew_f), mask)
        out[f"skew_a_u_{vname}"] = _unweighted_mean_safe(np.abs(skew_a), mask)
        out[f"kurt_f_u_{vname}"] = _unweighted_mean_safe(np.abs(kurt_f), mask)
        out[f"kurt_a_u_{vname}"] = _unweighted_mean_safe(np.abs(kurt_a), mask)
        out[f"crps_f_u_{vname}"] = _unweighted_mean(crps_f, mask)
        out[f"crps_a_u_{vname}"] = _unweighted_mean(crps_a, mask)

    return out

def compute_multi_obs_metrics(
        xa, xf, truth,
        hxf_mean_field, hxa_mean_field, truth_hx_field,
        var_names, Ne, store_fields=False
) -> dict:
    xf_mean = xf.mean(axis=3)
    xa_mean = xa.mean(axis=3)
    xf_std  = xf.std(axis=3, ddof=1)
    xa_std  = xa.std(axis=3, ddof=1) if xa.shape[3] > 1 else np.zeros_like(xf_std)

    err_hxf_field  = hxf_mean_field - truth_hx_field
    residual_field = hxa_mean_field - truth_hx_field

    abs_err_f_field = np.abs(xf_mean - truth)
    abs_err_a_field = np.abs(xa_mean - truth)
    bias_f_field    = xf_mean - truth
    bias_a_field    = xa_mean - truth

    # Nerger (2022)-style non-Gaussianity/skill diagnostics, full domain x nvar.
    # Uses the sorted O(Ne log Ne) CRPS estimator (not the O(Ne^2) pairwise one) —
    # the pairwise array would be nx*ny*nz*nvar*Ne^2 elements, far too large here.
    skew_f_field = ensemble_skew(xf, axis=3)
    kurt_f_field = ensemble_kurt(xf, axis=3)
    crps_f_field = crps_ensemble_sorted(xf, truth, axis=3)
    if xa.shape[3] > 1:
        skew_a_field = ensemble_skew(xa, axis=3)
        kurt_a_field = ensemble_kurt(xa, axis=3)
        crps_a_field = crps_ensemble_sorted(xa, truth, axis=3)
    else:
        skew_a_field = np.full_like(skew_f_field, np.nan)
        kurt_a_field = np.full_like(kurt_f_field, np.nan)
        crps_a_field = np.abs(xa_mean - truth)

    out = dict(
        hxf_mean_field=hxf_mean_field, hxa_mean_field=hxa_mean_field,
        truth_hx_field=truth_hx_field,
        err_hxf_field=err_hxf_field, residual_field=residual_field,
        abs_err_f_field=abs_err_f_field.astype(np.float32),
        abs_err_a_field=abs_err_a_field.astype(np.float32),
        bias_f_field=bias_f_field.astype(np.float32),
        bias_a_field=bias_a_field.astype(np.float32),
        spread_f_field=xf_std.astype(np.float32),
        spread_a_field=xa_std.astype(np.float32),
        skew_f_field=skew_f_field.astype(np.float32),
        skew_a_field=skew_a_field.astype(np.float32),
        kurt_f_field=kurt_f_field.astype(np.float32),
        kurt_a_field=kurt_a_field.astype(np.float32),
        crps_f_field=crps_f_field.astype(np.float32),
        crps_a_field=crps_a_field.astype(np.float32),
    )

    if store_fields:
        out["xf"] = xf.astype(np.float32)
        out["xa"] = xa.astype(np.float32)
        out["truth_state"] = truth.astype(np.float32)

    for iv, vname in enumerate(var_names):
        err_f = xf_mean[..., iv] - truth[..., iv]
        err_a = xa_mean[..., iv] - truth[..., iv]
        out[f"rmse_f_global_{vname}"] = float(np.sqrt((err_f ** 2).mean()))
        out[f"rmse_a_global_{vname}"] = float(np.sqrt((err_a ** 2).mean()))
        out[f"bias_f_global_{vname}"] = float(err_f.mean())
        out[f"bias_a_global_{vname}"] = float(err_a.mean())

        out[f"spread_f_global_{vname}"] = float(np.sqrt((Ne+1)/Ne * (xf_std[..., iv]**2).mean()))
        out[f"spread_a_global_{vname}"] = float(np.sqrt((Ne+1)/Ne * (xa_std[..., iv]**2).mean()))

        out[f"skew_f_global_{vname}"] = _nanmean_quiet(np.abs(skew_f_field[..., iv]))
        out[f"skew_a_global_{vname}"] = _nanmean_quiet(np.abs(skew_a_field[..., iv]))
        out[f"kurt_f_global_{vname}"] = _nanmean_quiet(np.abs(kurt_f_field[..., iv]))
        out[f"kurt_a_global_{vname}"] = _nanmean_quiet(np.abs(kurt_a_field[..., iv]))
        out[f"crps_f_global_{vname}"] = _nanmean_quiet(crps_f_field[..., iv])
        out[f"crps_a_global_{vname}"] = _nanmean_quiet(crps_a_field[..., iv])

    return out