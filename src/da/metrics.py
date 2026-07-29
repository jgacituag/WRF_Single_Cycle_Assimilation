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
    # float64 accumulator: a float32 sum over Ne members is wrong by ~1e-3 when one member
    # dominates (hydrometeors are typically 0 in all but a few members), and numpy only uses
    # pairwise summation when the reduction axis is contiguous, so the float32 result also
    # depended on the caller's memory layout. term2 is already float64 via the int weights.
    term1 = np.abs(ens - truth[..., np.newaxis]).mean(axis=-1, dtype=np.float64)
    weight = (2 * np.arange(1, Ne + 1) - Ne - 1)
    term2 = (weight * sorted_ens).sum(axis=-1) / (Ne ** 2)
    return term1 - term2

def _per_var(fn, ens, *extra):
    """Apply a member-axis reduction one state variable at a time, then restack.

    `fn(ens[..., iv], *(e[..., iv] for e in extra))` must return a (nx,ny,nz) field.
    Numerically identical to calling `fn` on the whole 5-D array, but the largest
    temporary is (nx,ny,nz,Ne) rather than (nx,ny,nz,Ne,nvar). At the full 558x898x11
    domain with Ne=59 that is 1.3 GB instead of 10.4 GB, and np.sort / dev**4 / the
    float64 CRPS weighting each allocate one of them.
    """
    nvar = ens.shape[-1]
    first = fn(ens[..., 0], *(e[..., 0] for e in extra))
    out = np.empty(first.shape + (nvar,), dtype=first.dtype)
    out[..., 0] = first
    for iv in range(1, nvar):
        out[..., iv] = fn(ens[..., iv], *(e[..., iv] for e in extra))
    return out

def compute_single_obs_metrics(
        xf_sub, xa_sub, truth_sub,
        ens_hx_sub, hxa_sub, truth_hx_sub,
        rloc, ox_s, oy_s, oz_s, yo, yo_clean, var_names, Ne, dbz_min=0.0
) -> dict:
    i0, j0, k0 = int(ox_s), int(oy_s), int(oz_s)
    mask = _unweighted_mask(rloc)                   
    weights = rloc                                  
    loc_wsum = float(np.nansum(weights))
    n_updated = int(mask.sum())

    hxf_mean_sub = ens_hx_sub.mean(axis=3)
    hxa_mean_sub = hxa_sub.mean(axis=3)
    hxf_std_sub  = ens_hx_sub.std(axis=3, ddof=1)
    hxa_std_sub  = hxa_sub.std(axis=3, ddof=1)

    # Members carrying signal at each subdomain point (prior). Strict > dbz_min matches the
    # clamped obs floor, so a fully clear-air point counts 0.
    n_active_f_sub = (ens_hx_sub > dbz_min).sum(axis=3)

    xf_mean = xf_sub.mean(axis=3)
    xa_mean = xa_sub.mean(axis=3)
    xf_std  = xf_sub.std(axis=3, ddof=1)
    xa_std  = xa_sub.std(axis=3, ddof=1)

    err_f_obs = hxf_mean_sub - truth_hx_sub
    err_a_obs = hxa_mean_sub - truth_hx_sub

    # Diagnostic fractional precipitation metric to map storm-edge conditions. Thresholded
    # on the config's dbz_min (not a hardcoded 0) so it stays consistent with the obs floor
    # the forward operator was clamped to.
    precip_fraction_f = float((hxf_mean_sub[mask] > dbz_min).mean())

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

        # Canonically-named aliases of spread_{f,a}_obs, matching spread_*_point_{vname}.
        # The originals are kept so existing notebooks keep resolving.
        spread_f_point_obs=float(hxf_std_sub[i0, j0, k0]),
        spread_a_point_obs=float(hxa_std_sub[i0, j0, k0]),
        spread_f_w_obs=_weighted_spread(hxf_std_sub, weights, Ne),
        spread_a_w_obs=_weighted_spread(hxa_std_sub, weights, Ne),
        spread_f_u_obs=_unweighted_spread(hxf_std_sub, mask, Ne),
        spread_a_u_obs=_unweighted_spread(hxa_std_sub, mask, Ne),

        # Signed error against truth. Distinct from dep_b/dep_a, which are yo - H(x) and so
        # carry the observation noise; these are H(x) - H(truth).
        bias_f_point_obs=float(err_f_obs[i0, j0, k0]),
        bias_a_point_obs=float(err_a_obs[i0, j0, k0]),
        bias_f_w_obs=_weighted_mean(err_f_obs, weights),
        bias_a_w_obs=_weighted_mean(err_a_obs, weights),
        bias_f_u_obs=_unweighted_mean(err_f_obs, mask),
        bias_a_u_obs=_unweighted_mean(err_a_obs, mask),

        n_active_f_point=float(n_active_f_sub[i0, j0, k0]),
        n_active_f_w=_weighted_mean(n_active_f_sub.astype(np.float32), weights),
        n_active_f_u=_unweighted_mean(n_active_f_sub.astype(np.float32), mask),

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

        # Signed ensemble-mean state value at the obs point (forecast, analysis, truth).
        # Lets a per-point single-obs cross-section be reconstructed from sweep output
        # (the rmse_*_point_* keys only store |mean - truth|, so sign/magnitude of e.g. w
        # are otherwise unrecoverable). Increment = mean_a_point - mean_f_point.
        out[f"mean_f_point_{vname}"] = float(xf_mean[i0, j0, k0, iv])
        out[f"mean_a_point_{vname}"] = float(xa_mean[i0, j0, k0, iv])
        out[f"truth_point_{vname}"]  = float(truth_sub[i0, j0, k0, iv])

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

        # Signed error over the localization volume. The _point_ variants are omitted --
        # they are mean_{f,a}_point_{vname} - truth_point_{vname}, both stored above.
        out[f"bias_f_w_{vname}"] = _weighted_mean(err_f, weights)
        out[f"bias_a_w_{vname}"] = _weighted_mean(err_a, weights)
        out[f"bias_f_u_{vname}"] = _unweighted_mean(err_f, mask)
        out[f"bias_a_u_{vname}"] = _unweighted_mean(err_a, mask)

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
        var_names, Ne, store_fields=False,
        ens_hxf=None, ens_hxa=None, dbz_min=0.0
) -> dict:
    """Full-domain metrics for multi_obs mode.

    `ens_hxf`/`ens_hxa` are the prior/posterior obs-space (reflectivity) ensembles,
    (nx, ny, nz, Ne). When supplied, reflectivity gets the same metric families as the
    state variables under the `_ref` suffix; the mean-only fields (err_hxf/residual) are
    emitted either way. They default to None so a caller that only has the means still
    works, but the runner always passes them -- without them there is no member axis to
    reduce and every `*_ref` key would be missing.
    """
    xf_mean = xf.mean(axis=3)
    xa_mean = xa.mean(axis=3)
    xf_std  = _per_var(lambda e: e.std(axis=3, ddof=1), xf)
    xa_std  = _per_var(lambda e: e.std(axis=3, ddof=1), xa) if xa.shape[3] > 1 else np.zeros_like(xf_std)

    err_hxf_field  = hxf_mean_field - truth_hx_field
    residual_field = hxa_mean_field - truth_hx_field

    abs_err_f_field = np.abs(xf_mean - truth)
    abs_err_a_field = np.abs(xa_mean - truth)
    bias_f_field    = xf_mean - truth
    bias_a_field    = xa_mean - truth

    # Nerger (2022)-style non-Gaussianity/skill diagnostics, full domain x nvar.
    # Uses the sorted O(Ne log Ne) CRPS estimator (not the O(Ne^2) pairwise one) —
    # the pairwise array would be nx*ny*nz*nvar*Ne^2 elements, far too large here.
    # Each reduction runs one state variable at a time via _per_var: at Ne=59 the
    # whole-array form peaks around +30 GB of temporaries, which OOMs the node.
    skew_f_field = _per_var(lambda e: ensemble_skew(e, axis=3), xf)
    kurt_f_field = _per_var(lambda e: ensemble_kurt(e, axis=3), xf)
    crps_f_field = _per_var(lambda e, t: crps_ensemble_sorted(e, t, axis=3), xf, truth)
    if xa.shape[3] > 1:
        skew_a_field = _per_var(lambda e: ensemble_skew(e, axis=3), xa)
        kurt_a_field = _per_var(lambda e: ensemble_kurt(e, axis=3), xa)
        crps_a_field = _per_var(lambda e, t: crps_ensemble_sorted(e, t, axis=3), xa, truth)
    else:
        skew_a_field = np.full_like(skew_f_field, np.nan)
        kurt_a_field = np.full_like(kurt_f_field, np.nan)
        crps_a_field = np.abs(xa_mean - truth)

    # Same diagnostics in obs (reflectivity) space, under the `_ref` suffix. These are
    # (nx,ny,nz) -- no trailing var axis -- so _per_var does not apply and the reductions
    # run directly. Each one allocates a (nx,ny,nz,Ne) temporary (np.sort, the abs-diff,
    # dev**4), i.e. the same peak as one extra state variable; they are freed as we go.
    ref = {}
    if ens_hxf is not None:
        ref["spread_f_ref_field"] = ens_hxf.std(axis=3, ddof=1)
        ref["skew_f_ref_field"]   = ensemble_skew(ens_hxf, axis=3)
        ref["kurt_f_ref_field"]   = ensemble_kurt(ens_hxf, axis=3)
        ref["crps_f_ref_field"]   = crps_ensemble_sorted(ens_hxf, truth_hx_field, axis=3)
        # Members carrying signal: strict > dbz_min matches the clamped obs floor, so a
        # fully clear-air point counts 0. int16 is ample for any realistic Ne.
        ref["n_active_f_field"]   = (ens_hxf > dbz_min).sum(axis=3).astype(np.int16)

        if ens_hxa is not None and ens_hxa.shape[3] > 1:
            ref["spread_a_ref_field"] = ens_hxa.std(axis=3, ddof=1)
            ref["skew_a_ref_field"]   = ensemble_skew(ens_hxa, axis=3)
            ref["kurt_a_ref_field"]   = ensemble_kurt(ens_hxa, axis=3)
            ref["crps_a_ref_field"]   = crps_ensemble_sorted(ens_hxa, truth_hx_field, axis=3)
        else:
            # Degenerate (e.g. Ne=1) posterior: skew/kurt undefined, CRPS -> |x-y|.
            ref["spread_a_ref_field"] = np.zeros_like(ref["spread_f_ref_field"])
            ref["skew_a_ref_field"]   = np.full_like(ref["skew_f_ref_field"], np.nan)
            ref["kurt_a_ref_field"]   = np.full_like(ref["kurt_f_ref_field"], np.nan)
            ref["crps_a_ref_field"]   = np.abs(residual_field)

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

    # abs_err/bias in obs space are deliberately not stored as `_ref` fields: they are
    # exactly err_hxf_field / residual_field (and their abs), already above. Only the
    # derived `_global_ref` scalars are emitted, so scalar iteration stays uniform.
    for _k, _v in ref.items():
        out[_k] = _v.astype(np.float32) if _v.dtype.kind == "f" else _v

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

    # Reflectivity globals, same reducers as the state loop above so `_global_ref` can be
    # read alongside `_global_{vname}` for vname in var_names.
    out["rmse_f_global_ref"] = float(np.sqrt((err_hxf_field ** 2).mean()))
    out["rmse_a_global_ref"] = float(np.sqrt((residual_field ** 2).mean()))
    out["bias_f_global_ref"] = float(err_hxf_field.mean())
    out["bias_a_global_ref"] = float(residual_field.mean())

    if ref:
        out["spread_f_global_ref"] = float(np.sqrt((Ne+1)/Ne * (ref["spread_f_ref_field"]**2).mean()))
        out["spread_a_global_ref"] = float(np.sqrt((Ne+1)/Ne * (ref["spread_a_ref_field"]**2).mean()))
        out["skew_f_global_ref"]   = _nanmean_quiet(np.abs(ref["skew_f_ref_field"]))
        out["skew_a_global_ref"]   = _nanmean_quiet(np.abs(ref["skew_a_ref_field"]))
        out["kurt_f_global_ref"]   = _nanmean_quiet(np.abs(ref["kurt_f_ref_field"]))
        out["kurt_a_global_ref"]   = _nanmean_quiet(np.abs(ref["kurt_a_ref_field"]))
        out["crps_f_global_ref"]   = _nanmean_quiet(ref["crps_f_ref_field"])
        out["crps_a_global_ref"]   = _nanmean_quiet(ref["crps_a_ref_field"])
        out["n_active_f_global"]   = float(ref["n_active_f_field"].mean())

    return out
