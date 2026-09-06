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

# ---------------------------------------------------------------------------
# Evaluation domains and protected reductions
# ---------------------------------------------------------------------------

STORM_THRESH_DBZ = 20.0


def domain_masks(truth_hx_field, obs_ijk=None, storm_thresh=STORM_THRESH_DBZ):
    """The three evaluation domains for an (nx, ny, nz) field.

    global  every cell
    storm   COLUMNS whose truth column-max reflectivity >= storm_thresh, broadcast to
            all levels. Not "cells with echo": the clear air above and below a storm
            column is inside this domain, which is the point -- it is where the
            increment lands without an observation to justify it.
    obs     the cells carrying an assimilated observation

    The chapter's argument rests on these three disagreeing: single-step and AOEI
    degrade reflectivity globally and improve it inside storm columns, and the sign
    disagrees between domains in 24 of 60 experiment-scheme pairs. In light mode the
    fields are gone, so if the scalars are not restricted here that comparison cannot
    be made at all.

    Notebooks/nbcommon.py imports this rather than defining its own, so the light-mode
    scalar and the notebook's field-recomputed value cannot drift apart.
    """
    m = {"global": np.ones(truth_hx_field.shape, bool)}
    m["storm"] = np.broadcast_to(
        (np.nanmax(truth_hx_field, axis=2) >= storm_thresh)[:, :, None],
        truth_hx_field.shape)
    if obs_ijk is not None:
        o = np.zeros(truth_hx_field.shape, bool)
        o[obs_ijk[0], obs_ijk[1], obs_ijk[2]] = True
        m["obs"] = o
    return m


def _finite_reduce(field, mask, kind, Ne=None):
    """Reduce `field` over `mask`, dropping non-finite cells. Returns (value, n_used).

    Every aggregate goes through here -- RMSE, bias, spread and CRPS included, not
    only the shape metrics that had protected reducers before. The state fields carry
    non-finite cells (N2 §12: 41-56 per hour in dataset A, a per-cell source mask
    propagated by `np.ma.filled(..., np.nan)`), so a plain `.mean()` returns NaN and
    the scalar is unusable. The notebooks worked around it by recomputing from the
    fields; in light mode there are no fields to recompute from, so the scalar is the
    only source and the protection has to be here.

    `n_used` is the denominator that was actually used -- the count that entered, not
    the domain size. It is written beside every scalar, because a mean over 1,523,027
    cells and one over 1,522,832 are not the same number and nothing else on disk
    would say which happened. The dropped count is `n_cells_{domain}` minus this.

    RMSE is sqrt(mean(err**2)) over the surviving cells, never mean(|err|).
    The squares accumulate in float64: at 1.5 M cells a float32 sum of dBZ**2 loses
    the last digits of the answer.
    """
    v = field[mask]
    fin = np.isfinite(v)
    n_used = int(fin.sum())
    if n_used == 0:
        return np.nan, 0
    g = v[fin].astype(np.float64)
    if kind == "rmse":
        return float(np.sqrt((g ** 2).mean())), n_used
    if kind == "mean":
        return float(g.mean()), n_used
    if kind == "absmean":
        return float(np.abs(g).mean()), n_used
    if kind == "spread":
        return float(np.sqrt((Ne + 1) / Ne * (g ** 2).mean())), n_used
    raise ValueError(f"unknown reduction {kind!r}")


# metric name -> (which field family, how it reduces)
_DOMAIN_METRICS = (
    ("rmse",   "err",    "rmse"),
    ("bias",   "err",    "mean"),
    ("spread", "std",    "spread"),
    ("crps",   "crps",   "mean"),
    ("skew",   "skew",   "absmean"),
    ("kurt",   "kurt",   "absmean"),
)


def _emit_domain_scalars(out, domains, var_key, fields, Ne):
    """Write every metric x domain x {prior, analysis} scalar for one variable.

    `fields` maps (family, 'f'|'a') -> an (nx, ny, nz) field. Missing entries are
    skipped rather than faked, so a caller without a posterior ensemble still gets the
    prior half instead of a file full of NaN.
    """
    for dname, dmask in domains.items():
        for metric, family, kind in _DOMAIN_METRICS:
            for side in ("f", "a"):
                fld = fields.get((family, side))
                if fld is None:
                    continue
                val, n_used = _finite_reduce(fld, dmask, kind, Ne=Ne)
                out[f"{metric}_{side}_{dname}_{var_key}"] = val
                out[f"n_{metric}_{side}_{dname}_{var_key}"] = n_used


def _untouched_mask(xa, xf):
    """Cells where the analysis equals the prior EXACTLY, in every member and variable.

    simple_letkf_wloc copies the prior verbatim at any grid point with no observation
    inside the localization cutoff, so this is the complement of the set the update
    reached. The notebooks recompute it from the ensembles; in light mode those are
    gone, which is why it is reduced to scalars here.

    Accumulated one variable at a time: the whole-array comparison would allocate an
    (nx, ny, nz, Ne, nvar) boolean -- 2.6 GB at the full domain with Ne=59.

    A plain `xa == xf` is WRONG here, and silently: NaN == NaN is False, so every cell
    the source data left non-finite would be reported as touched by an update that never
    reached it. On dataset A at 20 UTC that inflated n_touched by 53 of 7,188 -- the
    exact number of cells non-finite in the prior. Two NaNs in the same position mean
    the value did not change, so they count as equal.
    """
    acc = np.ones(xa.shape[:3], bool)
    for iv in range(xa.shape[-1]):
        a, f = xa[..., iv], xf[..., iv]
        same = (a == f) | (np.isnan(a) & np.isnan(f))
        acc &= same.all(axis=3)
    return acc


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

# Obs-space (reflectivity) fields. `output.store_ref_fields` alone keeps these and drops
# the per-state-variable ones; the `*_ref_field` keys are matched by suffix, not listed.
REF_FIELDS = ("hxf_mean_field", "hxa_mean_field", "truth_hx_field",
              "err_hxf_field", "residual_field", "n_active_f_field")


def _apply_storage_level(out, storage_level):
    """Drop metric FIELDS the storage level does not keep. Scalars always survive.

    The fields are computed either way -- every domain scalar is a reduction of one --
    so this changes what lands on disk and nothing else. A `full` multi-obs scheme file
    is ~520 MB; `ref` is ~21 MB; `light` is a few hundred kilobytes.

    The ensemble (`store_ensemble`) is deliberately outside this: it is orthogonal to
    which metric fields are kept, and it is the one thing here that scales with Ne.
    """
    if storage_level == "full":
        return out
    if storage_level not in ("ref", "light"):
        raise ValueError(f"storage_level must be light/ref/full, got {storage_level!r}")
    keep = {}
    for k, v in out.items():
        if not k.endswith("_field"):
            keep[k] = v                       # scalars, and the store_ensemble arrays
        elif storage_level == "ref" and (k in REF_FIELDS or "_ref_field" in k):
            keep[k] = v
    return keep


def compute_multi_obs_metrics(
        xa, xf, truth,
        hxf_mean_field, hxa_mean_field, truth_hx_field,
        var_names, Ne, store_ensemble=False, storage_level="full",
        ens_hxf=None, ens_hxa=None, dbz_min=0.0,
        obs_ijk=None, storm_thresh=STORM_THRESH_DBZ
) -> dict:
    """Full-domain metrics for multi_obs mode.

    `storage_level` selects what is KEPT, not what is computed:
      full   every metric field, state and reflectivity   (the historical behaviour)
      ref    the reflectivity fields only
      light  no fields at all -- per-domain scalars only
    The tag's REF / FULL flag must agree with it; see src/naming.py.

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

    if store_ensemble:
        out["xf"] = xf.astype(np.float32)
        out["xa"] = xa.astype(np.float32)
        out["truth_state"] = truth.astype(np.float32)

    # ---- domain-restricted scalars -----------------------------------------
    # These are the whole point of light mode: the fields above never reach disk, so
    # every number the chapter quotes has to survive as a scalar here. Each is a
    # reduction of a field that was computed anyway, so this costs passes over memory
    # and nothing else.
    domains = domain_masks(truth_hx_field, obs_ijk=obs_ijk, storm_thresh=storm_thresh)
    for dname, dmask in domains.items():
        out[f"n_cells_{dname}"] = int(np.count_nonzero(dmask))
    out["storm_thresh_dbz"] = float(storm_thresh)

    for iv, vname in enumerate(var_names):
        err_f = xf_mean[..., iv] - truth[..., iv]
        err_a = xa_mean[..., iv] - truth[..., iv]
        _emit_domain_scalars(out, domains, vname, {
            ("err",  "f"): err_f,          ("err",  "a"): err_a,
            ("std",  "f"): xf_std[..., iv], ("std",  "a"): xa_std[..., iv],
            ("crps", "f"): crps_f_field[..., iv], ("crps", "a"): crps_a_field[..., iv],
            ("skew", "f"): skew_f_field[..., iv], ("skew", "a"): skew_a_field[..., iv],
            ("kurt", "f"): kurt_f_field[..., iv], ("kurt", "a"): kurt_a_field[..., iv],
        }, Ne)

    # Reflectivity, same reducers and the same key shape, so `_global_ref` reads
    # alongside `_global_{vname}` for vname in var_names.
    ref_fields = {("err", "f"): err_hxf_field, ("err", "a"): residual_field}
    if ref:
        ref_fields.update({
            ("std",  "f"): ref["spread_f_ref_field"], ("std",  "a"): ref["spread_a_ref_field"],
            ("crps", "f"): ref["crps_f_ref_field"],   ("crps", "a"): ref["crps_a_ref_field"],
            ("skew", "f"): ref["skew_f_ref_field"],   ("skew", "a"): ref["skew_a_ref_field"],
            ("kurt", "f"): ref["kurt_f_ref_field"],   ("kurt", "a"): ref["kurt_a_ref_field"],
        })
    _emit_domain_scalars(out, domains, "ref", ref_fields, Ne)

    if ref:
        # n_active_f_global keeps its historical name and value; the two new domains
        # join it under the same pattern.
        for dname, dmask in domains.items():
            out[f"n_active_f_{dname}"] = float(
                ref["n_active_f_field"][dmask].mean()) if np.any(dmask) else np.nan

    # ---- what the update actually reached -----------------------------------
    # Two counts the notebooks recompute from the ensembles today, and cannot in light
    # mode. `frac_analysis_eq_prior` says how much of the domain the update never
    # touched at all; `frac_touched_no_obs` is the reach of the localization -- the
    # share of updated cells that carry no observation of their own, which is exactly
    # what separates §4.7 from the sweep, where every cell updated is the obs cell.
    untouched = _untouched_mask(xa, xf)
    touched = ~untouched
    n_touched = int(np.count_nonzero(touched))
    out["n_touched"] = n_touched
    out["n_untouched"] = int(np.count_nonzero(untouched))
    for dname, dmask in domains.items():
        n_d = int(np.count_nonzero(dmask))
        out[f"frac_analysis_eq_prior_{dname}"] = (
            float(np.count_nonzero(untouched & dmask) / n_d) if n_d else np.nan)
    if obs_ijk is not None:
        has_obs = domains["obs"]
        out["frac_touched_no_obs"] = (
            float(np.count_nonzero(touched & ~has_obs) / n_touched) if n_touched else np.nan)
        out["n_obs_cells"] = int(np.count_nonzero(has_obs))

    return _apply_storage_level(out, storage_level)
