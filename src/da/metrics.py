import numpy as np

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

    return out