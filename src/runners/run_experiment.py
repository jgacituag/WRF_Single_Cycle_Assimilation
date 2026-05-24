"""
src/runners/run_experiment.py
==============================
Three experiment modes, selected via sweep.obs_points.mode in the config:

  single_obs   One fixed observation point, all method combos.
  sweep        Every QC-passing stride point as an independent single-obs assimilation.
  multi_obs    All QC-passing stride points assimilated together in one Fortran call.
"""

import argparse
import faulthandler
import itertools
import math
import os
import pathlib
import shutil
import sys
import time
from multiprocessing import Pool
import numpy as np
import yaml

faulthandler.enable()

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "fortran"))

import da.core as core
from da.core import tenkf_update, aoei_update, letkf_update
from da.metrics import compute_single_obs_metrics, compute_multi_obs_metrics

_XF       = None   
_ENS_HX   = None   
_TRUTH    = None   
_TRUTH_HX = None   
_POS_KM   = None   
_NOISE    = None   
_YO_FIELD = None   

def _expand(val, is_int=False):
    if isinstance(val, dict):
        arr = np.linspace(val["start"], val["stop"], int(val["num"]))
        if is_int:
            arr = np.unique(np.round(arr).astype(int))
        return arr.tolist()
    if isinstance(val, list):
        return val
    return [val]

def _build_combos(sweep_cfg):
    methods = _expand(sweep_cfg.get("methods",  ["TEnKF"]))
    ntemps  = _expand(sweep_cfg.get("ntemp",    [1]), is_int=True)
    alphas  = _expand(sweep_cfg.get("alpha_s",  [2.0]))
    lxs     = _expand(sweep_cfg.get("loc_x",    [10.0]))
    lys     = _expand(sweep_cfg.get("loc_y",    [10.0]))
    lzs     = _expand(sweep_cfg.get("loc_z",    [4.0]))

    combos = []
    seen_single = set()
    for method, ntemp, alpha_s, lx, ly, lz in itertools.product(
            methods, ntemps, alphas, lxs, lys, lzs):
        if method in ("AOEI", "LETKF"):
            key = (method, float(alpha_s), float(lx), float(ly), float(lz))
            if key in seen_single:
                continue
            seen_single.add(key)
            combos.append((method, 1, float(alpha_s), float(lx), float(ly), float(lz)))
        else:
            combos.append((method, int(ntemp), float(alpha_s),
                           float(lx), float(ly), float(lz)))
    return combos

def _qc_pass(yo_val, hxf_max_val, qc_cfg):
    if not qc_cfg:
        return True
    dbz_min = float(qc_cfg.get("dbz_min", 5.0))
    fe   = bool(qc_cfg.get("filter_ensemble", True))
    ft   = bool(qc_cfg.get("filter_truth",    False))
    mode = qc_cfg.get("filter_mode", "and").lower()
    fail_e = fe and (float(hxf_max_val) < dbz_min)
    fail_t = ft and (float(yo_val)      < dbz_min)
    if fe and ft:
        return not (fail_e and fail_t) if mode == "or" else not (fail_e or fail_t)
    return not fail_e if fe else not fail_t

def _calc_hx_domain(state, var_idx, dbz_min=0.0):
    """Vectorised H(x) over the full domain via Fortran calc_ref_ens, securely floored."""
    from cletkf_wloc import common_da as cda
    vi = var_idx
    if state.ndim == 4:
        s = state[:, :, :, np.newaxis, :]
        ref = cda.calc_ref_ens(
            s[:,:,:,:,vi["qr"]].astype(np.float64),
            s[:,:,:,:,vi["qs"]].astype(np.float64),
            s[:,:,:,:,vi["qg"]].astype(np.float64),
            s[:,:,:,:,vi["T"] ].astype(np.float64),
            s[:,:,:,:,vi["P"] ].astype(np.float64),
            min_dbz=dbz_min,
        )
        return np.maximum(ref[:, :, :, 0], dbz_min).astype(np.float32)
    else:
        ref = cda.calc_ref_ens(
            state[:,:,:,:,vi["qr"]].astype(np.float64),
            state[:,:,:,:,vi["qs"]].astype(np.float64),
            state[:,:,:,:,vi["qg"]].astype(np.float64),
            state[:,:,:,:,vi["T"] ].astype(np.float64),
            state[:,:,:,:,vi["P"] ].astype(np.float64),
            min_dbz=dbz_min,
        )
        return np.maximum(ref, dbz_min).astype(np.float32)

def _setup(cfg, tm):
    global _XF, _ENS_HX, _TRUTH, _TRUTH_HX, _POS_KM, _NOISE, _YO_FIELD

    var_idx = cfg["state"]["var_idx"]
    qc_cfg  = cfg.get("qc", {})
    dbz_min = float(qc_cfg.get("dbz_min", 5.0))

    t0 = time.time()
    core._log(1, f"[setup tm={tm:02d}] loading {cfg['paths']['prepared']} ...")
    data = np.load(cfg["paths"]["prepared"])
    ens  = data["state_ensemble"] if "state_ensemble" in data else data["cross_sections"]

    if "pos_km" not in data:
        raise KeyError("'pos_km' not found in prepared .npz.")
    pos_km = data["pos_km"].astype(np.float32)   

    nx, ny, nz = ens.shape[:3]
    Ne_tot     = ens.shape[3]

    prior_size = cfg["sweep"].get("prior_size", None)
    all_others = [i for i in range(Ne_tot) if i != tm]
    if prior_size is not None:
        prior_size = int(prior_size)
        if prior_size > len(all_others):
            prior_size = len(all_others)
        all_others = all_others[:prior_size]
    Ne = len(all_others)

    truth = ens[:, :, :, tm, :].copy()                    
    xf    = np.asfortranarray(ens[:, :, :, all_others, :])
    del ens

    truth_hx = _calc_hx_domain(truth, var_idx, dbz_min=dbz_min)        
    ens_hx   = _calc_hx_domain(xf, var_idx, dbz_min=dbz_min)             

    sigma = float(np.sqrt(float(cfg["obs"]["obs_error_var"])))
    rng   = np.random.default_rng(42 + tm)
    add_noise = bool(cfg["obs"].get("add_noise", False))
    if add_noise:
        noise = rng.normal(0.0, sigma, (nx, ny, nz)).astype(np.float32)
    else:
        noise = np.zeros((nx, ny, nz), dtype=np.float32)

    do_clamp    = bool(qc_cfg.get("clamp_obs",        False))
    do_var_filt = bool(qc_cfg.get("filter_variance",  False))

    yo_field = (truth_hx + noise).astype(np.float32)   

    if do_clamp:
        np.maximum(yo_field, dbz_min, out=yo_field)    
        np.maximum(ens_hx,   dbz_min, out=ens_hx)     
        core._log(1, f"[setup tm={tm:02d}] obs and ens_hx clamped to {dbz_min} dBZ")

    stride = int(cfg["sweep"].get("stride", 1))

    if do_var_filt:
        ens_var = ens_hx.var(axis=3)                   
        pts = [(i, j, k) for i in range(0, nx, stride) for j in range(0, ny, stride) for k in range(0, nz) if ens_var[i, j, k] > 0.0]
        del ens_var
    else:
        ens_max = ens_hx.max(axis=3)                   
        pts = [(i, j, k) for i in range(0, nx, stride) for j in range(0, ny, stride) for k in range(0, nz) if _qc_pass(yo_field[i, j, k], ens_max[i, j, k], qc_cfg)]
        del ens_max

    _XF       = xf
    _ENS_HX   = ens_hx
    _TRUTH    = truth
    _TRUTH_HX = truth_hx
    _POS_KM   = pos_km
    _NOISE    = noise
    _YO_FIELD = yo_field

    return pts, Ne

def _subdomain_slices(i0, j0, k0, lx_km, ly_km, lz_km, pos_km, nx, ny, nz, cutoff_factor=4.0):
    di  = 1 if i0 + 1 < nx else -1
    dj  = 1 if j0 + 1 < ny else -1
    dx  = max(abs(float(pos_km[i0+di, j0, k0, 0] - pos_km[i0, j0, k0, 0])), 0.1)
    dy  = max(abs(float(pos_km[i0, j0+dj, k0, 1] - pos_km[i0, j0, k0, 1])), 0.1)

    half_i = int(np.ceil(cutoff_factor * lx_km / dx))
    half_j = int(np.ceil(cutoff_factor * ly_km / dy))

    i_min = max(0, i0 - half_i);  i_max = min(nx, i0 + half_i + 1)
    j_min = max(0, j0 - half_j);  j_max = min(ny, j0 + half_j + 1)

    z0    = float(pos_km[i0, j0, k0, 2])
    z_lev = pos_km[i0, j0, :, 2]              
    k_msk = np.abs(z_lev - z0) <= cutoff_factor * lz_km
    k_idx = np.where(k_msk)[0]
    if len(k_idx) == 0:
        k_min, k_max = 0, nz
    else:
        k_min = int(k_idx[0])
        k_max = int(k_idx[-1]) + 1

    return slice(i_min, i_max), slice(j_min, j_max), slice(k_min, k_max)

def _compute_rho(pos_km_sub, x0, y0, z0, lx_km, ly_km, lz_km):
    dx = pos_km_sub[:, :, :, 0] - x0
    dy = pos_km_sub[:, :, :, 1] - y0
    dz = pos_km_sub[:, :, :, 2] - z0

    d2 = np.zeros(dx.shape, dtype=np.float32)
    if lx_km > 0: d2 += (dx / lx_km) ** 2
    if ly_km > 0: d2 += (dy / ly_km) ** 2
    if lz_km > 0: d2 += (dz / lz_km) ** 2

    cutoff = (2.0 * np.sqrt(10.0 / 3.0)) ** 2
    return np.where(d2 <= cutoff, np.exp(-0.5 * d2), 0.0).astype(np.float32)

def _da_subdomain(xf_sub, yo, R0_val, ox_s, oy_s, oz_s,
                  pos_km_sub, loc_scales_km, var_idx, method, ntemp, alpha_s, dbz_min):
    yo_a = np.array([yo],     np.float32)
    R0_a = np.array([R0_val], np.float32)
    ox_a = np.array([ox_s],   np.int32)
    oy_a = np.array([oy_s],   np.int32)
    oz_a = np.array([oz_s],   np.int32)
    loc  = np.asarray(loc_scales_km, np.float32)

    if method == "TEnKF":
        return tenkf_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                            loc, var_idx, ntemp, alpha_s, pos_km_sub, dbz_min=dbz_min)["xa"]
    if method == "AOEI":
        return aoei_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                           loc, var_idx, pos_km_sub, dbz_min=dbz_min)["xa"]
    if method == "LETKF":
        return letkf_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                            loc, var_idx, pos_km_sub, dbz_min=dbz_min)["xa"]
    raise ValueError(f"Unknown method: {method}")

def _process_point(i0, j0, k0, combos, var_idx, R0_val,
                   cutoff_factor=4.0, return_fields=False, dbz_min=0.0):
    xf       = _XF
    ens_hx   = _ENS_HX
    truth    = _TRUTH
    truth_hx = _TRUTH_HX
    pos_km   = _POS_KM
    yo_field = _YO_FIELD
    nx, ny, nz, Ne, nvar = xf.shape
 
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
    x0       = float(pos_km[i0, j0, k0, 0])
    y0       = float(pos_km[i0, j0, k0, 1])
    z0       = float(pos_km[i0, j0, k0, 2])
    yo_clean = float(truth_hx[i0, j0, k0])
    yo       = float(yo_field[i0, j0, k0])
 
    n_c = len(combos)
 
    fixed_meta = dict(
        i        = np.full(n_c, i0,       np.int32),
        j        = np.full(n_c, j0,       np.int32),
        k        = np.full(n_c, k0,       np.int32),
        x_km     = np.full(n_c, x0,       np.float32),
        y_km     = np.full(n_c, y0,       np.float32),
        z_km     = np.full(n_c, z0,       np.float32),
        yo       = np.full(n_c, yo,       np.float32),
        yo_clean = np.full(n_c, yo_clean, np.float32),
    )
 
    method_arr  = np.empty(n_c, dtype="U8")
    ntemp_arr   = np.empty(n_c, np.int32)
    alpha_s_arr = np.empty(n_c, np.float32)
    lx_arr      = np.empty(n_c, np.float32)
    ly_arr      = np.empty(n_c, np.float32)
    lz_arr      = np.empty(n_c, np.float32)
 
    metrics_rows = [None] * n_c
    fields       = {} if return_fields else None
    sub_cache    = {}
 
    for c, (method, ntemp, alpha_s, lx_km, ly_km, lz_km) in enumerate(combos):
        method_arr[c]  = method
        ntemp_arr[c]   = ntemp
        alpha_s_arr[c] = alpha_s
        lx_arr[c]      = lx_km
        ly_arr[c]      = ly_km
        lz_arr[c]      = lz_km
 
        loc_key = (lx_km, ly_km, lz_km)
        if loc_key not in sub_cache:
            si, sj, sk = _subdomain_slices(
                i0, j0, k0, lx_km, ly_km, lz_km, pos_km, nx, ny, nz, cutoff_factor)
            xf_sub       = np.asfortranarray(xf[si, sj, sk, :, :])
            ens_hx_sub   = ens_hx[si, sj, sk, :]
            truth_sub    = truth[si, sj, sk, :]
            truth_hx_sub = truth_hx[si, sj, sk]
            pos_km_sub   = np.asfortranarray(pos_km[si, sj, sk, :])
            ox_s = i0 - si.start
            oy_s = j0 - sj.start
            oz_s = k0 - sk.start
            rho  = _compute_rho(pos_km_sub, x0, y0, z0, lx_km, ly_km, lz_km)
            sub_cache[loc_key] = (xf_sub, ens_hx_sub, truth_sub, truth_hx_sub,
                                  pos_km_sub, ox_s, oy_s, oz_s, rho)
        (xf_sub, ens_hx_sub, truth_sub, truth_hx_sub,
         pos_km_sub, ox_s, oy_s, oz_s, rho) = sub_cache[loc_key]
 
        xa_sub  = _da_subdomain(xf_sub, yo, R0_val, ox_s, oy_s, oz_s,
                                pos_km_sub, (lx_km, ly_km, lz_km),
                                var_idx, method, ntemp, alpha_s, dbz_min)
                                
        # Crucial propagation: Ensure posterior H(x) matches the dbz_min floor perfectly
        hxa_sub = _calc_hx_domain(xa_sub, var_idx, dbz_min=dbz_min)   
 
        metrics_rows[c] = compute_single_obs_metrics(
            xf_sub, xa_sub, truth_sub,
            ens_hx_sub, hxa_sub, truth_hx_sub,
            rho, ox_s, oy_s, oz_s, yo, yo_clean, var_names, Ne)
 
        if return_fields:
            fields[c] = xa_sub
 
    combo_meta = dict(
        method  = method_arr, ntemp   = ntemp_arr, alpha_s = alpha_s_arr,
        lx_km   = lx_arr,     ly_km   = ly_arr,    lz_km   = lz_arr,
    )
 
    metric_keys = list(metrics_rows[0].keys())
    metrics_flat = {
        k: np.array([metrics_rows[c][k] for c in range(n_c)], dtype=np.float32)
        for k in metric_keys
    }
 
    row = {**fixed_meta, **combo_meta, **metrics_flat}
    return row, fields

def _run_sweep_sequential(pts, combos, cfg, outdir, tag, tm, Ne):
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    dbz_min = float(cfg.get("qc", {}).get("dbz_min", 5.0))
    n_pts, n_c = len(pts), len(combos)
 
    core._log(1, f"[sweep tm={tm:02d}] {n_pts} pts x {n_c} combos = {n_pts*n_c} rows")
    all_rows = []
    t0 = time.time()
 
    for p_idx, (i0, j0, k0) in enumerate(pts):
        if p_idx % 500 == 0:
            elapsed = time.time() - t0
            rate    = p_idx / elapsed if p_idx > 0 else 0.0
            eta     = (n_pts - p_idx) / rate if rate > 0 else 0.0
            core._log(1, f"  [sweep] pt {p_idx}/{n_pts}  {rate:.1f} pts/s  ETA {eta/60:.0f} min")
            
        row, _ = _process_point(i0, j0, k0, combos, var_idx, R0_val, cutoff, dbz_min=dbz_min)
        all_rows.append(row)
 
    merged = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
    fname = f"{tag}_sweep_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]), **merged)
    sz = os.path.getsize(out) / 1e6
    core._log(1, f"[sweep tm={tm:02d}] saved {n_pts*n_c} rows  {sz:.1f} MB  {time.time()-t0:.1f}s -> {fname}")

def _worker_init():
    import ctypes
    omp_env = os.environ.get("OMP_NUM_THREADS", "unset")
    try:
        libgomp = ctypes.CDLL("libgomp.so.1")
        libgomp.omp_set_num_threads(1)
    except Exception as e:
        pass

def _sweep_worker(args):
    import os as _os
    _os.environ["OMP_NUM_THREADS"]      = "1"
    _os.environ["MKL_NUM_THREADS"]      = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
 
    pts_chunk, combos, var_idx, R0_val, cutoff, dbz_min = args
    all_rows = []
    for (i0, j0, k0) in pts_chunk:
        row, _ = _process_point(i0, j0, k0, combos, var_idx, R0_val, cutoff, dbz_min=dbz_min)
        all_rows.append(row)
    merged = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    return merged
 
def _run_sweep_parallel(pts, combos, cfg, outdir, tag, tm, Ne, n_workers):
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    dbz_min = float(cfg.get("qc", {}).get("dbz_min", 5.0))
    n_pts, n_c = len(pts), len(combos)

    import os as _os
    _os.environ["OMP_NUM_THREADS"]     = "1"
    _os.environ["MKL_NUM_THREADS"]     = "1"
    _os.environ["OPENBLAS_NUM_THREADS"]= "1"

    chunk_size  = max(1, n_pts // n_workers)
    chunks      = [pts[i:i+chunk_size] for i in range(0, n_pts, chunk_size)]
    worker_args = [(c, combos, var_idx, R0_val, cutoff, dbz_min) for c in chunks]
    n_chunks    = len(chunks)

    all_rows = []
    t0       = time.time()
 
    with Pool(processes=n_workers, initializer=_worker_init) as pool:
        for done, merged_chunk in enumerate(pool.imap_unordered(_sweep_worker, worker_args), start=1):
            all_rows.append(merged_chunk)
            elapsed   = time.time() - t0
            rows_done = sum(len(r["i"]) for r in all_rows)
            rate      = rows_done / n_c / elapsed if elapsed > 0 else 0.0
            eta       = (n_pts - rows_done // n_c) / rate if rate > 0 else 0.0
            core._log(1, f"  [sweep] chunk {done}/{n_chunks}  pt {rows_done//n_c}/{n_pts}  {rate:.1f} pts/s  ETA {eta/60:.0f} min")
 
    merged    = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
    fname = f"{tag}_sweep_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]), **merged)
    sz = os.path.getsize(out) / 1e6
    core._log(1, f"[sweep tm={tm:02d}] saved {n_pts*n_c} rows  {sz:.1f} MB  {time.time()-t0:.1f}s -> {fname}")

def _run_single_obs(combos, cfg, outdir, tag, tm, Ne):
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    dbz_min = float(cfg.get("qc", {}).get("dbz_min", 5.0))

    obs_loc    = cfg["sweep"]["obs_points"]["loc"]
    i0, j0, k0 = int(obs_loc["x"]), int(obs_loc["y"]), int(obs_loc["z"])

    row, _ = _process_point(
        i0, j0, k0, combos, var_idx, R0_val, cutoff,
        return_fields=False, dbz_min=dbz_min)
 
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
    fname = f"{tag}_single_obs_{i0}_{j0}_{k0}_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]), **row)

def _run_multi_obs(pts, combos, cfg, outdir, tag, tm, Ne):
    var_idx      = cfg["state"]["var_idx"]
    R0_val       = float(cfg["obs"]["obs_error_var"])
    store_fields = bool(cfg.get("store_fields", False))
    dbz_min      = float(cfg.get("qc", {}).get("dbz_min", 5.0))

    xf       = _XF
    truth    = _TRUTH
    truth_hx = _TRUTH_HX
    yo_field = _YO_FIELD
    pos_km   = _POS_KM
    nx, ny, nz, Ne_, nvar = xf.shape

    ix  = np.array([p[0] for p in pts], np.int32)
    iy  = np.array([p[1] for p in pts], np.int32)
    iz  = np.array([p[2] for p in pts], np.int32)
    yo_clean = truth_hx[ix, iy, iz].astype(np.float32)
    yo       = yo_field[ix, iy, iz].astype(np.float32)
    R0       = np.full(len(pts), R0_val, np.float32)

    ref_fname = f"{tag}_multi_obs_ref_Ne{Ne:03d}_tm{tm:02d}.npz"
    ref_out   = os.path.join(outdir, ref_fname)
    if not os.path.exists(ref_out):
        np.savez_compressed(ref_out,
            truth    = truth,
            xf_mean  = xf.mean(axis=3).astype(np.float32),
            truth_hx = truth_hx,
            yo       = yo,
            yo_clean = yo_clean,
            ix=ix, iy=iy, iz=iz,
            var_names = np.array(list(var_idx.keys())),
        )

    for (method, ntemp, alpha_s, lx_km, ly_km, lz_km) in combos:
        fname = (f"{tag}_multi_obs_{method}_Nt{ntemp:02d}"
                 f"_as{alpha_s:.1f}_Lx{lx_km}Ly{ly_km}Lz{lz_km}"
                 f"_Ne{Ne:03d}_tm{tm:02d}.npz")
        out = os.path.join(outdir, fname)

        if cfg.get("skip_existing", False) and os.path.exists(out):
            continue

        loc = np.array([lx_km, ly_km, lz_km], np.float32)
        t1 = time.time()

        core._log(3, f"  [multi_obs] calling {method} Nt={ntemp} alpha={alpha_s} loc=({lx_km},{ly_km},{lz_km})")
        if method == "TEnKF":
            res = tenkf_update(xf, yo, R0, ix, iy, iz, loc, var_idx, ntemp, alpha_s, np.asfortranarray(pos_km), dbz_min=dbz_min)
        elif method == "AOEI":
            res = aoei_update(xf, yo, R0, ix, iy, iz, loc, var_idx, np.asfortranarray(pos_km), dbz_min=dbz_min)
        elif method == "LETKF":
            res = letkf_update(xf, yo, R0, ix, iy, iz, loc, var_idx, np.asfortranarray(pos_km), dbz_min=dbz_min)
        else:
            raise ValueError(f"Unknown method: {method}")

        xa = res["xa"]
        core._log(3, f"  [multi_obs] DA done  xa.shape={xa.shape}  xa.dtype={xa.dtype}")

        # Crucial propagation: Ensure multi-obs final analysis matches the floor securely
        core._log(3, f"  [multi_obs] computing H(xa) domain-wide ...")
        hxa_ens = _calc_hx_domain(xa, var_idx, dbz_min=dbz_min)
        core._log(3, f"  [multi_obs] H(xa) done  shape={hxa_ens.shape}")
        hxa_mean_field = hxa_ens.mean(axis=3)
        hxf_mean_field = _ENS_HX.mean(axis=3)
        truth_hx_field = _TRUTH_HX

        var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]

        core._log(3, f"  [multi_obs] computing metrics ...")
        m = compute_multi_obs_metrics(
            xa, xf, truth,
            hxf_mean_field, hxa_mean_field, truth_hx_field,
            var_names, Ne, store_fields=store_fields)
        core._log(3, f"  [multi_obs] metrics done, saving -> {fname}")

        np.savez_compressed(out,
            method       = method, ntemp        = np.int32(ntemp), alpha_s      = np.float32(alpha_s),
            lx_km        = np.float32(lx_km), ly_km        = np.float32(ly_km), lz_km        = np.float32(lz_km),
            Ne           = np.int32(Ne), truth_member = np.int32(tm), var_names    = np.array(var_names),
            ref_file     = ref_fname, **m,
        )
        core._log(3, f"  [multi_obs] saved {fname}")
        del xa

def main():
    ap = argparse.ArgumentParser(description="WRF single-cycle assimilation experiment runner.")
    ap.add_argument("--config",  required=True, help="Path to YAML config file.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for sweep mode.")
    ap.add_argument("--verbose", type=int, default=None, help="Verbosity level 0-3.")
    ap.add_argument("--tm",      type=int, default=None, help="Truth member index.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    verbose = args.verbose if args.verbose is not None else int(cfg.get("verbose", 1))
    core.set_verbose(verbose)

    outdir = cfg["paths"]["outdir"]
    tag    = cfg.get("experiment_tag", "EXP")
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(outdir, f"{tag}_config.yaml"))

    if args.tm is not None:
        tm_list = [args.tm]
    else:
        tm_cfg = cfg["sweep"].get("truth_members", 0)
        if isinstance(tm_cfg, dict):
            start = int(tm_cfg["start"])
            stop  = int(tm_cfg["stop"])
            step  = int(tm_cfg.get("step", 1))
            tm_list = list(range(start, stop + 1, step))
        elif isinstance(tm_cfg, list):
            tm_list = [int(x) for x in tm_cfg]
        else:
            tm_list = [int(tm_cfg)]

    obs_cfg  = cfg["sweep"]["obs_points"]
    obs_mode = obs_cfg if isinstance(obs_cfg, str) else obs_cfg.get("mode", "sweep")

    n_workers = max(1, args.workers)
    t_start = time.time()

    if obs_mode == "sweep" and n_workers > 1:
        os.environ["OMP_NUM_THREADS"] = "1"

    combos  = _build_combos(cfg["sweep"])

    for tm in tm_list:
        core._log(1, f"[{tag}] === truth member {tm} ===")
        t_tm = time.time()
        pts, Ne = _setup(cfg, tm)

        if obs_mode == "single_obs":
            _run_single_obs(combos, cfg, outdir, tag, tm, Ne)
        elif obs_mode == "sweep":
            if n_workers == 1:
                _run_sweep_sequential(pts, combos, cfg, outdir, tag, tm, Ne)
            else:
                os.environ["OMP_NUM_THREADS"] = "1"
                _run_sweep_parallel(pts, combos, cfg, outdir, tag, tm, Ne, n_workers)
        elif obs_mode == "multi_obs":
            _run_multi_obs(pts, combos, cfg, outdir, tag, tm, Ne)
        else:
            raise ValueError(f"Unknown obs_mode '{obs_mode}'.")

        core._log(1, f"[{tag}] tm={tm} done in {time.time()-t_tm:.1f}s")

if __name__ == "__main__":
    main()