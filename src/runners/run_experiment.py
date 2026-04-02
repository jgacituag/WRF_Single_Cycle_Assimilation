"""
src/runners/run_experiment.py
==============================
Unified experiment runner for all WS experiments.

The YAML config controls everything: experiment type, obs construction,
sweep parameters, QC, methods, parallelism, and verbosity.

Usage
-----
  python src/runners/run_experiment.py --config configs/ws1.yaml
  python src/runners/run_experiment.py --config configs/ws2.yaml --workers 30
  python src/runners/run_experiment.py --config configs/ws2.yaml --verbose 1

Obs modes (set via sweep.obs_points in config)
-----------------------------------------------
  single              one fixed point -> one LETKF call per combo
  full_grid           every QC-passing point as independent single obs (WS-2)
  strided: N          all QC-passing points on ::N grid, one LETKF call
  all                 all QC-passing points, one LETKF call

Sweep dimensions (all support scalar, list, or {start, stop, num})
-------------------------------------------------------------------
  truth_members, prior_size, methods, ntemp, alpha_s, loc_x, loc_y, loc_z

Output files
------------
  Single-obs (mode: single or full_grid):
    One file per (combo, truth_member).  Each file is a table where every
    key is a 1-D array of length N_obs (number of QC-passing obs points).
    Row i corresponds to the experiment where point (obs_x[i], obs_y[i],
    obs_z[i]) was the single assimilated observation.  The prior ensemble
    is the same for all rows in a file.

    Filename: {tag}_{method}_Nt{nt}_as{as}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne}_qc{qc}_True{tm}.npz

    Keys stored: obs_x, obs_y, obs_z, yo, hxf_mean_obs, hxa_mean_obs,
      dep_b, dep_a, spread_f_obs, spread_a_obs, inc_obs,
      loc_weights_sum, n_updated,
      rmse_{f,a}_{w,u}_{var}, bias_{f,a}_{w,u}_{var},
      spread_{f,a}_{w,u}_{var},  xf_mean_{var}, xa_mean_{var}, truth_{var},
      rmse_{f,a}_obs_{w,u}, bias_{f,a}_obs_{w,u}

  Multi-obs (mode: strided or all):
    One file per (combo, truth_member). Stores full xf, xa matrices plus
    global scalar and (nx,ny,nz) field metrics.

    Filename: {tag}_{method}_Nt{nt}_as{as}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne}_str{stride}_qc{qc}_True{tm}.npz

  Config copy: {outdir}/{tag}_config.yaml  (written before any results)
"""

import argparse
import itertools
import os
import pathlib
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "fortran"))

import da.core as core
from da.core import (
    letkf_update, tenkf_update, aoei_update,
    atenkf_update, taoei_update,
    compute_loc_weights,
)
from da.metrics import (compute_single_obs_metrics, compute_multi_obs_metrics,
                        _hx_domain, _hx_domain_truth)


##### sweep parameter helpers ################################################

def _expand(val, is_int=False):
    """
    Expand a sweep parameter to a flat list.
      scalar       -> [scalar]
      [a, b, c]    -> [a, b, c]
      {start, stop, num}  -> linspace (inclusive both ends)
                             if is_int: rounded to unique ints
    """
    if isinstance(val, dict):
        import numpy as np
        arr = np.linspace(val["start"], val["stop"], int(val["num"]))
        if is_int:
            arr = np.unique(np.round(arr).astype(int))
        return arr.tolist()
    if isinstance(val, list):
        return val
    return [val]


def _expand_loc(cfg_val):
    """
    Expand a localization axis value.
    Accepts scalar, list, or {start,stop,num}.
    9999 or null/None means no localization.
    """
    vals = _expand(cfg_val if cfg_val is not None else 9999)
    return [float(v) for v in vals]


def _qc_code(qc_cfg):
    """Short code for QC settings used in filenames."""
    if not qc_cfg:
        return "none"
    fe = qc_cfg.get("filter_ensemble", True)
    ft = qc_cfg.get("filter_truth",    False)
    mode = qc_cfg.get("filter_mode", "and")
    if fe and ft:
        return f"ET_{mode}"
    if fe:
        return "E"
    if ft:
        return "T"
    return "none"


def _qc_pass(yo_val, hxf_mean_val, qc_cfg):
    if not qc_cfg:
        return True
    dbz_min = float(qc_cfg.get("dbz_min", 5.0))
    fe   = bool(qc_cfg.get("filter_ensemble", True))
    ft   = bool(qc_cfg.get("filter_truth",    False))
    mode = qc_cfg.get("filter_mode", "and").lower()
    fail_e = fe and (float(hxf_mean_val) < dbz_min)
    fail_t = ft and (float(yo_val)       < dbz_min)
    if fe and ft:
        return not (fail_e and fail_t) if mode == "or" \
               else not (fail_e or fail_t)
    return not fail_e if fe else not fail_t


##### observation operator ###################################################

def _calc_hx_domain(state_nvar, var_idx):
    """
    Compute reflectivity for a (nx,ny,nz,nvar) or (nx,ny,nz,Ne,nvar) array.
    Returns same shape without nvar axis.
    """
    from cletkf_wloc import common_da as cda
    vi = var_idx
    shape = state_nvar.shape
    if state_nvar.ndim == 4:            ##single member (nx,ny,nz,nvar)
        nx, ny, nz = shape[:3]
        out = np.full((nx, ny, nz), np.nan, np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    out[i,j,k] = cda.calc_ref(
                        state_nvar[i,j,k,vi["qr"]], state_nvar[i,j,k,vi["qs"]],
                        state_nvar[i,j,k,vi["qg"]], state_nvar[i,j,k,vi["T"]],
                        state_nvar[i,j,k,vi["P"]],
                    )
        return out
    else:                               ##ensemble (nx,ny,nz,Ne,nvar)
        nx, ny, nz, Ne = shape[:4]
        out = np.full((nx, ny, nz, Ne), np.nan, np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for m in range(Ne):
                        out[i,j,k,m] = cda.calc_ref(
                            state_nvar[i,j,k,m,vi["qr"]], state_nvar[i,j,k,m,vi["qs"]],
                            state_nvar[i,j,k,m,vi["qg"]], state_nvar[i,j,k,m,vi["T"]],
                            state_nvar[i,j,k,m,vi["P"]],
                        )
        return out


def _hx_point(state_nvar, i, j, k, var_idx):
    """H(x) at one grid point. state_nvar: (nvar,) or (Ne, nvar)."""
    from cletkf_wloc import common_da as cda
    vi = var_idx
    if state_nvar.ndim == 1:
        return float(cda.calc_ref(state_nvar[vi["qr"]], state_nvar[vi["qs"]],
                                  state_nvar[vi["qg"]], state_nvar[vi["T"]],
                                  state_nvar[vi["P"]]))
    return np.array([cda.calc_ref(state_nvar[m,vi["qr"]], state_nvar[m,vi["qs"]],
                                  state_nvar[m,vi["qg"]], state_nvar[m,vi["T"]],
                                  state_nvar[m,vi["P"]])
                     for m in range(state_nvar.shape[0])], np.float32)


##### method dispatcher ######################################################

def _run_method(method, xf, yo, R0, ox, oy, oz, loc_scales, var_idx, ntemp, alpha_s):
    if method == "LETKF":
        return letkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx)
    if method == "TEnKF":
        return tenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                            ntemp=ntemp, alpha_s=alpha_s)
    if method == "AOEI":
        return aoei_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx)
    if method == "ATEnKF":
        return atenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                             alpha_s=alpha_s)
    if method == "TAOEI":
        return taoei_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                            ntemp=ntemp, alpha_s=alpha_s)
    raise ValueError(f"Unknown method: {method}")


##### filename builders ######################################################

def _fmt(v):
    """Format a float for filenames: no trailing zeros."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _select_prior(ens, tm, prior_size=None):
    """
    Split ensemble into truth and prior.

    Parameters
    ----------
    ens        : (nx, ny, nz, Ne_tot, nvar)
    tm         : index of the truth member
    prior_size : number of prior members to use (None = all remaining,
                 sequential starting from index 0, skipping tm)

    Returns
    -------
    truth : (nx, ny, nz, nvar)
    xf    : (nx, ny, nz, Ne, nvar)  float32
    Ne    : int  actual prior ensemble size used
    """
    Ne_tot     = ens.shape[3]
    all_others = [i for i in range(Ne_tot) if i != tm]
    if prior_size is not None:
        if prior_size > len(all_others):
            raise ValueError(
                f"prior_size={prior_size} exceeds available members "
                f"({len(all_others)}) for truth member {tm}."
            )
        all_others = all_others[:prior_size]
    truth = ens[:, :, :, tm, :]
    xf    = ens[:, :, :, all_others, :].astype(np.float32)
    return truth, xf, len(all_others)


def _fname_single(tag, method, ntemp, alpha_s, lx, ly, lz, ne, qc, tm):
    return (f"{tag}_{method}_Nt{ntemp:02d}_as{_fmt(alpha_s)}"
            f"_Lx{_fmt(lx)}Ly{_fmt(ly)}Lz{_fmt(lz)}"
            f"_Ne{ne:03d}_qc{qc}_True{tm:02d}.npz")


def _fname_multi(tag, method, ntemp, alpha_s, lx, ly, lz,
                 ne, stride_str, qc, tm):
    return (f"{tag}_{method}_Nt{ntemp:02d}_as{_fmt(alpha_s)}"
            f"_Lx{_fmt(lx)}Ly{_fmt(ly)}Lz{_fmt(lz)}"
            f"_Ne{ne:03d}_str{stride_str}_qc{qc}_True{tm:02d}.npz")


##### per-truth-member worker ################################################

def _worker(args):
    (tm, ens, cfg, verbose) = args
    core.set_verbose(verbose)

    sweep    = cfg["sweep"]
    qc_cfg   = cfg.get("qc", {})
    var_idx  = cfg["state"]["var_idx"]
    outdir   = cfg["paths"]["outdir"]
    tag      = cfg.get("experiment_tag", "EXP")
    R0_val   = float(cfg["obs"]["obs_error_var"])
    qc       = _qc_code(qc_cfg)

    obs_cfg  = sweep["obs_points"]
    if isinstance(obs_cfg, str):
        obs_mode = obs_cfg
        obs_loc  = None
        stride   = 1
    elif isinstance(obs_cfg, dict):
        obs_mode = obs_cfg.get("mode", "single")
        obs_loc  = obs_cfg.get("loc", None)
        stride   = int(obs_cfg.get("stride", 2))

    ## single and full_grid are both single-obs modes (differ only in which points are iterated); strided and all are multi-obs modes.
    is_single_obs = obs_mode in ("single", "full_grid")

    methods     = _expand(sweep.get("methods",    ["LETKF"]))
    ntemps      = _expand(sweep.get("ntemp",      [1]), is_int=True)
    alphas      = _expand(sweep.get("alpha_s",    [2.0]))
    lxs         = _expand_loc(sweep.get("loc_x",  5.0))
    lys         = _expand_loc(sweep.get("loc_y",  5.0))
    lzs         = _expand_loc(sweep.get("loc_z",  5.0))
    prior_sizes = _expand(sweep["prior_size"], is_int=True) \
                  if sweep.get("prior_size") is not None else [None]

    ## variable names in index order (needed for per-variable metric keys)
    var_idx_inv = {v: k for k, v in var_idx.items()}
    var_names   = [var_idx_inv[i] for i in range(len(var_idx))]

    truth_base = ens[:, :, :, tm, :]           ##(nx, ny, nz, nvar)
    truth_hx   = _calc_hx_domain(truth_base, var_idx)   ##(nx,ny,nz)
    nx, ny, nz = truth_base.shape[:3]

    t0 = time.time()
    core._log(1, f"[truth {tm:02d}] start  domain={nx}x{ny}x{nz}"
                 f"  prior_sizes={prior_sizes}  mode={obs_mode}")

    combos = list(itertools.product(methods, ntemps, alphas, lxs, lys, lzs))
    saved  = []

    for prior_size in prior_sizes:

        _, xf, Ne = _select_prior(ens, tm, prior_size)
        ens_hx    = _calc_hx_domain(xf, var_idx)    ##(nx,ny,nz,Ne)
        ens_mean  = ens_hx.mean(axis=3)              ##(nx,ny,nz)

        core._log(2, f"  [truth {tm:02d}] Ne={Ne}")

        ##### build the list of QC-passing obs points #####################
        if obs_mode == "single":
            i0, j0, k0 = int(obs_loc["x"]), int(obs_loc["y"]), int(obs_loc["z"])
            qc_pts = [(i0, j0, k0)]
        elif obs_mode == "full_grid":
            qc_pts = [
                (i, j, k)
                for i in range(nx) for j in range(ny) for k in range(nz)
                if _qc_pass(truth_hx[i, j, k], ens_mean[i, j, k], qc_cfg)
            ]
            core._log(1, f"  [truth {tm:02d}] Ne={Ne}  full_grid: "
                         f"{len(qc_pts)}/{nx*ny*nz} pts pass QC")
        else:  ## strided / all - multi-obs
            s = stride if obs_mode == "strided" else 1
            qc_pts = [
                (i, j, k)
                for i in range(0, nx, s)
                for j in range(0, ny, s)
                for k in range(0, nz, s)
                if _qc_pass(truth_hx[i, j, k], ens_mean[i, j, k], qc_cfg)
            ]
            core._log(1, f"  [truth {tm:02d}] Ne={Ne}  {obs_mode}: "
                         f"{len(qc_pts)}/{nx*ny*nz} pts pass QC")

        if not qc_pts:
            core._log(1, f"  [truth {tm:02d}] Ne={Ne}  no valid obs, skipping")
            continue

        ##### multi-obs: one LETKF call, one output file per combo ########
        if not is_single_obs:
            ox_arr = np.array([p[0] for p in qc_pts], np.int32)
            oy_arr = np.array([p[1] for p in qc_pts], np.int32)
            oz_arr = np.array([p[2] for p in qc_pts], np.int32)
            yo_arr = np.array([truth_hx[p[0], p[1], p[2]] for p in qc_pts],
                               np.float32)
            R0     = np.full(len(yo_arr), R0_val, np.float32)
            stride_str = f"{stride}" if obs_mode == "strided" else "all"

            for (method, ntemp, alpha_s, lx, ly, lz) in combos:
                loc_scales = np.array([lx, ly, lz], np.float32)
                fname    = _fname_multi(tag, method, ntemp, alpha_s,
                                        lx, ly, lz, Ne, stride_str, qc, tm)
                out_path = os.path.join(outdir, fname)
                if os.path.exists(out_path) and cfg.get("skip_existing", False):
                    core._log(2, f"  [skip] {fname}")
                    saved.append(fname)
                    continue

                core._log(2, f"  {method} Ne={Ne} Nt={ntemp} as={alpha_s} "
                             f"L=[{lx},{ly},{lz}] nobs={len(yo_arr)}")

                res  = _run_method(method, xf, yo_arr, R0,
                                   ox_arr, oy_arr, oz_arr,
                                   loc_scales, var_idx, ntemp, alpha_s)
                save = compute_multi_obs_metrics(
                    xf=xf, xa=res["xa"],
                    truth=truth_base,
                    yo=yo_arr, ox=ox_arr, oy=oy_arr, oz=oz_arr,
                    var_idx=var_idx, var_names=var_names,
                )
                save["truth_member"] = np.int32(tm)
                save["Ne"]           = np.int32(Ne)
                np.savez_compressed(out_path, **save)
                saved.append(fname)
            continue   ## next prior_size

        ##### single-obs: iterate over QC-passing points, collect into table
        ##One output file per combo, containing all obs-point rows.
        ##################################################################
        for (method, ntemp, alpha_s, lx, ly, lz) in combos:
            loc_scales = np.array([lx, ly, lz], np.float32)
            fname    = _fname_single(tag, method, ntemp, alpha_s,
                                     lx, ly, lz, Ne, qc, tm)
            out_path = os.path.join(outdir, fname)
            if os.path.exists(out_path) and cfg.get("skip_existing", False):
                core._log(2, f"  [skip] {fname}")
                saved.append(fname)
                continue

            core._log(2, f"  {method} Ne={Ne} Nt={ntemp} as={alpha_s} "
                         f"L=[{lx},{ly},{lz}]  {len(qc_pts)} obs pts")

            ##Precompute fields that are identical for every obs point in
            ##this combo — xf and truth never change within the loop.
            hxf_mean_field = _hx_domain(xf, var_idx)           ##(nx,ny,nz)
            truth_hx_field = _hx_domain_truth(truth_base, var_idx)  ##(nx,ny,nz)
            xf_mean_state  = xf.mean(axis=3)   ##(nx,ny,nz,nvar) — also invariant

            ##Accumulate one row per obs point into lists, then stack.
            rows = []
            obs_x_list, obs_y_list, obs_z_list = [], [], []

            for (i0, j0, k0) in qc_pts:
                yo_val   = float(truth_hx[i0, j0, k0])
                ox_arr   = np.array([i0], np.int32)
                oy_arr   = np.array([j0], np.int32)
                oz_arr   = np.array([k0], np.int32)
                yo_arr   = np.array([yo_val], np.float32)
                R0       = np.array([R0_val],  np.float32)

                res = _run_method(method, xf, yo_arr, R0,
                                  ox_arr, oy_arr, oz_arr,
                                  loc_scales, var_idx, ntemp, alpha_s)

                ## localization weight field centred on this obs point
                rloc = compute_loc_weights(nx, ny, nz, i0, j0, k0, loc_scales)

                ## H(xf) ensemble at the obs point — shape (Ne,)
                hxf_at_obs = ens_hx[i0, j0, k0, :]    ##(Ne,)

                row = compute_single_obs_metrics(
                    xf=xf, xa=res["xa"],
                    truth=truth_base,
                    rloc=rloc,
                    hxf_at_obs=hxf_at_obs,
                    yo=yo_val,
                    var_idx=var_idx,
                    var_names=var_names,
                    hxf_mean_field=hxf_mean_field,
                    truth_hx_field=truth_hx_field,
                )
                rows.append(row)
                obs_x_list.append(i0)
                obs_y_list.append(j0)
                obs_z_list.append(k0)

            ##### stack rows into arrays and save ######################
            save = {
                "obs_x": np.array(obs_x_list, np.int32),
                "obs_y": np.array(obs_y_list, np.int32),
                "obs_z": np.array(obs_z_list, np.int32),
                "truth_member": np.int32(tm),
                "Ne": np.int32(Ne),
            }
            ## Every key in rows[0] is a scalar stack across obs points
            for key in rows[0]:
                save[key] = np.array([r[key] for r in rows], dtype=np.float32)

            np.savez_compressed(out_path, **save)
            saved.append(fname)
            core._log(2, f"  -> saved {fname}  ({len(rows)} rows)")

    elapsed = time.time() - t0
    core._log(1, f"[truth {tm:02d}] done  {len(saved)} files  {elapsed:.1f}s")
    return saved


##### main ###################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--verbose", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    verbose   = args.verbose if args.verbose is not None \
                else int(cfg.get("verbose", 1))
    outdir    = cfg["paths"]["outdir"]
    tag       = cfg.get("experiment_tag", "EXP")
    n_members = int(cfg["state"].get("n_members", 30))

    os.makedirs(outdir, exist_ok=True)

    ## copy config to output dir before anything else
    config_copy = os.path.join(outdir, f"{tag}_config.yaml")
    shutil.copy2(args.config, config_copy)
    print(f"[info] config saved -> {config_copy}")

    ## expand truth members
    tm_cfg = cfg["sweep"].get("truth_members", "all")
    if tm_cfg == "all":
        truth_members = list(range(n_members))
    else:
        truth_members = _expand(tm_cfg, is_int=True)

    ## load ensemble once in main process
    data = np.load(cfg["paths"]["prepared"])
    ens  = data["state_ensemble"] if "state_ensemble" in data \
           else data["cross_sections"]

    n_workers   = args.workers or len(truth_members)
    worker_args = [(tm, ens, cfg, verbose) for tm in truth_members]

    t_start = time.time()
    print(f"[{tag}] {len(truth_members)} truth members  "
          f"workers={n_workers}  verbose={verbose}")

    if n_workers == 1:
        all_saved = []
        for a in worker_args:
            all_saved.extend(_worker(a))
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_worker, worker_args)
        all_saved = [f for r in results for f in r]

    elapsed = time.time() - t_start
    print(f"[{tag}] done  {len(all_saved)} files  total={elapsed:.1f}s")


if __name__ == "__main__":
    main()