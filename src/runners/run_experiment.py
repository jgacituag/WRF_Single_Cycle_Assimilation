"""
src/runners/run_experiment.py
==============================
Three experiment modes, selected via sweep.obs_points.mode in the config:

  single_obs   One fixed observation point, all method combos. Spelled `point` in a
               tag, and with `obs_points.points: [[i,j,k], ...]` it becomes §6's point
               mode: the same assimilation at each listed cell, keeping the full
               ensemble in state and observation space, the LETKF weight matrix, and
               the state either side of the hydrometeor clamp, at every tempering step.
  sweep        Every QC-passing stride point as an independent single-obs assimilation.
  multi_obs    All QC-passing stride points assimilated together in one Fortran call.

What reaches disk is the `output:` block, resolved by naming.output_levels_of. The
metric fields are what the batch costs -- ~0.7 GB per multi_obs file on A and B, ~2.4 GB
on C, against a few kB of domain scalars -- so they are off by default and the tag has
to declare them. See configs/template.yaml.
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
from da.core import (tenkf_update, aoei_update, letkf_update,
                     clamp_mode_of, HYDRO_VARS)
from da.metrics import compute_single_obs_metrics, compute_multi_obs_metrics
from naming import (TagError, dep_band_of, obs_mode_of, output_levels_of,
                    storage_level_of, validate_experiment_tag)

_XF       = None   
_ENS_HX   = None   
_TRUTH    = None   
_TRUTH_HX = None   
_POS_KM   = None   
_NOISE    = None   
_YO_FIELD = None   
_HXF_MEAN = None   
_IDENTITY = None   

# The identity keys copied from the subset into every experiment output, so a result
# file says which dataset it came from without a filename having to be trusted. The
# absence of `config_index` is what made the physics-per-member assignment
# unverifiable and forced the dataset A rebuild.
IDENTITY_KEYS = ("dataset_id", "da_cycle_min", "dx_km", "physics", "upstream",
                 "source_run", "config_index", "config_index_note")

IDENTITY_MISSING_NOTE = (
    "not recoverable: the subset this run read carries no identity block. It predates "
    "src/extract_3d_subset.py writing one, and nothing in the experiment output can "
    "reconstruct it. -1 means 'not recorded', never 'configuration -1'."
)


def _read_identity(prepared, n_members):
    """The subset's identity block, or an explicit unrecoverable one.

    Never guessed from the path: a subset that does not carry `dataset_id` gets an
    empty string and an all -1 config_index with a note saying why, so a reader is
    left in no doubt about which of the two it is looking at.
    """
    ident = {}
    with np.load(prepared, allow_pickle=True) as f:
        present = [k for k in IDENTITY_KEYS if k in f.files]
        for k in present:
            ident[k] = f[k]
    if "dataset_id" not in ident:
        core._log(0, f"[WARN] {os.path.basename(prepared)} carries no identity block; "
                     f"writing an explicit 'unrecoverable' one into the output.")
        ident = dict(
            dataset_id=np.array(""), da_cycle_min=np.int16(-1),
            dx_km=np.float32(np.nan), physics=np.array(""),
            upstream=np.array(""), source_run=np.array(""),
            config_index=np.full(n_members, -1, np.int16),
            config_index_note=np.array(IDENTITY_MISSING_NOTE),
        )
    return ident

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

# _qc_pass used to apply the dbz filter one point at a time. It now lives vectorised
# inside _setup, next to the other two filters and next to the masks that record what
# each of them removed -- two copies of the same rule would eventually disagree.
#
# QC codes, unchanged, and now written into the output as `qc_first_filter`:
#   none | E | T | ET_and | ET_or

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

def _setup(cfg, tm, out_cfg=None):
    """Load the subset, build the prior, and select the observation set.

    Returns (pts, Ne, qc_info). `qc_info` is the QC bookkeeping every output file
    carries: how many points each filter removed, the active band, and -- when
    output.steps_departure is on -- the pre-filter departure record described in §5.
    """
    global _XF, _ENS_HX, _TRUTH, _TRUTH_HX, _POS_KM, _NOISE, _YO_FIELD
    global _HXF_MEAN, _IDENTITY

    out_cfg = out_cfg if out_cfg is not None else output_levels_of(cfg)

    var_idx = cfg["state"]["var_idx"]
    qc_cfg  = cfg.get("qc") or {}          # `qc: null` means no filtering at all
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

    # H(xf) ensemble mean. Needed by the departure band, by the pre-filter record, and
    # by the reference file's Jensen-gap pair, so it is computed once here rather than
    # three times downstream.
    hxf_mean = ens_hx.mean(axis=3)

    mask_from = cfg["sweep"].get("obs_mask_from", None)
    qc_info = {}

    if obs_mode_of(cfg) == "single_obs":
        # The observation set is named in the config, not swept for. Building the
        # candidate mesh and running QC over it would cost a pass over 1.5 M points to
        # produce a list this mode never looks at -- and, with steps_departure on, would
        # write a pre-filter record describing points that were never observed.
        pts = _point_list(cfg)
        qc_info.update(
            qc_first_filter=np.array("none:points named in the config"),
            qc_dep_band=np.array([np.nan, np.nan], np.float32),
            qc_dep_band_active=np.bool_(False),
            qc_dep_band_per_step=np.bool_(False),
            qc_dbz_min=np.float32(dbz_min), qc_clamp_obs=np.bool_(do_clamp),
            qc_stride=np.int32(stride),
            n_obs_candidates=np.int32(len(pts)),
            n_obs_after_first_filter=np.int32(len(pts)),
            n_obs_final=np.int32(len(pts)),
            n_obs_rejected_by_band=np.int32(0),
        )
        _HXF_MEAN = hxf_mean
        _IDENTITY = _read_identity(cfg["paths"]["prepared"], Ne_tot)
        _XF, _ENS_HX, _TRUTH, _TRUTH_HX = xf, ens_hx, truth, truth_hx
        _POS_KM, _NOISE, _YO_FIELD = pos_km, noise, yo_field
        core._log(1, f"[setup tm={tm:02d}] {len(pts)} named observation point(s) "
                     f"in {time.time()-t0:.1f}s")
        return pts, Ne, qc_info

    if mask_from is not None:
        # Reuse another run's observation set verbatim. QC (`ens_hx.var > 0`) depends on Ne,
        # so recomputing it would confound an Ne comparison with a change in which points are
        # observed. Variance over a superset of members is >0 wherever it is >0 over a subset,
        # so the smaller-Ne point set is a subset of the larger-Ne one and can be shared.
        with np.load(mask_from) as ref:
            ix, iy, iz = ref["ix"], ref["iy"], ref["iz"]
        if ix.max() >= nx or iy.max() >= ny or iz.max() >= nz:
            raise ValueError(f"obs_mask_from indices exceed domain {nx}x{ny}x{nz}: {mask_from}")
        cand = (np.asarray(ix, np.intp), np.asarray(iy, np.intp), np.asarray(iz, np.intp))
        keep = np.ones(len(ix), bool)
        qc_info["qc_first_filter"] = np.array(f"obs_mask_from:{os.path.basename(mask_from)}")
        core._log(1, f"[setup tm={tm:02d}] obs mask loaded from {os.path.basename(mask_from)}: "
                     f"{len(ix)} points (stride/QC in this config ignored)")
    else:
        gi, gj, gk = np.meshgrid(np.arange(0, nx, stride), np.arange(0, ny, stride),
                                 np.arange(nz), indexing="ij")
        cand = (gi.ravel(), gj.ravel(), gk.ravel())
        del gi, gj, gk
        if do_var_filt:
            keep = ens_hx.var(axis=3)[cand] > 0.0
            qc_info["qc_first_filter"] = np.array("filter_variance")
        elif not qc_cfg:
            # No qc block at all: _qc_pass short-circuited to True for every point, and
            # so does this. Without the special case the `filter_ensemble` default of
            # True would start filtering configs that never asked for it.
            keep = np.ones(len(cand[0]), bool)
            qc_info["qc_first_filter"] = np.array("none")
        else:
            ens_max = ens_hx.max(axis=3)
            fe   = bool(qc_cfg.get("filter_ensemble", True))
            ft   = bool(qc_cfg.get("filter_truth",    False))
            fmode = str(qc_cfg.get("filter_mode", "and")).lower()
            fail_e = (ens_max[cand] < dbz_min) if fe else np.zeros(len(cand[0]), bool)
            fail_t = (yo_field[cand] < dbz_min) if ft else np.zeros(len(cand[0]), bool)
            if fe and ft:
                keep = ~(fail_e & fail_t) if fmode == "or" else ~(fail_e | fail_t)
            elif fe:
                keep = ~fail_e
            elif ft:
                keep = ~fail_t
            else:
                keep = np.ones(len(cand[0]), bool)
            del ens_max, fail_e, fail_t
            code = ("ET_" + fmode) if (fe and ft) else ("E" if fe else ("T" if ft else "none"))
            qc_info["qc_first_filter"] = np.array(f"dbz_min:{code}")

    n_cand = len(cand[0])
    dep_all = (yo_field[cand] - hxf_mean[cand]).astype(np.float32)
    pass_first = keep.copy()

    # ---- §4: the departure band --------------------------------------------
    # A REJECTION rule on the SIGNED departure d = yo - mean(H(xf)), evaluated once,
    # here: after clamp_obs and after the variance/dbz filter, and before any
    # assimilation call. The band was fitted on the sweep's stored `dep_b`, which
    # N3-1D §0a confirmed is post-clamp, so evaluating it anywhere else would apply a
    # different rule than the one the chapter derived.
    #
    # Signed, not |d|: the damaging band sits just above zero, and a rule on |d| would
    # also reject departures near -5 dBZ, which are not damaging.
    #
    # Evaluated once on the original prior. The surviving set is then held fixed across
    # every tempering step -- re-evaluating per step is a different method, closer to an
    # iterative robust filter, and an observation rejected at step 2 has already
    # influenced step 1. `dep_band_per_step` exposes that as a config choice rather than
    # leaving it a hidden assumption; only the documented `false` is implemented.
    band = dep_band_of(cfg)
    per_step = bool(qc_cfg.get("dep_band_per_step", False))
    if per_step:
        raise NotImplementedError(
            "qc.dep_band_per_step: true is not implemented. Re-evaluating the band at "
            "every tempering step is a different method, not a setting: an observation "
            "rejected at step 2 has already influenced step 1. The key exists so the "
            "choice this run made is visible in the config.")
    pass_band = np.ones(n_cand, bool)
    if band is not None:
        lo, hi = band
        rejected = (dep_all >= lo) & (dep_all <= hi)
        pass_band = ~rejected
        keep = keep & pass_band
        n_removed = int((pass_first & rejected).sum())
        core._log(1, f"[setup tm={tm:02d}] departure band [{lo}, {hi}] dBZ REJECTS "
                     f"{n_removed} of {int(pass_first.sum())} QC-passing points "
                     f"({100.0 * n_removed / max(int(pass_first.sum()), 1):.1f} %); "
                     f"{int(keep.sum())} observations remain")
        if not keep.any():
            raise ValueError(f"departure band {band} rejected every observation")

    pts = [(int(a), int(b), int(c))
           for a, b, c in zip(cand[0][keep], cand[1][keep], cand[2][keep])]

    qc_info.update(
        qc_dep_band=np.array([np.nan, np.nan] if band is None else list(band), np.float32),
        qc_dep_band_active=np.bool_(band is not None),
        qc_dep_band_per_step=np.bool_(per_step),
        qc_dbz_min=np.float32(dbz_min),
        qc_clamp_obs=np.bool_(do_clamp),
        qc_stride=np.int32(stride),
        n_obs_candidates=np.int32(n_cand),
        n_obs_after_first_filter=np.int32(int(pass_first.sum())),
        n_obs_final=np.int32(len(pts)),
        n_obs_rejected_by_band=np.int32(int((pass_first & ~pass_band).sum())),
    )

    # ---- §5: the pre-filter departure record --------------------------------
    # Stored as a VECTOR, never as a histogram: fixed bin edges are irreversible and
    # re-binning would mean re-running the experiment, which is the one thing storing
    # intermediate state is meant to avoid. Stored BEFORE any filtering, with one mask
    # per filter, so the effect of a different band or a different variance threshold
    # stays reconstructable from the file rather than needing another run.
    #
    # "Before any filtering" means before any observation is REMOVED. The departure is
    # post-clamp, because clamp_obs changes a value and drops nothing, and because §4's
    # band is defined post-clamp -- storing a pre-clamp departure here would give a
    # column that means something different from every `dep_b` in the sweep.
    if out_cfg["steps_departure"]:
        qc_info.update(
            cand_i=cand[0].astype(np.int16), cand_j=cand[1].astype(np.int16),
            cand_k=cand[2].astype(np.int16), cand_dep_b=dep_all,
            cand_pass_first=pass_first, cand_pass_band=pass_band,
        )

    _HXF_MEAN = hxf_mean
    _IDENTITY = _read_identity(cfg["paths"]["prepared"], Ne_tot)

    _XF       = xf
    _ENS_HX   = ens_hx
    _TRUTH    = truth
    _TRUTH_HX = truth_hx
    _POS_KM   = pos_km
    _NOISE    = noise
    _YO_FIELD = yo_field

    core._log(1, f"[setup tm={tm:02d}] {len(pts)} observations of {n_cand} candidates "
                 f"in {time.time()-t0:.1f}s")
    return pts, Ne, qc_info

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
                  pos_km_sub, loc_scales_km, var_idx, method, ntemp, alpha_s, dbz_min,
                  step_hook=None, clamp_mode="per_step"):
    yo_a = np.array([yo],     np.float32)
    R0_a = np.array([R0_val], np.float32)
    ox_a = np.array([ox_s],   np.int32)
    oy_a = np.array([oy_s],   np.int32)
    oz_a = np.array([oz_s],   np.int32)
    loc  = np.asarray(loc_scales_km, np.float32)

    if method == "TEnKF":
        return tenkf_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                            loc, var_idx, ntemp, alpha_s, pos_km_sub,
                            dbz_min=dbz_min, step_hook=step_hook,
                            clamp_mode=clamp_mode)["xa"]
    if method == "AOEI":
        return aoei_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                           loc, var_idx, pos_km_sub,
                           dbz_min=dbz_min, step_hook=step_hook,
                           clamp_mode=clamp_mode)["xa"]
    if method == "LETKF":
        return letkf_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                            loc, var_idx, pos_km_sub,
                            dbz_min=dbz_min, step_hook=step_hook,
                            clamp_mode=clamp_mode)["xa"]
    raise ValueError(f"Unknown method: {method}")

def _process_point(i0, j0, k0, combos, var_idx, R0_val,
                   cutoff_factor=4.0, return_fields=False, dbz_min=0.0,
                   capture_steps=False, clamp_mode="per_step"):
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
    blocks       = {} if capture_steps else None
 
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
 
        rec = None
        if capture_steps:
            # `point` mode. The probe is the observation cell itself: the update is
            # driven from there, and it is the one cell where the state ensemble, the
            # observation-space ensemble and the weights all describe the same thing.
            rec = StepRecorder(
                dict(steps_departure=True, steps_cells=((ox_s, oy_s, oz_s),),
                     steps_fields=False),
                var_idx, dbz_min=dbz_min)

        xa_sub  = _da_subdomain(xf_sub, yo, R0_val, ox_s, oy_s, oz_s,
                                pos_km_sub, (lx_km, ly_km, lz_km),
                                var_idx, method, ntemp, alpha_s, dbz_min,
                                step_hook=rec, clamp_mode=clamp_mode)
        if capture_steps:
            blocks.update(_point_block(
                c, rec, method, ntemp, alpha_s, lx_km, ly_km, lz_km,
                ens_hx_sub[ox_s, oy_s, oz_s, :], Ne))
                                
        # Crucial propagation: Ensure posterior H(x) matches the dbz_min floor perfectly
        hxa_sub = _calc_hx_domain(xa_sub, var_idx, dbz_min=dbz_min)   
 
        metrics_rows[c] = compute_single_obs_metrics(
            xf_sub, xa_sub, truth_sub,
            ens_hx_sub, hxa_sub, truth_hx_sub,
            rho, ox_s, oy_s, oz_s, yo, yo_clean, var_names, Ne, dbz_min=dbz_min)
 
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
    if capture_steps:
        return row, fields, blocks
    return row, fields

def _run_sweep_sequential(pts, combos, cfg, outdir, tag, tm, Ne, qc_info):
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    dbz_min = float((cfg.get("qc") or {}).get("dbz_min", 5.0))
    clamp_mode = clamp_mode_of(cfg)
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
            
        row, _ = _process_point(i0, j0, k0, combos, var_idx, R0_val, cutoff,
                                dbz_min=dbz_min, clamp_mode=clamp_mode)
        all_rows.append(row)
 
    merged = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
    fname = f"{tag}_sweep_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]),
                        Ne=np.int32(Ne), truth_member=np.int32(tm),
                        **merged, **qc_info, **_IDENTITY)
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
 
    pts_chunk, combos, var_idx, R0_val, cutoff, dbz_min, clamp_mode = args
    all_rows = []
    for (i0, j0, k0) in pts_chunk:
        row, _ = _process_point(i0, j0, k0, combos, var_idx, R0_val, cutoff,
                                dbz_min=dbz_min, clamp_mode=clamp_mode)
        all_rows.append(row)
    merged = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    return merged
 
def _run_sweep_parallel(pts, combos, cfg, outdir, tag, tm, Ne, n_workers, qc_info):
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    dbz_min = float((cfg.get("qc") or {}).get("dbz_min", 5.0))
    clamp_mode = clamp_mode_of(cfg)
    n_pts, n_c = len(pts), len(combos)

    import os as _os
    _os.environ["OMP_NUM_THREADS"]     = "1"
    _os.environ["MKL_NUM_THREADS"]     = "1"
    _os.environ["OPENBLAS_NUM_THREADS"]= "1"

    chunk_size  = max(1, n_pts // n_workers)
    chunks      = [pts[i:i+chunk_size] for i in range(0, n_pts, chunk_size)]
    worker_args = [(c, combos, var_idx, R0_val, cutoff, dbz_min, clamp_mode)
                   for c in chunks]
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
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]),
                        Ne=np.int32(Ne), truth_member=np.int32(tm),
                        **merged, **qc_info, **_IDENTITY)
    sz = os.path.getsize(out) / 1e6
    core._log(1, f"[sweep tm={tm:02d}] saved {n_pts*n_c} rows  {sz:.1f} MB  {time.time()-t0:.1f}s -> {fname}")

def _run_single_obs(combos, cfg, outdir, tag, tm, Ne, qc_info):
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    dbz_min = float((cfg.get("qc") or {}).get("dbz_min", 5.0))
    clamp_mode = clamp_mode_of(cfg)

    obs_loc    = cfg["sweep"]["obs_points"]["loc"]
    i0, j0, k0 = int(obs_loc["x"]), int(obs_loc["y"]), int(obs_loc["z"])

    row, _ = _process_point(
        i0, j0, k0, combos, var_idx, R0_val, cutoff,
        return_fields=False, dbz_min=dbz_min, clamp_mode=clamp_mode)
 
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
    fname = f"{tag}_single_obs_{i0}_{j0}_{k0}_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]),
                        Ne=np.int32(Ne), truth_member=np.int32(tm),
                        **row, **qc_info, **_IDENTITY)

class StepRecorder:
    """The `step_hook` §5 and §6 attach to a run, and the arrays it collects.

    One instance per (combo, truth member). It is handed the whole state at each
    tempering step and keeps only what the output block asked for, so the runner never
    holds more than one step's ensemble at a time.

    Three switches, independent because their costs differ by four orders of magnitude:
      steps_departure  the departure VECTOR per step, float32  (~13 MB/step on C)
      steps_cells      the full ensemble at a few named cells   (kilobytes)
      steps_fields     the analysis metric fields per step      (expensive)
    """

    def __init__(self, out_cfg, var_idx, truth=None, dbz_min=0.0):
        self.cfg = out_cfg
        self.var_idx = var_idx
        self.truth = truth
        self.dbz_min = dbz_min
        self.cells = tuple(out_cfg["steps_cells"])
        self.dep, self.alpha, self.r_eff = [], [], []
        self.cell_pre, self.cell_post, self.cell_hx = [], [], []
        self.fields = {}

    @property
    def active(self):
        return (self.cfg["steps_departure"] or self.cfg["steps_fields"]
                or bool(self.cells))

    def __call__(self, stage, step, n_steps, alpha, R_eff, hxf, dep, x):
        # `hxf` is H(x) at the OBSERVATION points, which in multi_obs are not the
        # probed cells, so the obs-space record is taken from `x` at the cells instead.
        # It stays in the signature because the hook contract is da.core's, not this
        # class's.
        if stage == "pre_clamp":
            if self.cfg["steps_departure"]:
                # A vector, never a histogram: fixed bin edges are irreversible, and
                # re-binning would mean re-running the experiment.
                self.dep.append(np.asarray(dep, np.float32).copy())
                self.alpha.append(float(alpha))
                r = np.asarray(R_eff, np.float64)
                self.r_eff.append((float(r.mean()), float(r.min()), float(r.max())))
            if self.cells:
                self.cell_pre.append(self._at_cells(x))
            return

        # post_clamp
        if self.cells:
            self.cell_post.append(self._at_cells(x))
            self.cell_hx.append(self._hx_at_cells(x))
        if self.cfg["steps_fields"]:
            self._add_fields(step, x)

    def _at_cells(self, x):
        # copy=True is load-bearing: `x[i, j, k]` is a VIEW, and _clamp_hydro mutates the
        # array in place immediately after the pre_clamp call. Without the copy the
        # "before" and "after" arrays would alias and the clamp would appear never to
        # fire -- silently, and in exactly the direction that confirms the hypothesis.
        return np.stack([np.array(x[i, j, k], np.float32, copy=True)
                         for i, j, k in self.cells])

    def _hx_at_cells(self, x):
        """H(x) for every member at each probed cell, so the state and observation
        spaces are recorded at the same instant rather than reconstructed later."""
        from cletkf_wloc import common_da as cda
        vi = self.var_idx
        out = []
        for i, j, k in self.cells:
            sl = np.asarray(x[i:i+1, j:j+1, k:k+1], np.float64)
            ref = cda.calc_ref_ens(sl[..., vi["qr"]], sl[..., vi["qs"]], sl[..., vi["qg"]],
                                   sl[..., vi["T"]], sl[..., vi["P"]], min_dbz=self.dbz_min)
            out.append(np.maximum(ref[0, 0, 0], self.dbz_min).astype(np.float32))
        return np.stack(out)

    def _add_fields(self, step, x):
        """The analysis-side metric fields at one intermediate step.

        Only the analysis side: the prior fields do not change between steps, so
        recomputing them Nt times would multiply the most expensive part of the run for
        nothing. Skew/kurt/CRPS are deliberately left out -- at Nt=16 they would cost
        more than the assimilation itself; the ensemble at the probed cells is the
        cheaper way to see the distribution move.
        """
        xm = x.mean(axis=3)
        self.fields[f"step{step:02d}_bias_a_field"] = (xm - self.truth).astype(np.float32)
        self.fields[f"step{step:02d}_spread_a_field"] = np.stack(
            [x[..., iv].std(axis=3, ddof=1) for iv in range(x.shape[-1])],
            axis=-1).astype(np.float32)

    def arrays(self):
        """What to write into the output file. Empty when nothing was recorded."""
        out = {}
        if self.dep:
            r = np.asarray(self.r_eff, np.float32)
            out["steps_dep"] = np.stack(self.dep)
            out["steps_alpha"] = np.asarray(self.alpha, np.float32)
            out["steps_R_eff_mean"] = r[:, 0]
            out["steps_R_eff_min"] = r[:, 1]
            out["steps_R_eff_max"] = r[:, 2]
        if self.cell_post:
            out["steps_cells_ijk"] = np.asarray(self.cells, np.int32)
            # (n_steps, n_cells, Ne, nvar) before and after the hydrometeor clamp. If
            # the two coincide the clamp never fired and the "nonlinear projection after
            # a linear update" hypothesis dies in one figure.
            out["steps_cell_state_pre_clamp"] = np.stack(self.cell_pre)
            out["steps_cell_state_post_clamp"] = np.stack(self.cell_post)
            out["steps_cell_hx"] = np.stack(self.cell_hx)   # (n_steps, n_cells, Ne)
        out.update(self.fields)
        return out


def _write_ref_file(cfg, outdir, tag, tm, Ne, pts, qc_info, mode_token):
    """The per-truth-member reference file. Written at EVERY storage level.

    It carries `truth_hx_field` and `xf_mean`, which is where the domain masks come
    from and where the Jensen gap |h(mean(x_f)) - mean(h(x_f))| is computed. Without it
    §4.7.6 loses its central predictor and the domain restriction becomes impossible,
    so it is not something a storage level is allowed to drop. Everything in it is
    prior-only and identical across schemes, which is why it is one file per truth
    member rather than a copy inside each of them.

    Returns the path. The caller asserts on it before writing anything else.
    """
    var_idx = cfg["state"]["var_idx"]
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
    dbz_min = float((cfg.get("qc") or {}).get("dbz_min", 5.0))

    ref_fname = f"{tag}_{mode_token}_ref_Ne{Ne:03d}_tm{tm:02d}.npz"
    ref_out = os.path.join(outdir, ref_fname)
    if os.path.exists(ref_out):
        return ref_fname, ref_out

    ix = np.array([p[0] for p in pts], np.int32)
    iy = np.array([p[1] for p in pts], np.int32)
    iz = np.array([p[2] for p in pts], np.int32)
    xf_mean = _XF.mean(axis=3).astype(np.float32)

    # The Jensen gap's two halves, stored side by side rather than as their difference,
    # so a reader can see which one moved. h(mean(x_f)) is one extra forward-operator
    # call on a single field; mean(h(x_f)) is _HXF_MEAN, already computed in _setup.
    hx_of_xfmean = _calc_hx_domain(xf_mean, var_idx, dbz_min=dbz_min)

    np.savez_compressed(
        ref_out,
        truth=_TRUTH,
        xf_mean=xf_mean,
        truth_hx=_TRUTH_HX,                 # legacy name, kept: notebooks resolve it
        truth_hx_field=_TRUTH_HX,           # canonical name, matches the scheme files
        hxf_mean_field=_HXF_MEAN,           # mean(h(x_f))
        hx_of_xfmean_field=hx_of_xfmean,    # h(mean(x_f))
        yo=_YO_FIELD[ix, iy, iz].astype(np.float32),
        yo_clean=_TRUTH_HX[ix, iy, iz].astype(np.float32),
        ix=ix, iy=iy, iz=iz,
        var_names=np.array(var_names),
        Ne=np.int32(Ne), truth_member=np.int32(tm),
        **qc_info, **_IDENTITY,
    )
    core._log(1, f"[ref tm={tm:02d}] {ref_fname}  "
                 f"{os.path.getsize(ref_out)/1e6:.1f} MB")
    return ref_fname, ref_out


ECHO_DBZ = 5.0          # the truth contour §4.6 and §4.7 partition on, at the cell


def _hydro_mass_budget(xf, xa, truth, truth_hx, var_idx, Ne):
    """Domain hydrometeor mass, prior and posterior, split by the 5 dBZ truth contour.

    The clamp only ever raises a value, so it only ever ADDS mass; `per_step` should
    carry the largest posterior total and `never` the smallest, and the gap between
    them is the projection in the units the projection is made of.

    Reported as the ensemble MEAN summed over the domain (kg/kg x cells), so the number
    does not change meaning when Ne does. `mass_a_neg_*` is the part of the posterior
    that is below zero -- identically zero under `per_step`, and the mass the clamp
    would have to invent under the other two.

    Reductions are protected and float64, per §3: the subset carries non-finite cells
    that a plain sum would spread over the whole domain.
    """
    echo = np.asarray(truth_hx) >= ECHO_DBZ
    out = {"echo_thresh_dbz": np.float64(ECHO_DBZ),
           "n_cells_echo": np.int64(np.count_nonzero(echo)),
           "n_cells_clear": np.int64(np.count_nonzero(~echo))}
    for side, ens in (("f", xf), ("a", xa)):
        for q in HYDRO_VARS:
            v = ens[:, :, :, :, var_idx[q]]
            col = np.nansum(v, axis=3, dtype=np.float64) / Ne
            # The negative part costs a pass per member, and on the production path it
            # is identically zero -- a prior is a model state and a clamped posterior
            # has been floored. One nanmin says so for the price of one pass, which
            # takes the whole budget from ~8 s per scheme file to under 2.
            neg = None
            if np.nanmin(v) < 0.0:
                neg = np.zeros(col.shape, np.float64)
                for m in range(v.shape[3]):
                    neg += np.nan_to_num(np.minimum(v[:, :, :, m], 0.0), nan=0.0)
                neg /= Ne
            for name, sel in (("all", None), ("in", echo), ("out", ~echo)):
                c = col if sel is None else col[sel]
                out[f"mass_{side}_{q}_{name}"] = np.float64(c.sum())
                out[f"mass_{side}neg_{q}_{name}"] = np.float64(
                    0.0 if neg is None else (neg if sel is None else neg[sel]).sum())
    for q in HYDRO_VARS:
        t = np.asarray(truth[:, :, :, var_idx[q]], np.float64)
        for name, sel in (("all", None), ("in", echo), ("out", ~echo)):
            tt = t if sel is None else t[sel]
            out[f"mass_t_{q}_{name}"] = np.float64(np.nansum(tt))
    return out


def _run_multi_obs(pts, combos, cfg, outdir, tag, tm, Ne, qc_info, out_cfg):
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    storage = storage_level_of(cfg)
    dbz_min = float((cfg.get("qc") or {}).get("dbz_min", 5.0))
    clamp_mode = clamp_mode_of(cfg)

    xf       = _XF
    truth    = _TRUTH
    truth_hx = _TRUTH_HX
    yo_field = _YO_FIELD
    pos_km   = _POS_KM

    ix  = np.array([p[0] for p in pts], np.int32)
    iy  = np.array([p[1] for p in pts], np.int32)
    iz  = np.array([p[2] for p in pts], np.int32)
    yo       = yo_field[ix, iy, iz].astype(np.float32)
    R0       = np.full(len(pts), R0_val, np.float32)

    # Before anything else. Every scheme file below is a delta against this one, and at
    # `light` there is nothing else on disk that carries the truth field at all.
    ref_fname, ref_out = _write_ref_file(cfg, outdir, tag, tm, Ne, pts, qc_info,
                                         "multi_obs")
    assert os.path.exists(ref_out), (
        f"the reference file {ref_fname} was not written; refusing to write scheme "
        f"files that would have nothing to be compared against")

    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]

    for (method, ntemp, alpha_s, lx_km, ly_km, lz_km) in combos:
        fname = (f"{tag}_multi_obs_{method}_Nt{ntemp:02d}"
                 f"_as{alpha_s:.1f}_Lx{lx_km}Ly{ly_km}Lz{lz_km}"
                 f"_Ne{Ne:03d}_tm{tm:02d}.npz")
        out = os.path.join(outdir, fname)

        if cfg.get("skip_existing", False) and os.path.exists(out):
            continue

        loc = np.array([lx_km, ly_km, lz_km], np.float32)
        t1 = time.time()

        rec = StepRecorder(out_cfg, var_idx, truth=truth, dbz_min=dbz_min)
        hook = rec if rec.active else None

        core._log(3, f"  [multi_obs] calling {method} Nt={ntemp} alpha={alpha_s} loc=({lx_km},{ly_km},{lz_km})")
        if method == "TEnKF":
            res = tenkf_update(xf, yo, R0, ix, iy, iz, loc, var_idx, ntemp, alpha_s,
                               np.asfortranarray(pos_km), dbz_min=dbz_min,
                               step_hook=hook, clamp_mode=clamp_mode)
        elif method == "AOEI":
            res = aoei_update(xf, yo, R0, ix, iy, iz, loc, var_idx,
                              np.asfortranarray(pos_km), dbz_min=dbz_min,
                              step_hook=hook, clamp_mode=clamp_mode)
        elif method == "LETKF":
            res = letkf_update(xf, yo, R0, ix, iy, iz, loc, var_idx,
                               np.asfortranarray(pos_km), dbz_min=dbz_min,
                               step_hook=hook, clamp_mode=clamp_mode)
        else:
            raise ValueError(f"Unknown method: {method}")

        xa = res["xa"]
        core._log(3, f"  [multi_obs] DA done  xa.shape={xa.shape}  xa.dtype={xa.dtype}")

        # Crucial propagation: Ensure multi-obs final analysis matches the floor securely
        core._log(3, f"  [multi_obs] computing H(xa) domain-wide ...")
        hxa_ens = _calc_hx_domain(xa, var_idx, dbz_min=dbz_min)
        core._log(3, f"  [multi_obs] H(xa) done  shape={hxa_ens.shape}")
        hxa_mean_field = hxa_ens.mean(axis=3)

        core._log(3, f"  [multi_obs] computing metrics ...")
        m = compute_multi_obs_metrics(
            xa, xf, truth,
            _HXF_MEAN, hxa_mean_field, truth_hx,
            var_names, Ne,
            store_ensemble=out_cfg["store_ensemble"], storage_level=storage,
            ens_hxf=_ENS_HX, ens_hxa=hxa_ens, dbz_min=dbz_min,
            obs_ijk=(ix, iy, iz))
        core._log(3, f"  [multi_obs] metrics done, saving -> {fname}")

        extra = rec.arrays()
        # Unconditionally, at every storage level: what the floor at zero did, per
        # tempering step, and the hydrometeor mass budget it moved. A dozen numbers
        # against a 39 MB file, and without them the projection is only measurable in
        # a run built to measure it.
        extra.update(res["clamp"])
        extra.update(_hydro_mass_budget(xf, xa, truth, truth_hx, var_idx, Ne))
        if method == "AOEI" and out_cfg["steps_departure"]:
            # AOEI's whole content is the per-observation inflation, so the vector is
            # the diagnostic; a per-step mean would say nothing about it.
            extra["aoei_R_tilde"] = np.asarray(res["obs_error"], np.float32)

        np.savez_compressed(out,
            method       = method, ntemp        = np.int32(ntemp), alpha_s      = np.float32(alpha_s),
            lx_km        = np.float32(lx_km), ly_km        = np.float32(ly_km), lz_km        = np.float32(lz_km),
            Ne           = np.int32(Ne), truth_member = np.int32(tm), var_names    = np.array(var_names),
            ref_file     = ref_fname,
            storage_level = np.array(storage),
            store_ensemble = np.bool_(out_cfg["store_ensemble"]),
            **qc_info, **_IDENTITY, **extra, **m,
        )
        core._log(1, f"  [multi_obs] {fname}  {os.path.getsize(out)/1e6:.1f} MB  "
                     f"{time.time()-t1:.1f}s")
        del xa, hxa_ens, m, extra


def _point_block(c, rec, method, ntemp, alpha_s, lx, ly, lz, hxf_prior_at_obs, Ne):
    """Everything §6 asks for at one probed point, for one method combination.

    The weights are recomputed from the same inputs the Fortran saw, one step at a
    time. At the observation cell the localization distance is zero, so the weight is
    exactly 1 and rloc is the step's effective R -- no interpolation and no cutoff
    ambiguity. The step's H(x) ensemble is the prior's at step 0 and the previous
    step's post-clamp H(x) after that, which is precisely what the tempering loop feeds
    the next call, so nothing has to be replayed.
    """
    pre  = np.asarray(rec.cell_pre,  np.float32)[:, 0]      # (n_steps, Ne, nvar)
    post = np.asarray(rec.cell_post, np.float32)[:, 0]
    hx   = np.asarray(rec.cell_hx,   np.float32)[:, 0]      # (n_steps, Ne)
    n_steps = pre.shape[0]

    trans  = np.empty((n_steps, Ne, Ne), np.float32)
    transm = np.empty((n_steps, Ne),      np.float32)
    for it in range(n_steps):
        hxf_it = np.asarray(hxf_prior_at_obs if it == 0 else hx[it - 1], np.float64)
        hdxb = (hxf_it - hxf_it.mean())[None, :]            # (1 obs, Ne)
        rloc = np.array([rec.r_eff[it][0]], np.float64)     # d = 0 -> weight 1
        t, w = core.letkf_weights(hdxb, rloc, np.asarray(rec.dep[it], np.float64))
        trans[it], transm[it] = t, w

    pfx = f"c{c:02d}"
    return {
        f"{pfx}_method": np.array(method), f"{pfx}_ntemp": np.int32(ntemp),
        f"{pfx}_alpha_s": np.float32(alpha_s),
        f"{pfx}_lx_km": np.float32(lx), f"{pfx}_ly_km": np.float32(ly),
        f"{pfx}_lz_km": np.float32(lz),
        f"{pfx}_steps_alpha": np.asarray(rec.alpha, np.float32),
        f"{pfx}_steps_R_eff": np.asarray([r[0] for r in rec.r_eff], np.float32),
        f"{pfx}_steps_dep": np.concatenate(rec.dep).astype(np.float32),
        # Before and after the hydrometeor clamp. A nonlinear projection applied after
        # a linear update breaks consistency between the state and the covariances that
        # produced it, and would accumulate over iterations -- which fits the pattern
        # that hydrometeors improve while everything else degrades. If these two arrays
        # coincide the hypothesis dies in one figure.
        f"{pfx}_state_pre_clamp": pre, f"{pfx}_state_post_clamp": post,
        f"{pfx}_hx": hx,
        # The increment is X_f w, so `transm` says whether one member dominates the
        # update and `trans` says what the transform did to the spread. Watching
        # transm concentrate as Nt grows is collapse by weight degeneracy rather than
        # by the transform, and the two are indistinguishable from xa alone.
        f"{pfx}_trans": trans, f"{pfx}_transm": transm,
    }


def _point_list(cfg):
    """The observation points for `point` mode.

    `points: [[i,j,k], ...]` is the mode's own spelling; a lone `loc: {x,y,z}` is the
    single_obs form and still works, so an existing single_obs config runs unchanged.
    """
    obs = cfg["sweep"]["obs_points"]
    if isinstance(obs, dict) and obs.get("points"):
        pts = [tuple(int(q) for q in pt) for pt in obs["points"]]
        if any(len(pt) != 3 for pt in pts):
            raise ValueError("sweep.obs_points.points entries must be [i, j, k]")
        return pts
    loc = obs["loc"]
    return [(int(loc["x"]), int(loc["y"]), int(loc["z"]))]


def _run_point(cfg, outdir, tag, tm, Ne, qc_info, out_cfg, combos):
    """`point` mode: single_obs at a list of points, storing everything.

    An extension of single_obs rather than a fourth code path -- single_obs already
    assimilates one fixed observation over every method combination, and a fourth path
    would need maintaining and testing in parallel with it. The metrics row is byte for
    byte the one single_obs writes; what `point` adds is the per-step capture.

    Sixty members x nine fields x sixteen steps x three points is kilobytes, so it
    stores all of it and the user chooses nothing.
    """
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    dbz_min = float((cfg.get("qc") or {}).get("dbz_min", 5.0))
    clamp_mode = clamp_mode_of(cfg)
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
    points = _point_list(cfg)

    ref_fname, ref_out = _write_ref_file(cfg, outdir, tag, tm, Ne, points, qc_info,
                                         "point")
    assert os.path.exists(ref_out), (
        f"the reference file {ref_fname} was not written; refusing to write point "
        f"files that would have nothing to be compared against")

    for (i0, j0, k0) in points:
        t0 = time.time()
        row, _, blocks = _process_point(
            i0, j0, k0, combos, var_idx, R0_val, cutoff,
            return_fields=False, dbz_min=dbz_min, capture_steps=True,
            clamp_mode=clamp_mode)

        fname = f"{tag}_point_{i0}_{j0}_{k0}_Ne{Ne:03d}_tm{tm:02d}.npz"
        outp  = os.path.join(outdir, fname)
        np.savez_compressed(outp,
            var_names=np.array(var_names + ["ref"]),
            n_combos=np.int32(len(combos)),
            Ne=np.int32(Ne), truth_member=np.int32(tm), ref_file=ref_fname,
            clamp_mode=np.array(clamp_mode),
            xf_point=np.asarray(_XF[i0, j0, k0], np.float32),
            truth_point_state=np.asarray(_TRUTH[i0, j0, k0], np.float32),
            hxf_point=np.asarray(_ENS_HX[i0, j0, k0], np.float32),
            **row, **blocks, **qc_info, **_IDENTITY)
        core._log(1, f"  [point] {fname}  {os.path.getsize(outp)/1e3:.0f} kB  "
                     f"{time.time()-t0:.1f}s")


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

    # Before anything is loaded, created or copied. A run that takes six hours should
    # not discover at write time that its name is wrong, and a tag that disagrees with
    # its own config is worse than an ugly one, because it is believed.
    try:
        validate_experiment_tag(cfg, tag)
        out_cfg = output_levels_of(cfg, warn=lambda m: core._log(0, f"[DEPRECATED] {m}"))
        # Same reason as the tag: a misspelt clamp mode must not be discovered an hour
        # into the run, and must not silently fall back to the default.
        clamp_mode = clamp_mode_of(cfg)
    except (TagError, ValueError) as e:
        sys.exit(f"[{os.path.basename(args.config)}] {e}")

    core._log(1, f"[{tag}] storage: metric fields {storage_level_of(cfg)!r}"
                 f"  clamp_hydro {clamp_mode!r}"
                 f"  ensemble {out_cfg['store_ensemble']}"
                 f"  steps dep/cells/fields "
                 f"{out_cfg['steps_departure']}/{len(out_cfg['steps_cells'])}/"
                 f"{out_cfg['steps_fields']}")

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

    # `point` and `single_obs` are the same mode under two names -- the tag's spelling
    # and the runner's. naming.obs_mode_of resolves both to the runner's.
    obs_mode = obs_mode_of(cfg)

    n_workers = max(1, args.workers)
    t_start = time.time()

    if obs_mode == "sweep" and n_workers > 1:
        os.environ["OMP_NUM_THREADS"] = "1"

    combos  = _build_combos(cfg["sweep"])

    for tm in tm_list:
        core._log(1, f"[{tag}] === truth member {tm} ===")
        t_tm = time.time()
        pts, Ne, qc_info = _setup(cfg, tm, out_cfg)

        if obs_mode == "single_obs":
            # `points:` promotes single_obs to §6's point mode: the same assimilation,
            # at a list of cells, with the whole ensemble and the weights kept at each.
            if isinstance(cfg["sweep"]["obs_points"], dict) and \
                    cfg["sweep"]["obs_points"].get("points"):
                _run_point(cfg, outdir, tag, tm, Ne, qc_info, out_cfg, combos)
            else:
                _run_single_obs(combos, cfg, outdir, tag, tm, Ne, qc_info)
        elif obs_mode == "sweep":
            if n_workers == 1:
                _run_sweep_sequential(pts, combos, cfg, outdir, tag, tm, Ne, qc_info)
            else:
                os.environ["OMP_NUM_THREADS"] = "1"
                _run_sweep_parallel(pts, combos, cfg, outdir, tag, tm, Ne, n_workers,
                                    qc_info)
        elif obs_mode == "multi_obs":
            _run_multi_obs(pts, combos, cfg, outdir, tag, tm, Ne, qc_info, out_cfg)
        else:
            raise ValueError(f"Unknown obs_mode '{obs_mode}'.")

        core._log(1, f"[{tag}] tm={tm} done in {time.time()-t_tm:.1f}s")

if __name__ == "__main__":
    main()