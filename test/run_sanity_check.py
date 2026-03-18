"""
Quick sanity check: run all DA methods on a single observation point
with one truth member and print a comparison table.

Use this before committing to a full WS run to verify that:
  - all methods produce a posterior closer to the truth than the prior
  - AOEI inflates where expected
  - ATEnKF Ntemp_j is consistent with the inflation ratio
  - no obvious sign errors or unit mismatches

Usage
-----
  python test/run_sanity_check.py --config configs/ws2.yaml
  python test/run_sanity_check.py --config configs/ws2.yaml \\
      --truth 0 --x 10 --y 0 --z 15
"""

import argparse
import pathlib
import sys
 
import numpy as np
import yaml
 
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "fortran"))
 
from da.core import (
    tempering_schedule,
    letkf_update, tenkf_update, aoei_update,
    atenkf_update, taoei_update,
    aoei, _solve_ntemp,
)
 
 
def _expand(val, is_int=False):
    """Expand scalar / list / {start,stop,num} to a flat list."""
    if isinstance(val, dict):
        arr = np.linspace(val["start"], val["stop"], int(val["num"]))
        if is_int:
            arr = np.unique(np.round(arr).astype(int))
        return arr.tolist()
    if isinstance(val, list):
        return val
    return [val]
 
 
def _hx(state_3d, i, j, k, var_idx):
    """Reflectivity at one grid point for one member. state_3d: (nx,ny,nz,nvar)"""
    from cletkf_wloc import common_da as cda
    return float(cda.calc_ref(
        state_3d[i,j,k,var_idx["qr"]], state_3d[i,j,k,var_idx["qs"]],
        state_3d[i,j,k,var_idx["qg"]], state_3d[i,j,k,var_idx["T"]],
        state_3d[i,j,k,var_idx["P"]],
    ))
 
 
def _hx_ensemble(xf, i, j, k, var_idx):
    """H(x^f) for all members at one point. Returns (Ne,) array."""
    from cletkf_wloc import common_da as cda
    Ne = xf.shape[3]
    return np.array([cda.calc_ref(
        xf[i,j,k,m,var_idx["qr"]], xf[i,j,k,m,var_idx["qs"]],
        xf[i,j,k,m,var_idx["qg"]], xf[i,j,k,m,var_idx["T"]],
        xf[i,j,k,m,var_idx["P"]],
    ) for m in range(Ne)], np.float32)
 
 
def run_sanity_check(cfg, truth_member, ix, iy, iz):
    var_idx   = cfg["state"]["var_idx"]
    sweep     = cfg.get("sweep", {})
    R0_scalar = float(cfg["obs"]["obs_error_var"])
 
    # loc_scales: read from sweep.loc_x/y/z (unified config)
    # or from da.loc_scales (legacy), defaulting to [5,5,5]
    da_cfg = cfg.get("da", {})
    lx = float(_expand(sweep.get("loc_x", da_cfg.get("loc_scales", [5,5,5])[0]))[0])
    ly = float(_expand(sweep.get("loc_y", da_cfg.get("loc_scales", [5,5,5])[1]))[0])
    lz = float(_expand(sweep.get("loc_z", da_cfg.get("loc_scales", [5,5,5])[2]))[0])
    loc_scales = np.array([lx, ly, lz], np.float32)
 
    ntemp   = int(_expand(sweep.get("ntemp",   da_cfg.get("ntemp",   3)), is_int=True)[0])
    alpha_s = float(_expand(sweep.get("alpha_s", da_cfg.get("alpha_s", 2.0)))[0])
    methods = sweep.get("methods", da_cfg.get("methods",
                        ["LETKF","TEnKF","AOEI","ATEnKF","TAOEI"]))
 
    data = np.load(cfg["paths"]["prepared"])
    ens  = data["state_ensemble"] if "state_ensemble" in data \
           else data["cross_sections"]
 
    mask  = np.zeros(ens.shape[3], dtype=bool)
    mask[truth_member] = True
    truth = ens[:, :, :, mask,  :][:, :, :, 0, :]
    xf    = ens[:, :, :, ~mask, :].astype(np.float32)
    Ne    = xf.shape[3]
 
    yo_val     = _hx(truth, ix, iy, iz, var_idx)
    hxf_all    = _hx_ensemble(xf, ix, iy, iz, var_idx)
    hxf_mean   = float(hxf_all.mean())
    hxf_spread = float(hxf_all.std(ddof=1))
    dep        = yo_val - hxf_mean
    R0         = np.array([R0_scalar],  np.float32)
    yo         = np.array([yo_val],     np.float32)
    ox         = np.array([ix], np.int32)
    oy         = np.array([iy], np.int32)
    oz         = np.array([iz], np.int32)
 
    # AOEI diagnostics
    R_tilde     = float(aoei(yo, hxf_all[np.newaxis,:], R0)[0])
    infl_ratio  = R_tilde / R0_scalar
    ntemp_atenkf = _solve_ntemp(infl_ratio, alpha_s=1.0)
 
    # ## header ########################################################
    sep = "#" * 64
    print(f"\n{sep}")
    print(f"  SANITY CHECK — truth member {truth_member}  "
          f"obs ({ix},{iy},{iz})")
    print(sep)
    print(f"  yo          = {yo_val:8.3f} dBZ   (truth reflectivity)")
    print(f"  H(x̄^f)      = {hxf_mean:8.3f} dBZ   (prior ensemble mean)")
    print(f"  d = yo−H(x̄) = {dep:+8.3f} dBZ")
    print(f"  σ_H         = {hxf_spread:8.3f} dBZ   (prior spread)")
    print(f"  R0          = {R0_scalar:8.3f} dBZ²  (nominal obs variance)")
    print(f"  R_tilde     = {R_tilde:8.3f} dBZ²  (AOEI inflated variance)")
    print(f"  R_tilde/R0  = {infl_ratio:8.3f}        (inflation ratio)")
    print(f"  Ntemp_ATEnKF= {ntemp_atenkf:8d}         "
          f"(steps needed to reach R_tilde at step 1)")
    print()
 
    # ## per-method results #############################################
    print(f"  {'Method':<12}  {'xa_mean':>9}  {'xa_spread':>9}  "
          f"{'d_post':>9}  {'|d| reduc%':>10}  notes")
    print(f"  {'#'*12}  {'#'*9}  {'#'*9}  {'#'*9}  {'#'*10}")
 
    METHOD_FNS = {
        "LETKF":  lambda: letkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx),
        "TEnKF":  lambda: tenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                                        ntemp=ntemp, alpha_s=alpha_s),
        "AOEI":   lambda: aoei_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx),
        "ATEnKF": lambda: atenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                                         alpha_s=1.0),
        "TAOEI":  lambda: taoei_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                                        ntemp=ntemp, alpha_s=alpha_s),
    }
 
    from cletkf_wloc import common_da as cda
    for method in methods:
        if method not in METHOD_FNS:
            print(f"  {method:<12}  [unknown method]"); continue
 
        res = METHOD_FNS[method]()
        xa  = res["xa"]
 
        xa_hx = np.array([cda.calc_ref(
            xa[ix,iy,iz,m,var_idx["qr"]], xa[ix,iy,iz,m,var_idx["qs"]],
            xa[ix,iy,iz,m,var_idx["qg"]], xa[ix,iy,iz,m,var_idx["T"]],
            xa[ix,iy,iz,m,var_idx["P"]],
        ) for m in range(Ne)], np.float32)
 
        xa_mean   = float(xa_hx.mean())
        xa_spread = float(xa_hx.std(ddof=1))
        d_post    = yo_val - xa_mean
        pct_red   = (1.0 - abs(d_post) / max(abs(dep), 1e-6)) * 100
 
        # collect extra notes
        notes = []
        if method in ("AOEI", "ATEnKF", "TAOEI") and "obs_error" in res:
            r_used = float(np.asarray(res["obs_error"]).flat[0])
            notes.append(f"R_used={r_used:.1f}")
        if method == "ATEnKF" and "ntemps_per_obs" in res:
            notes.append(f"Nt_j={int(res['ntemps_per_obs'][0])}")
 
        flag = " ✓" if pct_red > 0 else " ✗ WORSE"
        print(f"  {method:<12}  {xa_mean:9.3f}  {xa_spread:9.3f}  "
              f"{d_post:+9.3f}  {pct_red:9.1f}%{flag}  "
              + "  ".join(notes))
 
    # ## tempering schedule preview #####################################
    print()
    print(f"  Tempering schedule (Ntemp={ntemp}, alpha_s={alpha_s}):")
    sched = tempering_schedule(ntemp, alpha_s)
    for i, a in enumerate(sched):
        print(f"    step {i+1}:  alpha={a:.5f}  R/alpha={R0_scalar/a:.2f} dBZ²")
 
    print(sep + "\n")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True)
    ap.add_argument("--truth",   type=int, default=0,  help="Truth member index")
    ap.add_argument("--x",       type=int, default=0,  help="Grid i index")
    ap.add_argument("--y",       type=int, default=0,  help="Grid j index")
    ap.add_argument("--z",       type=int, default=0,  help="Grid k index")
    args = ap.parse_args()
 
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
 
    run_sanity_check(cfg,
                     truth_member=args.truth,
                     ix=args.x, iy=args.y, iz=args.z)
 
 
if __name__ == "__main__":
    main()