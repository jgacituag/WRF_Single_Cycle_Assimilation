"""
src/runners/run_full3d_multicycle_exps.py
==========================================
Full-3D multi-method experiment runner (WS-3, WS-4, WS-5).

WS-3  Multiple obs, uniform ::2 grid, no QC filtering
WS-4  Multiple obs, uniform ::2 grid, near-zero departure filtering ON
WS-5  Localization/inflation tuning grid (rho_inf x L sweep)
      -- WS-5 reuses this runner with tuning=true in config

Usage
-----
    cd <repo_root>
    python src/runners/run_full3d_multicycle_exps.py --config configs/ws3.yaml
    python src/runners/run_full3d_multicycle_exps.py --config configs/ws4.yaml
    python src/runners/run_full3d_multicycle_exps.py --config configs/ws5.yaml
"""

import sys, os, pathlib, argparse, yaml, json
import numpy as np
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "fortran"))

from da.core import (
    tempering_schedule,
    compute_hxf,
    letkf_update,
    tenkf_update,
    aoei_update,
    atenkf_update,
)

# ── helpers ────────────────────────────────────────────────────────────────

def _save(outdir, tag, meta, **arrays):
    os.makedirs(outdir, exist_ok=True)
    meta["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(os.path.join(outdir, f"{tag}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    np.savez_compressed(os.path.join(outdir, f"{tag}.npz"), **arrays)
    print(f"  [save] {tag}")


def _load_member(ens, truth_member):
    mask = np.zeros(ens.shape[3], dtype=bool)
    mask[truth_member] = True
    truth = ens[:, :, :, mask,  :][:, :, :, 0, :]
    xf    = ens[:, :, :, ~mask, :].astype(np.float32)
    return truth, xf


def build_obs(truth, xf, ox_in, oy_in, oz_in, var_idx,
              sigma_dbz=5.0, dbz_min=5.0, dbz_max=70.0,
              detect_prob_min=0.20, filter_near_zero=False, k_sigma=5.0):
    """
    Build observation array from truth + QC.

    filter_near_zero=True  -> WS-4: drop obs where both yo and ensemble
                               mean are below dbz_min (no rain in either).
    filter_near_zero=False -> WS-3: only basic range + detectability QC.

    Parameters
    ----------
    ox_in, oy_in, oz_in : 1-D arrays of candidate indices along each axis.
                          The full candidate set is their Cartesian product.

    Returns (yo, ox, oy, oz) after QC, all length nobs_kept.
    """
    from cletkf_wloc import common_da as cda

    # Build Cartesian product of candidate indices
    gx, gy, gz = np.meshgrid(ox_in, oy_in, oz_in, indexing="ij")
    ox = gx.ravel().astype(int)
    oy = gy.ravel().astype(int)
    oz = gz.ravel().astype(int)
    nobs_cand = len(ox)
    Ne = xf.shape[3]

    yo   = np.empty(nobs_cand, np.float32)
    Hx_f = np.empty((nobs_cand, Ne), np.float32)

    for ii in range(nobs_cand):
        i, j, k = ox[ii], oy[ii], oz[ii]
        yo[ii] = cda.calc_ref(
            truth[i,j,k,var_idx["qr"]], truth[i,j,k,var_idx["qs"]],
            truth[i,j,k,var_idx["qg"]], truth[i,j,k,var_idx["T"]],
            truth[i,j,k,var_idx["P"]],
        )+ np.random.normal(0, np.sqrt(sigma_dbz))  # add obs error in dBZ
        for m in range(Ne):
            Hx_f[ii,m] = cda.calc_ref(
                xf[i,j,k,m,var_idx["qr"]], xf[i,j,k,m,var_idx["qs"]],
                xf[i,j,k,m,var_idx["qg"]], xf[i,j,k,m,var_idx["T"]],
                xf[i,j,k,m,var_idx["P"]],
            )

    keep = np.isfinite(yo) & np.all(np.isfinite(Hx_f), axis=1)
    keep &= (yo >= dbz_min) & (yo <= dbz_max)

    # Detectability: at least detect_prob_min fraction of members see rain
    p_detect = (Hx_f > dbz_min).mean(axis=1)
    keep &= (p_detect >= detect_prob_min)

    if filter_near_zero:
        # WS-4: drop points where both truth and ensemble mean are below threshold
        # (near-zero departure, no rain signal in either)
        hx_mean = Hx_f.mean(axis=1)
        keep &= ~((yo < dbz_min) & (hx_mean < dbz_min))

    if keep.sum() == 0:
        return (np.array([], np.float32),
                np.array([], int), np.array([], int), np.array([], int))

    yo, ox, oy, oz = yo[keep], ox[keep], oy[keep], oz[keep]

    # Deduplicate
    key = np.stack([ox, oy, oz], axis=1)
    _, ui = np.unique(key, axis=0, return_index=True)
    ui.sort()
    return yo[ui], ox[ui], oy[ui], oz[ui]


# ── main loop ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths  = cfg["paths"]
    st     = cfg["state"]
    obs    = cfg["obs"]
    da     = cfg["da"]
    tag    = cfg.get("experiment_tag", "exp")
    outdir = paths["outdir"]

    var_idx  = st["var_idx"]
    sigma_dbz = float(obs.get("sigma_dbz", 5.0))
    R0_scalar = sigma_dbz                         # VARIANCE
    stride    = int(obs.get("stride", 2))
    filter_nz = bool(obs.get("filter_near_zero", False))
    dbz_min   = float(obs.get("dbz_min", 5.0))

    ntemps    = da.get("ntemps",    [1])
    alphas    = da.get("alphas",    [2.0])
    loc_scales_list = da.get("loc_scales_list", [[5,5,5]])
    methods   = da.get("methods",   ["LETKF","TEnKF","AOEI","ATEnKF"])
    n_members = int(da.get("n_members", 30))

    # Load ensemble (single file for single-cycle experiments)
    data = np.load(paths["prepared"])
    ens  = data["state_ensemble"] if "state_ensemble" in data else data["cross_sections"]
    nx, ny, nz = ens.shape[0], ens.shape[1], ens.shape[2]

    print(f"\n{'='*60}")
    print(f"Experiment : {tag}")
    print(f"Domain     : {nx} x {ny} x {nz}")
    print(f"stride     : {stride}  filter_near_zero={filter_nz}")
    print(f"Methods    : {methods}")
    print(f"Ntemp list : {ntemps}  alphas : {alphas}")
    print(f"Loc scales : {loc_scales_list}")
    print(f"{'='*60}\n")

    # Candidate obs indices (strided grid, no double-stride)
    ox_cand = np.arange(0, nx, stride)
    oy_cand = np.arange(0, ny, stride)
    oz_cand = np.arange(0, nz, stride)

    for tm in range(n_members):
        truth, xf = _load_member(ens, tm)

        yo, ox_arr, oy_arr, oz_arr = build_obs(
            truth, xf, ox_cand, oy_cand, oz_cand, var_idx,
            sigma_dbz=sigma_dbz, dbz_min=dbz_min,
            filter_near_zero=filter_nz,
        )

        if len(yo) == 0:
            print(f"  truth={tm:02d}  -> no observations after QC, skipping")
            continue

        print(f"  truth={tm:02d}  nobs={len(yo)}")
        R0 = np.full(len(yo), R0_scalar, np.float32)

        for ntemp in ntemps:
            for alpha_s in alphas:
                for loc_list in loc_scales_list:
                    loc_scales = np.array(loc_list, np.float32)
                    loc_str    = "_".join(str(l) for l in loc_list)

                    for method in methods:
                        out_tag = (f"{tag}_Nt{ntemp:02d}_as{alpha_s:.1f}"
                                   f"_Loc{loc_str}_True{tm:02d}_{method}")
                        if os.path.exists(os.path.join(outdir, out_tag + ".npz")):
                            print(f"    [skip] {out_tag}")
                            continue

                        print(f"    {method}  Nt={ntemp} as={alpha_s} loc={loc_list}")

                        if method == "LETKF":
                            res = letkf_update(
                                xf, yo, R0, ox_arr, oy_arr, oz_arr, loc_scales, var_idx)
                        elif method == "TEnKF":
                            res = tenkf_update(
                                xf, yo, R0, ox_arr, oy_arr, oz_arr, loc_scales, var_idx,
                                ntemp=ntemp, alpha_s=alpha_s)
                        elif method == "AOEI":
                            res = aoei_update(
                                xf, yo, R0, ox_arr, oy_arr, oz_arr, loc_scales, var_idx)
                        elif method == "ATEnKF":
                            res = atenkf_update(
                                xf, yo, R0, ox_arr, oy_arr, oz_arr, loc_scales, var_idx,
                                ntemp=ntemp, alpha_s=alpha_s)
                        else:
                            raise ValueError(f"Unknown method: {method}")

                        _save(outdir, out_tag,
                              meta={"method":method, "ntemp":ntemp, "alpha_s":alpha_s,
                                    "loc_scales":loc_list, "truth_member":tm,
                                    "filter_near_zero":filter_nz, "nobs":len(yo)},
                              **res, yo=yo, xf=xf, truth=truth,
                              ox=ox_arr, oy=oy_arr, oz=oz_arr)

    print(f"\n[{tag} complete]")

if __name__ == "__main__":
    main()
