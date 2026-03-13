"""
src/runners/run_single_obs_exps.py
===================================
Single-observation experiment runner (WS-1 and WS-2).

WS-1  Ntemp sweep
    - Fixed obs location, fixed alpha_s
    - Runs LETKF (Ntemp=1) and TEnKF for each Ntemp in ntemp_sweep
    - Loops over all 30 truth members

WS-2  Obs position comparison
    - Three fixed observation positions (A near / B above / C below ensemble mean)
    - Runs all four methods: LETKF, TEnKF, AOEI, ATEnKF
    - Loops over all 30 truth members

Usage
-----
    cd <repo_root>
    python src/runners/run_single_obs_exps.py --config configs/ws1.yaml
    python src/runners/run_single_obs_exps.py --config configs/ws2.yaml
"""

import sys, os, pathlib, argparse, yaml, json
import numpy as np
from datetime import datetime

# ── path setup ─────────────────────────────────────────────────────────────
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


def _load_member(data, truth_member):
    """Split ensemble array into truth + prior ensemble."""
    mask = np.zeros(data.shape[3], dtype=bool)
    mask[truth_member] = True
    truth = data[:, :, :, mask,  :][:, :, :, 0, :]   # (nx,ny,nz,nvar)
    xf    = data[:, :, :, ~mask, :].astype(np.float32)  # (nx,ny,nz,Ne,nvar)
    return truth, xf


def _truth_yo(truth, x, y, z, var_idx):
    """Scalar reflectivity at one grid point from the truth."""
    from cletkf_wloc import common_da as cda
    return float(cda.calc_ref(
        truth[x, y, z, var_idx["qr"]],
        truth[x, y, z, var_idx["qs"]],
        truth[x, y, z, var_idx["qg"]],
        truth[x, y, z, var_idx["T"]],
        truth[x, y, z, var_idx["P"]],
    ))


# ── WS-1 ───────────────────────────────────────────────────────────────────

def run_ws1(cfg):
    print("\n" + "="*60)
    print("WS-1: Ntemp sweep")
    print("="*60)

    st     = cfg["state"]
    da     = cfg["da"]
    paths  = cfg["paths"]
    obs    = cfg["obs"]

    data       = np.load(paths["prepared"])
    ens        = data["state_ensemble"] if "state_ensemble" in data else data["cross_sections"]
    var_idx    = st["var_idx"]
    alpha_s    = float(da["alpha_s"])
    R0_scalar  = float(obs["sigma_dbz"])          # variance
    loc_scales = np.array(da["loc_scales"], np.float32)
    outdir     = paths["outdir"]
    ntemp_list = da["ntemp_sweep"]
    ox = int(obs["loc"]["x"])
    oy = int(obs["loc"]["y"])
    oz = int(obs["loc"]["z"])
    ox_arr = np.array([ox], np.int32)
    oy_arr = np.array([oy], np.int32)
    oz_arr = np.array([oz], np.int32)

    n_members = ens.shape[3]
    print(f"  obs location : ({ox},{oy},{oz})")
    print(f"  alpha_s      : {alpha_s}")
    print(f"  R0           : {R0_scalar:.2f} dBZ²")
    print(f"  truth members: all {n_members}")

    for tm in range(n_members):
        truth, xf = _load_member(ens, tm)
        yo        = np.array([_truth_yo(truth, ox, oy, oz, var_idx)], np.float32)
        R0        = np.array([R0_scalar], np.float32)

        print(f"\n  truth={tm:02d}  yo={yo[0]:.2f} dBZ")

        # LETKF baseline
        res = letkf_update(xf, yo, R0, ox_arr, oy_arr, oz_arr, loc_scales, var_idx)
        _save(outdir, f"WS1_LETKF_Nt01_True{tm:02d}",
              meta={"method":"LETKF","ntemp":1,"alpha_s":alpha_s,"truth_member":tm},
              **res, yo=yo, xf=xf, truth=truth)

        # TEnKF sweep
        for ntemp in ntemp_list:
            if ntemp == 1:
                continue   # already saved as LETKF
            res = tenkf_update(xf, yo, R0, ox_arr, oy_arr, oz_arr,
                               loc_scales, var_idx, ntemp=ntemp, alpha_s=alpha_s)
            _save(outdir, f"WS1_TEnKF_Nt{ntemp:02d}_True{tm:02d}",
                  meta={"method":"TEnKF","ntemp":ntemp,"alpha_s":alpha_s,"truth_member":tm},
                  **res, yo=yo, xf=xf, truth=truth)

    print("\n[WS-1 complete]")


# ── WS-2 ───────────────────────────────────────────────────────────────────

def run_ws2(cfg):
    print("\n" + "="*60)
    print("WS-2: Obs position comparison  (A / B / C)")
    print("="*60)

    st     = cfg["state"]
    da     = cfg["da"]
    paths  = cfg["paths"]
    obs    = cfg["obs"]

    data       = np.load(paths["prepared"])
    ens        = data["state_ensemble"] if "state_ensemble" in data else data["cross_sections"]
    var_idx    = st["var_idx"]
    alpha_s    = float(da["alpha_s"])
    ntemp      = int(da["ntemp"])
    R0_scalar  = float(obs["sigma_dbz"]) ** 2
    loc_scales = np.array(da["loc_scales"], np.float32)
    outdir     = paths["outdir"]

    # Three fixed positions from config
    positions  = cfg["obs_positions"]   # {A_near_mean: {x,y,z}, B_above_mean: ..., C_below_mean: ...}
    n_members  = ens.shape[3]

    methods = {
        "LETKF":  lambda xf, yo, R0, ox, oy, oz:
                    letkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx),
        "TEnKF":  lambda xf, yo, R0, ox, oy, oz:
                    tenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                                 ntemp=ntemp, alpha_s=alpha_s),
        "AOEI":   lambda xf, yo, R0, ox, oy, oz:
                    aoei_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx),
        "ATEnKF": lambda xf, yo, R0, ox, oy, oz:
                    atenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                                  ntemp=ntemp, alpha_s=alpha_s),
    }

    for tm in range(n_members):
        truth, xf = _load_member(ens, tm)
        print(f"\n  truth={tm:02d}")

        for pos_name, loc in positions.items():
            ox = int(loc["x"]); oy = int(loc["y"]); oz = int(loc["z"])
            ox_arr = np.array([ox], np.int32)
            oy_arr = np.array([oy], np.int32)
            oz_arr = np.array([oz], np.int32)
            yo     = np.array([_truth_yo(truth, ox, oy, oz, var_idx)], np.float32)
            R0     = np.array([R0_scalar], np.float32)

            for method_name, fn in methods.items():
                tag = f"WS2_{method_name}_{pos_name}_True{tm:02d}"
                # skip if already done
                if os.path.exists(os.path.join(outdir, tag + ".npz")):
                    print(f"    [skip] {tag}")
                    continue
                print(f"    {method_name}  pos={pos_name}  yo={yo[0]:.2f}")
                res = fn(xf, yo, R0, ox_arr, oy_arr, oz_arr)
                _save(outdir, tag,
                      meta={"method":method_name, "position":pos_name,
                            "obs_loc":[ox,oy,oz], "yo":float(yo[0]),
                            "ntemp":ntemp, "alpha_s":alpha_s, "truth_member":tm},
                      **res, yo=yo, xf=xf, truth=truth)

    print("\n[WS-2 complete]")


# ── dispatch ───────────────────────────────────────────────────────────────

RUNNERS = {"WS-1": run_ws1, "WS-2": run_ws2}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    exp = cfg.get("experiment", "WS-1")
    if exp not in RUNNERS:
        raise ValueError(f"Unknown experiment '{exp}'. Choose from {list(RUNNERS)}")
    RUNNERS[exp](cfg)

if __name__ == "__main__":
    main()
