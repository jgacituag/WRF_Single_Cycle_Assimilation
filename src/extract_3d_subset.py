"""
src/extract_3d_subset.py
========================
Extract a 3D WRF ensemble subset and save it as a compressed .npz file.

usage:

    python src/extract_3d_subset.py --config configs/build_3D_section.yaml

or imported in a notebook / script:

    from extract_3d_subset import process_data
    process_data("configs/build_3D_section.yaml")

Output array layout
-------------------
state_ensemble : (nx, ny, nz, Ne, 8)  float32
    Variable index mapping (last axis):
      0 – QGRAUP  [kg/kg]
      1 – QRAIN   [kg/kg]
      2 – QSNOW   [kg/kg]
      3 – T       [K]
      4 – P       [Pa]
      5 – UA      [m/s]
      6 – VA      [m/s]
      7 – WA      [m/s]
lats       : (ny, nx)      latitude  [°]
lons       : (ny, nx)      longitude [°]
z_heights  : (nz, ny, nx)  height above sea level [m]

YAML config schema
------------------
cross_sections_job:
  paths:
    pattern:    "/path/{member}/wrfout_d01_{date}"   # {member} and {date} are substituted
    output:     "/path/to/output/subset_{date}.npz"
    init_date:  "2023-12-16_19:00:00"
    end_date:   "2023-12-16_19:00:00"
    freq:       "1H"
  ensemble:
    mem_ini: 1
    mem_end: 30
    pad:     3     # zero-padding width for member number string
  subset_3d:
    timeidx: -1    # WRF time index (-1 = last)
    k_start: ~     # vertical level start (null = 0)
    k_end:   ~     # vertical level end   (null = top)
    j_start: ~     # south-north start
    j_end:   ~     # south-north end
    i_start: ~     # west-east start
    i_end:   ~     # west-east end
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset
from tqdm import tqdm


############## Helper functions ##############

def _expand_members(mem_ini: int, mem_end: int, pad: int) -> List[str]:
    """Return zero-padded member strings from mem_ini to mem_end inclusive."""
    return [str(i).zfill(pad) for i in range(mem_ini, mem_end + 1)]
 
 
def _resolve_paths(cfg: dict, dt) -> Tuple[List, List, str]:
    """
    Resolve file paths for a single date. Returns (members, nc_paths, out_path).
 
    Tokens available in pattern and output:
      {member}  — zero-padded member string
      {date}    — valid time, formatted with paths.date_fmt
      {init}    — init time string, taken literally from paths.init
    """
    p   = cfg["cross_sections_job"]["paths"]
    ens = cfg["cross_sections_job"]["ensemble"]
 
    pattern = p.get("pattern") or p.get("template")
    if pattern is None:
        raise ValueError("cross_sections_job.paths.pattern is required.")
 
    date_fmt = p.get("date_fmt", "%Y-%m-%d_%H:%M:%S")
    date_str = dt.strftime(date_fmt)
    init_str = p.get("start", "")
 
    members  = _expand_members(ens["mem_ini"], ens["mem_end"], ens.get("pad", 0))
    nc_paths = [pattern.format(member=m, date=date_str, init=init_str)
                for m in members]
    out_path = p["output"].format(date=date_str, init=init_str)
    return members, nc_paths, out_path
 
 
def _slices_from_cfg(sub_cfg: dict):
    """Build k, j, i slices from the subset_3d config block."""
    return (
        slice(sub_cfg.get("k_start"), sub_cfg.get("k_end")),
        slice(sub_cfg.get("j_start"), sub_cfg.get("j_end")),
        slice(sub_cfg.get("i_start"), sub_cfg.get("i_end")),
    )

def _nearest_ij(xlat: np.ndarray, xlong: np.ndarray,
                lat: float, lon: float) -> Tuple[int, int]:
    """
    Return (j, i) indices of the grid point nearest to (lat, lon).
    xlat, xlong: 2-D arrays of shape (ny, nx).
    """
    dist2 = (xlat - lat) ** 2 + (xlong - lon) ** 2
    j, i  = np.unravel_index(dist2.argmin(), dist2.shape)
    return int(j), int(i)

# ### wrfout format #######################################################################################
 
def _get_vars_wrfout(nc: Dataset, timeidx: int) -> dict:
    """
    Read variables from a native WRF output file using wrf-python.
    Returns arrays in WRF layout (nz, ny, nx), units ready for the Fortran DA.
    """
    import wrf
    return {
        "QGRAUP":   wrf.to_np(wrf.getvar(nc, "QGRAUP", timeidx=timeidx)),  # kg/kg
        "QRAIN":    wrf.to_np(wrf.getvar(nc, "QRAIN",  timeidx=timeidx)),  # kg/kg
        "QSNOW":    wrf.to_np(wrf.getvar(nc, "QSNOW",  timeidx=timeidx)),  # kg/kg
        "tk":       wrf.to_np(wrf.getvar(nc, "temp",   timeidx=timeidx)),  # K
        "pressure": wrf.to_np(wrf.getvar(nc, "pres",   timeidx=timeidx)),  # Pa
        "ua":       wrf.to_np(wrf.getvar(nc, "ua",     timeidx=timeidx)),  # m/s
        "va":       wrf.to_np(wrf.getvar(nc, "va",     timeidx=timeidx)),  # m/s
        "wa":       wrf.to_np(wrf.getvar(nc, "wa",     timeidx=timeidx)),  # m/s
        "z":        wrf.to_np(wrf.getvar(nc, "z",      timeidx=timeidx)),  # m
    }
 
 
def _probe_wrfout(nc_path: str, sub_cfg: dict) -> Tuple:
    """
    Probe dimensions and static fields from the first wrfout member.
    Returns (nz, ny, nx, lats_sub, lons_sub, z_heights_sub).
    """
    import wrf
    timeidx  = sub_cfg.get("timeidx", -1)
    k_slice, j_slice, i_slice = _slices_from_cfg(sub_cfg)
    with Dataset(nc_path) as nc:
        v    = _get_vars_wrfout(nc, timeidx)
        lat  = wrf.to_np(wrf.getvar(nc, "lat", timeidx=timeidx))
        lon  = wrf.to_np(wrf.getvar(nc, "lon", timeidx=timeidx))
        samp = v["tk"][k_slice, j_slice, i_slice]
        nz, ny, nx = samp.shape
        lats_sub   = lat[j_slice, i_slice]
        lons_sub   = lon[j_slice, i_slice]
        z_sub      = v["z"][k_slice, j_slice, i_slice]
    return nz, ny, nx, lats_sub, lons_sub, z_sub
 
 
def _fill_member_wrfout(nc_path: str, sub_cfg: dict,
                        out: np.ndarray, j: int) -> None:
    """Fill member j in out array from a wrfout file."""
    timeidx  = sub_cfg.get("timeidx", -1)
    k_slice, j_slice, i_slice = _slices_from_cfg(sub_cfg)
    with Dataset(nc_path) as nc:
        v = _get_vars_wrfout(nc, timeidx)
        # WRF layout (nz,ny,nx) -> transpose to (nx,ny,nz)
        out[:, :, :, j, 0] = v["QGRAUP"][k_slice, j_slice, i_slice].T
        out[:, :, :, j, 1] = v["QRAIN" ][k_slice, j_slice, i_slice].T
        out[:, :, :, j, 2] = v["QSNOW" ][k_slice, j_slice, i_slice].T
        out[:, :, :, j, 3] = v["tk"    ][k_slice, j_slice, i_slice].T
        out[:, :, :, j, 4] = v["pressure"][k_slice, j_slice, i_slice].T
        out[:, :, :, j, 5] = v["ua"    ][k_slice, j_slice, i_slice].T
        out[:, :, :, j, 6] = v["va"    ][k_slice, j_slice, i_slice].T
        out[:, :, :, j, 7] = v["wa"    ][k_slice, j_slice, i_slice].T
 
 
# ### post format ##########################################################################################
 
def _get_vars_post(nc: Dataset, k_slice, j_slice, i_slice) -> dict:
    """
    Read variables from a postprocessed CF-convention file.
 
    Unit conversions applied here so output matches wrfout conventions:
      QGRAUP / QRAIN / QSNOW : g/kg  -> kg/kg  (÷ 1000)
      PRESSURE               : hPa   -> Pa      (× 100)
      T, Umet, Vmet, W       : already K, m/s
      level_z                : already m
 
    Layout of postprocessed arrays: (XTIME, level_z, y, x)
    We take time index 0 and apply spatial slices -> (nz, ny, nx).
    """
    tidx = 0   # postprocessed files have exactly one time step
 
    def _get(varname, scale=1.0):
        arr = nc.variables[varname][tidx, k_slice, j_slice, i_slice]
        arr = np.ma.filled(arr, fill_value=np.nan).astype(np.float32)
        if scale != 1.0:
            arr = arr * scale
        return arr
 
    return {
        "QGRAUP":   _get("QGRAUP",   1e-3),   # g/kg -> kg/kg
        "QRAIN":    _get("QRAIN",    1e-3),
        "QSNOW":    _get("QSNOW",    1e-3),
        "tk":       _get("T"),                 # K
        "pressure": _get("PRESSURE", 100.0),  # hPa -> Pa
        "ua":       _get("Umet"),              # m/s
        "va":       _get("Vmet"),              # m/s
        "wa":       _get("W"),                 # m/s
    }
 
 
def _probe_post(nc_path: str, sub_cfg: dict) -> Tuple:
    """
    Probe dimensions and static fields from the first postprocessed member.
    Returns (nz, ny, nx, lats_sub, lons_sub, z_heights_sub).
    z_heights_sub is (nz, ny, nx) — level_z broadcast over the spatial subset.
    """
    k_slice, j_slice, i_slice = _slices_from_cfg(sub_cfg)
    with Dataset(nc_path) as nc:
        xlat  = nc.variables["XLAT"][j_slice, i_slice]   # (ny_sub, nx_sub)
        xlong = nc.variables["XLONG"][j_slice, i_slice]
        lev   = nc.variables["level_z"][k_slice]          # (nz_sub,)
 
        # probe shape from any 4-D variable
        samp     = nc.variables["T"][0, k_slice, j_slice, i_slice]
        nz, ny, nx = samp.shape
 
        # broadcast level_z to (nz, ny, nx) — height is the same everywhere
        z_sub = np.broadcast_to(
            lev[:, np.newaxis, np.newaxis],
            (nz, ny, nx)
        ).astype(np.float32).copy()
 
    return nz, ny, nx, xlat.astype(np.float32), xlong.astype(np.float32), z_sub
 
 
def _fill_member_post(nc_path: str, sub_cfg: dict,
                      out: np.ndarray, j: int) -> None:
    """Fill member j in out array from a postprocessed file."""
    k_slice, j_slice, i_slice = _slices_from_cfg(sub_cfg)
    with Dataset(nc_path) as nc:
        v = _get_vars_post(nc, k_slice, j_slice, i_slice)
        # layout (nz,ny,nx) -> transpose to (nx,ny,nz)
        out[:, :, :, j, 0] = v["QGRAUP"].T
        out[:, :, :, j, 1] = v["QRAIN" ].T
        out[:, :, :, j, 2] = v["QSNOW" ].T
        out[:, :, :, j, 3] = v["tk"    ].T
        out[:, :, :, j, 4] = v["pressure"].T
        out[:, :, :, j, 5] = v["ua"    ].T
        out[:, :, :, j, 6] = v["va"    ].T
        out[:, :, :, j, 7] = v["wa"    ].T

def ll_to_ij_post(nc_path: str, lat: float, lon: float) -> Tuple[int, int]:
    """
    Find nearest (j, i) grid indices for a given lat/lon in a postprocessed
    file.  Equivalent to wrf.ll_to_xy for wrfout files.
 
    Parameters
    ----------
    nc_path : path to any postprocessed member file
    lat, lon : target coordinates in degrees
 
    Returns
    -------
    (j, i) : 0-based grid indices  (j = south-north, i = west-east)
    """
    with Dataset(nc_path) as nc:
        xlat  = nc.variables["XLAT"][:]
        xlong = nc.variables["XLONG"][:]
    return _nearest_ij(xlat, xlong, lat, lon)

def process_data(config_path: str) -> None:
    """
    Extract 3D WRF ensemble subsets for all dates in the config and save
    each as a compressed .npz file.
 
    Parameters
    ----------
    config_path : str  path to the YAML configuration file
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
 
    fmt       = cfg["cross_sections_job"].get("format", "wrfout").lower()
    paths_cfg = cfg["cross_sections_job"]["paths"]
    date_ini  = paths_cfg.get("init_date")
    date_end  = paths_cfg.get("end_date")
    freq      = paths_cfg.get("freq", "1H")
    sub_cfg   = cfg["cross_sections_job"]["subset_3d"]
 
    if fmt not in ("wrfout", "post"):
        raise ValueError(f"format must be 'wrfout' or 'post', got '{fmt}'")
    if date_ini is None or date_end is None:
        raise ValueError("init_date and end_date must be set in the YAML config.")
 
    dates = pd.date_range(
        start=pd.to_datetime(date_ini, format="%Y-%m-%d_%H:%M:%S"),
        end=pd.to_datetime(date_end,   format="%Y-%m-%d_%H:%M:%S"),
        freq=freq,
    )
    print(f"[info] format={fmt}  {len(dates)} date(s)  ({date_ini} → {date_end})")
 
    for dt in dates:
        date_str = dt.strftime(cfg["cross_sections_job"]["paths"].get(
                               "date_fmt", "%Y-%m-%d_%H:%M:%S"))
        print(f"\n--- {date_str} ---")
 
        members, nc_paths, out_path = _resolve_paths(cfg, dt)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
 
        # ### probe first member ############################################################
        print(f"[info] probing from {nc_paths[0]}")
        if fmt == "wrfout":
            nz, ny, nx, lats_sub, lons_sub, z_sub = _probe_wrfout(nc_paths[0], sub_cfg)
        else:
            nz, ny, nx, lats_sub, lons_sub, z_sub = _probe_post(nc_paths[0], sub_cfg)
 
        Ne   = len(nc_paths)
        nvar = 8
        out  = np.zeros((nx, ny, nz, Ne, nvar), dtype=np.float32)
        print(f"[info] output shape: (nx={nx}, ny={ny}, nz={nz}, Ne={Ne}, nvar={nvar})")
        print("[info] variable order: [QGRAUP, QRAIN, QSNOW, T, P, UA, VA, WA]")
 
        # ### fill ensemble array ##########################################################
        for j, path in enumerate(tqdm(nc_paths, desc="members")):
            if not os.path.isfile(path):
                print(f"[warning] missing: {path}")
                out[:, :, :, j, :] = np.nan
                continue
            if fmt == "wrfout":
                _fill_member_wrfout(path, sub_cfg, out, j)
            else:
                _fill_member_post(path, sub_cfg, out, j)
 
        # ### drop all-NaN vertical levels ###########################################
        finite_z = np.isfinite(out).any(axis=(0, 1, 3, 4))
        n_dropped = int((~finite_z).sum())
        if n_dropped:
            print(f"[clean] dropping {n_dropped} all-NaN z-level(s) "
                  f"— consider adjusting k_start in the config.")
            out   = out[:, :, finite_z, :, :]
            z_sub = z_sub[finite_z, :, :]
 
        np.savez_compressed(
            out_path,
            state_ensemble=out,
            lats=lats_sub,
            lons=lons_sub,
            z_heights=z_sub,
        )
        print(f"[done] {out_path}  shape={out.shape}")
 
    print("\n[info] all dates processed.")
 
 
# ### CLI ######################################################################################################
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract a 3D WRF ensemble subset to .npz")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    process_data(args.config)
 