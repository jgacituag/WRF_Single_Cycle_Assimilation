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
      0 - QGRAUP  [kg/kg]
      1 - QRAIN   [kg/kg]
      2 - QSNOW   [kg/kg]
      3 - T       [K]
      4 - P       [Pa]
      5 - UA      [m/s]
      6 - VA      [m/s]
      7 - WA      [m/s]
lats       : (ny, nx)      latitude  [deg]
lons       : (ny, nx)      longitude [deg]
z_heights  : (nz, ny, nx)  height above sea level [m]
valid_mask : (nx, ny, nz)  bool
    False where at least one member carried a non-finite value in at least one
    variable. Those cells are NaN in EVERY member and variable in state_ensemble, so
    the ensemble size is uniform across the domain -- see VALID_MASK_NOTE, stored
    alongside as `valid_mask_note`.
members_read : (Ne,) bool   which member files were found (a missing one is all-NaN
    and is excluded from the union mask)
pos_km     : (nx, ny, nz, 3)  real-space position from the lower-left corner [km]
    pos_km[i, j, k, 0] = x  east  distance from corner (i=0, j=0) [km]
    pos_km[i, j, k, 1] = y  north distance from corner (i=0, j=0) [km]
    pos_km[i, j, k, 2] = z  height above sea level [km]
    x and y are computed using the equirectangular approximation from each
    point's actual lat/lon to the origin, giving proper x/y components that
    handle grid rotation. Valid for domains up to ~1000 km.

YAML config schema
------------------
cross_sections_job:
  # ---- dataset identity, written into every subset ----
  # dataset_id is REQUIRED and is checked against the output path. It is never parsed
  # back out of the filename, so a mistyped path fails instead of silently relabelling
  # the data. The output path must end in
  #     3D_subsets_{A|B|C|D}/subset_{same letter}_{YYYYMMDDHHMMSS}.npz
  dataset_id:   D          # A | B | C | D
  physics:      single     # multi | single
  da_cycle_min: 5          # upstream DA cycle length [minutes]
  upstream:     GUES       # which POST tree the prior came from
  # source_run is NOT a key: it is read from the pattern above (the directory above
  # POST), so it can never disagree with the files actually read. Override only if the
  # pattern does not follow that layout.
  # config_index: [...]    # 60 entries, the physics configuration each member used.
  #   Omit when it is not recorded: the subset then carries an all -1 array plus a note
  #   saying so. Never invent it -- -1 must always mean "not recorded".
  # dx_km is NOT configured; it is measured from pos_km at extraction time.
  paths:
    pattern:    "/path/{member}/wrfout_d01_{date}"   # {member} and {date} are substituted
    output:     "/path/to/data/3D_subsets_D/subset_D_{date}.npz"
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
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naming import TagError, validate_subset_path


# The note stored beside `valid_mask`, so nobody has to rediscover why a handful of
# cells are NaN in every member.
VALID_MASK_NOTE = (
    "False where AT LEAST ONE member carried a non-finite value in AT LEAST ONE "
    "variable. Those cells are set to NaN in every member and every variable, so the "
    "ensemble size is uniform across the domain. Diagnosed in N2 12: in dataset A the "
    "bad cells are 41-56 per hour, identical across all eight variables, never a whole "
    "column and never at a domain edge; 32 of 60 members carry 1-4 cells each, and "
    "those cells are masked in that member's own source netCDF, which "
    "np.ma.filled(..., np.nan) in _get_vars_post propagates one cell at a time. A cell "
    "valid in 28 members and not in the other 32 gives that column a different "
    "ensemble size than its neighbours, which biases every shape statistic and every "
    "covariance computed there. The union costs ~50 cells out of 1.5 M and removes the "
    "problem instead of leaving it to be managed downstream."
)


# The note stored beside an all -1 config_index, so a reader is never left guessing
# whether -1 is a configuration number.
CONFIG_INDEX_UNKNOWN = (
    "not recorded: no member-to-physics mapping was available at extraction time. "
    "-1 means 'not recorded', never 'configuration -1'."
)


def _source_run(cfg_job: dict) -> str:
    """The upstream WRF experiment a subset was built from.

    Taken from the input pattern -- the directory above POST -- rather than asked for
    again as a config key, so it cannot disagree with the files actually read. This is
    the one piece of provenance the chapter's "how each dataset was built" paragraph
    needs and that nothing on disk carried before.
    """
    explicit = cfg_job.get("source_run")
    if explicit:
        return str(explicit)
    pattern = (cfg_job.get("paths") or {}).get("pattern", "")
    parts = [q for q in pattern.split("/") if q]
    if "POST" in parts:
        i = parts.index("POST")
        if i > 0:
            return parts[i - 1]
    return "unknown"


def _union_mask(out: np.ndarray, members_read: np.ndarray) -> Tuple[np.ndarray, int]:
    """The union of every member's non-finite cells, and how many there are.

    Accumulated one member and one variable at a time: `np.isfinite(out).all(...)` on
    the whole array would allocate a boolean the size of the ensemble -- 2.7 GB on
    dataset C -- for a result that is (nx, ny, nz).

    Members whose file was missing entirely are excluded. They are all-NaN by
    construction, so including one would mask the whole domain.
    """
    valid = np.ones(out.shape[:3], bool)
    for j in np.flatnonzero(members_read):
        for v in range(out.shape[4]):
            valid &= np.isfinite(out[:, :, :, j, v])
    return valid, int((~valid).sum())


def _identity(cfg_job: dict, dataset_id: str, n_members: int, pos_km: np.ndarray) -> dict:
    """The identity block written into every subset.

    dataset_id comes from an explicit config key and is checked against the output
    path -- it is never parsed back out of the filename, so a mistyped path fails
    instead of silently relabelling the data.

    dx_km is MEASURED from pos_km rather than configured: it is a property of the grid
    that was just read, and a configured value could disagree with it.
    """
    ci = cfg_job.get("config_index")
    if ci is None:
        config_index = np.full(n_members, -1, np.int16)
        note = CONFIG_INDEX_UNKNOWN
    else:
        config_index = np.asarray(ci, np.int16)
        if config_index.shape != (n_members,):
            raise ValueError(
                f"cross_sections_job.config_index has {config_index.shape[0]} entries "
                f"but the ensemble has {n_members} members")
        note = "recorded at extraction time from cross_sections_job.config_index"

    physics = cfg_job.get("physics")
    if physics not in ("multi", "single"):
        raise ValueError(f"cross_sections_job.physics must be 'multi' or 'single', "
                         f"got {physics!r}")
    if cfg_job.get("da_cycle_min") is None:
        raise ValueError("cross_sections_job.da_cycle_min is required [minutes]")

    return dict(
        dataset_id=np.array(dataset_id),
        da_cycle_min=np.int16(cfg_job["da_cycle_min"]),
        dx_km=np.float32(np.median(np.diff(pos_km[:, 0, 0, 0]))),
        physics=np.array(physics),
        upstream=np.array(cfg_job.get("upstream", "unknown")),
        source_run=np.array(_source_run(cfg_job)),
        config_index=config_index,
        config_index_note=np.array(note),
    )


# ---------------------------------------------------------------------------
# Grid position helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray):
    """
    Equirectangular approximation of east-west and north-south distances [km].
    All inputs in degrees. Vectorised -- works on scalars or any shape.

    Returns
    -------
    dx : east-west distance  [km]  positive = east
    dy : north-south distance [km] positive = north
    """
    R        = 6371.0
    phi1     = np.radians(lat1)
    phi2     = np.radians(lat2)
    dphi     = phi2 - phi1
    dlambda  = np.radians(lon2 - lon1)
    phi_mean = (phi1 + phi2) / 2.0
    dx = R * dlambda * np.cos(phi_mean)
    dy = R * dphi
    return dx, dy


def _compute_pos_km(lats: np.ndarray, lons: np.ndarray,
                    z_heights: np.ndarray) -> np.ndarray:
    """
    Compute real-space grid-point positions in km from the lower-left corner.

    Parameters
    ----------
    lats      : (ny, nx)      latitude  [degrees]
    lons      : (ny, nx)      longitude [degrees]
    z_heights : (nz, ny, nx)  height above sea level [m]

    Returns
    -------
    pos_km : (nx, ny, nz, 3)  float32
        pos_km[i, j, k, 0] = x  east  distance from corner (i=0, j=0) [km]
        pos_km[i, j, k, 1] = y  north distance from corner (i=0, j=0) [km]
        pos_km[i, j, k, 2] = z  height above sea level [km]

    Method
    ------
    For every grid point (i,j), _haversine_km returns (dx, dy) directly
    as the east-west and north-south components from the origin to that
    point. Single vectorized call over the full (ny, nx) grid -- no loops.
    Handles grid rotation correctly since every point uses its own lat/lon.
    """
    nz_full, ny, nx = z_heights.shape

    lat0 = float(lats[0, 0])
    lon0 = float(lons[0, 0])

    # Single vectorized call -- x_2d, y_2d both (ny, nx)
    x_2d, y_2d = _haversine_km(lat0, lon0,
                                lats.astype(np.float64),
                                lons.astype(np.float64))

    # z: (nz, ny, nx) metres -> (nx, ny, nz) km via transpose
    z_km = z_heights.transpose(2, 1, 0).astype(np.float32) / 1000.0

    # x_2d, y_2d are (ny, nx) -- transpose to (nx, ny) then broadcast over nz
    pos_km = np.empty((nx, ny, nz_full, 3), dtype=np.float32)
    pos_km[:, :, :, 0] = x_2d.T[:, :, np.newaxis]
    pos_km[:, :, :, 1] = y_2d.T[:, :, np.newaxis]
    pos_km[:, :, :, 2] = z_km

    return pos_km


############## Helper functions ##############

def _expand_members(mem_ini: int, mem_end: int, pad: int) -> List[str]:
    """Return zero-padded member strings from mem_ini to mem_end inclusive."""
    return [str(i).zfill(pad) for i in range(mem_ini, mem_end + 1)]
 
 
def _resolve_paths(cfg: dict, dt) -> Tuple[List, List, str]:
    """
    Resolve file paths for a single date. Returns (members, nc_paths, out_path).
 
    Tokens available in pattern and output:
      {member}  -- zero-padded member string
      {date}    -- valid time, formatted with paths.date_fmt
      {init}    -- init time string, taken literally from paths.init
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

##### wrfout format #######################################################################################
 
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
 
 
##### post format ##########################################################################################
 
def _get_vars_post(nc: Dataset, k_slice, j_slice, i_slice) -> dict:
    """
    Read variables from a postprocessed CF-convention file.
 
    Unit conversions applied here so output matches wrfout conventions:
      QGRAUP / QRAIN / QSNOW : g/kg  -> kg/kg  (/ 1000)
      PRESSURE               : hPa   -> Pa      (x 100)
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
    z_heights_sub is (nz, ny, nx) -- level_z broadcast over the spatial subset.
    """
    k_slice, j_slice, i_slice = _slices_from_cfg(sub_cfg)
    with Dataset(nc_path) as nc:
        xlat  = nc.variables["XLAT"][j_slice, i_slice]   # (ny_sub, nx_sub)
        xlong = nc.variables["XLONG"][j_slice, i_slice]
        lev   = nc.variables["level_z"][k_slice]          # (nz_sub,)
 
        # probe shape from any 4-D variable
        samp     = nc.variables["T"][0, k_slice, j_slice, i_slice]
        nz, ny, nx = samp.shape
 
        # broadcast level_z to (nz, ny, nx) -- height is the same everywhere
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
    print(f"[info] format={fmt}  {len(dates)} date(s)  ({date_ini} -> {date_end})")
 
    # Validate every output path BEFORE reading a single member: extracting 60 members
    # takes long enough that discovering a malformed path at write time is a wasted job.
    job = cfg["cross_sections_job"]
    dataset_id = job.get("dataset_id")
    for dt in dates:
        _, _, probe_out = _resolve_paths(cfg, dt)
        try:
            validate_subset_path(probe_out, dataset_id)
        except TagError as e:
            raise SystemExit(f"[{os.path.basename(config_path)}] {e}")

    for dt in dates:
        date_str = dt.strftime(cfg["cross_sections_job"]["paths"].get(
                               "date_fmt", "%Y-%m-%d_%H:%M:%S"))
        print(f"\n--- {date_str} ---")

        members, nc_paths, out_path = _resolve_paths(cfg, dt)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
 
        ##### probe first member ############################################################
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

        # Compute pos_km once from the probed geometry
        pos_km = _compute_pos_km(lats_sub, lons_sub, z_sub)
        print(f"[info] pos_km shape: {pos_km.shape}  "
              f"x=[{pos_km[:,:,:,0].min():.1f}, {pos_km[:,:,:,0].max():.1f}] km  "
              f"y=[{pos_km[:,:,:,1].min():.1f}, {pos_km[:,:,:,1].max():.1f}] km  "
              f"z=[{pos_km[:,:,:,2].min():.2f}, {pos_km[:,:,:,2].max():.2f}] km")
 
        ##### fill ensemble array ##########################################################
        members_read = np.ones(Ne, bool)
        for j, path in enumerate(tqdm(nc_paths, desc="members")):
            if not os.path.isfile(path):
                print(f"[warning] missing: {path}")
                out[:, :, :, j, :] = np.nan
                members_read[j] = False
                continue
            if fmt == "wrfout":
                _fill_member_wrfout(path, sub_cfg, out, j)
            else:
                _fill_member_post(path, sub_cfg, out, j)
 
        ##### drop all-NaN vertical levels ###########################################
        finite_z = np.isfinite(out).any(axis=(0, 1, 3, 4))
        n_dropped = int((~finite_z).sum())
        if n_dropped:
            print(f"[clean] dropping {n_dropped} all-NaN z-level(s) "
                  f"-- consider adjusting k_start in the config.")
            out   = out[:, :, finite_z, :, :]
            z_sub = z_sub[finite_z, :, :]
 
        ##### mask the union of the members' bad cells ##############################
        # Fixed here rather than managed downstream. A cell valid in 28 members and not
        # in the other 32 gives that column a different ensemble size than its
        # neighbours; masking the union makes the ensemble size uniform, and the mask is
        # written into the subset so downstream code can assert on it instead of
        # rediscovering it. Applied to A, B, C and D alike.
        valid_mask, n_masked = _union_mask(out, members_read)
        if n_masked:
            out[~valid_mask] = np.nan
            frac = 100.0 * n_masked / valid_mask.size
            print(f"[clean] masked {n_masked} cell(s) ({frac:.4f} % of "
                  f"{valid_mask.size}) in every member: non-finite in at least one "
                  f"member's source file.")
        if not members_read.all():
            print(f"[clean] {int((~members_read).sum())} member(s) were missing and are "
                  f"all-NaN; they are excluded from the union mask.")

        ident = _identity(job, dataset_id, out.shape[3], pos_km)
        np.savez_compressed(
            out_path,
            state_ensemble=out,
            lats=lats_sub,
            lons=lons_sub,
            z_heights=z_sub,
            pos_km=pos_km,
            valid_mask=valid_mask,
            n_masked_cells=np.int64(n_masked),
            valid_mask_note=np.array(VALID_MASK_NOTE),
            members_read=members_read,
            **ident,
        )
        print(f"[done] {out_path}  shape={out.shape}")
        print(f"[id]   dataset_id={dataset_id}  da_cycle_min={int(ident['da_cycle_min'])}"
              f"  dx_km={float(ident['dx_km']):.4f}  physics={str(ident['physics'])}"
              f"  upstream={str(ident['upstream'])}  source_run={str(ident['source_run'])}"
              f"  config_index={'recorded' if (ident['config_index'] >= 0).any() else 'all -1 (not recorded)'}")
 
    print("\n[info] all dates processed.")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract a 3D WRF ensemble subset to .npz")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    process_data(args.config)