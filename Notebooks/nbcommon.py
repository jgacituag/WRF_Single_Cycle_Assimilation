"""
Notebooks/nbcommon.py
=====================
Shared layer for the analysis notebooks N1-N4.

The notebooks are meant to be configuration + narrative; everything mechanical lives
here. Most importantly, this module owns the *sign convention*: there is exactly one
function in the repo that subtracts an analysis from a prior, and it always subtracts
in the same order. See CONVENTION below and assert_convention(), which runs at import.

Usage in a notebook:

    import nbcommon as nb
    nb.banner()

Sections
    0  convention, banner, schema guard
    1  paths and run configuration
    2  figure style and output
    3  palettes, labels, variable tables
    4  physics (reflectivity forward operator)
    5  prior-ensemble access          (N1, N2)
    6  sweep loading and row alignment (N3, N4)
    7  THE convention: colname / raw / skill / skill_summary / publish / expect
    8  predictors                      (N3)
    9  binning and density primitives
   10  maps and 3-D projections
   11  multi-obs, guarded and optional (N4)
"""

import hashlib
import os
import pathlib
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm

__version__ = "1.0"

# ═════════════════════════════════════════════════════════════════════════════
# §0  Convention, banner, schema guard
# ═════════════════════════════════════════════════════════════════════════════

CONVENTION = """\
SIGN CONVENTION -- one rule, enforced by this module.

  Every quantity named `skill` is  prior - analysis.
  POSITIVE MEANS THE ANALYSIS IS BETTER THAN THE PRIOR.

  There is no other difference in this module. nbcommon exports no `drmse`, no
  `delta`, no `d_*`. skill() is the only function that subtracts a prior from an
  analysis, and it always subtracts in that order.

  `spread` is NOT a skill metric -- lower spread is not better -- and
  skill(metric='spread') raises.

  `bias`, `skew` and `kurt` are compared as magnitudes: skill = |prior| - |analysis|.
  (skew/kurt at the _w_/_u_ reductions are already stored as means of |.|, so the
  magnitude is not taken twice; see da/metrics.py.)

  Colour: CMAP_SKILL is blue at positive (better), red at negative (worse), always
  centred on 0. It is used with skill() output and nothing else.

  Column names are NATIVE and the variable suffix is LAST:
      crps_f_w_w   = loc-weighted CRPS of vertical wind
      rmse_f_u_u   = unweighted   RMSE of zonal wind
  Never build a column name by string surgery -- call colname()."""

# Sweep npz column count after the reflectivity-metrics change (315 -> 362).
# load_sweep() warns when a file predates it.
SCHEMA_NCOLS = 362


def banner(check_fortran=True):
    """Print the convention and the environment into the notebook's own output."""
    print(CONVENTION)
    print("-" * 78)
    print(f"nbcommon {__version__}   repo: {REPO}")
    print(f"figures -> {FIG_DIR}")
    print(f"derived -> {DERIVED}")
    if check_fortran:
        ok = verify_hx(n=500, quiet=True)
        print(f"reflectivity operator vs Fortran: {'OK' if ok else 'NOT CHECKED (see warning)'}")
    print("-" * 78)


# ═════════════════════════════════════════════════════════════════════════════
# §1  Paths and run configuration
# ═════════════════════════════════════════════════════════════════════════════

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIG_DIR = REPO / "Notebooks" / "figures"
DERIVED = DATA / "derived"

for _d in (FIG_DIR, DERIVED):
    _d.mkdir(parents=True, exist_ok=True)

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "fortran") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "fortran"))


def subset_path(hour, family="", date="20240319", stem="GUES_SINGLECONF"):
    """Path to a 3D subset npz. `hour` is '18'..'21'; `family` is '' or '_1HR'.

    The hour is always a parameter -- no notebook hardcodes a valid time.
    """
    stamp = f"{date}{hour}0000"
    d = DATA / f"3D_subsets_{date}_{stem}_{stamp}{family}"
    p = d / f"subset_{stamp}.npz"
    return str(p)


def run_dir(tag):
    return DATA / tag


def sweep_files(tag, tm=None):
    """Sweep npz files for a run, optionally restricted to given truth members."""
    d = run_dir(tag)
    if not d.is_dir():
        raise FileNotFoundError(f"no such run directory: {d}")
    files = sorted(str(p) for p in d.glob("*_sweep_Ne*_tm*.npz"))
    if tm is not None:
        want = {int(t) for t in np.atleast_1d(tm)}
        files = [f for f in files if int(f.rsplit("_tm", 1)[1][:2]) in want]
    return files


def run_config(tag):
    """The config yaml the runner copies into outdir. This is where R and dbz_min
    come from -- never a magic literal in a notebook."""
    import yaml
    d = run_dir(tag)
    cands = sorted(d.glob("*_config.yaml"))
    if not cands:
        raise FileNotFoundError(f"no *_config.yaml in {d}")
    with open(cands[0]) as f:
        return yaml.safe_load(f)


def obs_error_var(tag):
    return float(run_config(tag)["obs"]["obs_error_var"])


def dbz_min_of(tag):
    return float(run_config(tag).get("qc", {}).get("dbz_min", 0.0))


# ═════════════════════════════════════════════════════════════════════════════
# §2  Figure style and output
# ═════════════════════════════════════════════════════════════════════════════

MM = 1 / 25.4
COL_W = 84 * MM      # single-column width  [in]
PAGE_W = 174 * MM    # full-page width      [in]


def set_style():
    """The one and only rcParams block (QJRMS/Wiley)."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.size": 9,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.30,
        "pdf.fonttype": 42,   # embed TrueType, journal-safe
        "ps.fonttype": 42,
    })


set_style()


def save_fig(fig, name, dpi=600):
    """Save as vector PDF + 600-dpi PNG into Notebooks/figures/."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=dpi)
    print(f"saved {name}.pdf / {name}.png")


def panel_label(ax, letter, loc="upper left", pad=0.02):
    """(a)/(b) tag placed outside the title."""
    x, ha = (pad, "left") if "left" in loc else (1 - pad, "right")
    y, va = (1 - pad, "top") if "upper" in loc else (pad, "bottom")
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha=ha, va=va,
            fontweight="bold", fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))


# ═════════════════════════════════════════════════════════════════════════════
# §3  Palettes, labels, variable tables
# ═════════════════════════════════════════════════════════════════════════════

# The one dBZ scale. Of the four colormaps in use across the old notebooks this is
# the only properly stepped radar scale, and the only one paired with a BoundaryNorm.
DBZ_LEVELS = np.arange(0, 66, 5)
DBZ_CMAP = "Spectral_r"
DBZ_NORM = BoundaryNorm(DBZ_LEVELS, plt.get_cmap(DBZ_CMAP).N)

# CVD-validated sequential + diverging maps.
C_NT2, C_NT1, C_NONE, C_PRIOR = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
_SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("div_red_blue", ["#e34948", "#f0efec", C_NT2])
# CMAP_SKILL: blue = positive = better. Used with skill() output and nothing else.
CMAP_SKILL = CMAP_DIV
for _cm in (CMAP_SEQ, CMAP_DIV):
    _cm.set_bad("#ffffff", alpha=0.0)

_LOC_PALETTE = {0.1: "#d62728", 0.5: "#ff7f0e", 1.0: "#2ca02c", 2.0: "#1f77b4",
                3.0: "#9467bd", 4.0: "#8c564b", 5.0: "#e377c2"}
NTEMP_STYLES = {1: "-", 2: "--", 3: ":", 4: "-.", 5: (0, (3, 1, 1, 1))}
NTEMP_MARKERS = {1: "o", 2: "s", 3: "^", 4: "v", 5: "D"}
METHOD_COLORS = {"TEnKF": "#2a78d6", "LETKF": "#52514e", "AOEI": "#eb6834", "prior": "black"}
METHOD_LABELS = {"TEnKF": "TEnKF", "LETKF": "LETKF", "AOEI": "AOEI", "prior": "Prior"}


def loc_color(l):
    return _LOC_PALETTE.get(round(float(l), 3), "#444444")


def loc_label(l):
    return f"L = {float(l):g} km"


# Variable tables. VI is the single source of truth for the trailing-axis order and
# is the fix for the qg/qr/qs mislabelling bug in the old notebooks -- no notebook
# ever writes a variable-list literal again.
VI = {"qg": 0, "qr": 1, "qs": 2, "T": 3, "P": 4, "u": 5, "v": 6, "w": 7}
VARS = ["qg", "qr", "qs", "T", "P", "u", "v", "w"]
ALL_VARS = VARS + ["obs"]
HYDRO = ("qg", "qr", "qs")

RAW_UNITS = dict(qg="kg/kg", qr="kg/kg", qs="kg/kg", T="K", P="Pa",
                 u="m/s", v="m/s", w="m/s", obs="dBZ")
UNITS = {"qg": "g kg$^{-1}$", "qr": "g kg$^{-1}$", "qs": "g kg$^{-1}$",
         "T": "K", "P": "Pa", "u": "m s$^{-1}$", "v": "m s$^{-1}$",
         "w": "m s$^{-1}$", "obs": "dBZ"}
# Per-variable physical tolerance: the scale below which a difference is a tie, not
# a result. Used by skill_summary to resolve ties instead of a bare `> 0`.
TOL = dict(qg=1e-7, qr=1e-7, qs=1e-7, T=1e-3, P=1e-1,
           u=1e-4, v=1e-4, w=1e-4, obs=1e-3)
VAR_LABELS = {"obs": r"$Z$", "qg": r"$q_g$", "qr": r"$q_r$", "qs": r"$q_s$",
              "T": r"$T$", "P": r"$P$", "u": r"$u$", "v": r"$v$", "w": r"$w$"}
VAR_CMAPS = {"qg": "Purples", "qr": "Blues", "qs": "Greys", "T": "YlOrRd",
             "P": "RdPu", "u": "RdBu_r", "v": "RdBu_r", "w": "RdBu_r"}


def disp(x, v):
    """Raw value -> display units (hydrometeors kg/kg -> g/kg, else identity)."""
    return np.asarray(x) * 1e3 if v in HYDRO else np.asarray(x)


def disp_unit(v):
    return UNITS[v]


def combo_label(method, ntemp):
    return f"TEnKF Nt={int(ntemp)}" if method == "TEnKF" else METHOD_LABELS.get(method, method)


def combo_color(method, ntemp):
    if method != "TEnKF":
        return METHOD_COLORS.get(method, "#444444")
    shades = ["#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
    return shades[min(int(ntemp), 5) - 1]


def combo_style(method, ntemp):
    return NTEMP_STYLES.get(int(ntemp), "-") if method == "TEnKF" else "-"


# ═════════════════════════════════════════════════════════════════════════════
# §4  Physics
# ═════════════════════════════════════════════════════════════════════════════

PI = 3.14159265358979
RD = 287.0


def calc_ref_np(qr, qs, qg, T, P, min_dbz=0.0):
    """Tong & Xue (2006) reflectivity, pure NumPy.

    Mirrors the Fortran calc_ref/calc_ref_ens so notebooks work without the
    cpython-38 extension. verify_hx() pins the two together.
    """
    qr, qs, qg, T, P = (np.asarray(a, np.float64) for a in (qr, qs, qg, T, P))
    nor, nos, nog = 8.0e6, 2.0e6, 4.0e6
    ror, ros, rog, roi = 1000.0, 100.0, 913.0, 917.0
    ki2, kr2 = 0.176, 0.930
    pip = PI ** 1.75
    cf = 1e18 * 720.0 / (pip * nor ** 0.75 * ror ** 1.75)
    cf2 = 1e18 * 720.0 * ki2 * ros ** 0.25 / (pip * kr2 * nos ** 0.75 * roi ** 2.0)
    cf3 = 1e18 * 720.0 / (pip * nos ** 0.75 * ros ** 1.75)
    cf4 = (1e18 * 720.0 / (pip * nog ** 0.75 * rog ** 1.75)) ** 0.95
    ro = P / (RD * T)
    with np.errstate(invalid="ignore"):   # base<0 in branches discarded by np.where
        zr = np.where(qr > 0.0, cf * np.maximum(ro * qr, 0.0) ** 1.75, 0.0)
        zs = np.where(qs > 0.0,
                      np.where(T <= 273.16, cf2, cf3) * np.maximum(ro * qs, 0.0) ** 1.75,
                      0.0)
        zg = np.where(qg > 0.0, cf4 * np.maximum(ro * qg, 0.0) ** 1.6625, 0.0)
    z = np.maximum(zr + zs + zg, 1.0e-10)
    return np.maximum(10.0 * np.log10(z), min_dbz)


def hx(state, min_dbz=0.0):
    """Reflectivity from a state array whose LAST axis is the 8 variables.

    (..., 8) -> (...). Indexing goes through VI, never a positional literal.
    """
    s = np.asarray(state)
    return calc_ref_np(s[..., VI["qr"]], s[..., VI["qs"]], s[..., VI["qg"]],
                       s[..., VI["T"]], s[..., VI["P"]], min_dbz=min_dbz)


def verify_hx(n=3000, tol=1e-4, quiet=False):
    """Pin calc_ref_np to the Fortran operator on n random points.

    Warns and returns False (rather than raising) when cletkf_wloc is unimportable,
    so the notebooks stay usable outside the intermediate_exp environment.
    """
    try:
        from cletkf_wloc import common_da as cda
    except Exception as e:  # pragma: no cover - environment dependent
        warnings.warn(f"cletkf_wloc unavailable ({e}); using the NumPy operator "
                      f"unverified. Activate intermediate_exp to check it.")
        return False
    rng = np.random.default_rng(0)
    qr = np.abs(rng.normal(1e-3, 1e-3, n))
    qs = np.abs(rng.normal(5e-4, 5e-4, n))
    qg = np.abs(rng.normal(5e-4, 5e-4, n))
    T = rng.uniform(240.0, 300.0, n)
    P = rng.uniform(2e4, 1.0e5, n)
    ref_f = np.array([cda.calc_ref(a, b, c, d, e, min_dbz=0.0)
                      for a, b, c, d, e in zip(qr, qs, qg, T, P)])
    ref_np = calc_ref_np(qr, qs, qg, T, P, min_dbz=0.0)
    worst = float(np.abs(ref_f - ref_np).max())
    if not quiet:
        print(f"verify_hx: max |NumPy - Fortran| = {worst:.2e} dBZ over {n} points")
    if worst > tol:
        warnings.warn(f"calc_ref_np disagrees with Fortran by {worst:.3e} dBZ")
        return False
    return True


# Diagnostics come from the production module -- never redefined here.
from da.metrics import ensemble_skew, ensemble_kurt, crps_ensemble_sorted  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# §5  Prior-ensemble access  (N1, N2)
# ═════════════════════════════════════════════════════════════════════════════

SUBSET_KEYS = ("state_ensemble", "lats", "lons", "z_heights", "pos_km")


class CorruptSubset(Exception):
    """The npz is not a readable zip (truncated write)."""


class IncompleteSubset(Exception):
    """The npz is readable but missing keys the notebooks need."""


def peek_npz(path):
    """{key: (shape, dtype)} read from the zip + npy headers only.

    Zero decompression -- safe to call on all 35 subset directories. npz is DEFLATE,
    so actually reading a key costs a full 2.9-10.6 GB decompress.
    """
    from numpy.lib import format as npformat
    out = {}
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise CorruptSubset(f"{path}: {e}") from None
    with zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                version = npformat.read_magic(f)
                try:
                    shape, _fortran, dtype = npformat._read_array_header(f, version)
                except AttributeError:      # older numpy
                    shape, _fortran, dtype = npformat.read_array_header_1_0(f)
            out[name[:-4] if name.endswith(".npy") else name] = (shape, dtype)
    return out


def check_subset(path, need=("state_ensemble", "lats", "lons", "pos_km")):
    """('PASS'|'CORRUPT'|'MISSING-KEYS', detail) for the health-sweep table."""
    try:
        keys = peek_npz(path)
    except CorruptSubset as e:
        return "CORRUPT", str(e).split(": ", 1)[-1]
    except FileNotFoundError:
        return "MISSING", "file not found"
    missing = [k for k in need if k not in keys]
    if missing:
        return "MISSING-KEYS", "missing " + ", ".join(missing)
    return "PASS", "x".join(str(s) for s in keys["state_ensemble"][0])


def load_subset(path, keys=("lats", "lons", "pos_km", "z_heights")):
    """Load geometry only. Never touches state_ensemble unless it is listed."""
    try:
        zf = zipfile.ZipFile(path)
        zf.close()
    except zipfile.BadZipFile as e:
        raise CorruptSubset(f"{path}: {e}") from None
    out = {}
    with np.load(path) as f:
        for k in keys:
            if k not in f.files:
                raise IncompleteSubset(f"{os.path.basename(path)} has no '{k}' "
                                       f"(present: {sorted(f.files)})")
            out[k] = f[k]
    return out


def bbox_to_slices(lats, lons, lat_lim, lon_lim):
    """lat/lon box -> (islice, jslice) into the (nx, ny, ...) arrays.

    lats/lons are stored (ny, nx) while everything else is (nx, ny, ...); the
    transposition is handled here so no notebook writes .T by hand.
    """
    m = ((lats >= min(lat_lim)) & (lats <= max(lat_lim)) &
         (lons >= min(lon_lim)) & (lons <= max(lon_lim)))
    if not m.any():
        raise ValueError(f"empty box lat={lat_lim} lon={lon_lim}; data covers "
                         f"lat [{lats.min():.2f}, {lats.max():.2f}] "
                         f"lon [{lons.min():.2f}, {lons.max():.2f}]")
    jj, ii = np.where(m)          # rows are y (=j), cols are x (=i)
    return slice(int(ii.min()), int(ii.max()) + 1), slice(int(jj.min()), int(jj.max()) + 1)


def load_ensemble(path, members=None, ivars=None, bbox=None):
    """Sub-selected state_ensemble -> (nx, ny, nz, nm, nv).

    Note the peak is unavoidable: npz is DEFLATE, so numpy decompresses the whole
    (2.9 GB SINGLECONF / 10.6 GB 5-min) array before any slicing. What this saves is
    the *retained* footprint -- the full array is freed on return. For repeated use,
    go through prior_stats(), which caches to disk.
    """
    with np.load(path) as f:
        if "state_ensemble" not in f.files:
            raise IncompleteSubset(f"{os.path.basename(path)} has no 'state_ensemble'")
        arr = f["state_ensemble"]
        si, sj = bbox if bbox is not None else (slice(None), slice(None))
        arr = arr[si, sj]
        if members is not None:
            arr = arr[:, :, :, np.asarray(members), :]
        if ivars is not None:
            arr = arr[..., [VI[v] if isinstance(v, str) else v for v in ivars]]
        return np.ascontiguousarray(arr)


def _cache_key(path, bbox, min_dbz):
    b = "full" if bbox is None else f"{bbox[0].start}_{bbox[0].stop}_{bbox[1].start}_{bbox[1].stop}"
    h = hashlib.sha1(f"{os.path.abspath(path)}|{b}|{min_dbz}".encode()).hexdigest()[:12]
    return DERIVED / f"prior_{os.path.basename(os.path.dirname(path))}_{b}_{h}.npz"


def prior_stats(path, bbox=None, min_dbz=0.0, cache=True, verbose=True):
    """N2's workhorse: prior ensemble statistics from one subset file.

    Returns a small dict of (nx, ny, nz) fields -- dbz_mean, dbz_spread, n_active,
    dbz_skew, dbz_kurt -- plus (nx, ny) column-max per member and per-variable
    mean/spread. Cached to data/derived/, because every uncached call pays a full
    multi-GB decompress.
    """
    cf = _cache_key(path, bbox, min_dbz)
    if cache and cf.exists():
        with np.load(cf) as f:
            return {k: f[k] for k in f.files}

    if verbose:
        print(f"computing prior_stats for {os.path.basename(os.path.dirname(path))} "
              f"(uncached; decompressing) ...")
    ens = load_ensemble(path, bbox=bbox)            # (nx, ny, nz, Ne, 8)
    dbz = hx(ens, min_dbz=min_dbz).astype(np.float32)   # (nx, ny, nz, Ne)

    out = {
        "dbz_mean": dbz.mean(axis=3).astype(np.float32),
        "dbz_spread": dbz.std(axis=3, ddof=1).astype(np.float32),
        "n_active": (dbz > min_dbz).sum(axis=3).astype(np.int16),
        "dbz_skew": ensemble_skew(dbz, axis=3).astype(np.float32),
        "dbz_kurt": ensemble_kurt(dbz, axis=3).astype(np.float32),
        "dbz_colmax_members": dbz.max(axis=2).astype(np.float32),   # (nx, ny, Ne)
        "Ne": np.int32(ens.shape[3]),
        "min_dbz": np.float32(min_dbz),
    }
    for v in VARS:
        out[f"mean_{v}"] = ens[..., VI[v]].mean(axis=3).astype(np.float32)
        out[f"spread_{v}"] = ens[..., VI[v]].std(axis=3, ddof=1).astype(np.float32)
    del ens, dbz

    if cache:
        np.savez_compressed(cf, **out)
        if verbose:
            print(f"  cached -> {cf.name}")
    return out


def map_field(ax, lats, lons, field_xy, cmap=None, norm=None, rasterized=True, **kw):
    """pcolormesh of an (nx, ny) field on the (ny, nx) lat/lon grid.

    The transpose lives here. No notebook writes .T by hand -- at least one old
    notebook got it wrong.

    rasterized=True by default: a 307x451 mesh is ~140k vector quads per panel, and
    an eight-panel figure lands at ~90 MB of PDF. Rasterizing just the mesh keeps
    the text, axes and colorbar as vectors (which is what a journal actually checks)
    and brings the same figure to well under a megabyte.
    """
    f = np.asarray(field_xy)
    if f.shape == lats.shape:            # already (ny, nx)
        fT = f
    elif f.T.shape == lats.shape:        # the usual (nx, ny)
        fT = f.T
    else:
        raise ValueError(f"field {f.shape} matches neither lats {lats.shape} nor its transpose")
    return ax.pcolormesh(lons, lats, fT, cmap=cmap, norm=norm, shading="auto",
                         rasterized=rasterized, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# §6  Sweep loading and row alignment
# ═════════════════════════════════════════════════════════════════════════════

META_COLS = ["i", "j", "k", "x_km", "y_km", "z_km", "yo", "yo_clean",
             "method", "ntemp", "alpha_s", "lx_km", "ly_km", "lz_km"]
# Legacy aliases of spread_{f,a}_point_obs, kept in the npz for old notebooks.
DEPRECATED_COLS = {"spread_f_obs", "spread_a_obs"}
REDUCTIONS = ("point", "w", "u")
COMBO_FIELDS = ("method", "ntemp", "alpha_s", "lx_km", "ly_km", "lz_km")


def columns_for(metrics=("rmse", "crps"), vars=("obs",), reds=REDUCTIONS,
                stages=("f", "a"), extra=()):
    """Build the column whitelist for load_sweep.

    Not optional in practice: 362 float32 columns x ~1e6 rows x 7 combos is
    10-20 GB per truth member.
    """
    cols = list(META_COLS) + list(extra)
    for m in metrics:
        for v in vars:
            for r in reds:
                for s in stages:
                    try:
                        cols.append(colname(m, s, r, v))
                    except ColumnError:
                        # bias at _point_ for a state variable is synthesised by raw()
                        cols += [f"mean_{s}_point_{v}", f"truth_point_{v}"]
    return sorted(set(cols))


def load_sweep(path, columns=None, methods=None, verbose=True):
    """One sweep npz -> DataFrame of its 1-D columns."""
    d = np.load(path, allow_pickle=True)
    all1d = [k for k in d.files if d[k].ndim == 1 and k != "var_names"]
    if verbose and len(d.files) != SCHEMA_NCOLS:
        warnings.warn(
            f"{os.path.basename(path)} has {len(d.files)} keys, expected {SCHEMA_NCOLS}. "
            f"This file probably predates the reflectivity-metrics change; "
            f"`_ref`/`n_active`/`bias_*` columns will be missing.")
    want = [c for c in (columns if columns is not None else all1d)
            if c in all1d and c not in DEPRECATED_COLS]
    missing = sorted(set(columns or []) - set(all1d) - DEPRECATED_COLS)
    if missing:
        raise KeyError(f"{os.path.basename(path)} lacks columns: {missing[:8]}"
                       f"{' ...' if len(missing) > 8 else ''}")
    if verbose:
        nrow = d[want[0]].shape[0]
        print(f"  {os.path.basename(path)}: {nrow:,} rows x {len(want)} cols "
              f"(~{nrow * len(want) * 4 / 1e9:.2f} GB)")
    df = pd.DataFrame({c: d[c] for c in want})
    if "method" in df:
        df["method"] = df["method"].astype(str)   # U8 -> str, else == matches nothing
    if methods is not None:
        df = df[df["method"].isin(methods)].reset_index(drop=True)
    return df


def load_runs(spec, columns=None, verbose=True):
    """Load several runs into one tidy frame.

    spec = [{'tag': 'WS_...', 'loc': 0.1, 'hour': '18', 'tms': [0]}, ...]
    Adds run/loc/hour/Ne/tm columns.
    """
    frames = []
    for s in spec:
        for f in sweep_files(s["tag"], s.get("tms")):
            df = load_sweep(f, columns=columns, methods=s.get("methods"), verbose=verbose)
            base = os.path.basename(f)
            df["run"] = s["tag"]
            # NOT "loc": that name collides with the pandas .loc indexer, so
            # df.loc would silently be the indexer rather than the column.
            df["loc_km"] = float(s.get("loc", df["lx_km"].iloc[0]))
            df["hour"] = s.get("hour", "")
            df["Ne"] = int(base.split("_Ne")[1][:3])
            df["tm"] = int(base.rsplit("_tm", 1)[1][:2])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no sweep files matched spec {spec}")
    return pd.concat(frames, ignore_index=True)


def point_key(df, dims=None):
    """Stable per-point identifier, so combos align point-for-point.

    np.ravel_multi_index on the real domain dims, including tm. The old
    (i*1000+j)*100+k formula silently collides once ny >= 1000.
    """
    if dims is None:
        dims = (int(df["tm"].max()) + 1 if "tm" in df else 1,
                int(df["i"].max()) + 1, int(df["j"].max()) + 1, int(df["k"].max()) + 1)
    tm = df["tm"].to_numpy(np.int64) if "tm" in df else np.zeros(len(df), np.int64)
    return pd.Series(
        np.ravel_multi_index((tm, df["i"].to_numpy(np.int64),
                              df["j"].to_numpy(np.int64), df["k"].to_numpy(np.int64)), dims),
        index=df.index, name="point_key")


def combos_in(df):
    """Sorted list of the 6-tuple combos present."""
    g = df[list(COMBO_FIELDS)].drop_duplicates()
    return sorted(tuple(r) for r in g.itertuples(index=False, name=None))


class Aligned:
    """Combos restricted to their common point set.

    Both N3 and N4 aggregate through this object, which is what makes them
    incapable of averaging over different point sets.
    """

    def __init__(self, frames, combos, nan_counts, dropped):
        self.frames = frames
        self.combos = combos
        self.nan_counts = nan_counts
        self.dropped = dropped
        self.n_points = len(next(iter(frames.values()))) if frames else 0

    def __repr__(self):
        return (f"<Aligned {len(self.combos)} combos x {self.n_points:,} common points "
                f"({self.dropped:,} dropped by the intersection)>")

    def coverage(self):
        rows = []
        for c in self.combos:
            rows.append(dict(zip(COMBO_FIELDS, c),
                             n_points=self.n_points, n_nan=int(self.nan_counts.get(c, 0))))
        return pd.DataFrame(rows)


def align(df, combos=None, metric_cols=None):
    """Inner-join every combo on point_key.

    Raises if a combo appears twice (a duplicated alpha_s in the config is the usual
    cause) or if the intersection is empty.
    """
    df = df.copy()
    df["point_key"] = point_key(df)
    combos = list(combos) if combos is not None else combos_in(df)

    parts, keysets = {}, []
    for c in combos:
        m = np.ones(len(df), bool)
        for f, v in zip(COMBO_FIELDS, c):
            col = df[f].to_numpy()
            m &= (col == v) if f == "method" else np.isclose(col.astype(float), float(v))
        sub = df[m]
        if sub.empty:
            raise ValueError(f"combo {c} matched no rows")
        if sub["point_key"].duplicated().any():
            n = int(sub["point_key"].duplicated().sum())
            raise ValueError(
                f"combo {c} has {n} duplicated points. The usual cause is a config "
                f"whose alpha_s or loc list produced two rows that differ only in a "
                f"field not part of the combo key -- see _build_combos in "
                f"run_experiment.py.")
        parts[c] = sub.set_index("point_key").sort_index()
        keysets.append(set(parts[c].index))

    common = set.intersection(*keysets)
    if not common:
        raise ValueError("combos share no common points")
    idx = pd.Index(sorted(common), name="point_key")
    dropped = max(len(k) for k in keysets) - len(idx)

    frames, nan_counts = {}, {}
    for c in combos:
        f = parts[c].loc[idx]
        frames[c] = f
        cols = metric_cols or [k for k in f.columns
                               if k.startswith(("rmse_", "crps_", "bias_", "spread_",
                                                "skew_", "kurt_"))]
        nan_counts[c] = int(f[cols].isna().any(axis=1).sum()) if cols else 0
    return Aligned(frames, combos, nan_counts, dropped)


# ═════════════════════════════════════════════════════════════════════════════
# §7  THE convention
# ═════════════════════════════════════════════════════════════════════════════

# 'lower'      : smaller is better, compare raw
# 'lower_abs'  : smaller magnitude is better
# None         : not a skill metric
METRIC_ORIENT = {"rmse": "lower", "crps": "lower", "bias": "lower_abs",
                 "skew": "lower_abs", "kurt": "lower_abs", "spread": None}
METRIC_LABELS = {"rmse": "RMSE", "crps": "CRPS", "bias": "bias",
                 "skew": "skewness", "kurt": "excess kurtosis", "spread": "spread"}
RED_LABELS = {"point": "at obs point", "w": "loc-weighted", "u": "cutoff-zone mean"}


class ColumnError(KeyError):
    """The requested (metric, stage, reduction, variable) has no stored column."""


class ConsistencyError(AssertionError):
    """A published table did not reproduce. Carries .diff."""

    def __init__(self, msg, diff=None):
        super().__init__(msg)
        self.diff = diff


def colname(metric, stage, red, var):
    """The native column name. The variable suffix is always last."""
    if metric not in METRIC_ORIENT:
        raise ColumnError(f"unknown metric {metric!r}; known: {sorted(METRIC_ORIENT)}")
    if stage not in ("f", "a"):
        raise ColumnError(f"stage must be 'f' or 'a', got {stage!r}")
    if red not in REDUCTIONS:
        raise ColumnError(f"reduction must be one of {REDUCTIONS}, got {red!r}")
    if var not in ALL_VARS:
        raise ColumnError(f"unknown variable {var!r}; known: {ALL_VARS}")
    if metric == "bias" and red == "point" and var != "obs":
        raise ColumnError(
            f"bias_{stage}_point_{var} is not stored (da/metrics.py omits it "
            f"deliberately). raw() synthesises it from "
            f"mean_{stage}_point_{var} - truth_point_{var}.")
    return f"{metric}_{stage}_{red}_{var}"


def _stored_as_magnitude(metric, red):
    """skew/kurt at _w_/_u_ are already means of |.| (da/metrics.py); everything
    else -- including bias at every reduction -- is stored signed."""
    return metric in ("skew", "kurt") and red in ("w", "u")


def raw(df, metric, stage, red, var):
    """The stored value, no subtraction. Handles the synthesised bias-at-point."""
    if metric == "bias" and red == "point" and var != "obs":
        return (df[f"mean_{stage}_point_{var}"] - df[f"truth_point_{var}"]).rename(
            f"bias_{stage}_point_{var}")
    return df[colname(metric, stage, red, var)]


def skill(df, metric="rmse", var="obs", red="w"):
    """prior - analysis. POSITIVE MEANS THE ANALYSIS IS BETTER.

    The only function in this module that subtracts an analysis from a prior.
    """
    orient = METRIC_ORIENT.get(metric)
    if orient is None:
        raise ValueError(
            f"'{metric}' is not a skill metric -- lower {metric} is not better, so "
            f"'{metric} skill' has no meaning. Use raw(df, '{metric}', stage, red, var).")
    f = raw(df, metric, "f", red, var).astype(float)
    a = raw(df, metric, "a", red, var).astype(float)
    if orient == "lower_abs" and not _stored_as_magnitude(metric, red):
        f, a = f.abs(), a.abs()
    return (f - a).rename(f"skill_{metric}_{red}_{var}")


def skill_label(metric, var, red):
    """Axis text generated by the same call chain as the numbers, so a figure cannot
    be captioned against its own data."""
    unit = disp_unit(var)
    mag = " |.|" if METRIC_ORIENT.get(metric) == "lower_abs" else ""
    return (f"{METRIC_LABELS.get(metric, metric)}{mag} skill, {VAR_LABELS[var]}, "
            f"{RED_LABELS[red]} [{unit}]  (>0: analysis better)")


def _wilson(k, n, z=1.96):
    """Wilson score interval for a proportion -- O(1), unlike bootstrapping 1e6 rows."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def skill_summary(aligned, metric="rmse", var="obs", red="w"):
    """THE aggregation. Fixed schema, deterministic order, ties resolved by TOL[var].

    Both N3 and N4 call this and nothing else, so their headline numbers are the
    same numbers. n_nan is carried because divergent NaN rates between combos are
    the likeliest real cause of two notebooks disagreeing.
    """
    tol = TOL[var]
    rows = []
    for c in sorted(aligned.combos):
        f = aligned.frames[c]
        s = skill(f, metric=metric, var=var, red=red).to_numpy(float)
        good = np.isfinite(s)
        sv = s[good]
        n = int(sv.size)
        n_imp = int((sv > tol).sum())
        n_deg = int((sv < -tol).sum())
        lo, hi = _wilson(n_imp, n)
        rows.append({
            "run": str(f["run"].iloc[0]) if "run" in f else "",
            "loc_km": float(f["loc_km"].iloc[0]) if "loc_km" in f else float(c[3]),
            "method": c[0], "ntemp": int(c[1]), "alpha_s": float(c[2]),
            "metric": metric, "var": var, "red": red,
            "n_points": n, "n_nan": int((~good).sum()),
            "median_skill": float(np.median(sv)) if n else np.nan,
            "mean_skill": float(np.mean(sv)) if n else np.nan,
            "frac_improved": n_imp / n if n else np.nan,
            "frac_degraded": n_deg / n if n else np.nan,
            "ci_lo": lo, "ci_hi": hi,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["metric", "var", "red", "loc_km", "method", "ntemp"]
                           ).reset_index(drop=True)


_DIGEST_COLS = ["run", "loc_km", "method", "ntemp", "alpha_s", "metric", "var", "red",
                "n_points", "n_nan", "median_skill", "mean_skill",
                "frac_improved", "frac_degraded"]


def _canonical(df):
    d = df[[c for c in _DIGEST_COLS if c in df.columns]].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            # round to ~6 significant digits: float32 accumulation order varies with
            # row order, and too tight a digest cries wolf
            d[c] = d[c].map(lambda v: float(f"{v:.6g}") if np.isfinite(v) else np.nan)
    return d.sort_values(list(d.columns)).reset_index(drop=True)


def consistency_digest(df):
    return hashlib.sha1(_canonical(df).to_csv(index=False).encode()).hexdigest()[:16]


def publish(name, df, verbose=True):
    """Persist a canonical table and return its digest (the string N4 pastes)."""
    p = DERIVED / f"{name}.csv"
    _canonical(df).to_csv(p, index=False)
    dig = consistency_digest(df)
    if verbose:
        print(f"published {p.name}  digest={dig}")
    return dig


def expect(name, df, digest=None):
    """Two independent checks: against the pasted digest, and against the stored CSV.

    Raises ConsistencyError carrying a row-aligned .diff.
    """
    got = consistency_digest(df)
    p = DERIVED / f"{name}.csv"
    if not p.exists():
        raise ConsistencyError(f"{p.name} not found -- run the notebook that publishes "
                               f"'{name}' first.")
    ref = pd.read_csv(p)
    cur = _canonical(df)
    try:
        pd.testing.assert_frame_equal(ref, cur, check_dtype=False, rtol=1e-6, atol=0)
    except AssertionError as e:
        key = [c for c in ("metric", "var", "red", "loc_km", "method", "ntemp")
               if c in ref.columns and c in cur.columns]
        diff = ref.merge(cur, on=key, how="outer", suffixes=("_published", "_here"))
        for c in ("median_skill", "frac_improved"):
            if f"{c}_published" in diff:
                diff[f"{c}_delta"] = diff[f"{c}_here"] - diff[f"{c}_published"]
        raise ConsistencyError(f"'{name}' does not reproduce:\n{e}", diff=diff) from None
    if digest is not None and got != digest:
        raise ConsistencyError(
            f"'{name}' digest mismatch: pasted {digest}, computed {got}. The stored "
            f"CSV matches, so the publishing notebook was re-run with a different "
            f"configuration and its digest literal is stale.")
    return got


def assert_convention():
    """Runs at import. If someone flips the sign, every notebook fails at once."""
    d = pd.DataFrame({"rmse_f_w_obs": [5.0, 1.0], "rmse_a_w_obs": [3.0, 4.0],
                      "skew_f_point_w": [-2.0, -2.0], "skew_a_point_w": [0.5, 0.5],
                      "skew_f_w_w": [2.0, 2.0], "skew_a_w_w": [0.5, 0.5]})
    s = skill(d, "rmse", "obs", "w")
    assert s.iloc[0] == 2.0, f"skill must be prior-analysis; got {s.iloc[0]} for 5.0 -> 3.0"
    assert s.iloc[1] == -3.0, f"skill sign wrong for a degradation; got {s.iloc[1]}"
    # magnitude comparison for skew at _point_ (signed storage): |-2.0| - |0.5| = 1.5
    assert skill(d, "skew", "w", "point").iloc[0] == 1.5
    # ... but _w_/_u_ skew is ALREADY a mean of |.|, so the magnitude is not retaken
    assert skill(d, "skew", "w", "w").iloc[0] == 1.5
    # the two naming hazards
    assert colname("crps", "f", "w", "w") == "crps_f_w_w"
    assert colname("rmse", "f", "u", "u") == "rmse_f_u_u"
    for bad in ("spread",):
        try:
            skill(d, bad, "obs", "w")
        except ValueError:
            pass
        else:
            raise AssertionError(f"skill('{bad}') must raise")
    try:
        colname("bias", "f", "point", "qg")
    except ColumnError:
        pass
    else:
        raise AssertionError("colname('bias','f','point','qg') must raise")


assert_convention()


# ═════════════════════════════════════════════════════════════════════════════
# §8  Predictors  (N3)
# ═════════════════════════════════════════════════════════════════════════════

# (column, axis label, percentile clip)
PREDICTORS = [
    ("n_active_f_point", "members with signal", (0, 100)),
    ("frac_floor", "fraction of members at the floor", (0, 100)),
    ("spread_f_point_obs", r"prior spread $\sigma_H$ [dBZ]", (0.5, 99.5)),
    ("abs_skew_f", r"prior $|$skewness$|$", (0.5, 99)),
    ("kurt_f_point_obs", "prior excess kurtosis", (0.5, 99)),
    ("abs_dep_b", r"$|$innovation$|$ [dBZ]", (0.5, 99.5)),
    ("norm_innov", r"normalised innovation $|d|/\sqrt{\sigma_H^2+R}$", (0, 99.5)),
    ("z_km", "height [km]", (0, 100)),
]


def add_predictors(df, R, Ne=None, dbz_min=0.0):
    """Prior-condition predictors, all derived from stored sweep columns.

    After the reflectivity-metrics change these are native columns, so nothing here
    recomputes H(x) -- which is what made the old build_obs_predictors.py redundant.
    R must come from run_config(tag)['obs']['obs_error_var'].
    """
    df = df.copy()
    if Ne is None:
        Ne = int(df["Ne"].iloc[0]) if "Ne" in df else 59
    df["abs_dep_b"] = df["dep_b"].abs()
    df["norm_innov"] = df["abs_dep_b"] / np.sqrt(df["spread_f_point_obs"] ** 2 + float(R))
    df["z_km"] = df["z_km"] if "z_km" in df else np.nan
    df["frac_floor"] = 1.0 - df["n_active_f_point"] / float(Ne)
    df["abs_skew_f"] = df["skew_f_point_obs"].abs()
    return df


# ═════════════════════════════════════════════════════════════════════════════
# §9  Binning and density primitives
# ═════════════════════════════════════════════════════════════════════════════

def lims(v, lo=0.5, hi=99.5):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (0.0, 1.0)


def binned_stats(x, y, edges, min_count=10):
    """Per-bin mean and std, blanked where the count is too low."""
    x, y = np.asarray(x), np.asarray(y)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(len(centers), np.nan)
    stds = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), int)
    for b in range(len(centers)):
        m = (x >= edges[b]) & (x < edges[b + 1]) & np.isfinite(y)
        counts[b] = m.sum()
        if counts[b] >= min_count:
            means[b] = np.nanmean(y[m])
            stds[b] = np.nanstd(y[m])
    return centers, means, stds, counts


def binned_mean(x, y, c, xr, yr, grid=24, mincnt=5):
    """2-D binned mean of c over (x, y), NaN where the count is too low."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    H, xe, ye = np.histogram2d(x[m], y[m], bins=grid, range=[xr, yr], weights=c[m])
    N, _, _ = np.histogram2d(x[m], y[m], bins=grid, range=[xr, yr])
    with np.errstate(invalid="ignore"):
        g = H / N
    g[N < mincnt] = np.nan
    return g.T, xe, ye


def prob_hexbin(ax, x, y, mask, xlim, ylim, gridsize=45, mincnt=40, **kw):
    """P(mask) per hexagonal bin, on a 0..1 diverging scale."""
    return ax.hexbin(x, y, C=np.asarray(mask, float), reduce_C_function=np.mean,
                     gridsize=gridsize, cmap=CMAP_DIV, vmin=0.0, vmax=1.0,
                     mincnt=mincnt, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                     linewidths=0.0, **kw)


def skill_hexbin(ax, x, y, s, xlim, ylim, gridsize=45, mincnt=40, vmax=None, **kw):
    """Mean SKILL per bin. Colour is not a free choice here: CMAP_SKILL centred at 0,
    so blue always means the analysis is better."""
    if "cmap" in kw or "norm" in kw:
        raise ValueError("skill_hexbin fixes the colormap and norm so that blue always "
                         "means 'analysis better'. Use ax.hexbin directly for other data.")
    s = np.asarray(s, float)
    if vmax is None:
        finite = s[np.isfinite(s)]
        vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    vmax = vmax or 1.0
    return ax.hexbin(x, y, C=s, reduce_C_function=np.nanmean, gridsize=gridsize,
                     cmap=CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                     mincnt=mincnt, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                     linewidths=0.0, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# §10  Maps and 3-D projections
# ═════════════════════════════════════════════════════════════════════════════

AOI = dict(lat=(-41.5, -25.3), lon=(-68.6, -55.4))


RASTER_ZORDER = 2.5


def setup_map(ax, extent=None, labels=True, rasterize_below=RASTER_ZORDER):
    """Cartopy basemap. Returns False (and leaves ax alone) if cartopy is absent.

    Everything below `rasterize_below` is flattened to pixels on export: the data
    mesh, the land/ocean fill and the 50 m coastline/border/state geometry. Text,
    gridline labels, titles and colorbars stay vector.

    This matters more than it sounds. The NaturalEarth 50 m cultural geometry alone
    is tens of thousands of paths per panel, so an eight-panel map figure exports at
    ~59 MB of PDF without it and well under 1 MB with it. Pass
    rasterize_below=None to keep everything vector.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:      # pragma: no cover
        warnings.warn("cartopy not available; drawing without a basemap")
        return False
    ax.add_feature(cfeature.LAND, facecolor="#f0efec", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#dce9f5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="black", zorder=2)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "50m", facecolor="none"),
        edgecolor="black", linewidth=0.4, linestyle=":", zorder=2)
    e = extent or (AOI["lon"][0], AOI["lon"][1], AOI["lat"][0], AOI["lat"][1])
    ax.set_extent(e, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=labels, linestyle="--", alpha=0.4, color="gray",
                      linewidth=0.4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    if rasterize_below is not None:
        ax.set_rasterization_zorder(rasterize_below)
    return True


def draw_box(ax, lat_lim, lon_lim, **kw):
    """Outline a lat/lon sub-box on a map."""
    import cartopy.crs as ccrs
    style = dict(c="#d62728", lw=1.6, transform=ccrs.PlateCarree(), zorder=6)
    style.update(kw)
    la, lo = sorted(lat_lim), sorted(lon_lim)
    return ax.plot([lo[0], lo[1], lo[1], lo[0], lo[0]],
                   [la[0], la[0], la[1], la[1], la[0]], **style)


def coarse_map_km(sub, col, pos_km, stride=20, agg="max"):
    """Grid scattered sweep points onto a coarse km map."""
    g = sub.copy()
    g["bi"] = (g["i"] // stride).astype(int)
    g["bj"] = (g["j"] // stride).astype(int)
    a = g.groupby(["bi", "bj"])[col].agg(agg).reset_index()
    nbi, nbj = int(g["bi"].max()) + 1, int(g["bj"].max()) + 1
    grid = np.full((nbi, nbj), np.nan)
    grid[a["bi"].to_numpy(), a["bj"].to_numpy()] = a[col].to_numpy()
    xb = np.clip(np.arange(nbi) * stride, 0, pos_km.shape[0] - 1)
    yb = np.clip(np.arange(nbj) * stride, 0, pos_km.shape[1] - 1)
    return pos_km[xb, 0, 0, 0], pos_km[0, yb, 0, 1], grid


def _proj3(field3d, mode="max"):
    """(nx,ny,nz) -> three projections (over z, over y, over x)."""
    fn = np.nanmax if mode == "max" else np.nanmean
    return fn(field3d, axis=2), fn(field3d, axis=1), fn(field3d, axis=0)


def _edges(v):
    """Cell centres -> edges, so pcolormesh renders non-uniform z correctly."""
    v = np.asarray(v, float)
    d = np.diff(v)
    return np.concatenate([[v[0] - d[0] / 2], v[:-1] + d / 2, [v[-1] + d[-1] / 2]])


def _panel(ax, img, xc, yc, xlabel="", ylabel="", title="", rasterized=True, **kw):
    """pcolormesh with centre->edge conversion. img is (len(yc), len(xc)).

    rasterized=True for the same reason as map_field: keep the mesh out of the
    vector layer so the PDF stays a sane size.
    """
    im = ax.pcolormesh(_edges(xc), _edges(yc), img, shading="flat",
                       rasterized=rasterized, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return im


def scales(fields, v, anom_vars=()):
    """Per-variable colour-scale policy, shared across a row so panels compare."""
    a = np.concatenate([np.asarray(f).ravel() for f in fields])
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(cmap="viridis", vmin=0.0, vmax=1.0)
    if v in ("obs", "DBZ"):
        return dict(cmap=DBZ_CMAP, norm=BoundaryNorm(DBZ_LEVELS, plt.get_cmap(DBZ_CMAP).N))
    if v in ("u", "v", "w") or v in anom_vars:
        lim = float(np.percentile(np.abs(a), 99)) or 1.0
        return dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
    if v in HYDRO:
        hi = float(np.percentile(a[a > 0], 99.5)) if np.any(a > 0) else 1.0
        return dict(cmap="YlGnBu", vmin=0.0, vmax=hi if hi > 0 else 1.0)
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    return dict(cmap="viridis", vmin=lo, vmax=hi if hi > lo else lo + 1.0)


def sym_scale(fields):
    """Symmetric diverging scale for increments."""
    a = np.concatenate([np.asarray(f).ravel() for f in fields])
    a = a[np.isfinite(a)]
    lim = float(np.percentile(np.abs(a), 99.5)) if a.size else 1.0
    return dict(cmap="RdBu_r", vmin=-(lim or 1.0), vmax=(lim or 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# §11  Multi-obs, guarded and optional
# ═════════════════════════════════════════════════════════════════════════════

def has_multiobs(tag):
    d = run_dir(tag)
    return d.is_dir() and any(d.glob("*_multi_obs_*_Ne*_tm*.npz"))


def multiobs_files(tag):
    return sorted(str(p) for p in run_dir(tag).glob("*_multi_obs_*_Ne*_tm*.npz")
                  if "_ref_" not in os.path.basename(p))


_SLICE_CACHE = {}


def sl(path, key, j):
    """The j-slice of a 4-D multi-obs field, cached.

    Biggest single performance win available here: the stored fields are DEFLATE and
    a full key is ~176 MB to decompress, while one slice is a few hundred kB.
    """
    ck = (path, key, int(j))
    if ck not in _SLICE_CACHE:
        with np.load(path) as f:
            _SLICE_CACHE[ck] = np.asarray(f[key][:, int(j)], dtype=np.float64)
    return _SLICE_CACHE[ck]


def mo_globals(path, metrics=("rmse", "crps"), vars=ALL_VARS):
    """The *_global_* scalars from one multi-obs npz, as a tidy frame.

    Reflectivity uses the `_ref` suffix there (vs `_obs` in the sweep), so this
    normalises it to 'obs' for consistency with the rest of the module.
    """
    rows = []
    with np.load(path, allow_pickle=True) as f:
        for m in metrics:
            for v in vars:
                key_v = "ref" if v == "obs" else v
                kf, ka = f"{m}_f_global_{key_v}", f"{m}_a_global_{key_v}"
                if kf not in f.files or ka not in f.files:
                    continue
                pf, pa = float(f[kf]), float(f[ka])
                if METRIC_ORIENT.get(m) == "lower_abs":
                    pf, pa = abs(pf), abs(pa)
                rows.append(dict(method=str(f["method"]), ntemp=int(f["ntemp"]),
                                 loc_km=float(f["lx_km"]), metric=m, var=v,
                                 prior=pf, analysis=pa, skill=pf - pa))
    return pd.DataFrame(rows)


def field_rmse(abs_err, mask=None):
    """Per-variable RMSE of a (nx,ny,nz,nvar) absolute-error field."""
    out = np.full(abs_err.shape[-1], np.nan)
    for vi in range(abs_err.shape[-1]):
        f = abs_err[..., vi]
        vals = f[mask] if mask is not None else f[np.isfinite(f)]
        if vals.size:
            out[vi] = np.sqrt(np.nanmean(vals ** 2))
    return out


def domain_masks(truth_hx, obs_ijk=None, storm_thresh=20.0):
    """global / storm / obs evaluation domains for a (nx,ny,nz) field."""
    m = {"global": np.ones(truth_hx.shape, bool)}
    m["storm"] = np.broadcast_to(
        (np.nanmax(truth_hx, axis=2) >= storm_thresh)[:, :, None], truth_hx.shape)
    if obs_ijk is not None:
        o = np.zeros(truth_hx.shape, bool)
        o[obs_ijk[0], obs_ijk[1], obs_ijk[2]] = True
        m["obs"] = o
    return m
