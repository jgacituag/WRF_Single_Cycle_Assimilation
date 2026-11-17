"""
Notebooks/nbcommon.py
=====================
Shared layer for the notebooks N0-N5: paths, run configuration, figure style and
output, palettes and variable tables, the reflectivity operator, prior-ensemble and
sweep access, maps, and the two languages of the figures. No analysis logic: the
metrics live in the notebooks.

Usage in a notebook:

    import nbcommon as nb

Sections
    1  paths and run configuration
    2  figure style
    3  palettes, labels, variable tables
    4  physics (reflectivity forward operator)
    5  prior-ensemble access
    6  sweep loading
    7  maps
    8  languages: vocabulary, dataset labels, fmt, localize, save
"""

import os
import pathlib
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib.ticker as mticker

# ═════════════════════════════════════════════════════════════════════════════
# 1  Paths and run configuration
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


DATASETS = ("A", "B", "C", "D")

# What each dataset letter is, for the assertion messages below.
DS_DESC = {
    "A": "4 km, 5 min DA, multi-physics (rebuild)",
    "B": "4 km, 1 h DA, multi-physics",
    "C": "2 km, 5 min DA, multi-physics",
    "D": "4 km, 5 min DA, single-physics",
}


def subset_path(hour, dataset, date="20240319"):
    """Path to a 3D subset npz: data/3D_subsets_{DS}/subset_{DS}_{stamp}.npz.

    `hour` is '18'..'21'; `dataset` is one of A/B/C/D.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    stamp = f"{date}{hour}0000"
    return str(DATA / f"3D_subsets_{dataset}" / f"subset_{dataset}_{stamp}.npz")


# ═════════════════════════════════════════════════════════════════════════════
# 1b  The analysis window and the run files
# ═════════════════════════════════════════════════════════════════════════════

# ONE analysis window, used by every notebook and every section. A 5.5 deg square
# centred at 36.0 S. It lives here rather than in a notebook cell so that "inside the
# window" cannot mean two different boxes in two different figures.
WIN_LAT = (-38.75, -33.25)
WIN_LON = (-62.00, -56.50)

DS_ALL = DATASETS


N2_HOURS = ("18", "19", "20", "21")


def dataset_of(path):
    """The `dataset_id` stored INSIDE a subset or run npz, or None if it has none.

    Read from the file, never parsed from its name. Cheap: np.load reads one small
    member of the zip, not the gigabytes next to it.
    """
    try:
        with np.load(path, allow_pickle=False) as f:
            if "dataset_id" not in f.files:
                return None
            return str(f["dataset_id"])
    except Exception:                                    # noqa: BLE001
        return None


def assert_dataset(path, expected, what="file"):
    """Fail loudly when a file is not the dataset the notebook believed it was.

    The whole point of writing `dataset_id` into the data: a mis-migrated or
    mis-copied file is then caught at load, instead of being analysed as the wrong
    dataset and quietly changing a published number. Files without a `dataset_id`
    are passed through with a warning rather than an error -- there is nothing to
    check them against.
    """
    got = dataset_of(path)
    if got is None:
        warnings.warn(
            f"{os.path.basename(path)} carries no dataset_id. Believing the caller: {expected}.")
        return expected
    if expected is not None and got != expected:
        raise ValueError(
            f"dataset mismatch: {path}\n"
            f"  the notebook expected dataset {expected} ({DS_DESC.get(expected, '?')})\n"
            f"  the file says              {got} ({DS_DESC.get(got, '?')})\n"
            f"  Refusing to analyse it as {expected}.")
    return got


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
# 2  Figure style
# ═════════════════════════════════════════════════════════════════════════════

MM = 1 / 25.4
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
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

set_style()


def panel_label(ax, letter, loc="upper left", pad=0.02, fontsize=9):
    """(a)/(b) tag placed outside the title."""
    x, ha = (pad, "left") if "left" in loc else (1 - pad, "right")
    y, va = (1 - pad, "top") if "upper" in loc else (pad, "bottom")
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha=ha, va=va,
            fontweight="bold", fontsize=fontsize,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))


# ═════════════════════════════════════════════════════════════════════════════
# 3  Palettes, labels, variable tables
# ═════════════════════════════════════════════════════════════════════════════

# The one dBZ scale: a stepped radar scale with its BoundaryNorm.
DBZ_LEVELS = np.arange(0, 66, 5)
DBZ_CMAP = "Spectral_r"
DBZ_NORM = BoundaryNorm(DBZ_LEVELS, plt.get_cmap(DBZ_CMAP).N)

# CVD-validated sequential + diverging maps.
C_NT2, C_NT1, C_NONE = "#8B0000", "#008080", "#FF8C00"
_SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)
CMAP_SEQ.set_bad("#ffffff", alpha=0.0)
# The diverging map: RdBu_r, red at positive. Left as a name, not a Colormap instance:
# mutating the registered instance (set_bad) would change it for every other user of
# "RdBu_r" in the process, and its default bad colour is already fully transparent,
# which is what the NaN cells need.
CMAP_DIV = "RdBu_r"
# CMAP_SKILL: red = positive = better.
CMAP_SKILL = CMAP_DIV

METHOD_COLORS = {"TEnKF": "#2a78d6", "LETKF": "#52514e", "AOEI": "#eb6834", "prior": "black"}


# Variable tables. VI is the single source of truth for the trailing-axis order: no
# notebook writes a variable-list literal.
VI = {"qg": 0, "qr": 1, "qs": 2, "T": 3, "P": 4, "u": 5, "v": 6, "w": 7}
VARS = ["qg", "qr", "qs", "T", "P", "u", "v", "w"]
HYDRO = ("qg", "qr", "qs")

# Per-variable physical tolerance: the scale below which a change of |error| is a tie,
# not a result.
TOL = dict(qg=1e-7, qr=1e-7, qs=1e-7, T=1e-3, P=1e-1,
           u=1e-4, v=1e-4, w=1e-4, obs=1e-3)


def combo_color(method, ntemp):
    if method != "TEnKF":
        return METHOD_COLORS.get(method, "#444444")
    shades = ["#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
    return shades[min(int(ntemp), 5) - 1]


# ═════════════════════════════════════════════════════════════════════════════
# 4  Physics
# ═════════════════════════════════════════════════════════════════════════════

PI = 3.14159265358979
RD = 287.0


def calc_ref_np(qr, qs, qg, T, P, min_dbz=0.0):
    """Tong & Xue (2006) reflectivity, pure NumPy.

    Mirrors the Fortran calc_ref/calc_ref_ens so notebooks work without the
    compiled extension.
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


# Diagnostics come from the production module -- never redefined here.
from da.metrics import ensemble_skew, ensemble_kurt  # noqa: E402

# The three evaluation domains, likewise. The runner reduces the light-mode scalars over
# exactly these masks, so a local copy here would let a notebook's field-recomputed
# number and the scalar on disk drift apart without either being obviously wrong.
from da.metrics import STORM_THRESH_DBZ, domain_masks  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# 5  Prior-ensemble access
# ═════════════════════════════════════════════════════════════════════════════


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


def load_subset(path, keys=("lats", "lons", "pos_km", "z_heights"), dataset=None):
    """Geometry from a subset npz. Pass `dataset` to assert what the file says it is."""
    if dataset is not None:
        assert_dataset(path, dataset)
    return _load_subset_arrays(path, keys)


def _load_subset_arrays(path, keys=("lats", "lons", "pos_km", "z_heights")):
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


def load_ensemble(path, members=None, ivars=None, bbox=None):
    """Sub-selected state_ensemble -> (nx, ny, nz, nm, nv).

    Note the peak is unavoidable: npz is DEFLATE, so numpy decompresses the whole
    (2.9 GB SINGLECONF / 10.6 GB 5-min) array before any slicing. What this saves is
    the *retained* footprint -- the full array is freed on return.
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


# The reflectivity histogram grid. hx() clamps at min_dbz, so min_dbz is the floor
# and everything below it piles into the first bin; 0.25 dBZ is fine enough that any
# coarser binning, quantile or exceedance fraction can be taken from the counts.
DBZ_HIST_EDGES = np.arange(0.0, 80.0 + 0.25, 0.25)


def map_field(ax, lats, lons, field_xy, cmap=None, norm=None, rasterized=True, **kw):
    """pcolormesh of an (nx, ny) field on the (ny, nx) lat/lon grid.

    The transpose lives here, so no notebook writes .T by hand.

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
# 6  Sweep loading
# ═════════════════════════════════════════════════════════════════════════════

# Aliases of spread_{f,a}_point_obs that the runner still writes; never loaded.
DEPRECATED_COLS = {"spread_f_obs", "spread_a_obs"}


def load_sweep(path, columns=None, methods=None, verbose=True):
    """One sweep npz -> DataFrame of its 1-D columns."""
    d = np.load(path, allow_pickle=True)
    all1d = [k for k in d.files if d[k].ndim == 1 and k != "var_names"]
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


# ═════════════════════════════════════════════════════════════════════════════
# 7  Maps
# ═════════════════════════════════════════════════════════════════════════════

AOI = dict(lat=(-41.5, -25.3), lon=(-68.6, -55.4))


RASTER_ZORDER = 2.5

def setup_map(ax, extent=None, labels=True, xlocs=None, ylocs=None,
              left_labels=True, bottom_labels=True, rasterize_below=RASTER_ZORDER,
              label_size=6):
    """Cartopy basemap. Returns False (and leaves ax alone) if cartopy is absent.

    `label_size` is the font size [pt] of the latitude/longitude labels.

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
    gl = ax.gridlines(draw_labels=labels, linestyle="--", alpha=0.4,
                      color="gray", linewidth=0.4)
    gl.top_labels = gl.right_labels = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    if xlocs is not None:
        gl.xlocator = mticker.FixedLocator(xlocs)
    if ylocs is not None:
        gl.ylocator = mticker.FixedLocator(ylocs)
    gl.xlabel_style = {"size": label_size, "rotation": 45, "ha": "right"}
    gl.ylabel_style = {"size": label_size}
    if rasterize_below is not None:
        ax.set_rasterization_zorder(rasterize_below)
    return True


# ═════════════════════════════════════════════════════════════════════════════
# 8  Languages: vocabulary, dataset labels, fmt, localize, save
# ═════════════════════════════════════════════════════════════════════════════
# Every figure is saved twice, figures/es/<stem>.png and figures/en/<stem>.png. This
# section holds data (the vocabulary and the labels) and three presentation helpers,
# nothing else: no analysis. Labels specific to one notebook live in its own
# LABELS = {"es": {...}, "en": {...}} cell.

import re as _re

LANGS = ("es", "en")
SHOW = "es"                     # the language shown inline when save() is not told otherwise
FIG_DPI = 300

# The codes A, B, C stay in file names and in the data; figures, legends and printed
# tables use the label, identical in both languages. D has no agreed label and is not
# used in the chapter: asking for it raises KeyError rather than inventing one.
DS_LABEL = {"A": "4 km–5 min", "B": "4 km–1 h", "C": "2 km–5 min"}

# Shared vocabulary, as decided in the review. Scheme names (TEnKF-T1 ... TEnKF-T10,
# AOEI) are the same in both languages and are not listed here.
VOCAB = {
    "es": {
        "ss":           "skill score (SS)",
        "ss_rmse":      r"$\mathrm{SS}_{\mathrm{RMSE}}$",
        "ss_crps":      r"$\mathrm{SS}_{\mathrm{CRPS}}$",
        "frac_imp":     "fracción que mejora",
        "p_deg":        "probabilidad de degradación",
        "n_echo":       r"miembros con $Z > 0$ dBZ",
        "n_echo_sym":   r"$N_{Z>0}$",
        "nt":           r"pasos de temperado $N_t$",
        "innov":        r"innovación $d$ [dBZ]",
        "cum_weight":   "peso acumulado",
        "truth":        "verdad",
        "obs":          "observación",
        "ens_mean":     "media del ensamble",
        "ens_spread":   "dispersión",
        "single_obs":   "experimentos de observación única",
        "multi_obs":    "experimentos de observaciones múltiples",
        "dom_reached":  "celdas alcanzadas",
        "dom_storm":    "tormenta",
        "dom_observed": "celdas observadas",
    },
    "en": {
        "ss":           "skill score (SS)",
        "ss_rmse":      r"$\mathrm{SS}_{\mathrm{RMSE}}$",
        "ss_crps":      r"$\mathrm{SS}_{\mathrm{CRPS}}$",
        "frac_imp":     "fraction improving",
        "p_deg":        "probability of degradation",
        "n_echo":       r"members with $Z > 0$ dBZ",
        "n_echo_sym":   r"$N_{Z>0}$",
        "nt":           r"tempering steps $N_t$",
        "innov":        r"innovation $d$ [dBZ]",
        "cum_weight":   "cumulative weight",
        "truth":        "truth",
        "obs":          "observation",
        "ens_mean":     "ensemble mean",
        "ens_spread":   "spread",
        "single_obs":   "single-observation experiments",
        "multi_obs":    "multiple-observation experiments",
        "dom_reached":  "reached cells",
        "dom_storm":    "storm",
        "dom_observed": "observed cells",
    },
}
assert set(VOCAB) == set(LANGS) and set(VOCAB["es"]) == set(VOCAB["en"]), \
    "VOCAB: the two languages do not have the same keys"

# A decimal point is a point between two digits, so the full stop of a sentence and
# the dots of a mathtext command are left alone.
_DECIMAL_POINT = _re.compile(r"(?<=\d)\.(?=\d)")
# West is "O" (oeste) in Spanish; S, N and E are the same in both languages.
_WEST = _re.compile("(°|\\^\\{\\\\circ\\}\\$?)W\\b")


def _check_lang(lang):
    if lang not in LANGS:
        raise ValueError(f"language {lang!r}; expected one of {LANGS}")


def _to_spanish(s):
    """A tick label in Spanish: decimal comma, and 'O' for west in longitudes."""
    return _WEST.sub(r"\1O", _DECIMAL_POINT.sub(",", s))


def fmt(x, lang, spec="g"):
    """A number as text for a figure, in language `lang`.

    `spec` is a format() spec: ".2f", ".1%", "+.2f", ",d"... Spanish uses the decimal
    comma and a thin space between thousands; English the point and the comma. The
    minus sign is the one on the axis ticks (U+2212 when axes.unicode_minus is set), so
    a number in an annotation and the same number on an axis look the same.
    """
    _check_lang(lang)
    s = format(x, spec)
    if plt.rcParams["axes.unicode_minus"]:
        s = s.replace("-", "−")
    if lang == "es":
        s = s.replace(",", "\x00")              # thousands, before touching the decimal
        s = s.replace(".", ",").replace("\x00", " ")
    return s


class _SpanishFormatter(mticker.Formatter):
    """Wraps an axis formatter and rewrites its text for Spanish (see _to_spanish).

    Everything is delegated to the original formatter, so its choices (number of
    decimals, offset, scientific notation, fixed or categorical labels) are kept and
    only the text it produces is rewritten.
    """

    def __init__(self, base):
        self.base = base

    def set_axis(self, axis):
        super().set_axis(axis)
        self.base.set_axis(axis)

    def set_locs(self, locs):
        self.base.set_locs(locs)

    def __call__(self, x, pos=None):
        return _to_spanish(self.base(x, pos))

    def format_ticks(self, values):
        return [_to_spanish(s) for s in self.base.format_ticks(values)]

    def format_data(self, value):
        return _to_spanish(self.base.format_data(value))

    def format_data_short(self, value):
        return _to_spanish(self.base.format_data_short(value))

    def get_offset(self):
        return _to_spanish(self.base.get_offset())


def _axes_axis(ax):
    """The coordinate axes of an Axes: x and y, plus z on a 3-D Axes."""
    out = [ax.xaxis, ax.yaxis]
    if hasattr(ax, "zaxis"):
        out.append(ax.zaxis)
    return out


def localize(fig, lang):
    """One pass over every axis of `fig`, colorbars included.

    In Spanish it wraps the major and minor formatter of every axis so that ticks use
    the decimal comma (and 'O' for west on longitudes); in English it leaves, or puts
    back, the original formatter. Colorbars are in fig.axes, so the same pass covers
    them. Call it right before saving: a later set_xticklabels would undo it. Numbers
    inside text (titles, legends, annotations) are not touched here; write them with
    fmt().

    Cartopy maps label latitude and longitude with their Gridliner formatters, not the
    axis ones; those are wrapped the same way.
    """
    _check_lang(lang)

    def swap(f):
        wrapped = isinstance(f, _SpanishFormatter)
        if lang == "es" and not wrapped:
            return _SpanishFormatter(f)
        if lang == "en" and wrapped:
            return f.base
        return f

    for ax in fig.axes:
        for axis in _axes_axis(ax):
            axis.set_major_formatter(swap(axis.get_major_formatter()))
            axis.set_minor_formatter(swap(axis.get_minor_formatter()))
        # GeoAxes.gridlines() adds the Gridliner as one more artist of the axes.
        for gl in ax.artists:
            if hasattr(gl, "xformatter") and hasattr(gl, "yformatter"):
                gl.xformatter = swap(gl.xformatter)
                gl.yformatter = swap(gl.yformatter)
    return fig


# File names in figures/en/. The notebooks give one stem, sometimes in Spanish; the
# English copy gets an English name. Fragments are replaced in this order (the longer
# ones first, so "mapa_eco_falso" is not eaten by "eco_falso"). A new Spanish stem
# needs its fragment here, or figures/en/ gets a Spanish name again.
STEM_EN = (
    ("Cap4/caso_radar_19mar", "Ch4/radar_case_19mar"),
    ("corte_vertical", "cross_section"),
    ("mapa_diagnostico", "diagnostic_map"),
    ("posterior_contra_prior", "posterior_vs_prior"),
    ("observacion_unica_conteos", "single_obs_counts"),
    ("observacion_unica", "single_obs"),
    ("mapa_eco_falso", "false_echo_map"),
    ("eco_falso", "false_echo"),
    ("regla_observadas", "rule_observed"),
    ("observadas", "observed"),
    ("regiones", "regions"),
    ("ventaja", "advantage"),
    ("particiones", "partitions"),
    ("resultado", "result"),
    ("seleccion", "selection"),
    ("barrido", "sweep"),
    ("espacial", "spatial"),
    ("predictores", "predictors"),
    ("dispersion", "spread"),
    ("grupos", "groups"),
    ("conteos", "counts"),
)


def stem_for(stem, lang):
    """The file stem of a figure in language `lang` (see STEM_EN)."""
    _check_lang(lang)
    if lang == "en":
        for es, en in STEM_EN:
            stem = stem.replace(es, en)
    return stem


def save(fig, stem, show=None, dpi=FIG_DPI, langs=None):
    """Draw and save a figure in both languages (or only in `langs`).

    `fig` is a function lang -> Figure: the figure cell defines `def draw(lang): ...`,
    with every label taken from LABELS[lang], VOCAB[lang] and fmt(x, lang), and calls
    save(draw, "N3_fig1_headline"). An already drawn Figure is not enough, because its
    text is in one language only.

    For each language: draw, apply localize(), save figures/<lang>/<stem>.png and close
    the figure, except the one in language `show` (SHOW if not given), which is left
    open so the notebook displays it. show=False leaves none open, for batches of many
    figures.

    `stem` is the name without extension and may include subfolders
    ("N3s/points/clear/clear_120_80_5_tm03"). In figures/en/ it goes through
    stem_for(), which turns a Spanish stem into English. Returns the two paths.

    `langs` restricts the languages drawn and saved, for exploratory batches
    (langs=("en",)); None is all of LANGS.
    """
    if not callable(fig) or isinstance(fig, plt.Figure):
        raise TypeError("save() expects a function lang -> Figure, not an already drawn "
                        "figure, so that it can draw it once per language")
    stem = str(stem)
    if stem.startswith("/") or stem.endswith((".png", ".pdf")):
        raise ValueError(f"stem {stem!r}: must be relative and without extension")
    show = SHOW if show is None else show
    if show is not False:
        _check_lang(show)
    langs = LANGS if langs is None else tuple(langs)
    for lang in langs:
        _check_lang(lang)
    paths = {}
    for lang in langs:
        f = fig(lang)
        if not isinstance(f, plt.Figure):
            raise TypeError(f"the drawing function returned {type(f).__name__}, "
                            f"not a Figure")
        localize(f, lang)
        p = FIG_DIR / lang / f"{stem_for(stem, lang)}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        f.savefig(p, dpi=dpi, bbox_inches="tight")
        paths[lang] = p
        if lang != show:
            plt.close(f)
    print("  ".join(str(p.relative_to(FIG_DIR.parent)) for p in paths.values()))
    return paths
