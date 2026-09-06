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
import matplotlib.ticker as mticker
__version__ = "2.0"

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

# What each dataset letter is, for captions and for the assertion messages below.
# The chapter's old "A" was the single-physics run and is now D; the new A is a
# multi-physics rebuild. Anything that still assumes SINGLECONF == A is wrong.
DS_DESC = {
    "A": "4 km, 5 min DA, multi-physics (rebuild)",
    "B": "4 km, 1 h DA, multi-physics",
    "C": "2 km, 5 min DA, multi-physics",
    "D": "4 km, 5 min DA, single-physics",
}


def subset_path(hour, dataset, date="20240319"):
    """Path to a 3D subset npz: data/3D_subsets_{DS}/subset_{DS}_{stamp}.npz.

    `hour` is '18'..'21'; `dataset` is one of A/B/C/D. The hour is always a parameter
    -- no notebook hardcodes a valid time -- and so is the dataset, which used to be
    smuggled in as a directory stem plus a filename suffix.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    stamp = f"{date}{hour}0000"
    return str(DATA / f"3D_subsets_{dataset}" / f"subset_{dataset}_{stamp}.npz")


# ═════════════════════════════════════════════════════════════════════════════
# 1b  The analysis window, and the N2 diagnostic caches
# ═════════════════════════════════════════════════════════════════════════════

# ONE analysis window, used by every notebook and every section. A 5.5 deg square
# centred at 36.0 S. It lives here rather than in a notebook cell so that "inside the
# window" cannot mean two different boxes in two different figures.
WIN_LAT = (-38.75, -33.25)
WIN_LON = (-62.00, -56.50)

# The four datasets. A, B and C carry the chapter; D appears only in the
# physics-diversity contrast against A -- a fourth row for a secondary question would
# widen every panel in the main figures.
DS_MAIN = ("A", "B", "C")
DS_ALL = ("A", "B", "C", "D")

DS_ROLE = {
    "A": "main",
    "B": "isolates the DA cycle against A",
    "C": "isolates resolution against A",
    "D": "isolates physics diversity against A (secondary)",
}

N2_HOURS = ("18", "19", "20", "21")


def _n2_cache(kind, ds, hour):
    return DERIVED / f"n2_{kind}_{ds}_20240319{hour}0000.npz"


def n2_available(kind, ds, hours=N2_HOURS):
    return [h for h in hours if _n2_cache(kind, ds, h).exists()]


def identity_block(ds, hour="19"):
    """The identity a dataset's own file reports. Read, never assumed.

    This is also the material for the 4.2 paragraph on how each dataset was built:
    the upstream run, its DA cycle, its grid spacing and whether its physics were
    varied across members.
    """
    p = subset_path(hour, ds)
    with np.load(p, allow_pickle=False) as f:
        out = {}
        for k in ("dataset_id", "physics", "da_cycle_min", "dx_km", "upstream",
                  "source_run"):
            if k in f.files:
                v = f[k]
                out[k] = str(v) if v.dtype.kind in "US" else v.item()
            else:
                # "" rather than None: publish() sorts the table, and a None mixed with
                # strings in an object column raises on comparison in pandas.
                out[k] = ""
        ci = f["config_index"] if "config_index" in f.files else None
    out["config_index"] = ("not recorded" if ci is None or (ci < 0).all()
                           else f"{len(set(ci.tolist()))} configurations")
    out["file"] = os.path.basename(p)
    return out


def n2_departures(ds, hours=N2_HOURS):
    """4 · per-member departure statistics, the four hours CONCATENATED.

    Sums and counts are added across hours and divided at the end, which is what
    concatenating the four observation sets means. Averaging four per-hour means would
    instead weight an hour with few observations like an hour with many.

    Returns one row per candidate truth member with, for each of the echo and no-echo
    branches, the mean departure over the positive branch, the mean over the negative
    branch, and the fraction of observations of each sign. A single mean would conflate
    how strong the departures of one sign are with how many there are.

    NOT the same statistic as n2_loo_members(). This one is restricted to the OBSERVATION
    NETWORK -- at least one of the OTHER 59 members carrying echo, the set `filter_variance`
    builds -- and to reflectivity. n2_loo_members() covers every point where the 60
    members are not identical, which additionally includes the points where only the truth
    member carries echo, and it covers qtot and w as well. The numbers therefore differ,
    and both are kept: this one is the network the runs actually assimilate.
    """
    got = n2_available("depart", ds, hours)
    if not got:
        raise FileNotFoundError(f"no n2_depart cache for {ds}; run "
                                f"src/build_n2_diagnostics.py --ds {ds}")
    acc = None
    for h in got:
        with np.load(_n2_cache("depart", ds, h)) as f:
            d = {k: f[k] for k in f.files if not k.startswith("_")}
        acc = d if acc is None else {k: (acc[k] + d[k] if k != "ne_tot" else acc[k])
                                     for k in acc}
    rows = []
    ne_tot = int(acc["ne_tot"])
    for m in range(ne_tot):
        r = {"member": m, "hours": len(got)}
        for tag in ("echo", "clear"):
            n = acc[f"{tag}_n"][m]
            npos, nneg = acc[f"{tag}_npos"][m], acc[f"{tag}_nneg"][m]
            r[f"{tag}_n"] = int(n)
            # Both fractions, not one: they sum to 1 only where no departure is exactly
            # zero, and the residual is worth being able to see rather than assume away.
            r[f"{tag}_pos_frac"] = npos / n if n else np.nan
            r[f"{tag}_neg_frac"] = nneg / n if n else np.nan
            r[f"{tag}_zero_frac"] = (n - npos - nneg) / n if n else np.nan
            r[f"{tag}_pos_mean"] = acc[f"{tag}_spos"][m] / npos if npos else np.nan
            r[f"{tag}_neg_mean"] = acc[f"{tag}_sneg"][m] / nneg if nneg else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    # the magnitude axis of the 4 figure: how strong a departure is, regardless of
    # sign, over the echo branch -- the branch the censoring does not bound.
    df["echo_mean_mag"] = (df["echo_pos_frac"] * df["echo_pos_mean"].abs()
                           + (1 - df["echo_pos_frac"]) * df["echo_neg_mean"].abs())
    df["dataset"] = ds
    return df


def n2_pairs(ds, hours=N2_HOURS, offset=30):
    """5 · the offset-k pairing, on FULL fields and on ANOMALIES, with the sign.

    The two readings are opposite, so the answer is meaningless without saying which
    was computed:

      positive on anomalies -> true twins. 60 members are ~30 independent draws, any
                               n = 60 in a sign test is wrong, and the sampling noise
                               in the cross-covariances is a factor sqrt(2) larger
                               than the nominal size suggests.
      negative on anomalies -> antithetic pairs, a deliberate variance-reduction
                               technique. The ensemble spans the same subspace with
                               better-behaved moments and the effective size is NOT 30.

    The null for the anomaly correlation is not 0 but -1/(Ne-1): anomalies about the
    ensemble mean sum to zero, so the average off-diagonal correlation is forced
    slightly negative. It is reported alongside, or the baseline looks like a result.
    """
    got = n2_available("pairs", ds, hours)
    if not got:
        raise FileNotFoundError(f"no n2_pairs cache for {ds}")
    rows = []
    for h in got:
        with np.load(_n2_cache("pairs", ds, h)) as f:
            for tag in ("full", "anom"):
                C = f[f"corr_{tag}"]
                n = C.shape[0]
                pair = np.array([C[m, m + offset] for m in range(n - offset)])
                other = ~np.eye(n, dtype=bool)
                for m in range(n - offset):
                    other[m, m + offset] = other[m + offset, m] = False
                rows.append(dict(dataset=ds, hour=h, basis=tag, Ne=n,
                                 pair_mean=float(pair.mean()),
                                 pair_min=float(pair.min()),
                                 pair_max=float(pair.max()),
                                 other_mean=float(np.nanmean(C[other])),
                                 null=-1.0 / (n - 1) if tag == "anom" else np.nan,
                                 n_partner_is_closest=int(sum(
                                     1 for m in range(n)
                                     if int(np.argmax(np.where(np.eye(n, dtype=bool)[m],
                                                               -np.inf, C[m])))
                                     == (m + offset) % n))))
    return pd.DataFrame(rows)


def n2_pair_index(ds, hours=N2_HOURS, offset=30):
    """(pair_id, partner) per member, for stratifying a subsample over the 30 pairs.

    Quantiles taken over the 60 members would count each pair twice and over-weight
    whichever extreme a pair sits at; take them over the pairs instead.
    """
    ne = int(n2_pairs(ds, hours, offset)["Ne"].iloc[0])
    pair_id = np.array([m % offset for m in range(ne)])
    partner = np.array([(m + offset) % ne for m in range(ne)])
    return pair_id, partner


def n2_effective_members(ds, hours=N2_HOURS, offset=30, n_members=60):
    """How many INDEPENDENT truth members a dataset really has, from n2_pairs().

    Returns (n_eff, verdict, evidence). The verdict is read off the sign of the
    anomaly correlation of the offset-k pairs, because the two signs mean opposite
    things and only one of them costs the sign test half its n:

      "twins"      pair_mean > other_mean on ANOMALIES -> member m and m+offset are
                   near-copies. n_eff = offset: a test that uses all `n_members`
                   differences is counting each draw twice.
      "antithetic" pair_mean < the null -> a deliberate variance-reduction pairing.
                   The ensemble still spans the space with `n_members` distinct
                   draws, so n_eff = n_members.
      "none"       the pairs are indistinguishable from any other pair.
                   n_eff = n_members.

    Exists so that no notebook writes `n = 60` or `n = 30` as a literal: the number
    follows the pairing that N2 measured, and if a rebuilt ensemble ever breaks the
    pairing the sign tests downstream widen on their own.
    """
    t = n2_pairs(ds, hours, offset)
    a = t[t.basis == "anom"]
    if a.empty:
        raise ValueError(f"n2_pairs({ds!r}) returned no anomaly rows")
    pair, other = float(a["pair_mean"].mean()), float(a["other_mean"].mean())
    null = float(a["null"].mean())
    ne = int(a["Ne"].iloc[0])
    if ne != n_members:
        raise ValueError(
            f"dataset {ds}: n2_pairs reports {ne} members, the caller assumed "
            f"{n_members}. One of the two is looking at the wrong ensemble.")
    # The margin is against the OTHER-pair mean, not against zero: anomalies about the
    # ensemble mean are forced slightly negative (the null), so zero is not the
    # baseline and a test against it would call every ensemble antithetic.
    if pair > other + 0.05:
        verdict, n_eff = "twins", int(offset)
    elif pair < null - 0.05:
        verdict, n_eff = "antithetic", int(n_members)
    else:
        verdict, n_eff = "none", int(n_members)
    evidence = dict(dataset=ds, basis="anom", offset=int(offset), n_members=ne,
                    pair_mean=pair, other_mean=other, null=null,
                    partner_is_closest=f"{a['n_partner_is_closest'].mean():.2f}/{ne}",
                    verdict=verdict, n_eff=n_eff)
    return n_eff, verdict, evidence


def n2_nonfinite(ds, hours=N2_HOURS):
    """6 · where the non-finite cells are, per hour."""
    rows = []
    for h in n2_available("nonfinite", ds, hours):
        with np.load(_n2_cache("nonfinite", ds, h)) as f:
            r = dict(dataset=ds, hour=h)
            r["cells_any_var"] = int(f["n_cells_any_var"])
            r["cells_all_vars"] = int(f["n_cells_all_vars"])
            r["columns"] = int(f["n_columns"])
            r["full_columns"] = int(f["n_full_columns"])
            r["at_domain_edge"] = int(f["n_edge"])
            r["by_level"] = f["by_level"].tolist()
            r["per_var"] = {v: int(f[f"n_cells_{v}"]) for v in VARS}
            rows.append(r)
    return pd.DataFrame(rows)


def n2_state(ds, hour="19"):
    """2/3 · composite-variable moments and window histograms for one hour."""
    cf = _n2_cache("state", ds, hour)
    if not cf.exists():
        raise FileNotFoundError(f"no n2_state cache for {ds} {hour}Z")
    with np.load(cf) as f:
        return {k: f[k] for k in f.files if not k.startswith("_")}


# ── §8 · leave-one-out departures ────────────────────────────────────────────
N2_LOO_QUANTITIES = ("qtot", "w", "dbz")
N2_LOO_BRANCHES = ("echo", "clear", "all")
N2_LOO_SCOPES = ("dom", "win")
N2_LOO_LABEL = {"qtot": r"$q_g+q_r+q_s$", "w": "$w$", "dbz": "$Z$"}
N2_LOO_UNIT = {"qtot": "kg/kg", "w": "m/s", "dbz": "dBZ"}

# Only these keys are summed across the four hours: counts and sums concatenate, a mean
# or a skewness does not. The moments are merged separately, by _moment_merge.
_LOO_SUMMED = ("n_", "npos_", "nneg_", "spos_", "sneg_", "hist_", "below_", "above_",
               "ntot_", "lhpos_", "lhneg_", "lbelow_", "labove_", "lzero_")


def _n2_loo_check(ds, hour, d, ref=None):
    """Refuse to add two departure caches that are not the same statistic.

    Four hours are pooled by adding counts on a SHARED bin grid; if a rebuilt cache ever
    lands on a different grid, a different window or a different ensemble size, the sum
    is silently wrong rather than loudly absent. So it is checked instead of assumed.
    """
    if int(d["schema"]) != 1:
        raise ValueError(f"n2_loodep {ds} {hour}Z is schema {int(d['schema'])}; this "
                         f"reader knows schema 1")
    if str(d["dataset"]) != ds:
        raise ValueError(f"n2_loodep for {ds} {hour}Z says it is dataset "
                         f"{str(d['dataset'])} -- the file and the request disagree")
    if int(d["ne_tot"]) != 60:
        raise ValueError(
            f"n2_loodep {ds} {hour}Z has ne_tot={int(d['ne_tot'])}. The statistic is "
            f"'each of the 60 against the other 59'; a 59 here means the block ran "
            f"AFTER the truth-member hold-out and is measuring something else.")
    if ref is not None:
        for q in N2_LOO_QUANTITIES:
            if not np.array_equal(d[f"edges_{q}"], ref[f"edges_{q}"]):
                raise ValueError(f"{ds} {hour}Z bins {q} on a different grid; the hours "
                                 f"cannot be added")
        for k in ("win_i0", "win_i1", "win_j0", "win_j1"):
            if int(d[k]) != int(ref[k]):
                raise ValueError(f"{ds} {hour}Z has a different analysis window ({k})")
    p = subset_path(hour, ds)
    if os.path.isfile(p) and "_src_size" in d:
        st = os.stat(p)
        if int(d["_src_size"]) != st.st_size:
            warnings.warn(f"n2_loodep {ds} {hour}Z was built from a different "
                          f"{os.path.basename(p)}; rebuild with --force")


def _n2_loo_load(ds, hours):
    """The four hourly caches, checked against each other, with counts already summed."""
    got = n2_available("loodep", ds, hours)
    if not got:
        raise FileNotFoundError(f"no n2_loodep cache for {ds}; run "
                                f"src/build_n2_diagnostics.py --ds {ds}")
    acc, per_hour, ref = None, [], None
    for h in got:
        with np.load(_n2_cache("loodep", ds, h)) as f:
            d = {k: f[k] for k in f.files}
        _n2_loo_check(ds, h, d, ref)
        ref = ref or d
        per_hour.append((h, d))
        if acc is None:
            acc = dict(d)
        else:
            for k in acc:
                if k.startswith(_LOO_SUMMED):
                    acc[k] = acc[k] + d[k]
    return got, acc, per_hour


def _moment_merge(a, b):
    """Chan-Golub-LeVeque combination of two centred moment sets (n, mean, M2, M3, M4).

    The exact way to pool a skewness or an excess kurtosis across the four hours. Four
    per-hour skewnesses cannot be averaged: a skewness is a ratio of moments, and the
    mean of four ratios is the ratio of the pooled moments only when the four samples
    are the same size and the same distribution -- which four hours of a growing storm
    are precisely not.
    """
    if b["n"] == 0:
        return a
    if a["n"] == 0:
        return b
    na, nb_ = float(a["n"]), float(b["n"])
    n = na + nb_
    d = b["mean"] - a["mean"]
    d2 = d * d
    M2 = a["M2"] + b["M2"] + d2 * na * nb_ / n
    M3 = (a["M3"] + b["M3"] + d2 * d * na * nb_ * (na - nb_) / n ** 2
          + 3.0 * d * (na * b["M2"] - nb_ * a["M2"]) / n)
    M4 = (a["M4"] + b["M4"]
          + d2 * d2 * na * nb_ * (na * na - na * nb_ + nb_ * nb_) / n ** 3
          + 6.0 * d2 * (na * na * b["M2"] + nb_ * nb_ * a["M2"]) / n ** 2
          + 4.0 * d * (na * b["M3"] - nb_ * a["M3"]) / n)
    return dict(n=n, mean=a["mean"] + d * nb_ / n, M2=M2, M3=M3, M4=M4)


def n2_loo_members(ds, hours=N2_HOURS, scope="dom", quantities=N2_LOO_QUANTITIES,
                   branches=N2_LOO_BRANCHES):
    """8 · per-member leave-one-out departure statistics, the four hours CONCATENATED.

    One row per (candidate truth member, quantity, branch). Counts and sums are ADDED
    across the hours and divided once at the end, which is what concatenating four
    observation sets means; averaging four per-hour means would weight an hour with a
    handful of convective points like an hour full of them.

    The departure is d_m = x_m - mean(x_{k != m}) over all 60 members, taken before the
    truth member is held out, at every point where the 60 members are not all identical.
    Where they ARE identical every d_m is exactly zero and the point says nothing about
    any member, so it is dropped and counted (`n_dead`) rather than averaged in. For
    reflectivity that removes 42 % of the domain -- clear air sitting at the clamp; for
    qtot and w it removes almost nothing, because a WRF field is never bit-identical
    across 60 members.

    `branch` is set by the TRUTH member's reflectivity, for all three quantities alike:
    'echo' is h(x_m) > min_dbz, 'clear' its complement. For dbz the two are asymmetric BY
    CONSTRUCTION -- a clear truth gives d = -(60/59) * mean(prior) <= 0, so the positive
    branch is empty -- and pooling them buries a censoring artefact inside what looks
    like a property of the member.

    `scope` is 'dom' (the whole subset) or 'win' (the analysis window the rest of N2
    uses). Both live in the same file: this is a switch, not a rebuild.

    Columns: dataset, member, quantity, branch, scope, hours, n, n_pos, n_neg, n_zero,
    pos_frac, neg_frac, zero_frac, pos_mean, neg_mean, mean, mean_abs. `pos_mean` and
    `neg_mean` are the two rows of the §8 figure. `mean_abs = (spos - sneg) / n` is the
    mean of |d|, exact even when some departures are exactly zero -- unlike a
    pos_frac*|pos_mean| + (1 - pos_frac)*|neg_mean| reconstruction, which quietly assumes
    none are.
    """
    got, acc, _ = _n2_loo_load(ds, hours)
    ne = int(acc["ne_tot"])
    rows = []
    for q in quantities:
        for b in branches:
            k = f"{q}_{b}_{scope}"
            n, npos, nneg = acc[f"n_{k}"], acc[f"npos_{k}"], acc[f"nneg_{k}"]
            spos, sneg = acc[f"spos_{k}"], acc[f"sneg_{k}"]
            for m in range(ne):
                nm, npm, nnm = int(n[m]), int(npos[m]), int(nneg[m])
                rows.append(dict(
                    dataset=ds, member=m, quantity=q, branch=b, scope=scope,
                    hours=len(got), n=nm, n_pos=npm, n_neg=nnm,
                    n_zero=nm - npm - nnm,
                    pos_frac=npm / nm if nm else np.nan,
                    neg_frac=nnm / nm if nm else np.nan,
                    zero_frac=(nm - npm - nnm) / nm if nm else np.nan,
                    pos_mean=spos[m] / npm if npm else np.nan,
                    neg_mean=sneg[m] / nnm if nnm else np.nan,
                    mean=(spos[m] + sneg[m]) / nm if nm else np.nan,
                    mean_abs=(spos[m] - sneg[m]) / nm if nm else np.nan))
    return pd.DataFrame(rows)


def n2_loo_hist(ds, quantity, hours=N2_HOURS, branch="all", scope="dom"):
    """8 · the pooled departure distribution, the four hours CONCATENATED.

    Counts add across the hours because they are counts on the SAME fixed edges -- the
    grid is a module constant of the builder, identical in every dataset and every hour,
    which is what lets four files be summed without a second pass over 50 GB.

    Returns the linear histogram with its two out-of-range tails (so the counts always
    add up), the dropped zero-spread and non-finite point counts, and the shape numbers
    -- mean, sd, skewness, excess kurtosis -- pooled EXACTLY through the centred power
    sums rather than by averaging four per-hour skewnesses. The skewness and kurtosis
    normalisations are Nerger (2022) Eqs. 25-26, the same ones ensemble_skew and
    ensemble_kurt use, so a pooled number is comparable with a per-point field.

    Percentiles come back two ways, because they are the one statistic that does not
    pool: `pct_by_hour` is exact, one row per hour; `pct` is the four-hour value read off
    the summed histogram with hist_quantile, accurate to `pct_tol`, the bin width.

    `lhist_pos` / `lhist_neg` bin log10|d| by sign. For qtot the linear grid cannot carry
    the distribution at all -- its quartiles sit at -9e-6, -3e-7 and -5e-10 kg/kg while
    the top 0.1 % reaches +3e-3, seven decades apart, so three quartiles land in the one
    bin at the origin. `quantum` is the float32 resolution of the subtraction on this
    branch: a percentile below it would be reporting the storage grid of the subset file
    rather than the ensemble.
    """
    got, acc, per_hour = _n2_loo_load(ds, hours)
    k = f"{quantity}_{branch}_{scope}"
    edges = acc[f"edges_{quantity}"]
    mom = {"n": 0.0, "mean": 0.0, "M2": 0.0, "M3": 0.0, "M4": 0.0}
    for _, d in per_hour:
        mom = _moment_merge(mom, {"n": float(d[f"ntot_{k}"]), "mean": float(d[f"mean_{k}"]),
                                  "M2": float(d[f"M2_{k}"]), "M3": float(d[f"M3_{k}"]),
                                  "M4": float(d[f"M4_{k}"])})
    n = mom["n"]
    var = mom["M2"] / (n - 1) if n > 1 else np.nan
    counts = acc[f"hist_{k}"]
    out = dict(
        dataset=ds, quantity=quantity, branch=branch, scope=scope, hours=len(got),
        counts=counts, edges=edges,
        below=int(acc[f"below_{k}"]), above=int(acc[f"above_{k}"]),
        n=int(acc[f"ntot_{k}"]),
        n_live=int(acc[f"nlive_{quantity}_{scope}"]),
        n_dead=int(acc[f"ndead_{quantity}_{scope}"]),
        n_bad=int(acc[f"nbad_{quantity}_{scope}"]),
        n_points=int(acc[f"npts_{scope}"]),
        mean=mom["mean"], std=np.sqrt(var), M2=mom["M2"], M3=mom["M3"], M4=mom["M4"],
        skew=(mom["M3"] / n) / var ** 1.5 if n > 1 else np.nan,
        exkurt=(mom["M4"] / n) / (mom["M2"] / n) ** 2 - 3.0 if n > 1 else np.nan,
        quantum=float(acc[f"quantum_{k}"]),
        pct_q=acc["pct_q"],
        pct=np.array([hist_quantile(counts, edges, float(p) / 100.0) for p in acc["pct_q"]]),
        pct_tol=float(np.diff(edges)[0]),
        pct_by_hour=pd.DataFrame([dict(hour=h, **{f"q{p:g}": v for p, v in
                                                  zip(d["pct_q"], d[f"pct_{k}"])})
                                  for h, d in per_hour]),
        lhist_pos=acc[f"lhpos_{quantity}_{branch}_{scope}"] if branch == "all" else None,
        lhist_neg=acc[f"lhneg_{quantity}_{branch}_{scope}"] if branch == "all" else None,
        lmag_edges=acc["lmag_edges"],
        l_zero=int(acc[f"lzero_{k}"]) if branch == "all" else None,
    )
    assert out["counts"].sum() + out["below"] + out["above"] == out["n"], \
        f"{ds} {k}: histogram lost points"
    return out


def n2_loo_shape(ds_list, quantities=N2_LOO_QUANTITIES, hours=N2_HOURS,
                 branches=("all",), scope="dom"):
    """The table beside the §7 distribution figure: the numbers a histogram cannot carry.

    Thin loop over n2_loo_hist. The standardised excess kurtosis of the same variable is
    carried alongside where n2_state has it, because the raw pooled departure mixes
    points of different ensemble spread -- a mixture of Gaussians of different widths is
    leptokurtic even when every point is perfectly Gaussian, so the raw number alone
    cannot separate the mixture from the shape.
    """
    rows = []
    for ds in np.atleast_1d(ds_list):
        st = n2_state(ds, hours[1]) if quantities else None
        for q in quantities:
            for b in branches:
                h = n2_loo_hist(ds, q, hours=hours, branch=b, scope=scope)
                rows.append(dict(
                    dataset=ds, quantity=q, branch=b, n=h["n"], n_dead=h["n_dead"],
                    mean=h["mean"], sd=h["std"], skew=h["skew"], exkurt=h["exkurt"],
                    exkurt_std=(float(st[f"pkurt_{q}"]) if st is not None
                                and f"pkurt_{q}" in st else np.nan),
                    quantum=h["quantum"], pct_tol=h["pct_tol"],
                    q01=h["pct"][0], q50=h["pct"][4], q999=h["pct"][-1]))
    return pd.DataFrame(rows)


def win_box(ax, lat=None, lon=None, **kw):
    """Outline the analysis window on a map axis.

    Every figure that reports something "inside the window" draws the same box from the
    same two constants, so a reader can see that the box in one panel is the box in the
    next.
    """
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    lat = WIN_LAT if lat is None else lat
    lon = WIN_LON if lon is None else lon
    style = dict(edgecolor="k", facecolor="none", lw=0.8, ls="--", zorder=5)
    style.update(kw)
    ax.add_patch(mpatches.Rectangle(
        (min(lon), min(lat)), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]),
        transform=ccrs.PlateCarree(), **style))
    return ax


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
    dataset and quietly changing a published number. Files written before the
    identity backfill carry no `dataset_id` and are passed through with a warning
    rather than an error -- there is nothing to check them against.
    """
    got = dataset_of(path)
    if got is None:
        warnings.warn(
            f"{os.path.basename(path)} carries no dataset_id; it predates the identity "
            f"backfill (src/backfill_identity.py). Believing the caller: {expected}.")
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
# 2  Figure style and output
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
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
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
# 3  Palettes, labels, variable tables
# ═════════════════════════════════════════════════════════════════════════════

# The one dBZ scale. Of the four colormaps in use across the old notebooks this is
# the only properly stepped radar scale, and the only one paired with a BoundaryNorm.
DBZ_LEVELS = np.arange(0, 66, 5)
DBZ_CMAP = "Spectral_r"
DBZ_NORM = BoundaryNorm(DBZ_LEVELS, plt.get_cmap(DBZ_CMAP).N)

# CVD-validated sequential + diverging maps.
C_NT2, C_NT1, C_NONE, C_PRIOR = "#8B0000", "#008080", "#FF8C00", "#4682B4"
_SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)
CMAP_SEQ.set_bad("#ffffff", alpha=0.0)
# THE diverging map, for skill and for increments alike: RdBu_r, RED AT POSITIVE.
# It is the stock map rather than a hand-rolled one because scales() and sym_scale()
# already return RdBu_r for wind and increment panels -- two diverging maps in one
# chapter is how a reader learns that colour carries no fixed meaning. Left as a name,
# not a Colormap instance: mutating the registered instance (set_bad) would change it
# for every other user of "RdBu_r" in the process, and its default bad colour is
# already fully transparent, which is what the NaN cells need.
CMAP_DIV = "RdBu_r"
# CMAP_SKILL: red = positive = better. Used with skill() output and nothing else.
CMAP_SKILL = CMAP_DIV

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
         "T": "K", "P": "hPa", "u": "m s$^{-1}$", "v": "m s$^{-1}$",
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
    if v == "P":
        return np.asarray(x) / 1e2
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
# 4  Physics
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

# The three evaluation domains, likewise. The runner reduces the light-mode scalars over
# exactly these masks, so a local copy here would let a notebook's field-recomputed
# number and the scalar on disk drift apart without either being obviously wrong.
from da.metrics import STORM_THRESH_DBZ, domain_masks  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# 5  Prior-ensemble access  (N1, N2)
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


def _src_stamp(path):
    """Size and mtime of the source npz, stored inside every cache file."""
    st = os.stat(path)
    return {"_src_size": np.int64(st.st_size), "_src_mtime": np.float64(st.st_mtime)}


def _load_cached(cf, path, verbose=True):
    """Cached dict, or None if there is no cache or its source has since changed.

    The cache key is built from the *path*, so re-extracting a subset in place --
    which N1 does whenever a job is re-run -- would otherwise keep serving the
    statistics of the file that used to be there. Caches written before this stamp
    existed carry no `_src_*` keys and are trusted, as they always were.
    """
    if not cf.exists():
        return None
    with np.load(cf) as f:
        d = {k: f[k] for k in f.files}
    size, mtime = d.pop("_src_size", None), d.pop("_src_mtime", None)
    if size is not None:
        st = os.stat(path)
        if int(size) != st.st_size or abs(float(mtime) - st.st_mtime) > 1.0:
            if verbose:
                print(f"  {cf.name}: source changed since it was cached; recomputing")
            return None
    return d


def _save_cached(cf, out, path):
    np.savez_compressed(cf, **out, **_src_stamp(path))


def _cache_stem(path):
    """The part of a cache filename that names its source.

    The subset FILE stem, not its parent directory. Under the old layout the parent
    directory carried the hour (`3D_subsets_20240319_GUES_SINGLECONF_20240319180000`);
    under the dataset-first layout it is just `3D_subsets_D`, so keying on it would
    leave four hours of caches distinguishable only by their hashes.
    """
    return os.path.splitext(os.path.basename(path))[0]


def _cache_key(path, bbox, min_dbz, truth_member=None):
    b = "full" if bbox is None else f"{bbox[0].start}_{bbox[0].stop}_{bbox[1].start}_{bbox[1].stop}"
    # truth_member is part of the key: a 59-member prior and a 60-member ensemble are
    # different statistics and must never be served from each other's cache. The
    # tm=None form keeps the old key exactly, so every cache written before this
    # parameter existed still hits.
    tm = "" if truth_member is None else f"|tm{int(truth_member)}"
    h = hashlib.sha1(f"{os.path.abspath(path)}|{b}|{min_dbz}{tm}".encode()).hexdigest()[:12]
    suffix = "" if truth_member is None else f"_tm{int(truth_member):02d}"
    return DERIVED / f"prior_{_cache_stem(path)}_{b}{suffix}_{h}.npz"


def prior_stats(path, bbox=None, min_dbz=0.0, cache=True, verbose=True, dataset=None,
                truth_member=None):
    """N2's workhorse: prior ensemble statistics from one subset file.

    Returns a small dict of (nx, ny, nz) fields -- dbz_mean, dbz_spread, n_active,
    dbz_skew, dbz_kurt -- plus (nx, ny) column-max per member and per-variable
    mean/spread. Cached to data/derived/, because every uncached call pays a full
    multi-GB decompress.

    `truth_member` holds one member out, so what comes back is THE PRIOR the
    assimilation actually saw: Ne = 59, and n_active can be at most 59. With
    truth_member=None every member is used and n_active reaches 60 -- which is an
    ensemble statistic, not a prior one. N2 passes 0, the truth member every
    multi-obs run in this repository was produced with.
    """
    if dataset is not None:
        assert_dataset(path, dataset)
    cf = _cache_key(path, bbox, min_dbz, truth_member)
    if cache:
        hit = _load_cached(cf, path, verbose=verbose)
        if hit is not None:
            return hit

    if verbose:
        print(f"computing prior_stats for {os.path.basename(os.path.dirname(path))} "
              f"(uncached; decompressing) ...")
    ens = load_ensemble(path, bbox=bbox)            # (nx, ny, nz, Ne_tot, 8)
    if truth_member is not None:
        keep = [m for m in range(ens.shape[3]) if m != int(truth_member)]
        ens = ens[:, :, :, keep, :]
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
        "truth_member": np.int32(-1 if truth_member is None else truth_member),
    }
    for v in VARS:
        out[f"mean_{v}"] = ens[..., VI[v]].mean(axis=3).astype(np.float32)
        out[f"spread_{v}"] = ens[..., VI[v]].std(axis=3, ddof=1).astype(np.float32)
    del ens, dbz

    if cache:
        _save_cached(cf, out, path)
        if verbose:
            print(f"  cached -> {cf.name}")
    return out


def section_stats(path, j, min_dbz=0.0, cache=True, verbose=True, dataset=None):
    """Per-member reflectivity on one j row -> (nx, nz, Ne), plus its moments.

    prior_stats keeps the members only after the column max is taken, so a vertical
    section can be drawn from the ensemble mean but not from what the members
    actually produce. This pays one more decompress for a single row and keeps
    everything: (nx, nz, Ne) is ~0.8 MB for a 307 x 11 x 60 subset, so the whole
    per-member section fits in the cache and any further statistic is free.

    Returns dbz_members plus dbz_mean / dbz_max / dbz_spread / n_active /
    dbz_skew / dbz_kurt, all (nx, nz), and the state variables' row mean/spread.
    """
    if dataset is not None:
        assert_dataset(path, dataset)
    h = hashlib.sha1(f"{os.path.abspath(path)}|{int(j)}|{min_dbz}".encode()).hexdigest()[:12]
    cf = DERIVED / f"section_{_cache_stem(path)}_j{int(j)}_{h}.npz"
    if cache:
        hit = _load_cached(cf, path, verbose=verbose)
        if hit is not None:
            return hit

    if verbose:
        print(f"computing section_stats j={j} for "
              f"{os.path.basename(os.path.dirname(path))} (uncached; decompressing) ...")
    # The npz inflates whole either way; the row slice is what stays resident.
    ens = load_ensemble(path, bbox=(slice(None), slice(int(j), int(j) + 1)))
    ens = ens[:, 0]                                          # (nx, nz, Ne, 8)
    dbz = hx(ens, min_dbz=min_dbz).astype(np.float32)        # (nx, nz, Ne)

    out = {
        "dbz_members": dbz,
        "dbz_mean": dbz.mean(axis=2).astype(np.float32),
        "dbz_max": dbz.max(axis=2).astype(np.float32),
        "dbz_spread": dbz.std(axis=2, ddof=1).astype(np.float32),
        "n_active": (dbz > min_dbz).sum(axis=2).astype(np.int16),
        "dbz_skew": ensemble_skew(dbz, axis=2).astype(np.float32),
        "dbz_kurt": ensemble_kurt(dbz, axis=2).astype(np.float32),
        "j": np.int32(j),
        "Ne": np.int32(ens.shape[2]),
        "min_dbz": np.float32(min_dbz),
    }
    for v in VARS:
        out[f"mean_{v}"] = ens[..., VI[v]].mean(axis=2).astype(np.float32)
        out[f"spread_{v}"] = ens[..., VI[v]].std(axis=2, ddof=1).astype(np.float32)
    del ens, dbz

    if cache:
        _save_cached(cf, out, path)
        if verbose:
            print(f"  cached -> {cf.name}")
    return out


# The reflectivity histogram grid. hx() clamps at min_dbz, so min_dbz is the floor
# and everything below it piles into the first bin; 0.25 dBZ is fine enough that any
# coarser binning, quantile or exceedance fraction can be taken from the counts.
DBZ_HIST_EDGES = np.arange(0.0, 80.0 + 0.25, 0.25)


def dbz_histogram(path, edges=None, min_dbz=0.0, bbox=None, cache=True, verbose=True, dataset=None):
    """Counts of every per-member reflectivity value in one subset.

    prior_stats keeps moments and column maxima, so the *distribution* of H(x) over
    the full (nx, ny, nz, Ne) field cannot be recovered from its cache. This pays
    one more decompress and keeps only the counts: ~3 kB per file, against the
    1.3 GB the reflectivity field itself would need, and enough to draw any
    histogram, quantile or exceedance fraction afterwards.

    Returns counts (len(edges) - 1), edges, n_total, n_below and n_above -- the two
    tails being values outside the binned range, so the counts always add up.
    """
    if dataset is not None:
        assert_dataset(path, dataset)
    edges = DBZ_HIST_EDGES if edges is None else np.asarray(edges, float)
    b = "full" if bbox is None else f"{bbox[0].start}_{bbox[0].stop}_{bbox[1].start}_{bbox[1].stop}"
    h = hashlib.sha1(f"{os.path.abspath(path)}|{b}|{min_dbz}|{edges[0]}|{edges[-1]}|"
                     f"{len(edges)}".encode()).hexdigest()[:12]
    cf = DERIVED / f"dbzhist_{_cache_stem(path)}_{b}_{h}.npz"
    if cache:
        hit = _load_cached(cf, path, verbose=verbose)
        if hit is not None:
            return hit

    if verbose:
        print(f"computing dbz_histogram for {os.path.basename(os.path.dirname(path))} "
              f"(uncached; decompressing) ...")
    ens = load_ensemble(path, bbox=bbox)
    dbz = hx(ens, min_dbz=min_dbz).astype(np.float32)
    del ens

    counts, _ = np.histogram(dbz, bins=edges)
    out = {
        "counts": counts.astype(np.int64),
        "edges": edges,
        "n_total": np.int64(dbz.size),
        "n_below": np.int64((dbz < edges[0]).sum()),
        "n_above": np.int64((dbz > edges[-1]).sum()),
        "Ne": np.int32(dbz.shape[-1]),
        "min_dbz": np.float32(min_dbz),
    }
    del dbz
    assert int(out["counts"].sum()) + int(out["n_below"]) + int(out["n_above"]) \
        == int(out["n_total"]), "histogram lost points"

    if cache:
        _save_cached(cf, out, path)
        if verbose:
            print(f"  cached -> {cf.name}")
    return out


def hist_rebin(counts, edges, factor):
    """Coarsen a histogram by an integer factor. Returns (counts, edges)."""
    factor = int(factor)
    n = (len(counts) // factor) * factor
    return (np.asarray(counts)[:n].reshape(-1, factor).sum(axis=1),
            np.asarray(edges)[:n + 1:factor])


def hist_quantile(counts, edges, q, floor=None):
    """q-quantile of a binned sample, linearly interpolated inside the bin.

    `floor` restricts the sample to values at or above it, which is how the
    thresholded distributions in N2 are summarised: the quantile is then over echo
    values only, not over a sample dominated by clamped clear air.
    """
    counts, edges = np.asarray(counts, float), np.asarray(edges, float)
    if floor is not None:
        counts = np.where(edges[:-1] >= floor, counts, 0.0)
    tot = counts.sum()
    if tot == 0:
        return np.nan
    cum = np.cumsum(counts)
    target = np.atleast_1d(q) * tot
    out = []
    for t in target:
        k = int(np.searchsorted(cum, t, side="left"))
        k = min(k, len(counts) - 1)
        below = cum[k] - counts[k]
        frac = (t - below) / counts[k] if counts[k] else 0.0
        out.append(edges[k] + frac * (edges[k + 1] - edges[k]))
    return out[0] if np.isscalar(q) else np.array(out)


def hist_frac_above(counts, edges, thresh, n_total=None, n_above=0):
    """Fraction of the whole sample at or above `thresh`."""
    counts, edges = np.asarray(counts, float), np.asarray(edges, float)
    tot = float(n_total) if n_total is not None else counts.sum() + float(n_above)
    return float((counts[edges[:-1] >= thresh].sum() + float(n_above)) / tot)


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
# 6  Sweep loading and row alignment
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
            # A mis-migrated run is caught here rather than being pooled into the wrong
            # dataset and quietly moving a published number.
            if s.get("ds") is not None:
                assert_dataset(f, s["ds"])
            df = load_sweep(f, columns=columns, methods=s.get("methods"), verbose=verbose)
            base = os.path.basename(f)
            df["run"] = s["tag"]
            # NOT "loc": that name collides with the pandas .loc indexer, so
            # df.loc would silently be the indexer rather than the column.
            df["loc_km"] = float(s.get("loc", df["lx_km"].iloc[0]))
            df["hour"] = s.get("hour", "")
            if s.get("ds") is not None:
                df["ds"] = s["ds"]
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


def _combo_list(x):
    return list(x.combos) if isinstance(x, Aligned) else list(x)


def combo_key(combos, method, ntemp=None, loc=None):
    """The 6-tuple for a (method, ntemp[, loc]) pair.

    Replaces the list comprehension every N3/N4 cell used to open-code, which
    silently took [0] of the matches and so picked an arbitrary combo whenever the
    selection was ambiguous. Here an ambiguous selection raises.
    """
    cands = [k for k in _combo_list(combos)
             if k[0] == method
             and (ntemp is None or int(k[1]) == int(ntemp))
             and (loc is None or np.isclose(float(k[3]), float(loc)))]
    if not cands:
        raise KeyError(f"no combo method={method!r} ntemp={ntemp} loc={loc}; "
                       f"present: {_combo_list(combos)}")
    if len(cands) > 1:
        raise KeyError(f"combo method={method!r} ntemp={ntemp} loc={loc} is ambiguous: "
                       f"{cands}. Pass loc= to disambiguate.")
    return cands[0]


def combo_order(combos):
    """Presentation order: TEnKF by increasing ntemp first, then the other methods.

    The plain sorted() order of combos_in puts AOEI before TEnKF and so breaks the
    'no tempering -> more tempering' reading of every axis it labels.
    """
    return sorted(_combo_list(combos),
                  key=lambda k: (0 if k[0] == "TEnKF" else 1, k[0],
                                 int(k[1]), float(k[3])))


def combo_slug(combo):
    """Filename/column-safe tag for a combo, e.g. 'TEnKF_Nt2', 'AOEI_Nt1'."""
    return f"{combo[0]}_Nt{int(combo[1])}"


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
# 7  THE convention
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


def skill_summary_by_tm(aligned, metric="rmse", var="obs", red="w"):
    """skill_summary, split by truth member.

    Same skill() call and the same tie tolerance, so the pooled table is this table
    aggregated over tm -- the two cannot drift apart.

    Why it exists: one truth member is one draw of the OSSE -- a different truth, a
    different prior (leave-one-out) and a different observation-noise realisation.
    Wilson intervals treat the ~1e6 spatially correlated points of a single member as
    independent samples, so they shrink as 1/sqrt(points) and understate how much the
    answer moves when the experiment is redrawn. The spread of these per-member
    numbers does not shrink with points and is the honest error bar, exactly as
    METHODOLOGY_CHECKLIST.md sec.9 requires. A difference between two update schemes
    smaller than the across-tm spread is not a result.

    Requires the 'tm' column that load_runs adds.
    """
    tol = TOL[var]
    rows = []
    for c in sorted(aligned.combos):
        f = aligned.frames[c]
        if "tm" not in f.columns:
            raise ColumnError(
                "frames carry no 'tm' column -- skill_summary_by_tm needs the truth "
                "member, which load_runs adds from the *_tmNN.npz filename.")
        s = skill(f, metric=metric, var=var, red=red).to_numpy(float)
        tm = f["tm"].to_numpy()
        good = np.isfinite(s)
        for t in np.unique(tm):
            in_tm = tm == t
            sv = s[in_tm & good]
            n = int(sv.size)
            rows.append({
                "run": str(f["run"].iloc[0]) if "run" in f else "",
                "loc_km": float(f["loc_km"].iloc[0]) if "loc_km" in f else float(c[3]),
                "method": c[0], "ntemp": int(c[1]), "alpha_s": float(c[2]),
                "metric": metric, "var": var, "red": red, "tm": int(t),
                "n_points": n, "n_nan": int((in_tm & ~good).sum()),
                "median_skill": float(np.median(sv)) if n else np.nan,
                "mean_skill": float(np.mean(sv)) if n else np.nan,
                "frac_improved": float((sv > tol).mean()) if n else np.nan,
                "frac_degraded": float((sv < -tol).mean()) if n else np.nan,
            })
    return pd.DataFrame(rows).sort_values(
        ["metric", "var", "red", "method", "ntemp", "tm"]).reset_index(drop=True)


_DIGEST_COLS = ["run", "loc_km", "method", "ntemp", "alpha_s", "metric", "var", "red",
                "n_points", "n_nan", "median_skill", "mean_skill",
                "frac_improved", "frac_degraded"]


def _canonical(df, cols=None):
    """Project a table onto the columns that define it, rounded and row-sorted.

    `cols=None` keeps the sweep's column set: that is what every N3 table is, and
    changing it would move digests that are pasted into other notebooks. A table
    that is NOT a sweep row -- the multi-obs tables of N4 are keyed by dataset,
    hour and domain, none of which is in _DIGEST_COLS -- passes its own columns
    instead, so publishing does not silently drop the columns it exists to carry.
    """
    keep = list(cols) if cols is not None else _DIGEST_COLS
    if cols is not None:
        # An explicit list is a promise about the table; a typo in it must not turn
        # into a quietly narrower CSV.
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise ColumnError(f"publish(cols=...) names columns the table does not "
                              f"have: {missing}")
    d = df[[c for c in keep if c in df.columns]].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            # round to ~6 significant digits: float32 accumulation order varies with
            # row order, and too tight a digest cries wolf
            d[c] = d[c].map(lambda v: float(f"{v:.6g}") if np.isfinite(v) else np.nan)
    return d.sort_values(list(d.columns)).reset_index(drop=True)


def consistency_digest(df, cols=None):
    return hashlib.sha1(_canonical(df, cols).to_csv(index=False).encode()).hexdigest()[:16]


def publish(name, df, cols=None, verbose=True):
    """Persist a canonical table and return its digest (the string N4 pastes).

    `cols` names the columns to keep; see _canonical. Pass list(df.columns) for a
    table that is not a sweep row.
    """
    p = DERIVED / f"{name}.csv"
    out = _canonical(df, cols)
    out.to_csv(p, index=False)
    dig = consistency_digest(df, cols)
    if verbose:
        print(f"published {p.name}  ({len(out):,} rows x {out.shape[1]} cols)  "
              f"digest={dig}")
    return dig


def expect(name, df, digest=None, cols=None):
    """Two independent checks: against the pasted digest, and against the stored CSV.

    Raises ConsistencyError carrying a row-aligned .diff.
    """
    got = consistency_digest(df, cols)
    p = DERIVED / f"{name}.csv"
    if not p.exists():
        raise ConsistencyError(f"{p.name} not found -- run the notebook that publishes "
                               f"'{name}' first.")
    ref = pd.read_csv(p)
    cur = _canonical(df, cols)
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
# 8  Predictors  (N3)
# ═════════════════════════════════════════════════════════════════════════════

# (column, axis label, percentile clip)
PREDICTORS = [
    ("n_active_f_point", "members with signal", (0, 100)),
    ("frac_floor", "fraction of members at the floor", (0, 100)),
    ("spread_f_point_obs", r"prior spread $\sigma_H$", (0.5, 99.5)),
    ("abs_skew_f", r"prior $|$skewness$|$", (0.5, 99)),
    ("skew_f_point_obs", r"prior skewness", (0.5, 99)),
    ("kurt_f_point_obs", "prior excess kurtosis", (0.5, 99)),
    ("abs_dep_b", r"$|$innovation$|$", (0.5, 99.5)),
    ("dep_b", r"innovation", (0.5, 99.5)),
    ("norm_innov", r"normalised innovation", (0, 99.5)),
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
# 9  Binning and density primitives
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


def prob_hexbin(ax, x, y, mask, xlim, ylim, gridsize=45, mincnt=1, **kw):
    """P(mask) per hexagonal bin, on a 0..1 diverging scale."""
    return ax.hexbin(x, y, C=np.asarray(mask, float), reduce_C_function=np.mean,
                     gridsize=gridsize, cmap="RdBu_r", vmin=0.0, vmax=1.0,
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
# 10  Maps and 3-D projections
# ═════════════════════════════════════════════════════════════════════════════

AOI = dict(lat=(-41.5, -25.3), lon=(-68.6, -55.4))


RASTER_ZORDER = 2.5

def setup_map(ax, extent=None, labels=True, xlocs=None, ylocs=None,
              left_labels=True, bottom_labels=True, rasterize_below=RASTER_ZORDER):
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
    gl = ax.gridlines(draw_labels=labels, linestyle="--", alpha=0.4,
                      color="gray", linewidth=0.4)
    gl.top_labels = gl.right_labels = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    if xlocs is not None:
        gl.xlocator = mticker.FixedLocator(xlocs)
    if ylocs is not None:
        gl.ylocator = mticker.FixedLocator(ylocs)
    gl.xlabel_style = {"size": 6, "rotation": 45, "ha": "right"}
    gl.ylabel_style = {"size": 6}
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
# 11  Multi-obs, guarded and optional
# ═════════════════════════════════════════════════════════════════════════════

def has_multiobs(tag):
    d = run_dir(tag)
    return d.is_dir() and any(d.glob("*_multi_obs_*_Ne*_tm*.npz"))


def run_dataset(tag, expected=None):
    """The dataset_id the files of a run actually carry, checked for agreement.

    N4 reads multi-obs files directly rather than through load_runs, so this is its
    hook: one call per run says whether every file in that directory agrees with itself
    and with what the notebook believed.
    """
    found = {}
    for p in sorted(run_dir(tag).glob("*_Ne*_tm*.npz")):
        found.setdefault(dataset_of(p), []).append(os.path.basename(p))
    if len(found) > 1:
        raise ValueError(f"{tag}: its files disagree about which dataset they are: " +
                         ", ".join(f"{k}({len(v)} files)" for k, v in found.items()))
    got = next(iter(found), None)
    if got is None:
        return None
    if expected is not None and got != expected:
        raise ValueError(f"{tag}: the notebook expected dataset {expected} "
                         f"({DS_DESC.get(expected, '?')}) but its files say {got} "
                         f"({DS_DESC.get(got, '?')}).")
    return got


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


# domain_masks now comes from da.metrics (imported above). It used to be defined here
# with the same body; the runner needs it too, and two copies of the definition that
# decides which cells a published number was averaged over is one copy too many.
