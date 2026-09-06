"""
src/build_n2_diagnostics.py
===========================
Everything N2 needs that costs a full decompress, computed in ONE pass per
(dataset, hour) and cached to data/derived/.

    python src/build_n2_diagnostics.py            # all datasets, all hours
    python src/build_n2_diagnostics.py --ds A     # one dataset
    python src/build_n2_diagnostics.py --force    # ignore existing caches

A subset npz is DEFLATE, so numpy inflates the whole array (2.9 GB for the 4 km
datasets, 10.6 GB for C) before anything can be sliced. Calling prior_stats, then a
departures routine, then a composite-moments routine would pay that three times over.
This pays it once and writes:

  prior_subset_{DS}_{stamp}_full_tm00_{h}.npz   the 59-member prior, in prior_stats'
                                                own cache format, so nb.prior_stats
                                                serves it without recomputing
  n2_depart_{DS}_{stamp}.npz     §4  per-member departure sums, for the 4-hour pooling
  n2_state_{DS}_{stamp}.npz      §2/§3  composite-variable moments and histograms
  n2_pairs_{DS}_{stamp}.npz      §5  member-by-member correlation, full and anomaly
  n2_nonfinite_{DS}_{stamp}.npz  §6  where the non-finite cells are

The §4 sums are stored per hour and NOT averaged here: the four hours are concatenated,
which is a sum of counts and a sum of values, not a mean of means.
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Notebooks"))
sys.path.insert(0, str(REPO / "src"))

import nbcommon as nb   # noqa: E402

DATASETS = ("A", "B", "C", "D")
HOURS = ("18", "19", "20", "21")
MIN_DBZ = 0.0
TRUTH_MEMBER = 0          # the member every multi-obs run in this repo held out

# The analysis window: one 5.5 deg square centred at 36.0 S, used by every section.
WIN_LAT = (-38.75, -33.25)
WIN_LON = (-62.00, -56.50)

# §2/§3 composite variables. NOT sqrt(u^2+v^2): that is a nonlinear transform, so its
# distribution is Rice-skewed even when u and v are jointly Gaussian, and its skewness
# would mix the transform's contribution with the ensemble's. Summing hydrometeors is
# linear, so qtot carries no such artefact.
COMPOSITES = ("qtot", "w", "u", "v")

# Fixed, generous histogram edges so every dataset lands on the SAME bins and the
# curves are comparable without a second pass over 50 GB. Fine enough that
# nb.hist_rebin can coarsen them afterwards; the tails outside are counted separately,
# so the counts always add up.
HIST_EDGES = {
    "qtot": np.linspace(0.0, 5.0e-3, 5001),    # kg/kg, 1e-6 bins
    "w":    np.linspace(-30.0, 30.0, 6001),    # m/s, 0.01 bins
    "u":    np.linspace(-80.0, 80.0, 6401),    # m/s -- u reaches >60 at 12 km (the jet)
    "v":    np.linspace(-80.0, 80.0, 6401),
}

# ---------------------------------------------------------------------------
# §8 · leave-one-out departures.  Configuration.
# ---------------------------------------------------------------------------
# The three quantities the §7 distribution figure and the §8 member selection share.
LOO_QUANTITIES = ("qtot", "w", "dbz")

# The branch is set by the TRUTH member's reflectivity, for all three quantities alike:
# "echo" is h(x_m) > min_dbz at that point, "clear" its complement. "all" MUST come last
# -- _pool_stats partitions its input in place, and for "all" that input is a view on the
# array the other two branches select from.
LOO_BRANCHES = ("echo", "clear", "all")
LOO_SCOPES = ("dom", "win")

# Dyadic grids: the bin width is a negative power of two and the endpoints are exact, so
# the float64 edges stored here and the ones numpy rebuilds from `range=` are bit-
# identical, and 0.0 is exactly an edge (index nbin // 2). That last point is what lets
# the histogram's sign split be checked against the per-member npos/nneg counters, which
# are computed by a different code path. Every nbin is 2^k or 2^k * 5, so nb.hist_rebin
# coarsens without silently dropping a remainder.
#
# The ranges are set from the measured maxima over dataset A, 19Z, with headroom for C
# (2 km, deeper graupel cores): max|d| is 0.0173 kg/kg for qtot, 44.85 m/s for w and
# 66.13 dBZ for dbz. The w range in particular has to clear 45, not 30.
LOO_GRID = {                                # (lo, hi, nbin) -> width
    "qtot": (-0.03125, 0.03125, 65536),     # 2^-20 = 9.537e-7 kg/kg  (+-31.25 g/kg)
    "w":    (-80.0, 80.0, 20480),           # 2^-7  = 0.0078125 m/s
    "dbz":  (-80.0, 80.0, 10240),           # 2^-6  = 0.015625 dBZ
}

# A linear grid cannot carry qtot: its departures run from a q75 of -4.9e-10 kg/kg to a
# q99.9 of +3.1e-3, so three quartiles land in the single bin at the origin. The log-
# magnitude companion splits |d| by sign and bins log10|d| instead, which is the only
# axis on which that distribution has a visible body. Kept for all three quantities (it
# costs two 360-bin histograms) but only for the pooled branch.
LOO_LMAG = (-16.0, 2.0, 360)                # log10|d|, 0.05 decades
LOO_PCT = np.array([0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9])
LOO_SCHEMA = 1

# The standardised ensemble perturbation, (x_m - mean_m) / spread_m, pooled over the
# window. THIS is the non-Gaussianity picture: dividing by the pointwise spread makes
# every point contribute unit variance, so what is left is shape alone. A histogram of
# the raw values instead would pool spatial variability with ensemble spread -- a
# storm-scale gradient across the window would show up as a fat tail and be read as
# prior non-Gaussianity, which is the confound this section exists to avoid.
# Points with zero spread (clear air, every member identical) are undefined here and
# are excluded; their count is recorded so the exclusion is visible.
PERT_EDGES = np.linspace(-8.0, 8.0, 3201)


def derived(name):
    return nb.DERIVED / name


def stamp_of(hour):
    return f"20240319{hour}0000"


def composite(ens, name):
    """One composite variable from (nx, ny, nz, Ne, 8). Linear combinations only."""
    if name == "qtot":
        return ens[..., nb.VI["qg"]] + ens[..., nb.VI["qr"]] + ens[..., nb.VI["qs"]]
    return ens[..., nb.VI[name]]


# ---------------------------------------------------------------------------
# §4 · member departures
# ---------------------------------------------------------------------------

def member_departures(dbz, min_dbz=MIN_DBZ):
    """Per-member departure sums over the observation network, for one hour.

    For member m taken as truth, the other 59 are the prior and

        d = h(x_m) - mean(h(x_{k != m}))

    with NO observation noise: the noise is an arbitrary draw seeded on the member
    index, not a property of the member, and leaving it in would make the ordering
    partly random.

    The observation network is the one the runs actually use (`filter_variance`):
    points where the prior's obs-space variance is non-zero. With reflectivity floored
    at min_dbz that is exactly "at least one of the other 59 members carries echo", so
    it is computed as a count rather than a variance -- same set, one pass.

    Echo and no-echo are kept apart because the censoring makes them asymmetric BY
    CONSTRUCTION: where the truth has no echo the departure is bounded below by
    -mean(prior) and cannot go past it, while the positive branch is unbounded. Pooling
    them would hide a structural bias inside what looks like a property of the member.

    Returns sums and counts, not means -- the four hours are concatenated later, and a
    mean of means would weight an hour with few observations like an hour with many.
    """
    ne_tot = dbz.shape[3]
    s1 = dbz.sum(axis=3, dtype=np.float64)                  # (nx,ny,nz)
    n_act = (dbz > min_dbz).sum(axis=3).astype(np.int16)    # (nx,ny,nz)

    out = {k: np.zeros(ne_tot, np.float64) for k in
           ("echo_n", "echo_npos", "echo_spos", "echo_nneg", "echo_sneg",
            "clear_n", "clear_npos", "clear_spos", "clear_nneg", "clear_sneg")}

    for m in range(ne_tot):
        truth = dbz[:, :, :, m]
        # leave-one-out prior mean and network, both from the totals
        prior_mean = (s1 - truth) / (ne_tot - 1)
        network = (n_act - (truth > min_dbz)) >= 1
        if not network.any():
            continue
        d = (truth - prior_mean)[network]
        echo = (truth[network] > min_dbz)

        for tag, sel in (("echo", echo), ("clear", ~echo)):
            dd = d[sel]
            if dd.size == 0:
                continue
            pos, neg = dd > 0, dd < 0
            out[f"{tag}_n"][m] += dd.size
            out[f"{tag}_npos"][m] += int(pos.sum())
            out[f"{tag}_spos"][m] += float(dd[pos].sum())
            out[f"{tag}_nneg"][m] += int(neg.sum())
            out[f"{tag}_sneg"][m] += float(dd[neg].sum())
    return out


# ---------------------------------------------------------------------------
# §5 · the offset-30 pairing
# ---------------------------------------------------------------------------

def pair_correlations(colmax):
    """Member-by-member correlation of the column-max reflectivity field.

    Computed BOTH ways, because they answer different questions and the earlier
    result did not say which one it was:

      full     correlate the fields as they are. Every member contains the same storm,
               so this is high for any pair and says little about the ensemble.
      anomaly  subtract the 60-member mean field first. THIS is the one that decides
               whether members m and m+30 are twins (positive) or an antithetic pair
               (negative), and the sign changes the conclusion about the effective
               ensemble size.

    colmax : (nx, ny, Ne)
    """
    x = colmax.reshape(-1, colmax.shape[-1]).astype(np.float64)      # (npix, Ne)
    out = {}
    for tag, arr in (("full", x), ("anom", x - x.mean(axis=1, keepdims=True))):
        a = arr - arr.mean(axis=0, keepdims=True)
        sd = a.std(axis=0)
        sd[sd == 0] = np.nan
        out[f"corr_{tag}"] = ((a / sd).T @ (a / sd) / a.shape[0]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# §8 · leave-one-out departures of qtot, w and dbz
# ---------------------------------------------------------------------------

def loo_field(x, scale=True):
    """(x_m - the mean of the other Ne-1 members), from the ensemble anomaly.

    For member m taken as truth the leave-one-out departure is algebraically

        d_m = x_m - (S - x_m) / (Ne - 1) = Ne / (Ne - 1) * (x_m - xbar_Ne)

    -- the anomaly about the FULL ensemble mean, rescaled. There is no 60-iteration
    truth loop anywhere in this module: the loop and this differ only by round-off, and
    the loop costs sixty passes over a 1.3 GB array to arrive at the same numbers.

    Returns (d, live, mu). `d` overwrites `x` in place, so the caller must pass an array
    it owns; `mu` is kept because np.spacing(|mu|) is the float32 resolution of the
    subtraction and therefore the floor under which a percentile of `d` is reporting the
    storage grid of the subset file rather than the ensemble.
    `live` is "all Ne members finite AND the ensemble is not constant". Where the
    Ne members are bit-identical the float64 mean is exactly that value, so every d_m is
    exactly 0: "drop the zero departures" and "drop the zero-spread points" are the same
    set, and only the second one costs a comparison instead of a reduction. Recording it
    as a mask also keeps the exclusion countable -- for reflectivity it removes 42 % of
    the domain (clear air at the clamp), for qtot and w essentially nothing (0.003 %,
    the non-finite cells of §6), and those two facts are worth being able to state.

    Non-finiteness needs no (nx, ny, nz, Ne) temporary: one NaN or Inf among the Ne
    makes the float64 sum non-finite, so ~isfinite(mu) IS "some member is non-finite".
    """
    ne = x.shape[3]
    mu = x.mean(axis=3, dtype=np.float64)
    live = np.isfinite(mu) & (x.max(axis=3) > x.min(axis=3))
    x -= mu.astype(np.float32)[..., None]
    if scale:
        x *= np.float32(ne / (ne - 1.0))
    return x, live, mu


def _pool_stats(sel, grid, logmag=False, nblocks=32):
    """Pooled histogram, exact centred moments and exact percentiles of one 1-D sample.

    TAKES OWNERSHIP of `sel`: the percentile is taken with overwrite_input=True, which
    partitions the array in place rather than copying a gigabyte, so nothing may read
    `sel` afterwards.

    Two passes and a partition. Pass one is the mean; pass two is blocked and does the
    histogram, the two tails and the CENTRED power sums together. Centred, because raw
    power sums of a 3e8-sample would have to be differenced against a large mean at the
    end, and the whole point of keeping M2/M3/M4 is that they pool across the four hours
    exactly (Chan, Golub & LeVeque) without that cancellation -- four per-hour skewnesses
    cannot be averaged, since a skewness is a ratio of moments and the mean of four
    ratios is not the ratio of the pooled moments unless the four samples are the same
    size and the same distribution, which four hours of a growing storm are not.

    float64 for every accumulator: a float32 sum of 3e8 similar-magnitude values loses
    about 28 bits, the same order as the sum-to-zero residual this cache is checked
    against. The blocks are cast to float64 before np.histogram as well, so numpy builds
    float64 bin edges and they match the stored grid bit for bit -- with a float32 input
    it would build float32 edges and the stored grid would be a different grid by an ulp.
    """
    lo, hi, nbin = grid
    nlm = LOO_LMAG[2]
    n = int(sel.size)
    out = {"hist": np.zeros(nbin, np.int64), "below": np.int64(0), "above": np.int64(0),
           "ntot": np.int64(n), "mean": np.nan, "M2": np.nan, "M3": np.nan,
           "M4": np.nan, "pct": np.full(LOO_PCT.size, np.nan)}
    if logmag:
        out.update(lhpos=np.zeros(nlm, np.int64), lhneg=np.zeros(nlm, np.int64),
                   lbelow=np.int64(0), labove=np.int64(0), lzero=np.int64(0))
    if n == 0:
        return out

    flat = sel.reshape(-1)
    mean = float(flat.sum(dtype=np.float64) / n)
    hist = np.zeros(nbin, np.int64)
    below = above = 0
    lhp, lhn = np.zeros(nlm, np.int64), np.zeros(nlm, np.int64)
    lbe = lab = lzero = 0
    m2 = m3 = m4 = 0.0

    for blk in np.array_split(flat, nblocks):
        b = blk.astype(np.float64)
        hist += np.histogram(b, bins=nbin, range=(lo, hi))[0]
        below += int(np.count_nonzero(b < lo))
        above += int(np.count_nonzero(b > hi))
        if logmag:
            npos = int(np.count_nonzero(b > 0.0))
            nneg = int(np.count_nonzero(b < 0.0))
            lzero += b.size - npos - nneg
            for sgn, acc in ((1.0, lhp), (-1.0, lhn)):
                v = sgn * b
                v = np.log10(v[v > 0.0])
                acc += np.histogram(v, bins=nlm, range=LOO_LMAG[:2])[0]
                lbe += int(np.count_nonzero(v < LOO_LMAG[0]))
                lab += int(np.count_nonzero(v > LOO_LMAG[1]))
        b -= mean
        bb = b * b
        m2 += float(bb.sum())
        m3 += float((bb * b).sum())
        m4 += float((bb * bb).sum())

    # last: this reorders `flat`, and therefore `sel`
    out["pct"] = np.percentile(flat, LOO_PCT, overwrite_input=True).astype(np.float64)
    out.update(hist=hist, below=np.int64(below), above=np.int64(above),
               ntot=np.int64(n), mean=np.float64(mean),
               M2=np.float64(m2), M3=np.float64(m3), M4=np.float64(m4))
    if logmag:
        out.update(lhpos=lhp, lhneg=lhn, lbelow=np.int64(lbe), labove=np.int64(lab),
                   lzero=np.int64(lzero))
    return out


def loo_departures(ens, dbz_all, si, sj, min_dbz=MIN_DBZ, check=True):
    """§8 · leave-one-out departures of qtot, w and dbz over the WHOLE ensemble.

    Called BEFORE the truth member is held out: every one of the 60 members is a
    candidate truth here, and holding one out first would leave 59 candidates measured
    against a 58-member prior -- a different statistic wearing the same name.

    Two spatial scopes are written side by side, the full subset domain and the analysis
    window: the second is a reduction over an array already in RAM, so storing both makes
    the notebook a switch rather than an 80-minute rebuild.

    Three branches per quantity, all defined by the truth member's REFLECTIVITY rather
    than by the quantity itself. For dbz that is the censoring split: where the truth
    carries no echo the departure is -(Ne/(Ne-1)) * mean(prior), bounded above by zero,
    while the echo branch is unbounded, so pooling them hides a structural asymmetry
    inside what looks like a property of the member. For qtot and w the same mask asks a
    different and useful question -- what the departures look like where the truth member
    has convection -- and costs nothing, because the mask is already built.

    Sums and counts are stored, never means: the four hours are concatenated downstream,
    which is an addition of counts and of values, not an average of four averages.
    """
    ne = ens.shape[3]
    nx, ny, nz = ens.shape[:3]
    out = {"schema": np.int32(LOO_SCHEMA), "ne_tot": np.int32(ne),
           "loo_factor": np.float64(ne / (ne - 1.0)),
           "min_dbz": np.float32(min_dbz),
           "shape": np.array([nx, ny, nz, ne], np.int32),
           "win_i0": np.int32(si.start), "win_i1": np.int32(si.stop),
           "win_j0": np.int32(sj.start), "win_j1": np.int32(sj.stop),
           "pct_q": LOO_PCT.astype(np.float64),
           "lmag_edges": np.linspace(*LOO_LMAG[:2], LOO_LMAG[2] + 1),
           "quantities": np.array(LOO_QUANTITIES),
           "branches": np.array(LOO_BRANCHES),
           "scopes": np.array(LOO_SCOPES)}
    for q in LOO_QUANTITIES:
        lo, hi, nbin = LOO_GRID[q]
        out[f"edges_{q}"] = np.linspace(lo, hi, nbin + 1)

    echo_all = dbz_all > min_dbz                        # (nx,ny,nz,Ne) bool
    scope_sl = {"dom": (slice(None), slice(None)), "win": (si, sj)}

    for q in LOO_QUANTITIES:
        if q == "qtot":
            x = composite(ens, "qtot")                  # a fresh array already
        elif q == "dbz":
            x = dbz_all.copy()                          # never mutate the caller's
        else:
            x = np.array(composite(ens, q))             # composite() returns a view
        d, live, mu = loo_field(x)
        if check:
            _check_identity(q, ens, dbz_all, d, min_dbz)

        for s in LOO_SCOPES:
            sl = scope_sl[s]
            ds_, lv = d[sl], live[sl]
            npts = int(lv.size)
            nlive = int(lv.sum())
            out[f"npts_{s}"] = np.int64(npts)
            out[f"nlive_{q}_{s}"] = np.int64(nlive)
            # a point is dead either because every member agrees (nothing to say about
            # any member) or because a member is non-finite (§6); the two are separated
            # so a growing nbad shows up as a data problem rather than as clear air
            nbad = int((~np.isfinite(ds_[..., 0])).sum())
            out[f"nbad_{q}_{s}"] = np.int64(nbad)
            out[f"ndead_{q}_{s}"] = np.int64(npts - nlive - nbad)

            dm = ds_[lv]                                # (nlive, Ne)
            em = echo_all[sl][lv]                       # (nlive, Ne)
            # The float32 resolution of the subtraction, per point. Reported per BRANCH
            # because it varies by five decades across the domain with the magnitude of
            # the field itself: for qtot it is ~1e-14 in clear air and ~1e-9 inside the
            # storm, so a single domain median would say there is no quantization floor
            # exactly where there is one.
            qsp = np.spacing(np.abs(mu[sl][lv]).astype(np.float32)) * (ne / (ne - 1.0))
            for b in LOO_BRANCHES:                      # "all" LAST -- see LOO_BRANCHES
                key = f"{q}_{b}_{s}"
                sub = (qsp if b == "all" else
                       qsp[em.any(axis=1) if b == "echo" else (~em).any(axis=1)])
                out[f"quantum_{key}"] = np.float64(np.median(sub) if sub.size else np.nan)
                if b == "all":
                    out[f"n_{key}"] = np.full(ne, nlive, np.int64)
                    pos, neg = dm > 0, dm < 0
                    sel = dm.reshape(-1)                # a view; safe only because last
                else:
                    m = em if b == "echo" else ~em
                    out[f"n_{key}"] = np.count_nonzero(m, axis=0).astype(np.int64)
                    pos, neg = (dm > 0) & m, (dm < 0) & m
                    sel = dm[m]                         # a copy
                out[f"npos_{key}"] = np.count_nonzero(pos, axis=0).astype(np.int64)
                out[f"nneg_{key}"] = np.count_nonzero(neg, axis=0).astype(np.int64)
                out[f"spos_{key}"] = np.sum(dm, axis=0, where=pos, dtype=np.float64)
                out[f"sneg_{key}"] = np.sum(dm, axis=0, where=neg, dtype=np.float64)
                del pos, neg
                st = _pool_stats(sel, LOO_GRID[q], logmag=(b == "all"))
                for k, v in st.items():
                    out[f"{k}_{key}"] = v
                del sel
            del dm, em, qsp
        del d, x, live, mu
    del echo_all
    return out


def _check_identity(q, ens, dbz_all, d, min_dbz, n=512, seed=0):
    """Pin d against the explicit leave-one-out loop on a random subsample.

    The whole cache rests on d_m = Ne/(Ne-1) * (x_m - xbar); this is the one place the
    identity is checked against the definition rather than asserted in a docstring.
    """
    rng = np.random.RandomState(seed)
    ii, jj, kk = (rng.randint(0, s, n) for s in d.shape[:3])
    if q == "qtot":
        xs = (ens[ii, jj, kk, :, nb.VI["qg"]] + ens[ii, jj, kk, :, nb.VI["qr"]]
              + ens[ii, jj, kk, :, nb.VI["qs"]]).astype(np.float64)
    elif q == "dbz":
        xs = dbz_all[ii, jj, kk, :].astype(np.float64)
    else:
        xs = ens[ii, jj, kk, :, nb.VI[q]].astype(np.float64)
    ne = xs.shape[1]
    ref = xs - (xs.sum(axis=1, keepdims=True) - xs) / (ne - 1)   # the definition itself
    got = d[ii, jj, kk, :].astype(np.float64)
    ok = np.isfinite(ref) & np.isfinite(got)
    scale = max(float(np.abs(ref[ok]).max()), 1e-30)
    err = float(np.abs(ref[ok] - got[ok]).max())
    assert err <= 1e-5 * scale, \
        f"{q}: leave-one-out identity broken, max error {err:.3e} on a scale of {scale:.3e}"


# ---------------------------------------------------------------------------
# §6 · where the non-finite cells are
# ---------------------------------------------------------------------------

def nonfinite_audit(ens):
    """Locate non-finite cells in the raw subset, per variable.

    The multi-obs state fields carry the same count of non-finite cells for all eight
    variables, which points at the extraction rather than the analysis: an analysis
    artefact would not hit eight physically different variables identically. If the
    cells are the same (i, j, k) in every variable, they are bad columns or levels in
    the source and can be fixed there.
    """
    nx, ny, nz, ne, nv = ens.shape
    out = {}
    per_var_cells = []
    for v, name in enumerate(nb.VARS):
        bad = ~np.isfinite(ens[..., v])                 # (nx,ny,nz,Ne)
        cell = bad.any(axis=3)                          # (nx,ny,nz)
        out[f"n_bad_{name}"] = np.int64(bad.sum())
        out[f"n_cells_{name}"] = np.int64(cell.sum())
        per_var_cells.append(cell)
    same = np.logical_and.reduce(per_var_cells)
    anyv = np.logical_or.reduce(per_var_cells)
    out["n_cells_all_vars"] = np.int64(same.sum())
    out["n_cells_any_var"] = np.int64(anyv.sum())
    i, j, k = np.nonzero(anyv)
    out["bad_i"], out["bad_j"], out["bad_k"] = (i.astype(np.int32), j.astype(np.int32),
                                                k.astype(np.int32))
    out["by_level"] = np.bincount(k, minlength=nz).astype(np.int64)
    out["n_edge"] = np.int64(((i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)).sum())
    out["shape"] = np.array([nx, ny, nz, ne, nv], np.int32)
    # a column is bad at every level -> an extraction column, not a stray cell
    if i.size:
        cols = {}
        for a, b in zip(i.tolist(), j.tolist()):
            cols[(a, b)] = cols.get((a, b), 0) + 1
        out["n_columns"] = np.int64(len(cols))
        out["n_full_columns"] = np.int64(sum(1 for c in cols.values() if c == nz))
    else:
        out["n_columns"] = np.int64(0)
        out["n_full_columns"] = np.int64(0)
    return out


# ---------------------------------------------------------------------------

def run_one(ds, hour, force=False):
    path = nb.subset_path(hour, ds)
    if not os.path.isfile(path):
        print(f"  {ds} {hour}Z: no subset at {path}")
        return
    st = stamp_of(hour)
    targets = {
        "depart": derived(f"n2_depart_{ds}_{st}.npz"),
        "state": derived(f"n2_state_{ds}_{st}.npz"),
        "pairs": derived(f"n2_pairs_{ds}_{st}.npz"),
        "nonfinite": derived(f"n2_nonfinite_{ds}_{st}.npz"),
        "loodep": derived(f"n2_loodep_{ds}_{st}.npz"),
    }
    prior_cf = nb._cache_key(path, None, MIN_DBZ, TRUTH_MEMBER)
    if not force and prior_cf.exists() and all(t.exists() for t in targets.values()):
        print(f"  {ds} {hour}Z: cached")
        return

    nb.assert_dataset(path, ds)
    t0 = time.time()
    # The geometry first: lats/lons are small members of the same zip, so this costs no
    # inflate, and an empty analysis window then fails in a tenth of a second instead of
    # after a two-minute decompress. Both the §8 and the §2/§3 blocks use this one call,
    # so the window they record cannot drift apart.
    with np.load(path) as f:
        lats, lons = f["lats"], f["lons"]
    si, sj = nb.bbox_to_slices(lats, lons, WIN_LAT, WIN_LON)
    print(f"  {ds} {hour}Z: decompressing {os.path.getsize(path)/1e9:.1f} GB ...")
    ens = nb.load_ensemble(path)                      # (nx,ny,nz,60,8)
    ne_tot = ens.shape[3]
    print(f"       {ens.shape}  {time.time()-t0:.0f}s")

    # --- §6, on the raw array before anything is held out or masked ---------
    if force or not targets["nonfinite"].exists():
        np.savez_compressed(targets["nonfinite"], **nonfinite_audit(ens),
                            **nb._src_stamp(path))

    dbz_all = nb.hx(ens, min_dbz=MIN_DBZ).astype(np.float32)     # (nx,ny,nz,60)

    # --- §4, over all 60 members as candidate truths ------------------------
    if force or not targets["depart"].exists():
        d = member_departures(dbz_all, MIN_DBZ)
        np.savez_compressed(targets["depart"], **d,
                            ne_tot=np.int32(ne_tot), **nb._src_stamp(path))

    # --- §5, on the 60-member ensemble: a question about how it was built ---
    if force or not targets["pairs"].exists():
        np.savez_compressed(targets["pairs"],
                            **pair_correlations(dbz_all.max(axis=2)),
                            **nb._src_stamp(path))

    # --- §8, all 60 members as candidate truths, on qtot, w and dbz ---------
    # This MUST stay above the hold-out: the statistic is "each of the 60 against the
    # other 59", and on a 59-member array it would silently become "each of 59 against
    # the other 58" under the same key names.
    if force or not targets["loodep"].exists():
        assert ens.shape[3] == ne_tot, "loo_departures must run before the hold-out"
        np.savez_compressed(targets["loodep"],
                            **loo_departures(ens, dbz_all, si, sj, MIN_DBZ),
                            dataset=np.array(ds), **nb._src_stamp(path))
        print(f"       loodep -> {targets['loodep'].name} {time.time()-t0:.0f}s")

    # --- the 59-member PRIOR: everything below holds truth member 0 out -----
    keep = [m for m in range(ne_tot) if m != TRUTH_MEMBER]
    ens = ens[:, :, :, keep, :]
    dbz = dbz_all[:, :, :, keep]
    del dbz_all

    if force or not prior_cf.exists():
        out = {
            "dbz_mean": dbz.mean(axis=3).astype(np.float32),
            "dbz_spread": dbz.std(axis=3, ddof=1).astype(np.float32),
            "n_active": (dbz > MIN_DBZ).sum(axis=3).astype(np.int16),
            "dbz_skew": nb.ensemble_skew(dbz, axis=3).astype(np.float32),
            "dbz_kurt": nb.ensemble_kurt(dbz, axis=3).astype(np.float32),
            "dbz_colmax_members": dbz.max(axis=2).astype(np.float32),
            "Ne": np.int32(dbz.shape[3]),
            "min_dbz": np.float32(MIN_DBZ),
            "truth_member": np.int32(TRUTH_MEMBER),
        }
        for v in nb.VARS:
            out[f"mean_{v}"] = ens[..., nb.VI[v]].mean(axis=3).astype(np.float32)
            out[f"spread_{v}"] = ens[..., nb.VI[v]].std(axis=3, ddof=1).astype(np.float32)
        nb._save_cached(prior_cf, out, path)
        print(f"       prior_stats cache -> {prior_cf.name}")

    # --- §2/§3, composite variables -----------------------------------------
    if force or not targets["state"].exists():
        out = {"win_i0": np.int32(si.start), "win_i1": np.int32(si.stop),
               "win_j0": np.int32(sj.start), "win_j1": np.int32(sj.stop),
               "Ne": np.int32(dbz.shape[3])}
        out["pert_edges"] = PERT_EDGES
        for name in COMPOSITES:
            c = composite(ens, name)                       # (nx,ny,nz,59)
            out[f"skew_{name}"] = nb.ensemble_skew(c, axis=3).astype(np.float32)
            out[f"kurt_{name}"] = nb.ensemble_kurt(c, axis=3).astype(np.float32)
            out[f"mean_{name}"] = c.mean(axis=3).astype(np.float32)
            out[f"spread_{name}"] = c.std(axis=3, ddof=1).astype(np.float32)
            # the distribution INSIDE the window, pooled over x, y, z and members
            w = c[si, sj]
            w = w[np.isfinite(w)]
            edges = HIST_EDGES[name]
            cnt, _ = np.histogram(w, bins=edges)
            out[f"hist_{name}"] = cnt.astype(np.int64)
            out[f"edges_{name}"] = edges
            out[f"below_{name}"] = np.int64((w < edges[0]).sum())
            out[f"above_{name}"] = np.int64((w > edges[-1]).sum())
            out[f"n_{name}"] = np.int64(w.size)
            # size-independent shape numbers, for the table the curves cannot carry
            out[f"wmean_{name}"] = np.float64(w.mean())
            out[f"wstd_{name}"] = np.float64(w.std(ddof=1))
            for q in (1, 5, 25, 50, 75, 95, 99):
                out[f"q{q:02d}_{name}"] = np.float64(np.percentile(w, q))
            del w

            # standardised ensemble perturbations inside the window
            cw = c[si, sj]
            mu = cw.mean(axis=3, keepdims=True)
            sd = cw.std(axis=3, ddof=1, keepdims=True)
            live = np.isfinite(sd) & (sd > 0)
            z = np.where(live, (cw - mu) / np.where(live, sd, 1.0), np.nan)
            z = z[np.isfinite(z)]
            pcnt, _ = np.histogram(z, bins=PERT_EDGES)
            out[f"phist_{name}"] = pcnt.astype(np.int64)
            out[f"pbelow_{name}"] = np.int64((z < PERT_EDGES[0]).sum())
            out[f"pabove_{name}"] = np.int64((z > PERT_EDGES[-1]).sum())
            out[f"pn_{name}"] = np.int64(z.size)
            out[f"pdead_{name}"] = np.int64(int(live.size - live.sum()))
            # shape numbers of the standardised perturbation: 0 and 0 for a Gaussian
            out[f"pskew_{name}"] = np.float64(((z - z.mean()) ** 3).mean() / z.std() ** 3)
            out[f"pkurt_{name}"] = np.float64(((z - z.mean()) ** 4).mean() / z.std() ** 4 - 3.0)
            del c, cw, mu, sd, live, z
        np.savez_compressed(targets["state"], **out, **nb._src_stamp(path))

    print(f"  {ds} {hour}Z: done in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", nargs="*", default=list(DATASETS))
    ap.add_argument("--hours", nargs="*", default=list(HOURS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for ds in args.ds:
        for hour in args.hours:
            run_one(ds, hour, force=args.force)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
