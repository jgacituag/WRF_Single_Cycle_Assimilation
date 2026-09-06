"""
test/test_install.py
====================
Installation smoke test. Verifies that a checkout is actually able to run a DA
experiment: the Fortran extension is built and importable, its three exported
symbols behave, and all three DA methods produce a sane analysis.

Deliberately data-free and config-free -- it allocates a tiny synthetic 6x6x4
Ne=10 state in memory, so it passes on a fresh clone with no .npz, no YAML and no
access to the 76 GB subset tree. That is the whole point: it answers "is this
install working?", not "is the science right?" (see test_ensemble_stats.py for
the numerical checks on the diagnostics).

Run with:  python test/test_install.py    (from repo root)
"""
import sys
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "fortran"))

import numpy as np

VAR_IDX = {"qg": 0, "qr": 1, "qs": 2, "T": 3, "P": 4, "u": 5, "v": 6, "w": 7}


# ── 1. the compiled extension ───────────────────────────────────────────────

def test_fortran_extension_imports():
    """The .so is gitignored, so a fresh clone must run src/build_fortran.sh first.
    A failure here is a build problem, not a code problem."""
    try:
        from cletkf_wloc import common_da as cda
    except ImportError as e:
        raise SystemExit(
            f"\nFAIL  cannot import cletkf_wloc: {e}\n"
            f"      The compiled extension is gitignored. Build it with:\n"
            f"          conda activate intermediate_exp && bash src/build_fortran.sh\n"
            f"      Note the .so is tied to the interpreter it was built against\n"
            f"      (the shipped one is cpython-38).\n")
    for name in ("calc_ref", "calc_ref_ens", "simple_letkf_wloc"):
        assert hasattr(cda, name), f"cletkf_wloc.common_da is missing {name}"
    print("PASS  cletkf_wloc imports; common_da exposes calc_ref/calc_ref_ens/simple_letkf_wloc")


# ── 2. the reflectivity forward operator ────────────────────────────────────

def _sample_hydro():
    """A single moist, warm, near-surface point that produces a solid echo."""
    return dict(qr=2.0e-3, qs=1.0e-3, qg=1.0e-3, T=285.0, P=90000.0)


def test_calc_ref_scalar():
    from cletkf_wloc import common_da as cda
    p = _sample_hydro()
    ref = cda.calc_ref(p["qr"], p["qs"], p["qg"], p["T"], p["P"], min_dbz=0.0)
    assert np.isfinite(ref), f"calc_ref returned {ref}"
    assert ref > 0.0, f"expected an echo from {p}, got {ref} dBZ"

    # the floor is a floor, not a clip of the physics: raising it must raise the output
    hi = cda.calc_ref(1e-12, 1e-12, 1e-12, 285.0, 90000.0, min_dbz=7.5)
    assert np.isclose(hi, 7.5), f"min_dbz floor not applied: {hi}"
    print(f"PASS  calc_ref: finite echo ({ref:.2f} dBZ) and min_dbz floor honoured")


def test_calc_ref_ens_matches_scalar():
    """calc_ref_ens is what the runner uses domain-wide; it must agree with the
    per-point calc_ref that compute_hxf uses. A mismatch here would silently
    decouple the sweep and multi_obs paths."""
    from cletkf_wloc import common_da as cda
    rng = np.random.default_rng(0)
    nx, ny, nz, nbv = 3, 4, 2, 1
    shape = (nx, ny, nz, nbv)
    qr = np.abs(rng.normal(1e-3, 5e-4, shape))
    qs = np.abs(rng.normal(5e-4, 2e-4, shape))
    qg = np.abs(rng.normal(5e-4, 2e-4, shape))
    T = np.full(shape, 285.0)
    P = np.full(shape, 90000.0)

    ens = cda.calc_ref_ens(qr, qs, qg, T, P, min_dbz=0.0)
    assert ens.shape == shape, f"calc_ref_ens shape {ens.shape} != {shape}"

    worst = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                one = cda.calc_ref(qr[i, j, k, 0], qs[i, j, k, 0], qg[i, j, k, 0],
                                   T[i, j, k, 0], P[i, j, k, 0], min_dbz=0.0)
                worst = max(worst, abs(one - ens[i, j, k, 0]))
    assert worst < 1e-6, f"calc_ref_ens disagrees with calc_ref by {worst}"
    print(f"PASS  calc_ref_ens matches per-point calc_ref (max diff {worst:.2e} dBZ)")


# ── 3. a synthetic single-observation assimilation ──────────────────────────

def _synthetic_case(nx=6, ny=6, nz=4, Ne=10, seed=0):
    """Tiny moist ensemble plus one observation at the domain centre, chosen so the
    departure is large enough that AOEI must fire."""
    from da.core import compute_hxf
    rng = np.random.default_rng(seed)
    xf = np.zeros((nx, ny, nz, Ne, 8), dtype=np.float32)
    xf[..., VAR_IDX["qr"]] = np.abs(rng.normal(2.0e-3, 8.0e-4, (nx, ny, nz, Ne)))
    xf[..., VAR_IDX["qs"]] = np.abs(rng.normal(1.0e-3, 4.0e-4, (nx, ny, nz, Ne)))
    xf[..., VAR_IDX["qg"]] = np.abs(rng.normal(1.0e-3, 4.0e-4, (nx, ny, nz, Ne)))
    xf[..., VAR_IDX["T"]] = 285.0 + rng.normal(0, 1.0, (nx, ny, nz, Ne))
    xf[..., VAR_IDX["P"]] = 90000.0 + rng.normal(0, 200.0, (nx, ny, nz, Ne))
    # u/v/w start at zero on purpose: they are unobserved and can only move through
    # the cross-covariance with the hydrometeors, which is what we want to exercise.
    xf = np.asfortranarray(xf)

    pos_km = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    pos_km[..., 0] = np.arange(nx)[:, None, None] * 1.0
    pos_km[..., 1] = np.arange(ny)[None, :, None] * 1.0
    pos_km[..., 2] = np.arange(nz)[None, None, :] * 0.5
    pos_km = np.asfortranarray(pos_km)

    ox = np.array([nx // 2], np.int32)
    oy = np.array([ny // 2], np.int32)
    oz = np.array([nz // 2], np.int32)
    hxf = compute_hxf(xf, ox, oy, oz, VAR_IDX, dbz_min=0.0)
    yo = np.array([hxf.mean() + 12.0], np.float32)
    R0 = np.array([5.0], np.float32)
    loc = np.array([2.0, 2.0, 2.0], np.float32)
    return dict(xf=xf, pos_km=pos_km, ox=ox, oy=oy, oz=oz,
                yo=yo, R0=R0, loc=loc, hxf=hxf)


def _check_analysis(xa, xf, label):
    assert xa.shape == xf.shape, f"{label}: xa shape {xa.shape} != {xf.shape}"
    assert np.all(np.isfinite(xa)), f"{label}: non-finite values in xa"
    for q in ("qr", "qs", "qg"):
        lo = float(xa[..., VAR_IDX[q]].min())
        assert lo >= 0.0, f"{label}: negative {q} in the analysis ({lo})"


def _departure(xa, c):
    """|yo - H(xa)| at the observation point."""
    from da.core import compute_hxf
    hxa = compute_hxf(np.asfortranarray(xa), c["ox"], c["oy"], c["oz"], VAR_IDX, dbz_min=0.0)
    return float(abs(c["yo"][0] - hxa.mean()))


def test_letkf_update():
    from da.core import letkf_update
    c = _synthetic_case()
    res = letkf_update(c["xf"], c["yo"], c["R0"], c["ox"], c["oy"], c["oz"],
                       c["loc"], VAR_IDX, c["pos_km"], dbz_min=0.0)
    _check_analysis(res["xa"], c["xf"], "LETKF")
    d_b, d_a = float(abs(res["dep"][0])), _departure(res["xa"], c)
    assert d_a < d_b, f"LETKF moved away from the obs: |dep| {d_b:.3f} -> {d_a:.3f}"
    print(f"PASS  letkf_update: |dep| {d_b:.3f} -> {d_a:.3f} dBZ, bounds and shape OK")


def test_tenkf_update():
    from da.core import tenkf_update
    c = _synthetic_case()
    res = tenkf_update(c["xf"], c["yo"], c["R0"], c["ox"], c["oy"], c["oz"],
                       c["loc"], VAR_IDX, 3, 2.0, c["pos_km"], dbz_min=0.0)
    _check_analysis(res["xa"], c["xf"], "TEnKF")
    assert res["alpha_weights"].shape == (3,)
    d_b, d_a = float(abs(res["deps"][0, 0])), _departure(res["xa"], c)
    assert d_a < d_b, f"TEnKF moved away from the obs: |dep| {d_b:.3f} -> {d_a:.3f}"
    print(f"PASS  tenkf_update (Nt=3): |dep| {d_b:.3f} -> {d_a:.3f} dBZ over 3 steps")


def test_aoei_update_inflates():
    from da.core import aoei_update
    c = _synthetic_case()
    res = aoei_update(c["xf"], c["yo"], c["R0"], c["ox"], c["oy"], c["oz"],
                      c["loc"], VAR_IDX, c["pos_km"], dbz_min=0.0)
    _check_analysis(res["xa"], c["xf"], "AOEI")
    R_t, R_0 = float(res["obs_error"][0]), float(res["obs_error_raw"][0])
    assert R_t >= R_0, f"AOEI floor violated: {R_t} < {R_0}"
    assert R_t > R_0, f"AOEI should have fired on a +12 dBZ departure: {R_t} == {R_0}"
    print(f"PASS  aoei_update: inflated R {R_0:.1f} -> {R_t:.1f} on a large departure")


def test_clamp_modes_differ_only_where_they_should():
    """The three `clamp_hydro` settings, on one synthetic case.

    Four properties, and all four have to hold or the flag is not the control the
    chapter would be leaning on:

      never    leaves negative hydrometeors in the posterior;
      per_step leaves none;
      Nt=1     `final` == `per_step` BIT FOR BIT -- with one step there is nothing to
               defer, so any difference would be a bug in the flag rather than a
               finding about the filter;
      the counters count the same pairs whether or not the floor is applied, so
      `never`'s bookkeeping is a measurement of its own trajectory and not of nothing.
    """
    from da.core import tenkf_update, CLAMP_MODES, HYDRO_VARS
    IHY = [VAR_IDX[q] for q in HYDRO_VARS]

    def run(mode, nt):
        c = _synthetic_case()
        return tenkf_update(c["xf"], c["yo"], c["R0"], c["ox"], c["oy"], c["oz"],
                            c["loc"], VAR_IDX, nt, 2.0, c["pos_km"], dbz_min=0.0,
                            clamp_mode=mode)

    assert CLAMP_MODES == ("per_step", "final", "never")
    r = {m: run(m, 1) for m in CLAMP_MODES}
    neg = {m: int((r[m]["xa"][:, :, :, :, IHY] < 0).sum()) for m in CLAMP_MODES}
    assert neg["per_step"] == 0, f"per_step left {neg['per_step']} negative hydrometeors"
    assert neg["final"] == 0, f"final left {neg['final']} negative hydrometeors"
    assert neg["never"] > 0, (
        "never left no negative hydrometeors, so this case cannot tell the three "
        "modes apart and the test proves nothing")
    assert np.array_equal(r["per_step"]["xa"], r["final"]["xa"]), (
        "at Nt=1 `final` and `per_step` are the same operation and must agree exactly")

    cp, cn = r["per_step"]["clamp"], r["never"]["clamp"]
    assert bool(cp["clamp_applied"][0]) and not bool(cn["clamp_applied"][0])
    assert int(cp["clamp_n_pairs"][0]) == int(cn["clamp_n_pairs"][0]), (
        f"the counters disagree across modes at step 0 on an identical trajectory: "
        f"{int(cp['clamp_n_pairs'][0])} vs {int(cn['clamp_n_pairs'][0])}")
    assert float(cp["clamp_mass_total"][0]) > 0

    # Nt=3: `final` clamps once, so its first two steps are measured but not applied.
    f3 = run("final", 3)
    assert list(f3["clamp"]["clamp_applied"]) == [False, False, True], (
        f"`final` applied the floor at {list(f3['clamp']['clamp_applied'])}, "
        f"not only at the last step")
    assert int((f3["xa"][:, :, :, :, IHY] < 0).sum()) == 0
    print(f"PASS  clamp_hydro: per_step/final/never differ as specified "
          f"({neg['never']} negative hydrometeor values left by `never`, 0 by the "
          f"other two; Nt=1 final == per_step bit for bit)")


def test_clamp_mode_is_validated():
    from da.core import clamp_mode_of
    assert clamp_mode_of({}) == "per_step"
    assert clamp_mode_of({"assimilation": None}) == "per_step"
    assert clamp_mode_of({"assimilation": {"clamp_hydro": "never"}}) == "never"
    for bad in ("perstep", "Final", "", "true"):
        try:
            clamp_mode_of({"assimilation": {"clamp_hydro": bad}})
        except ValueError:
            continue
        raise AssertionError(f"clamp_hydro={bad!r} was accepted; a misspelt mode would "
                             f"silently run production behaviour under another name")
    print("PASS  clamp_hydro: defaults to per_step, and 4 misspellings are refused")


def test_localization_reaches_points():
    """simple_letkf_wloc returns n_updated; zero would mean the localization cutoff
    excluded every grid point and the 'analysis' is just a copy of the prior."""
    c = _synthetic_case()
    idx = (c["ox"].astype(np.intp), c["oy"].astype(np.intp), c["oz"].astype(np.intp))
    from cletkf_wloc import common_da as cda
    nx, ny, nz, Ne, nvar = c["xf"].shape
    dep = (c["yo"] - c["hxf"].mean(axis=1)).astype(np.float32)
    _, n_updated = cda.simple_letkf_wloc(
        nx=nx, ny=ny, nz=nz, nbv=Ne, nvar=nvar, nobs=1,
        hxf=np.asfortranarray(c["hxf"]), xf=c["xf"], dep=dep,
        ox=c["pos_km"][idx[0], idx[1], idx[2], 0].astype(np.float32),
        oy=c["pos_km"][idx[0], idx[1], idx[2], 1].astype(np.float32),
        oz=c["pos_km"][idx[0], idx[1], idx[2], 2].astype(np.float32),
        locs=c["loc"], oerr=c["R0"], pos_km=c["pos_km"])
    total = nx * ny * nz
    assert n_updated > 0, "localization updated no grid points at all"
    print(f"PASS  simple_letkf_wloc: updated {int(n_updated)}/{total} grid points")


# ── 3b. the LETKF weights, which §6 stores and nothing else can recover ─────

def test_letkf_weights_reproduce_fortran():
    """da.core.letkf_weights is a NumPy transcription of common_letkf.f90's letkf_core,
    used to store the weight matrix at a probed point -- simple_letkf_wloc returns xa
    and a counter, so the weights are otherwise unrecoverable from a run's output.

    The check is end to end: rebuild the analysis at every updated grid point from the
    weights alone and compare against what the Fortran produced. If the transcription
    drifts from the Fortran, the stored weights stop describing the run that happened,
    and nothing downstream would notice."""
    from da.core import letkf_weights, local_obs_at
    from cletkf_wloc import common_da as cda
    c = _synthetic_case()
    xf, pos_km, hxf = c["xf"], c["pos_km"], c["hxf"]
    nx, ny, nz, Ne, nvar = xf.shape
    idx = (c["ox"].astype(np.intp), c["oy"].astype(np.intp), c["oz"].astype(np.intp))
    ox, oy, oz = (pos_km[idx[0], idx[1], idx[2], d].astype(np.float32) for d in range(3))
    dep = (c["yo"] - hxf.mean(axis=1)).astype(np.float32)

    xa_f, _ = cda.simple_letkf_wloc(
        nx=nx, ny=ny, nz=nz, nbv=Ne, nvar=nvar, nobs=1,
        hxf=np.asfortranarray(hxf), xf=xf, dep=dep, ox=ox, oy=oy, oz=oz,
        locs=c["loc"], oerr=c["R0"], pos_km=pos_km)

    hdxb_all = (hxf - hxf.mean(axis=1)[:, None]).astype(np.float64)
    worst, probed = 0.0, 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                sel, rloc = local_obs_at(i, j, k, ox, oy, oz, c["R0"], c["loc"], pos_km)
                if len(sel) == 0:
                    assert np.array_equal(xa_f[i, j, k], xf[i, j, k]), \
                        "a cell with no local obs was modified"
                    continue
                T, wa = letkf_weights(hdxb_all[sel], rloc, dep[sel].astype(np.float64))
                xfm = xf[i, j, k].mean(axis=0)
                xa_py = xfm[None, :] + (T + wa[:, None]).T @ (xf[i, j, k] - xfm)
                scale = max(float(np.abs(xa_f[i, j, k]).max()), 1e-30)
                worst = max(worst, float(np.abs(xa_py - xa_f[i, j, k]).max() / scale))
                probed += 1
    assert probed > 0, "no cell was updated, so nothing was checked"
    assert worst < 1e-5, f"weights do not reproduce the Fortran analysis: {worst:.2e}"
    print(f"PASS  letkf_weights: rebuilt the analysis at {probed} cells from the "
          f"weights alone (worst relative diff {worst:.1e})")


# ── 3c. light mode: the scalars must equal what the fields would have given ─

def _multi_obs_case(seed=1):
    """A tiny multi_obs-shaped case: prior, analysis, truth, and both H(x) fields."""
    from da.core import letkf_update
    from da.metrics import compute_multi_obs_metrics
    c = _synthetic_case(seed=seed)
    xf = c["xf"]
    nx, ny, nz, Ne, nvar = xf.shape
    rng = np.random.default_rng(seed)
    truth = (xf.mean(axis=3) + rng.normal(0, 1e-4, (nx, ny, nz, nvar))).astype(np.float32)
    truth[..., VAR_IDX["T"]] += rng.normal(0, 0.5, (nx, ny, nz))
    res = letkf_update(xf, c["yo"], c["R0"], c["ox"], c["oy"], c["oz"],
                       c["loc"], VAR_IDX, c["pos_km"], dbz_min=0.0)
    xa = np.asfortranarray(res["xa"])

    def hx(ens):
        from cletkf_wloc import common_da as cda
        r = cda.calc_ref_ens(ens[..., VAR_IDX["qr"]].astype(np.float64),
                             ens[..., VAR_IDX["qs"]].astype(np.float64),
                             ens[..., VAR_IDX["qg"]].astype(np.float64),
                             ens[..., VAR_IDX["T"]].astype(np.float64),
                             ens[..., VAR_IDX["P"]].astype(np.float64), min_dbz=0.0)
        return np.maximum(r, 0.0).astype(np.float32)

    ens_hxf, ens_hxa = hx(xf), hx(xa)
    truth_hx = hx(truth[:, :, :, None, :])[:, :, :, 0]
    var_names = [k for k, _ in sorted(VAR_IDX.items(), key=lambda x: x[1])]
    obs_ijk = (c["ox"].astype(np.intp), c["oy"].astype(np.intp), c["oz"].astype(np.intp))
    return dict(xa=xa, xf=xf, truth=truth, ens_hxf=ens_hxf, ens_hxa=ens_hxa,
                truth_hx=truth_hx, var_names=var_names, Ne=Ne, obs_ijk=obs_ijk,
                hxf_mean=ens_hxf.mean(axis=3), hxa_mean=ens_hxa.mean(axis=3))


def _metrics(case, storage_level, storm_thresh):
    from da.metrics import compute_multi_obs_metrics
    return compute_multi_obs_metrics(
        case["xa"], case["xf"], case["truth"],
        case["hxf_mean"], case["hxa_mean"], case["truth_hx"],
        case["var_names"], case["Ne"], storage_level=storage_level,
        ens_hxf=case["ens_hxf"], ens_hxa=case["ens_hxa"], dbz_min=0.0,
        obs_ijk=case["obs_ijk"], storm_thresh=storm_thresh)


def test_light_scalars_match_full_fields():
    """The check that licenses trusting light mode: a domain scalar computed with the
    fields dropped must equal the same quantity recomputed from the fields in full
    mode. If these disagree, every light run in the batch is unverifiable."""
    from da.metrics import domain_masks
    case = _multi_obs_case()
    # a threshold this synthetic echo actually straddles, so `storm` is a real subset
    thresh = float(np.median(np.nanmax(case["truth_hx"], axis=2)))
    light = _metrics(case, "light", thresh)
    full = _metrics(case, "full", thresh)

    assert not [k for k in light if k.endswith("_field")], "light mode kept a field"
    assert [k for k in full if k.endswith("_field")], "full mode dropped every field"

    masks = domain_masks(case["truth_hx"], obs_ijk=case["obs_ijk"], storm_thresh=thresh)
    assert 0 < masks["storm"].sum() < masks["global"].sum(), \
        "the storm domain is empty or the whole grid; the test would prove nothing"

    worst, checked = 0.0, 0
    for iv, v in enumerate(case["var_names"]):
        err = full["abs_err_a_field"][..., iv]
        for dname, m in masks.items():
            vals = err[m]
            vals = vals[np.isfinite(vals)]
            ref = float(np.sqrt((vals.astype(np.float64) ** 2).mean()))
            got = light[f"rmse_a_{dname}_{v}"]
            assert light[f"n_rmse_a_{dname}_{v}"] == vals.size, \
                f"n_rmse_a_{dname}_{v} is not the count that entered"
            worst = max(worst, abs(got - ref) / max(abs(ref), 1e-30))
            checked += 1
    assert worst < 1e-9, f"light and full disagree by {worst:.2e} relative"
    print(f"PASS  light scalars match the full fields over "
          f"{len(masks)} domains x {len(case['var_names'])} variables "
          f"({checked} comparisons, worst {worst:.1e} relative)")


def test_domains_disagree_and_counts_are_written():
    """The three domains must be able to disagree -- that disagreement is what the
    chapter's argument rests on -- and every scalar must carry the denominator that
    produced it."""
    case = _multi_obs_case()
    thresh = float(np.median(np.nanmax(case["truth_hx"], axis=2)))
    m = _metrics(case, "light", thresh)
    assert m["n_cells_global"] > m["n_cells_storm"] > 0
    assert m["n_cells_obs"] == len(case["obs_ijk"][0])
    assert m["rmse_f_global_ref"] != m["rmse_f_storm_ref"], \
        "global and storm reflectivity RMSE are identical; the restriction did nothing"
    assert 0.0 <= m["frac_analysis_eq_prior_global"] <= 1.0
    assert m["n_touched"] + m["n_untouched"] == m["n_cells_global"]
    print(f"PASS  domains: global {m['n_cells_global']} / storm {m['n_cells_storm']} / "
          f"obs {m['n_cells_obs']} cells, RMSE_ref {m['rmse_f_global_ref']:.2f} vs "
          f"{m['rmse_f_storm_ref']:.2f} dBZ, {m['frac_touched_no_obs']:.1%} of touched "
          f"cells carry no obs")


def test_nan_cells_are_not_counted_as_touched():
    """A cell the update never reached, but which the source data left non-finite, must
    still count as untouched. `xa == xf` gets this wrong -- NaN == NaN is False -- and
    the error is invisible in light mode, where the ensembles are not on disk to check
    against. On dataset A at 20 UTC it inflated n_touched by 53 of 7,188, which is
    exactly the number of cells non-finite in the prior."""
    from da.metrics import _untouched_mask
    rng = np.random.default_rng(3)
    xf = rng.normal(size=(3, 3, 2, 5, 4)).astype(np.float32)
    xa = xf.copy()

    assert _untouched_mask(xa, xf).all(), "an unmodified copy read as touched"

    # a masked source cell: NaN in the prior, and copied verbatim into the analysis
    xf[0, 1, 0, 2, 3] = np.nan
    xa[0, 1, 0, 2, 3] = np.nan
    m_nan = _untouched_mask(xa, xf)
    assert m_nan.all(), \
        f"a NaN cell was counted as touched ({int((~m_nan).sum())} cells)"

    # and a cell that genuinely moved must still read as touched
    xa[2, 2, 1, 4, 1] += 1.0
    m = _untouched_mask(xa, xf)
    assert m[0, 1, 0], "the NaN cell flipped to touched once another cell changed"
    assert not m[2, 2, 1], "a genuinely changed cell was counted as untouched"
    assert int((~m).sum()) == 1, f"expected exactly 1 touched cell, got {int((~m).sum())}"
    print("PASS  untouched mask: a NaN cell stays untouched, a changed cell does not")


def test_nan_cells_do_not_poison_the_scalars():
    """The state fields carry non-finite cells, and in light mode the scalar is the
    only source. A plain .mean() returns NaN there -- which is what every
    rmse_*_global_* in the runs already on disk actually is."""
    case = _multi_obs_case()
    thresh = float(np.median(np.nanmax(case["truth_hx"], axis=2)))
    clean = _metrics(case, "light", thresh)

    poisoned = dict(case)
    truth = case["truth"].copy()
    truth[0, 0, 0, VAR_IDX["w"]] = np.nan
    truth[1, 2, 1, VAR_IDX["w"]] = np.nan
    poisoned["truth"] = truth
    got = _metrics(poisoned, "light", thresh)

    assert np.isfinite(got["rmse_f_global_w"]), \
        "a NaN cell poisoned the protected reduction"
    assert got["n_rmse_f_global_w"] == clean["n_rmse_f_global_w"] - 2, \
        "the dropped cells are not reflected in the count that entered"
    assert np.isfinite(got["rmse_f_global_qg"]), "an unrelated variable was affected"
    naive = float(np.mean((case["xf"].mean(axis=3)[..., VAR_IDX["w"]]
                           - truth[..., VAR_IDX["w"]]) ** 2))
    assert not np.isfinite(naive), "the naive reduction was expected to be NaN here"
    print(f"PASS  protected reductions: rmse_f_global_w = "
          f"{got['rmse_f_global_w']:.4f} over {got['n_rmse_f_global_w']} of "
          f"{got['n_cells_global']} cells, where a plain .mean() gives nan")


# ── 4. tempering schedule and AOEI algebra ──────────────────────────────────
# Salvaged verbatim from the old test_da_core.py (the half that still passes).

def test_schedule_sums_to_one():
    from da.core import tempering_schedule
    for nt in [1, 2, 3, 5, 10]:
        for a in [0.0, 0.5, 2.0, 5.0]:
            s = tempering_schedule(nt, a)
            assert len(s) == nt
            assert abs(s.sum() - 1.0) < 1e-5, f"Nt={nt} as={a}: sum={s.sum()}"
    print("PASS  tempering_schedule: sums to 1")


def test_schedule_equal_weights_at_zero():
    from da.core import tempering_schedule
    for nt in [1, 3, 5, 10]:
        s = tempering_schedule(nt, 0.0)
        assert np.allclose(s, 1.0 / nt, atol=1e-5), f"Not equal at as=0: {s}"
    print("PASS  tempering_schedule: equal weights when alpha_s=0")


def test_schedule_back_loaded():
    from da.core import tempering_schedule
    s = tempering_schedule(5, 2.0)
    assert np.all(np.diff(s) > 0), f"Not back-loaded: {s}"
    print(f"PASS  tempering_schedule: back-loaded  {np.round(s, 4)}")


def test_aoei_floor():
    from da.core import aoei
    rng = np.random.default_rng(0)
    for _ in range(100):
        nobs = rng.integers(1, 10)
        R0 = np.abs(rng.standard_normal(nobs)) * 10 + 1.0
        yo = rng.standard_normal(nobs) * 5
        hxf = rng.standard_normal((nobs, 15)) * 3
        r = aoei(yo, hxf, R0)
        assert np.all(r >= R0 - 1e-6), f"Floor violated: {r} < {R0}"
    print("PASS  aoei: floor guarantee (R_tilde >= R0) for 100 random cases")


TESTS = [
    test_fortran_extension_imports,
    test_calc_ref_scalar,
    test_calc_ref_ens_matches_scalar,
    test_letkf_update,
    test_tenkf_update,
    test_aoei_update_inflates,
    test_localization_reaches_points,
    test_clamp_modes_differ_only_where_they_should,
    test_clamp_mode_is_validated,
    test_letkf_weights_reproduce_fortran,
    test_light_scalars_match_full_fields,
    test_domains_disagree_and_counts_are_written,
    test_nan_cells_are_not_counted_as_touched,
    test_nan_cells_do_not_poison_the_scalars,
    test_schedule_sums_to_one,
    test_schedule_equal_weights_at_zero,
    test_schedule_back_loaded,
    test_aoei_floor,
]

if __name__ == "__main__":
    print(f"Installation check  ({REPO})\n" + "-" * 60)
    for t in TESTS:
        t()
    print("-" * 60)
    print("Install OK -- this checkout can run DA experiments.")
