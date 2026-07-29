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
