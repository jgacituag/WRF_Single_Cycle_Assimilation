"""
tests/test_ensemble_stats.py
Nerger (2022)-style CRPS / skewness / kurtosis diagnostics — unit tests.
Run with:  python test/test_ensemble_stats.py   (from repo root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from da.metrics import (
    ensemble_skew, ensemble_kurt, crps_ensemble, crps_ensemble_sorted,
    compute_single_obs_metrics, compute_multi_obs_metrics,
)


# ── exact closed-form limiting ensembles (Nerger Appendix A recipes) ────────
#
# Nerger's Table A1 gives asymptotic (~5-6% deviation at Ne=100) limits for these
# three point-mass ensembles. Rather than test against an approximate tolerance,
# the exact closed forms below were re-derived by hand directly from Eq. 25/26 for
# each recipe and confirmed numerically — so these assert *exact* equality:
#
#   - one low outlier (Ne-1 at a, one at a-d): skew = -(Ne-1)(Ne-2)/Ne^1.5,
#     kurt = [(Ne-1)^3+1]/[Ne(Ne-1)] - 3
#   - two symmetric outliers in a cluster (Ne-2 at a, one at a-d, one at a+d):
#     skew = 0, kurt = Ne/2 - 3 exactly (not just in the large-Ne limit)
#   - two equal-size groups (Ne/2 at a-d, Ne/2 at a+d, needs even Ne):
#     skew = 0, kurt = -2 exactly
#
# (Note: as extracted from the paper, the "max. kurt"/"min. kurt" table rows'
# listed kurt-limit values don't match their own listed recipe when evaluated
# exactly — most likely a table/PDF-extraction artifact. The exact algebra here
# is self-contained and doesn't depend on that table.)

def test_one_outlier_exact():
    for Ne in (24, 25, 59, 100):
        x = np.zeros(Ne); x[0] = -1.0
        sk_exact = -(Ne - 1) * (Ne - 2) / Ne ** 1.5
        ku_exact = ((Ne - 1) ** 3 + 1) / (Ne * (Ne - 1)) - 3
        assert np.isclose(ensemble_skew(x, axis=0), sk_exact)
        assert np.isclose(ensemble_kurt(x, axis=0), ku_exact)
    print("PASS  ensemble_skew/kurt: one-outlier exact closed form")


def test_two_outlier_cluster_exact():
    for Ne in (24, 25, 59, 100):
        x = np.zeros(Ne); x[0] = -1.0; x[1] = 1.0
        assert np.isclose(ensemble_skew(x, axis=0), 0.0, atol=1e-8)
        assert np.isclose(ensemble_kurt(x, axis=0), Ne / 2 - 3)
    print("PASS  ensemble_skew/kurt: two-outlier-in-cluster exact closed form (kurt=Ne/2-3)")


def test_two_group_split_exact():
    for Ne in (24, 58, 100):  # even Ne only — odd Ne can't split exactly in half
        x = np.concatenate([np.full(Ne // 2, -1.0), np.full(Ne // 2, 1.0)])
        assert np.isclose(ensemble_skew(x, axis=0), 0.0, atol=1e-8)
        assert np.isclose(ensemble_kurt(x, axis=0), -2.0)
    print("PASS  ensemble_skew/kurt: two-equal-group split exact closed form (kurt=-2, exact per the paper's own prose)")


def test_skew_kurt_nan_on_zero_variance():
    x = np.full(30, 5.0)
    assert np.isnan(ensemble_skew(x, axis=0))
    assert np.isnan(ensemble_kurt(x, axis=0))
    print("PASS  ensemble_skew/kurt: NaN (not an error) for a zero-variance ensemble")


def test_skew_kurt_no_inf_on_float32_overflow():
    """Regression guard: a mostly-zero float32 ensemble (typical for a clipped hydrometeor
    variable like qg/qr/qs) with one moderately large outlier can overflow dev**4 in float32
    before the ratio is even taken. This must map to NaN, not leak Inf downstream (an earlier
    version of ensemble_skew/kurt only special-cased an *exactly* zero variance and missed
    this — it silently produced Inf, which then poisoned every global/spatial average that
    consumed it, e.g. skew_f_global_qg, via plain np.nanmean/plain multiply not stripping Inf)."""
    import warnings
    x = np.zeros(40, dtype=np.float32)
    x[0] = 1e10
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any stray RuntimeWarning fails the test
        sk = ensemble_skew(x, axis=0)
        ku = ensemble_kurt(x, axis=0)
    assert not np.isinf(sk) and not np.isinf(ku)
    print(f"PASS  ensemble_skew/kurt: no Inf leak on float32 overflow (skew={sk:.3f}, kurt={ku})")


# ── CRPS ─────────────────────────────────────────────────────────────────────

def test_crps_degenerate_ensemble_equals_abs_error():
    ens = np.full(40, 3.0)
    truth = np.array(5.0)
    assert np.isclose(crps_ensemble(ens, truth, axis=0), 2.0)
    assert np.isclose(crps_ensemble_sorted(ens, truth, axis=0), 2.0)
    print("PASS  crps_ensemble(_sorted): degenerate ensemble reduces to |mean-truth|")


def test_crps_sorted_matches_pairwise():
    """The O(Ne log Ne) sorted estimator must numerically match the O(Ne^2)
    pairwise reference — production code (metrics.py) relies on the sorted
    variant for the full-domain multi-obs case, so this is a regression guard."""
    rng = np.random.default_rng(1)
    for Ne in (5, 24, 25, 59, 100):
        ens = rng.normal(0, 3, size=(4, 5, Ne, 2))  # extra batch + var dims, axis=2 = member
        truth = rng.normal(0, 3, size=(4, 5, 2))
        a = crps_ensemble(ens, truth, axis=2)
        b = crps_ensemble_sorted(ens, truth, axis=2)
        assert np.allclose(a, b, atol=1e-6), (Ne, np.abs(a - b).max())
    print("PASS  crps_ensemble_sorted matches crps_ensemble (O(Ne log Ne) == O(Ne^2))")


def test_crps_ne1_equals_abs_error():
    """CRPS for a single-member 'ensemble' has no division-by-variance risk (unlike
    skew/kurt) — it should just reduce to the plain absolute error."""
    ens = np.array([7.0])
    truth = np.array(4.0)
    assert np.isclose(crps_ensemble_sorted(ens, truth, axis=0), 3.0)
    print("PASS  crps_ensemble_sorted: Ne=1 reduces to |x-y| (no NaN, unlike skew/kurt)")


# ── integration: compute_single_obs_metrics / compute_multi_obs_metrics ─────

def test_single_obs_metrics_smoke():
    rng = np.random.default_rng(0)
    var_names = ["qg", "qr", "qs", "T", "P", "u", "v", "w"]
    ni, nj, nk, Ne = 7, 7, 5, 40
    xf_sub = rng.normal(0, 1, (ni, nj, nk, Ne, len(var_names))).astype(np.float32)
    xa_sub = xf_sub * 0.8 + rng.normal(0, 0.3, xf_sub.shape).astype(np.float32)
    truth_sub = rng.normal(0, 1, (ni, nj, nk, len(var_names))).astype(np.float32)
    ens_hx_sub = rng.uniform(0, 40, (ni, nj, nk, Ne)).astype(np.float32)
    hxa_sub = ens_hx_sub * 0.8 + rng.normal(0, 2, ens_hx_sub.shape).astype(np.float32)
    truth_hx_sub = rng.uniform(0, 40, (ni, nj, nk)).astype(np.float32)

    # zero-variance point (all members identical) exercises the NaN-guard path
    xf_sub[3, 3, 2, :, 0] = 0.0
    xa_sub[3, 3, 2, :, 0] = 0.0

    rloc = np.zeros((ni, nj, nk), dtype=np.float32)
    rloc[2:5, 2:5, 1:4] = 0.7
    rloc[3, 3, 2] = 1.0

    out = compute_single_obs_metrics(
        xf_sub, xa_sub, truth_sub, ens_hx_sub, hxa_sub, truth_hx_sub,
        rloc, ox_s=3, oy_s=3, oz_s=2, yo=20.0, yo_clean=20.0, var_names=var_names, Ne=Ne,
    )
    assert np.isnan(out["skew_f_point_qg"])          # exactly at the zero-variance point
    assert np.isfinite(out["skew_f_w_qg"])            # NaN-guard excludes it from the local mean
    assert np.isfinite(out["crps_f_point_obs"])
    assert np.isfinite(out["kurt_a_u_w"])
    print(f"PASS  compute_single_obs_metrics: {len(out)} keys, NaN-guard + finiteness OK")


def test_multi_obs_metrics_smoke():
    rng = np.random.default_rng(0)
    var_names = ["qg", "qr", "qs", "T", "P", "u", "v", "w"]
    nx, ny, nz, Ne = 12, 15, 6, 30
    xf = rng.normal(0, 1, (nx, ny, nz, Ne, len(var_names))).astype(np.float32)
    xa = xf * 0.7 + rng.normal(0, 0.3, xf.shape).astype(np.float32)
    truth = rng.normal(0, 1, (nx, ny, nz, len(var_names))).astype(np.float32)
    hxf_mean_field = rng.uniform(0, 40, (nx, ny, nz)).astype(np.float32)
    hxa_mean_field = hxf_mean_field * 0.8
    truth_hx_field = rng.uniform(0, 40, (nx, ny, nz)).astype(np.float32)

    out = compute_multi_obs_metrics(xa, xf, truth, hxf_mean_field, hxa_mean_field,
                                     truth_hx_field, var_names, Ne=Ne, store_fields=False)
    assert out["skew_f_field"].shape == (nx, ny, nz, len(var_names))
    assert out["crps_f_field"].shape == (nx, ny, nz, len(var_names))
    assert np.isfinite(out["crps_f_global_w"])
    print(f"PASS  compute_multi_obs_metrics: {len(out)} keys, shapes/finiteness OK")

    # Ne=1 analysis ensemble: skew/kurt undefined (NaN), CRPS still well-defined
    out1 = compute_multi_obs_metrics(xa[:, :, :, :1, :], xf, truth, hxf_mean_field,
                                      hxa_mean_field, truth_hx_field, var_names, Ne=1,
                                      store_fields=False)
    assert np.isnan(out1["skew_a_global_w"])
    assert np.isfinite(out1["crps_a_global_w"])
    print("PASS  compute_multi_obs_metrics: Ne=1 degenerate analysis ensemble handled correctly")


if __name__ == "__main__":
    print("Running tests\n" + "-" * 40)
    test_one_outlier_exact()
    test_two_outlier_cluster_exact()
    test_two_group_split_exact()
    test_skew_kurt_nan_on_zero_variance()
    test_skew_kurt_no_inf_on_float32_overflow()
    test_crps_degenerate_ensemble_equals_abs_error()
    test_crps_sorted_matches_pairwise()
    test_crps_ne1_equals_abs_error()
    test_single_obs_metrics_smoke()
    test_multi_obs_metrics_smoke()
    print("-" * 40)
    print("All tests passed.")
