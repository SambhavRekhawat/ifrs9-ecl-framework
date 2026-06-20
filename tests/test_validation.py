"""Tests for the validation framework: reconciliation checks and ECL sensitivity."""
import numpy as np
import polars as pl

from src.validation import checks, ecl_sensitivity


def _ecl(s1=0.11, s2=6.9, s3=18.207, wl=0.1821, cov=0.289):
    return {"weighted_lgd": wl,
            "summary": {"coverage_pct": cov,
                        "by_stage": [{"stage": 1, "coverage_pct": s1},
                                     {"stage": 2, "coverage_pct": s2},
                                     {"stage": 3, "coverage_pct": s3}]}}


def test_coverage_monotonic_pass_and_fail():
    assert checks.coverage_monotonic_by_stage(_ecl())["passed"]
    assert not checks.coverage_monotonic_by_stage(_ecl(s1=9.0))["passed"]


def test_stage3_equals_weighted_lgd():
    assert checks.stage3_equals_weighted_lgd(_ecl(), 0.5)["passed"]
    assert not checks.stage3_equals_weighted_lgd(_ecl(s3=25.0), 0.5)["passed"]


def test_portfolio_coverage_range():
    assert checks.portfolio_coverage_in_range(_ecl(cov=0.3), 0.05, 2.0)["passed"]
    assert not checks.portfolio_coverage_in_range(_ecl(cov=9.0), 0.05, 2.0)["passed"]


def test_pd_discrimination():
    m = {"best_model": "lgbm", "models": {"lgbm": {"auc": 0.89}}}
    assert checks.pd_discrimination(m, 0.70)["passed"]
    assert not checks.pd_discrimination(m, 0.95)["passed"]


def test_effective_pd_calibration():
    # calibrated PD close to TTC -> pass; wildly off -> fail
    assert checks.effective_pd_calibration(0.0085, 0.0105, [0.3, 3.0])["passed"]
    assert not checks.effective_pd_calibration(0.20, 0.0105, [0.3, 3.0])["passed"]


def test_run_all_returns_six_checks():
    ecl = _ecl()
    scn = {"ordering_ok": True, "pit_pd_12m": {}}
    pdm = {"best_model": "lgbm", "models": {"lgbm": {"auc": 0.89}}, "calibration": []}
    cfg = {"min_auc": 0.7, "coverage_range_pct": [0.05, 2.0],
           "stage3_lgd_tolerance_pct": 0.5, "calibration_ratio_band": [0.3, 3.0]}
    res = checks.run_all(ecl, scn, pdm, cfg)
    assert len(res) == 6


def _loans(n=400):
    rng = np.random.default_rng(0)
    return pl.DataFrame({"loan_id": [f"L{i}" for i in range(n)],
                         "pd12": np.clip(rng.beta(1.6, 150, n), 1e-4, None),
                         "upb": rng.uniform(100000, 400000, n), "rate": rng.uniform(3, 6, n),
                         "term": rng.uniform(200, 360, n),
                         "stage": rng.choice([1, 2, 3], n, p=[0.95, 0.04, 0.01])})


def test_sensitivity_directions():
    H = 60
    pit = {"base": [0.0074] * H, "upside": [0.0038] * H, "downside": [0.0406] * H}
    w = {"base": 0.5, "upside": 0.25, "downside": 0.25}
    cfg = {"cpr_values": [0.0, 0.20], "horizon_values": [60],
           "downturn_lgd_values": [0.30, 0.45],
           "weight_variants": {"base": w, "severe": {"base": 0.3, "upside": 0.1, "downside": 0.6}}}
    out = ecl_sensitivity.run_grid(_loans(), pit, w, 0.126, 0.35, "downside", 0.10, H, cfg)
    g = {(r["lever"], r["value"]): r["total_ecl"] for r in out["grid"]}
    assert g[("annual_cpr", "0.2")] < g[("annual_cpr", "0.0")]            # prepay lowers ECL
    assert g[("downturn_lgd", "0.45")] > g[("downturn_lgd", "0.3")]       # higher LGD raises ECL
    assert g[("scenario_weights", "severe")] > g[("scenario_weights", "base")]
