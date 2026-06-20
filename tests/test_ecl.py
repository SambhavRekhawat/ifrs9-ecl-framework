"""Tests for the ECL engine: term-structure helpers and the core computation."""
import numpy as np
import polars as pl

from src.ecl_engine import term_structure as T, ecl as E


def test_monthly_hazard_reproduces_annual_pd():
    h = T.annual_to_monthly_hazard(0.12)
    assert abs((1 - (1 - h) ** 12) - 0.12) < 1e-9


def test_cpr_to_smm_bounds():
    assert abs(T.cpr_to_smm(0.0)) < 1e-12
    smm = T.cpr_to_smm(0.10)
    assert 0 < smm < 0.10 and abs((1 - (1 - smm) ** 12) - 0.10) < 1e-9


def test_scenario_multipliers_anchored_to_one():
    pit = {"base": [0.0074] * 3, "upside": [0.0038] * 3, "downside": [0.0406] * 3}
    wp = [0.0148] * 3
    w = {"base": 0.5, "upside": 0.25, "downside": 0.25}
    mult = T.scenario_multipliers(pit, wp)
    assert abs(sum(w[s] * mult[s][0] for s in w) - 1.0) < 1e-9


def test_single_loan_ecl_hand_value():
    loans = pl.DataFrame({"loan_id": ["L1"], "pd12": [0.12], "upb": [100000.0],
                          "rate": [6.0], "term": [360.0], "stage": [2]})
    res = E.compute_ecl(loans, {"base": np.ones(12)}, {"base": 1.0}, {"base": 0.25},
                        smm=0.0, horizon=12)
    r = res["per_loan"].to_dicts()[0]
    assert 2700 < r["ecl"] < 3000
    assert abs(r["ecl_12m"] - r["ecl_lifetime"]) < 1e-6   # 12-month horizon


def test_stage3_is_lgd_times_balance():
    loans = pl.DataFrame({"loan_id": ["D"], "pd12": [0.02], "upb": [200000.0],
                          "rate": [5.0], "term": [300.0], "stage": [3]})
    res = E.compute_ecl(loans, {"base": np.ones(60), "down": np.ones(60)},
                        {"base": 0.6, "down": 0.4}, {"base": 0.15, "down": 0.35},
                        smm=0.0, horizon=60)
    expected = (0.6 * 0.15 + 0.4 * 0.35) * 200000
    assert abs(res["per_loan"]["ecl"][0] - expected) < 1e-6


def test_coverage_rises_by_stage():
    n = 600
    loans = pl.DataFrame({
        "loan_id": [f"L{i}" for i in range(n)],
        "pd12": [0.01] * n, "upb": [200000.0] * n, "rate": [5.0] * n, "term": [300.0] * n,
        "stage": [1] * 200 + [2] * 200 + [3] * 200})
    res = E.compute_ecl(loans, {"base": np.ones(60)}, {"base": 1.0}, {"base": 0.2},
                        smm=T.cpr_to_smm(0.10), horizon=60)
    cov = {r["stage"]: r["coverage_pct"] for r in res["summary"]["by_stage"]}
    assert cov[1] < cov[2] < cov[3]
