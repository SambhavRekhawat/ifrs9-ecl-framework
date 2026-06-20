"""
tests/test_pit.py
================
Tests the Stage-6 math: Vasicek TTC<->PIT inverse, isotonic recalibration to
the true base rate, and recovery of the macro -> Z relationship.
"""

import numpy as np
import polars as pl
from datetime import date

from src.pit_calibration import vasicek, calibration, ttc_pit


def test_vasicek_inverse_roundtrip():
    rho, ttc = 0.15, 0.02
    for dr in [0.005, 0.02, 0.08, 0.20]:
        z = vasicek.implied_z(dr, ttc, rho)
        assert abs(float(vasicek.vasicek_pit(ttc, z, rho)) - dr) < 1e-9


def test_vasicek_monotone_and_sign():
    rho, ttc = 0.15, 0.02
    boom = float(vasicek.vasicek_pit(ttc, 2.0, rho))
    mid = float(vasicek.vasicek_pit(ttc, 0.0, rho))
    stress = float(vasicek.vasicek_pit(ttc, -2.0, rho))
    assert boom < mid < stress          # PIT PD falls as the economy (Z) improves
    assert boom < ttc < stress          # boom below, stress above the TTC anchor


def test_isotonic_calibration_matches_base_rate():
    rng = np.random.default_rng(0)
    n = 40000
    y = (rng.random(n) < 0.01).astype(int)             # true 1% base rate
    scores = np.clip(0.1 + 0.8 * y + rng.normal(0, 0.1, n), 0, 1)  # inflated but ranks well
    iso = calibration.fit_calibrator(scores, y)
    summ = calibration.calibration_summary(scores, y, iso)
    assert abs(summ["mean_calibrated_pd"] - summ["observed_default_rate"]) < 0.005


def test_drop_incomplete_window_removes_censored_tail():
    # 24 periods: last 12 are an all-default censoring artifact (DR=1.0).
    periods = [date(2022 + (m // 12), (m % 12) + 1, 1) for m in range(24)]
    rows_p, rows_t = [], []
    for i, p in enumerate(periods):
        t = (np.zeros(100, dtype=int) if i < 12 else np.ones(100, dtype=int))
        t[:1] = 1  # a couple of real defaults early too
        rows_p += [p] * 100
        rows_t.append(t)
    labeled = pl.DataFrame({"reporting_period": rows_p, "target": np.concatenate(rows_t)})
    clean = ttc_pit.drop_incomplete_window(labeled, horizon_months=12)
    dr = ttc_pit.default_rate_by_period(clean, min_n=10)
    assert dr.filter(pl.col("dr") >= 0.999).height == 0   # no all-default periods remain
    assert dr.height == 12                                # only complete-window periods kept


def test_macro_to_z_recovers_unemployment_link():
    rng = np.random.default_rng(0)
    rho, ttc, N = 0.15, 0.02, 4000
    periods = [date(2018 + (m // 12), (m % 12) + 1, 1) for m in range(36)]
    unemp = np.linspace(3.5, 8, 36) + rng.normal(0, 0.15, 36)
    dr_true = vasicek.vasicek_pit(ttc, -0.5 * (unemp - 5), rho)
    period_col = [p for p in periods for _ in range(N)]            # contiguous blocks
    target = np.concatenate([(rng.random(N) < dr_true[i]).astype(int) for i in range(36)])
    labeled = pl.DataFrame({"reporting_period": period_col, "target": target})
    drp = ttc_pit.add_z_factor(ttc_pit.default_rate_by_period(labeled, min_n=100), ttc, rho)
    macro = pl.DataFrame({"reporting_period": periods, "unemployment": unemp,
                          "hpi_yoy": rng.normal(5, 1, 36), "gdp_yoy": rng.normal(2, 0.5, 36),
                          "treasury_10y": rng.normal(2.5, 0.3, 36)})
    reg = ttc_pit.fit_macro_to_z(drp, macro, ["unemployment", "hpi_yoy", "gdp_yoy", "treasury_10y"])
    assert reg["r2"] > 0.5
    assert reg["coefficients"]["unemployment"] < 0   # more unemployment -> lower Z (stress)
