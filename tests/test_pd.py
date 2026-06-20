"""
tests/test_pd.py
===============
Tests the highest-risk PD logic: the forward-looking label (with censoring),
WOE/IV, and the ranking metrics.
"""

import numpy as np
import polars as pl
from datetime import date

from src.pd_model import target as T
from src.pd_model import woe, metrics


def _months(n):
    return [date(2018 + (m // 12), (m % 12) + 1, 1) for m in range(n)]


def test_label_event_within_horizon():
    # Loan goes 90+ (delq 3) at month index 9; earlier performing rows -> target 1.
    df = pl.DataFrame({
        "loan_id": ["A"] * 15,
        "reporting_period": _months(15),
        "delq_num": [0] * 9 + [3] + [0] * 5,
    })
    out = T.build_target(df, horizon=12, default_dpd=3).sort("reporting_period")
    # month 0 is performing and the event (month 9) is within the next 12 -> 1
    assert out.filter(pl.col("reporting_period") == date(2018, 1, 1))["target"][0] == 1
    # the delinquent month itself is excluded from the performing population
    assert date(2018, 10, 1) not in out["reporting_period"].to_list()


def test_label_clean_loan_is_zero():
    df = pl.DataFrame({
        "loan_id": ["B"] * 20,
        "reporting_period": _months(20),
        "delq_num": [0] * 20,
    })
    out = T.build_target(df, horizon=12, default_dpd=3)
    # first row has a full 12-month clean window -> target 0
    assert out.sort("reporting_period")["target"][0] == 0


def test_label_censored_rows_dropped():
    # 10-month clean loan: NO row has a full 12-month forward window -> all dropped.
    df = pl.DataFrame({
        "loan_id": ["C"] * 10,
        "reporting_period": _months(10),
        "delq_num": [0] * 10,
    })
    out = T.build_target(df, horizon=12, default_dpd=3)
    assert out.height == 0


def test_woe_no_nan_and_iv_positive():
    rng = np.random.default_rng(0)
    x = pl.Series("f", rng.normal(0, 1, 2000))
    y = pl.Series("y", (rng.random(2000) < (0.1 + 0.3 * (x.to_numpy() > 0))).astype(int))
    m = woe.fit_woe(x, y, bins=5)
    t = woe.transform_woe(x, m)
    assert t.null_count() == 0
    assert m["iv"] >= 0


def test_woe_constant_feature_no_crash():
    x = pl.Series("c", [1.0] * 100)
    y = pl.Series("y", [0, 1] * 50)
    m = woe.fit_woe(x, y, bins=5)  # must not raise
    assert "woe" in m


def test_metrics_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    r = metrics.evaluate(y, p)
    assert r["auc"] == 1.0
    assert r["gini"] == 1.0
    assert r["ks"] == 1.0
