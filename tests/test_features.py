"""
tests/test_features.py
=====================
Deterministic unit tests for the feature transforms (no database needed).
"""

import polars as pl
from datetime import date

from src.feature_engineering import transforms


def _panel():
    return pl.DataFrame({
        "loan_id": ["A"] * 5,
        "reporting_period": [date(2018, m, 1) for m in range(1, 6)],
        "upb_current": [200000.0, 198000, 196000, 194000, 192000],
        "upb_orig": [200000.0] * 5,
        "ltv_orig": [80.0] * 5,
        "int_rate_current": [4.5] * 5,
        "int_rate_orig": [4.0] * 5,
        "delq_status": ["00", "00", "01", "02", "03"],
        "total_principal_current": [2000.0] * 5,
    })


def _out():
    return transforms.add_features(_panel(), balance_months=2, delq_months=2,
                                   windows=[3, 6], dpd30_min=1).sort("reporting_period")


def test_months_on_book():
    assert _out()["months_on_book"].to_list() == [0, 1, 2, 3, 4]


def test_delq_num_parsing():
    assert _out()["delq_num"].to_list() == [0, 0, 1, 2, 3]


def test_current_ltv_proxy():
    # last row: 80 * 192000/200000 = 76.8
    assert abs(_out()["current_ltv"].to_list()[-1] - 76.8) < 1e-6


def test_rate_spread():
    assert abs(_out()["rate_spread_vs_orig"].to_list()[0] - 0.5) < 1e-9


def test_rolling_max_delq():
    assert _out()["max_delq_6m"].to_list()[-1] == 3


def test_cum_principal():
    assert _out()["cum_principal_paid"].to_list() == [2000, 4000, 6000, 8000, 10000]


def test_ever_30dpd_flag():
    out = _out()
    assert out["ever_30dpd_6m"].to_list()[-1] == 1
    assert out["ever_30dpd_6m"].to_list()[0] == 0


def test_zero_upb_early_months_become_null():
    # Fannie Mae reports Current Actual UPB as 0 in a loan's first months;
    # balance-derived features must be null there, not a fake 100% paydown.
    panel = pl.DataFrame({
        "loan_id": ["X"] * 4,
        "reporting_period": [date(2018, m, 1) for m in range(1, 5)],
        "upb_current": [0.0, 0.0, 80000.0, 79500.0],
        "upb_orig": [80000.0] * 4,
        "ltv_orig": [80.0] * 4,
        "int_rate_current": [4.0] * 4,
        "int_rate_orig": [4.0] * 4,
        "delq_status": ["00"] * 4,
        "total_principal_current": [0.0, 0.0, 0.0, 500.0],
    })
    out = transforms.add_features(panel, balance_months=2, delq_months=2,
                                  windows=[3, 6]).sort("reporting_period")
    assert out["current_ltv"].to_list()[:2] == [None, None]
    assert out["upb_paydown_ratio"].to_list()[:2] == [None, None]
    assert out["current_ltv"].to_list()[2] is not None
    assert "_upb_valid" not in out.columns