"""
tests/test_prepay.py
===================
Tests the competing-risk prepayment label: prepay within horizon -> 1, full
clean window -> 0, default-first -> censored, terminal month excluded,
end-of-data censoring.
"""

import polars as pl
from datetime import date

from src.prepayment_model import target as T


def _months(start_year, start_month, n):
    out = []
    y, m = start_year, start_month
    for _ in range(n):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _feature_df(loan_id, periods):
    return pl.DataFrame({"loan_id": [loan_id] * len(periods),
                         "reporting_period": periods,
                         "delq_num": [0] * len(periods)})


def test_prepay_within_horizon_labels_one():
    periods = _months(2020, 1, 6)                       # Jan..Jun
    df = _feature_df("A", periods)
    term = pl.DataFrame({"loan_id": ["A"], "prepay_period": [date(2020, 6, 1)],
                         "default_period": [None], "last_period": [date(2020, 6, 1)]})
    out = T.build_prepay_target(df, term, horizon=12)
    assert out.height == 5                              # Jan..May (terminal Jun excluded)
    assert out["target"].sum() == 5                     # all prepay within 12m
    assert date(2020, 6, 1) not in out["reporting_period"].to_list()


def test_default_first_is_censored():
    periods = _months(2020, 1, 4)                       # Jan..Apr, defaults Apr
    df = _feature_df("B", periods)
    term = pl.DataFrame({"loan_id": ["B"], "prepay_period": [None],
                         "default_period": [date(2020, 4, 1)], "last_period": [date(2020, 4, 1)]})
    out = T.build_prepay_target(df, term, horizon=12)
    assert out.height == 0                              # competing default -> all censored


def test_clean_survivor_labels_zero_then_censors_tail():
    periods = _months(2020, 1, 36)                      # 3 years, never terminates
    df = _feature_df("C", periods)
    term = pl.DataFrame({"loan_id": ["C"], "prepay_period": [None],
                         "default_period": [None], "last_period": [date(2022, 12, 1)]})
    out = T.build_prepay_target(df, term, horizon=12).sort("reporting_period")
    assert out["target"].sum() == 0                     # no prepay ever
    assert out["target"][0] == 0                        # early row: full clean window
    # last 12 months have no complete forward window -> dropped
    assert out["reporting_period"].max() <= date(2021, 12, 1)
