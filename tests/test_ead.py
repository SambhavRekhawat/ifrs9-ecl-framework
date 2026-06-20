"""Tests for the EAD amortization engine and validation logic."""
from datetime import date

import numpy as np
import polars as pl

from src.ead_model import amortization as A, ead_model as E


def test_monthly_payment_known_value():
    # $100k, 6% annual, 360 months -> ~599.55
    assert abs(float(A.monthly_payment(100000, 6.0, 360)) - 599.55) < 0.5


def test_remaining_balance_known_values():
    assert abs(float(A.remaining_balance(100000, 6.0, 360, 12)) - 98772) < 50
    assert float(A.remaining_balance(100000, 6.0, 360, 360)) < 1.0   # fully paid at maturity


def test_zero_rate_is_straight_line():
    # 0% loan pays down linearly
    assert abs(float(A.remaining_balance(120000, 0.0, 120, 60)) - 60000) < 1.0


def test_guards_zero_upb_and_term():
    assert float(A.remaining_balance(0, 5.0, 360, 12)) == 0.0
    assert float(A.remaining_balance(100000, 5.0, 0, 12)) == 0.0


def test_validate_reproduces_scheduled_paydown():
    # Build a clean amortizing loan; the projection must match actual paydown.
    upb0, rate, term = 300000.0, 5.0, 360
    rows = []
    for m in range(30):
        t = 2018 * 12 + m
        rows.append(("L1", date(t // 12, t % 12 + 1, 1),
                     float(A.remaining_balance(upb0, rate, term, m)), rate, float(term - m)))
    panel = pl.DataFrame(rows, schema=["loan_id", "reporting_period", "upb_current",
                                        "note_rate", "remaining_term"], orient="row")
    res = E.validate(panel, [3, 6, 12])
    for r in res:
        assert r["median_abs_pct_err"] < 0.001          # near-exact
        assert abs(r["actual_to_scheduled"] - 1.0) < 0.005


def test_project_ead_curtailment_scales():
    base = E.project_ead(200000, 5.0, 360, 12, curtailment=1.0)
    faster = E.project_ead(200000, 5.0, 360, 12, curtailment=0.95)
    assert np.all(faster <= base) and faster[-1] < base[-1]
