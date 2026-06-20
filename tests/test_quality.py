"""
tests/test_quality.py
=====================
Fast, deterministic unit tests for the data-quality checks.
Run with: pytest tests/test_quality.py
"""

import polars as pl
from datetime import date

from src.quality_checks import checks, drift, scorecard


def test_uniqueness_detects_duplicate():
    df = pl.DataFrame({"loan_id": ["A", "B", "B"]})
    res = checks.check_uniqueness(df, "t", ["loan_id"])
    assert res[0].status == "FAIL"
    assert res[0].value == 1.0


def test_non_negative_flags_negative():
    df = pl.DataFrame({"upb": [100.0, -5.0, 50.0]})
    res = checks.check_non_negative(df, "t", ["upb"])
    assert res[0].status == "FAIL"
    assert res[0].value == 1.0


def test_numeric_range_warns_out_of_range():
    df = pl.DataFrame({"fico_orig": [700.0, 200.0, 800.0]})
    res = checks.check_numeric_ranges(df, "t", {"fico_orig": [300, 850]})
    assert res[0].status == "WARN"


def test_completeness_passes_when_full():
    df = pl.DataFrame({"loan_id": ["A", "B"], "x": [1, 2]})
    res = checks.check_completeness(df, "t", ["loan_id"], max_null_pct=5.0)
    assert any(r.metric == "null_pct" and r.status == "PASS" for r in res)


def test_vintage_coverage_finds_gap():
    df = pl.DataFrame({"vintage": ["2018Q1", "2018Q3"]})  # missing 2018Q2
    res = drift.check_vintage_coverage(df)
    assert res[0].status == "WARN"
    assert "2018Q2" in res[0].message


def test_psi_zero_for_identical():
    s = pl.Series([float(i % 100) for i in range(500)])
    assert abs(drift.psi(s, s)) < 1e-6


def test_scorecard_grades():
    results = [
        checks.CheckResult("t", "x", None, "m", 0, 0, "PASS", ""),
        checks.CheckResult("t", "x", None, "m", 1, 0, "FAIL", ""),
    ]
    card = scorecard.build_scorecard(results)
    assert card["overall"]["FAIL"] == 1
    assert card["overall"]["score"] == 50.0  # pass-rate: 1 of 2 scored checks pass