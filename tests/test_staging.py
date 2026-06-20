"""Tests for the IFRS 9 staging / SICR rules engine."""
import polars as pl

from src.staging import sicr


def _frame():
    return pl.DataFrame({
        "loan_id": ["a", "b", "c", "d", "e", "f"],
        "delq_num": [0, 1, 0, 0, 3, 4],
        "pd_orig": [0.010, 0.010, 0.010, 0.005, 0.010, 0.010],
        "pd_now":  [0.010, 0.010, 0.040, 0.020, 0.010, 0.500],
    })


def _staged():
    return sicr.assign_stage(_frame(), default_dpd=3, backstop_dpd=1, pd_rel=2.0, pd_abs=0.01)


def test_performing_is_stage1():
    st = _staged()
    assert st.filter(pl.col("loan_id") == "a")["stage"][0] == 1


def test_30dpd_backstop_is_stage2():
    st = _staged()
    row = st.filter(pl.col("loan_id") == "b")
    assert row["stage"][0] == 2 and row["sicr_backstop_30dpd"][0]


def test_pd_deterioration_relative_and_absolute():
    st = _staged()
    assert st.filter(pl.col("loan_id") == "c")["stage"][0] == 2   # 4x relative
    assert st.filter(pl.col("loan_id") == "d")["stage"][0] == 2   # +1.5pp absolute


def test_default_90dpd_is_stage3_and_takes_precedence():
    st = _staged()
    # 'f' is both deteriorated and 90+DPD -> Stage 3 wins
    assert st.filter(pl.col("loan_id") == "e")["stage"][0] == 3
    assert st.filter(pl.col("loan_id") == "f")["stage"][0] == 3


def test_distribution_sums_to_n():
    dist = sicr.stage_distribution(_staged())
    assert dist["n"].sum() == 6
    assert abs(dist["pct"].sum() - 100.0) < 0.1


def test_migration_matrix_counts_common_loans():
    st = _staged()
    prev = st.select(["loan_id", "stage"])
    curr = st.select(["loan_id"]).with_columns(pl.lit(1).alias("stage"))
    mm = sicr.migration_matrix(prev, curr)
    assert mm["n"].sum() == 6
