"""
src/feature_engineering/transforms.py
=====================================
Pure functions that turn a loan-month panel (one vintage) into engineered
features. Everything is computed WITHIN each loan (partitioned by loan_id,
ordered by reporting_period), so it is safe to process one vintage at a time.

Features produced (mapped to Phase 4 requirements):
  loan_age              - passed through from source
  months_on_book        - sequential month index per loan (0-based)
  current_ltv           - balance-based current LTV proxy
  upb_paydown_ratio     - share of original balance repaid (payment behaviour)
  cum_principal_paid    - cumulative principal paid (payment behaviour)
  rate_spread_vs_orig   - current rate minus original rate (interest-rate spread)
  delq_num              - delinquency level in months past due
  balance_change_Nm     - balance trend over N months
  delq_change_Nm        - delinquency trend over N months
  max_delq_Wm           - rolling max delinquency over W months
  count_30dpd_Wm        - rolling count of 30+ DPD months over W months
  ever_30dpd_Wm         - rolling flag: any 30+ DPD in last W months
"""

from __future__ import annotations

import polars as pl


def add_features(df: pl.DataFrame, balance_months: int, delq_months: int,
                 windows: list[int], dpd30_min: int = 1) -> pl.DataFrame:
    df = df.sort(["loan_id", "reporting_period"])

    # Delinquency status -> integer months past due (non-numeric like 'XX' -> null).
    df = df.with_columns(
        pl.when(pl.col("delq_status").str.contains(r"^[0-9]+$"))
          .then(pl.col("delq_status").cast(pl.Int64, strict=False))
          .otherwise(None)
          .alias("delq_num")
    )

    # Fannie Mae reports Current Actual UPB as 0 in a loan's earliest months
    # (the balance is not yet populated). Treat non-positive current balance as
    # MISSING so balance-derived features don't misreport (e.g. 100% paydown).
    df = df.with_columns(
        pl.when(pl.col("upb_current") > 0).then(pl.col("upb_current"))
          .otherwise(None).alias("_upb_valid")
    )

    # Level / ratio features.
    df = df.with_columns([
        pl.int_range(0, pl.len()).over("loan_id").cast(pl.Int64).alias("months_on_book"),
        pl.when(pl.col("upb_orig") > 0)
          .then(pl.col("ltv_orig") * pl.col("_upb_valid") / pl.col("upb_orig"))
          .otherwise(None).cast(pl.Float64).alias("current_ltv"),
        pl.when(pl.col("upb_orig") > 0)
          .then(1 - pl.col("_upb_valid") / pl.col("upb_orig"))
          .otherwise(None).cast(pl.Float64).alias("upb_paydown_ratio"),
        (pl.col("int_rate_current") - pl.col("int_rate_orig")).cast(pl.Float64).alias("rate_spread_vs_orig"),
        pl.col("total_principal_current").cum_sum().over("loan_id").cast(pl.Float64).alias("cum_principal_paid"),
    ])

    # Trend features.
    df = df.with_columns([
        (pl.col("_upb_valid") / pl.col("_upb_valid").shift(balance_months).over("loan_id") - 1)
            .cast(pl.Float64).alias(f"balance_change_{balance_months}m"),
        (pl.col("delq_num") - pl.col("delq_num").shift(delq_months).over("loan_id"))
            .cast(pl.Float64).alias(f"delq_change_{delq_months}m"),
    ])

    # Rolling delinquency features.
    roll_exprs = []
    for w in windows:
        roll_exprs.append(
            pl.col("delq_num").rolling_max(window_size=w, min_samples=1).over("loan_id")
              .cast(pl.Int64).alias(f"max_delq_{w}m")
        )
    biggest = max(windows)
    roll_exprs.append(
        (pl.col("delq_num") >= dpd30_min).cast(pl.Int64)
          .rolling_sum(window_size=biggest, min_samples=1).over("loan_id")
          .cast(pl.Int64).alias(f"count_30dpd_{biggest}m")
    )
    df = df.with_columns(roll_exprs)
    df = df.with_columns(
        (pl.col(f"count_30dpd_{biggest}m") > 0).cast(pl.Int64).alias(f"ever_30dpd_{biggest}m")
    )
    return df.drop("_upb_valid")