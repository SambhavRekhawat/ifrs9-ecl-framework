"""
src/pd_model/target.py
=====================
Builds the PD modelling label: "does this performing loan reach default
(90+ DPD) within the next H months?"

Careful handling:
  - We only keep PERFORMING observations (current delinquency < default_dpd).
  - target = 1 if the loan reaches 90+ DPD in the next H months.
  - target = 0 only if a FULL H-month window is observed with no event.
  - rows whose forward window is cut off by the end of data (right-censored,
    no event) are dropped (target = null), so we never mislabel "no data" as
    "no default".
"""

from __future__ import annotations

import polars as pl


def build_target(df: pl.DataFrame, horizon: int, default_dpd: int) -> pl.DataFrame:
    df = df.sort(["loan_id", "reporting_period"])

    event = (pl.col("delq_num").fill_null(0) >= default_dpd).cast(pl.Int8)
    df = df.with_columns(event.alias("_event"))

    # Forward window: any event in the next `horizon` months (excludes current).
    fwd_cols = [pl.col("_event").shift(-k).over("loan_id") for k in range(1, horizon + 1)]
    df = df.with_columns(pl.max_horizontal(fwd_cols).alias("_fwd_event"))

    # Months of data remaining for this loan (to know if the window is complete).
    df = df.with_columns(
        (pl.col("reporting_period").dt.year() * 12 + pl.col("reporting_period").dt.month()).alias("_midx")
    )
    df = df.with_columns(pl.col("_midx").max().over("loan_id").alias("_midx_max"))
    df = df.with_columns((pl.col("_midx_max") - pl.col("_midx")).alias("_months_left"))

    df = df.with_columns(
        pl.when(pl.col("_fwd_event") == 1).then(1)
          .when((pl.col("_fwd_event").fill_null(0) == 0) & (pl.col("_months_left") >= horizon)).then(0)
          .otherwise(None)
          .alias("target")
    )

    # Keep performing observations with a usable label.
    df = df.filter(pl.col("delq_num").fill_null(0) < default_dpd)
    df = df.filter(pl.col("target").is_not_null())

    return df.drop([c for c in ["_event", "_fwd_event", "_midx", "_midx_max", "_months_left"] if c in df.columns])
