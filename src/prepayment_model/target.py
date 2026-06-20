"""
src/prepayment/target.py
========================
Builds the prepayment label: "does this active, performing loan PREPAY within
the next H months?" - treating default as a competing risk.

Rules (cause-specific, competing-risk aware):
  - target = 1 if the loan prepays within H months (and not after defaulting).
  - target = 0 if it survives the full H-month window without prepaying.
  - target = null (dropped) if it exits via DEFAULT first (competing risk) or
    the window is right-censored by the end of data.

Observation population = active (not yet terminated) performing loans
(delq_num < 3), the same risk set the PD model uses - so default and prepayment
are proper competing risks from one population.
"""

from __future__ import annotations

import polars as pl


def _mi(col: pl.Expr) -> pl.Expr:
    return col.dt.year() * 12 + col.dt.month()


def build_prepay_target(df: pl.DataFrame, terminal: pl.DataFrame, horizon: int) -> pl.DataFrame:
    term = terminal.with_columns([
        _mi(pl.col("prepay_period").cast(pl.Date)).alias("prepay_mi"),
        _mi(pl.col("default_period").cast(pl.Date)).alias("default_mi"),
        _mi(pl.col("last_period").cast(pl.Date)).alias("last_mi"),
    ]).select(["loan_id", "prepay_mi", "default_mi", "last_mi"])

    df = df.join(term, on="loan_id", how="left")
    df = df.with_columns(_mi(pl.col("reporting_period")).alias("t_mi"))
    H = horizon

    prepay_in = (pl.col("prepay_mi").is_not_null()
                 & (pl.col("prepay_mi") > pl.col("t_mi"))
                 & (pl.col("prepay_mi") <= pl.col("t_mi") + H))
    default_in = (pl.col("default_mi").is_not_null()
                  & (pl.col("default_mi") > pl.col("t_mi"))
                  & (pl.col("default_mi") <= pl.col("t_mi") + H))
    prepay_first = prepay_in & (pl.col("default_mi").is_null()
                                | (pl.col("prepay_mi") <= pl.col("default_mi")))
    default_first = default_in & (pl.col("prepay_mi").is_null()
                                  | (pl.col("default_mi") < pl.col("prepay_mi")))
    full_window = pl.col("last_mi") >= pl.col("t_mi") + H

    df = df.with_columns(
        pl.when(prepay_first).then(1)
          .when(default_first).then(None)                       # competing risk -> censor
          .when(~prepay_in & ~default_in & full_window).then(0)
          .otherwise(None)
          .alias("target")
    )

    active = ((pl.col("prepay_mi").is_null() | (pl.col("t_mi") < pl.col("prepay_mi")))
              & (pl.col("default_mi").is_null() | (pl.col("t_mi") < pl.col("default_mi"))))
    df = df.filter((pl.col("delq_num").fill_null(0) < 3) & active)
    df = df.filter(pl.col("target").is_not_null())

    return df.drop([c for c in ["prepay_mi", "default_mi", "last_mi", "t_mi"] if c in df.columns])
