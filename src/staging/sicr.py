"""
src/staging/sicr.py
==================
IFRS 9 stage allocation rules (pure, vectorised over a Polars frame).

Stage 3  loan is credit-impaired / in default        -> lifetime ECL
Stage 2  significant increase in credit risk (SICR)   -> lifetime ECL
Stage 1  none of the above (performing)               -> 12-month ECL

SICR triggers (any one):
  * Quantitative: PD has deteriorated since origination
        PD_now / PD_orig >= rel_threshold   OR   PD_now - PD_orig >= abs_threshold
  * Backstop: 30+ DPD (IFRS 9 rebuttable presumption)            delq_num >= backstop_dpd
  * Qualitative (optional): a watchlist/forbearance flag column

Default (Stage 3): 90+ DPD (delq_num >= default_dpd) or a credit-event flag.
Stage 3 takes precedence over Stage 2.
"""

from __future__ import annotations

import polars as pl

_EPS = 1e-6


def assign_stage(df: pl.DataFrame, *, default_dpd: int, backstop_dpd: int,
                 pd_rel: float, pd_abs: float, delq_col: str = "delq_num",
                 pd_now_col: str = "pd_now", pd_orig_col: str = "pd_orig",
                 qual_col: str | None = None, default_flag_col: str | None = None) -> pl.DataFrame:
    is_default = pl.col(delq_col) >= default_dpd
    if default_flag_col and default_flag_col in df.columns:
        is_default = is_default | pl.col(default_flag_col)

    deteriorated = (
        (pl.col(pd_now_col) / pl.max_horizontal(pl.col(pd_orig_col), pl.lit(_EPS)) >= pd_rel)
        | ((pl.col(pd_now_col) - pl.col(pd_orig_col)) >= pd_abs)
    )
    backstop = pl.col(delq_col) >= backstop_dpd
    sicr = backstop | deteriorated
    if qual_col and qual_col in df.columns:
        sicr = sicr | pl.col(qual_col)

    stage = (pl.when(is_default).then(3)
               .when(sicr).then(2)
               .otherwise(1).alias("stage"))
    return df.with_columns([
        stage,
        is_default.alias("is_default"),
        backstop.alias("sicr_backstop_30dpd"),
        deteriorated.alias("sicr_pd_deterioration"),
    ])


def stage_distribution(staged: pl.DataFrame) -> pl.DataFrame:
    n = staged.height
    return (staged.group_by("stage").agg(pl.len().alias("n"))
            .sort("stage")
            .with_columns((pl.col("n") / max(n, 1) * 100).round(2).alias("pct")))


def migration_matrix(prev: pl.DataFrame, curr: pl.DataFrame,
                     key: str = "loan_id") -> pl.DataFrame:
    """Counts of stage transitions for loans present in both snapshots."""
    j = (prev.select([key, pl.col("stage").alias("stage_from")])
         .join(curr.select([key, pl.col("stage").alias("stage_to")]), on=key, how="inner"))
    return (j.group_by(["stage_from", "stage_to"]).agg(pl.len().alias("n"))
            .sort(["stage_from", "stage_to"]))
