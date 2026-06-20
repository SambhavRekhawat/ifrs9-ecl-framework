"""
src/ecl_engine/ecl_data.py
========================
Assembles the per-loan ECL inputs for the reporting snapshot:
  pd12 + stage   <- staging pipeline (Stage 10)
  upb/rate/term  <- loan_monthly latest row per loan (for EAD + discounting)
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from config.settings import settings
from src.ingestion import db
from src.staging import sicr, staging_data
from src.utils.logger import get_logger

log = get_logger(__name__)


def _ead_inputs(period: str | None) -> pl.DataFrame:
    from sqlalchemy import inspect
    engine = db.get_engine()
    mcols = {c["name"] for c in inspect(engine).get_columns("loan_monthly")}
    rate_col = "int_rate_current" if "int_rate_current" in mcols else "int_rate_orig"
    term_col = ("remaining_months_to_maturity" if "remaining_months_to_maturity" in mcols
                else "remaining_months_to_legal_maturity")
    where_period = f"AND reporting_period = '{period}'" if period else ""
    sql = f"""
        WITH ranked AS (
            SELECT loan_id, reporting_period, upb_current,
                   {rate_col} AS note_rate, {term_col} AS remaining_term,
                   ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY reporting_period DESC) AS rn
            FROM loan_monthly
            WHERE upb_current > 0 AND {term_col} > 0 {where_period}
        )
        SELECT loan_id, upb_current, note_rate, remaining_term FROM ranked WHERE rn = 1
    """
    with engine.connect() as conn:
        pdf = pd.read_sql(sql, conn)
    for c in ("upb_current", "note_rate", "remaining_term"):
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    df = pl.from_pandas(pdf).drop_nulls(["upb_current", "note_rate", "remaining_term"])
    log.info("EAD inputs: %s loans (rate=%s, term=%s)", f"{df.height:,}", rate_col, term_col)
    return df


def stage_snapshot(period: str | None = None) -> pl.DataFrame:
    """Staged snapshot: loan_id, pd_orig, pd_now, delq_num, stage, trigger flags."""
    scfg = settings.config["staging"]
    snap = staging_data.staged_snapshot(period)
    return sicr.assign_stage(
        snap, default_dpd=scfg["default_dpd"], backstop_dpd=scfg["backstop_dpd"],
        pd_rel=scfg["sicr_pd_rel_threshold"], pd_abs=scfg["sicr_pd_abs_threshold"])


def attach_ead(staged: pl.DataFrame, period: str | None = None) -> pl.DataFrame:
    """Join EAD inputs to a staged snapshot -> loans frame for the ECL engine."""
    ead = _ead_inputs(period)
    loans = (staged.select(["loan_id", pl.col("pd_now").alias("pd12"), "stage"])
             .join(ead.rename({"upb_current": "upb", "note_rate": "rate",
                               "remaining_term": "term"}), on="loan_id", how="inner"))
    log.info("ECL input set: %s loans (of %s staged, %s with EAD inputs)",
             f"{loans.height:,}", f"{staged.height:,}", f"{ead.height:,}")
    return loans


def load_loans(period: str | None = None) -> pl.DataFrame:
    """Per-loan frame: loan_id, pd12, upb, rate, term, stage."""
    return attach_ead(stage_snapshot(period), period)
