"""
src/ingestion/transform.py
==========================
Splits one loan-level Parquet file into the two warehouse tables:

  loan_master  : one row per loan  (static origination/acquisition fields)
  loan_monthly : many rows per loan (the monthly performance time series)

Column names are renamed to clean snake_case DB names here.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.ingestion import schema
from src.utils.logger import get_logger

log = get_logger(__name__)


def _rename(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Keep only `cols` (+ vintage) and rename them to DB snake_case names."""
    keep = [c for c in cols if c in df.columns]
    if "vintage" in df.columns:
        keep = keep + ["vintage"]
    out = df.select(keep)
    mapping = {c: schema.db_name(c) for c in keep if c != "vintage"}
    return out.rename(mapping)


def split_parquet(parquet_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (loan_master_df, loan_monthly_df) for one vintage parquet."""
    df = pl.read_parquet(parquet_path)
    log.info("Splitting %s (%s rows).", parquet_path.name, f"{df.height:,}")

    loan_id = "Loan Identifier"
    period = "Monthly Reporting Period"

    # --- loan_monthly: the full time series (key + dynamic fields) ---
    monthly = _rename(df, schema.LOAN_MONTHLY_COLS)

    # --- loan_master: one row per loan, taken from its earliest reporting period ---
    # Sort so the first row per loan is the earliest month, then keep first.
    master_src = df.sort([loan_id, period]).unique(subset=[loan_id], keep="first")
    master = _rename(master_src, schema.LOAN_MASTER_COLS)

    log.info("  -> loan_master: %s loans | loan_monthly: %s rows",
             f"{master.height:,}", f"{monthly.height:,}")
    return master, monthly
