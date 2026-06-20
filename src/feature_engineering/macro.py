"""
src/feature_engineering/macro.py
================================
Optional macroeconomic features. Joins a `macro_data` table (FRED + FHFA series,
keyed by reporting_period) onto the panel IF that table exists.

Until the macro data is ingested (a later step, needed for Stage 6 PIT
calibration), this safely no-ops and leaves the panel unchanged.
"""

from __future__ import annotations

import polars as pl
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from src.utils.logger import get_logger

log = get_logger(__name__)


def macro_table_exists(engine: Engine) -> bool:
    try:
        return inspect(engine).has_table("macro_data")
    except Exception:
        return False


def join_macro(df: pl.DataFrame, engine: Engine) -> pl.DataFrame:
    """Left-join macro features on reporting_period; return df unchanged if absent."""
    if not macro_table_exists(engine):
        log.info("No macro_data table yet - skipping macro features (will add in the macro step).")
        return df
    with engine.connect() as conn:
        macro = pl.read_database("SELECT * FROM macro_data", connection=conn)
    if "reporting_period" not in macro.columns:
        log.warning("macro_data has no reporting_period column; skipping join.")
        return df
    return df.join(macro, on="reporting_period", how="left")
