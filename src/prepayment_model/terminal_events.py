"""
src/prepayment/terminal_events.py
================================
Pulls each loan's terminal events from loan_monthly (the feature store doesn't
carry the exit reason). For every loan we find:
  - prepay_period  : first month with a prepay zero-balance code (01)
  - default_period : first month with a credit-event zero-balance code (02/03/09/15)
  - last_period    : last observed month (for censoring)

These drive the competing-risk prepayment label.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from config.settings import settings
from src.ingestion import db
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_terminal_events() -> pl.DataFrame:
    cfg = settings.config["prepay"]
    prepay_in = ", ".join(f"'{c}'" for c in cfg["prepay_zbc"])
    default_in = ", ".join(f"'{c}'" for c in cfg["default_zbc"])
    sql = f"""
        SELECT loan_id,
               MIN(CASE WHEN trim(CAST(zero_balance_code AS VARCHAR)) IN ({prepay_in})
                        THEN reporting_period END) AS prepay_period,
               MIN(CASE WHEN trim(CAST(zero_balance_code AS VARCHAR)) IN ({default_in})
                        THEN reporting_period END) AS default_period,
               MAX(reporting_period) AS last_period
        FROM loan_monthly
        GROUP BY loan_id
    """
    engine = db.get_engine()
    # Read via pandas: it handles columns that are NULL for the first rows then
    # dated later, which trips up Polars' read_database schema inference.
    with engine.connect() as conn:
        pdf = pd.read_sql(sql, conn)
    for c in ("prepay_period", "default_period", "last_period"):
        pdf[c] = pd.to_datetime(pdf[c], errors="coerce")
    df = pl.from_pandas(pdf)
    log.info("Loaded terminal events for %s loans (%s prepaid, %s defaulted)",
             f"{df.height:,}",
             f"{df['prepay_period'].is_not_null().sum():,}",
             f"{df['default_period'].is_not_null().sum():,}")
    return df