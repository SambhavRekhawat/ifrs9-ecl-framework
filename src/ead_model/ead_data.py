"""
src/ead_model/ead_data.py
========================
Pulls a sampled loan-month panel (current balance, note rate, remaining term)
from loan_monthly to validate the amortization engine against actual paydown.
Sampling is by hashed loan_id so whole loan histories are kept intact.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from config.settings import settings
from src.ingestion import db
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_ead_panel() -> pl.DataFrame:
    from sqlalchemy import inspect
    cfg = settings.config["ead"]
    mod = int(cfg["sample_loan_mod"])
    engine = db.get_engine()
    mcols = {c["name"] for c in inspect(engine).get_columns("loan_monthly")}

    rate_col = "int_rate_current" if "int_rate_current" in mcols else "int_rate_orig"
    if "remaining_months_to_maturity" in mcols:
        term_col = "remaining_months_to_maturity"
    elif "remaining_months_to_legal_maturity" in mcols:
        term_col = "remaining_months_to_legal_maturity"
    else:
        raise KeyError("No remaining-term column found in loan_monthly.")

    sql = f"""
        SELECT loan_id, reporting_period,
               upb_current,
               {rate_col} AS note_rate,
               {term_col} AS remaining_term
        FROM loan_monthly
        WHERE upb_current > 0 AND {term_col} > 0
          AND mod(abs(hashtext(loan_id)), {mod}) = 0
    """
    with engine.connect() as conn:
        pdf = pd.read_sql(sql, conn)
    for c in ("upb_current", "note_rate", "remaining_term"):
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    pdf["reporting_period"] = pd.to_datetime(pdf["reporting_period"], errors="coerce")
    df = pl.from_pandas(pdf).drop_nulls(["upb_current", "note_rate", "remaining_term"])
    log.info("EAD panel: %s rows across %s loans (rate=%s, term=%s)",
             f"{df.height:,}", f"{df['loan_id'].n_unique():,}", rate_col, term_col)
    return df