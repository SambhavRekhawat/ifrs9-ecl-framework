"""
src/eda/queries.py
==================
Aggregation queries for EDA. The monthly table has ~20M rows, so we let
PostgreSQL do the heavy GROUP BY work and only pull back small summary tables.

Definitions (from Fannie Mae glossary / statistical summaries, see config 'eda'):
  default     -> zero_balance_code in {02,03,09,15} (completed credit dispositions)
  prepayment  -> zero_balance_code = 01 (prepaid/matured)
  30+ DPD     -> Current Loan Delinquency Status >= 1
  90+ DPD     -> Current Loan Delinquency Status >= 3
"""

from __future__ import annotations

import polars as pl
from sqlalchemy.engine import Engine

from config.settings import settings

_E = settings.config["eda"]
_DPD30 = int(_E["dpd30_min_months"])
_DPD90 = int(_E["dpd90_min_months"])
_MAXAGE = int(_E["seasoning_max_age"])
_DEFCODES = ", ".join(f"'{c}'" for c in _E["default_zero_balance_codes"])
_PREPAY = _E["prepay_zero_balance_code"]

# A reusable SQL fragment: TRUE when delq_status is numeric and >= n months.
def _dpd(n: int) -> str:
    return f"(delq_status ~ '^[0-9]+$' AND CAST(delq_status AS INTEGER) >= {n})"


def _run(engine: Engine, sql: str) -> pl.DataFrame:
    with engine.connect() as conn:
        return pl.read_database(sql, connection=conn)


def master_sample(engine: Engine) -> pl.DataFrame:
    """Static fields for distribution charts (one row per loan, ~small enough)."""
    return _run(engine,
        "SELECT fico_orig, ltv_orig, cltv_orig, dti_orig, upb_orig, int_rate_orig, "
        "state, vintage FROM loan_master")


def delinquency_timeseries(engine: Engine) -> pl.DataFrame:
    return _run(engine, f"""
        SELECT reporting_period,
               count(*) AS active,
               sum(CASE WHEN {_dpd(_DPD30)} THEN 1 ELSE 0 END) AS dpd30,
               sum(CASE WHEN {_dpd(_DPD90)} THEN 1 ELSE 0 END) AS dpd90
        FROM loan_monthly
        WHERE reporting_period IS NOT NULL
        GROUP BY reporting_period
        ORDER BY reporting_period
    """)


def vintage_outcomes(engine: Engine) -> pl.DataFrame:
    return _run(engine, f"""
        SELECT vintage,
               count(DISTINCT loan_id) AS loans,
               count(DISTINCT CASE WHEN zero_balance_code IN ({_DEFCODES}) THEN loan_id END) AS defaults,
               count(DISTINCT CASE WHEN zero_balance_code = '{_PREPAY}' THEN loan_id END) AS prepaid
        FROM loan_monthly
        WHERE vintage IS NOT NULL
        GROUP BY vintage
        ORDER BY vintage
    """)


def seasoning_curve(engine: Engine) -> pl.DataFrame:
    return _run(engine, f"""
        SELECT CAST(loan_age AS INTEGER) AS loan_age,
               count(*) AS n,
               sum(CASE WHEN {_dpd(_DPD90)} THEN 1 ELSE 0 END) AS dpd90
        FROM loan_monthly
        WHERE loan_age IS NOT NULL AND loan_age BETWEEN 0 AND {_MAXAGE}
        GROUP BY CAST(loan_age AS INTEGER)
        ORDER BY loan_age
    """)


def cohort_matrix(engine: Engine) -> pl.DataFrame:
    return _run(engine, f"""
        SELECT vintage, CAST(loan_age AS INTEGER) AS age,
               count(*) AS n,
               sum(CASE WHEN {_dpd(_DPD90)} THEN 1 ELSE 0 END) AS dpd90
        FROM loan_monthly
        WHERE vintage IS NOT NULL AND loan_age BETWEEN 0 AND {_MAXAGE}
        GROUP BY vintage, CAST(loan_age AS INTEGER)
    """)
