"""
src/ingestion/metadata.py
=========================
Writes the audit trail (ingestion_log) and the data dictionary
(metadata_catalog) into the warehouse.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.ingestion import schema
from src.utils.logger import get_logger

log = get_logger(__name__)


def log_ingestion(engine: Engine, vintage: str, source_file: str,
                  rows_master: int, rows_monthly: int,
                  status: str, message: str = "", sample_fraction: float | None = None) -> None:
    """Insert one audit row describing an ingestion run."""
    stmt = text(
        "INSERT INTO ingestion_log "
        "(vintage, source_file, rows_master, rows_monthly, sample_fraction, status, message, finished_at) "
        "VALUES (:v, :f, :rm, :rmo, :sf, :s, :m, now())"
    )
    with engine.begin() as conn:
        conn.execute(stmt, {"v": vintage, "f": source_file, "rm": rows_master,
                            "rmo": rows_monthly, "sf": sample_fraction,
                            "s": status, "m": message})


def populate_catalog(engine: Engine) -> None:
    """Fill metadata_catalog with a description of every warehouse column."""
    rows = []
    for col in schema.STATIC_COLUMNS:
        rows.append(("loan_master", schema.db_name(col), col,
                     _dtype(col), True))
    for col in schema.DYNAMIC_COLUMNS:
        rows.append(("loan_monthly", schema.db_name(col), col,
                     _dtype(col), False))

    stmt = text(
        "INSERT INTO metadata_catalog (table_name, db_column, official_name, data_type, is_static) "
        "VALUES (:t, :d, :o, :dt, :s) "
        "ON CONFLICT (table_name, db_column) DO NOTHING"
    )
    with engine.begin() as conn:
        for t, d, o, dt, s in rows:
            conn.execute(stmt, {"t": t, "d": d, "o": o, "dt": dt, "s": s})
    log.info("Populated metadata_catalog with %d column descriptions.", len(rows))


def _dtype(col: str) -> str:
    if col in schema.DATE_COLUMNS:
        return "date"
    if col in schema.NUMERIC_COLUMNS:
        return "numeric"
    return "text"