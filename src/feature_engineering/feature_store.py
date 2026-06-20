"""
src/feature_engineering/feature_store.py
========================================
A simple, fast, Parquet-based feature store: one file per vintage under
data/feature_store/. Far quicker to write on a CPU laptop than a 20M-row
database insert, and Stage 5 can read it back instantly with Polars.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

STORE_DIR: Path = settings.project_root / "data" / "feature_store"


def _path(vintage: str) -> Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return STORE_DIR / f"{vintage}.parquet"


def write_vintage(df: pl.DataFrame, vintage: str) -> Path:
    """Write (overwrite) one vintage's features. Idempotent."""
    p = _path(vintage)
    df.write_parquet(p)
    log.info("Feature store: wrote %s rows for %s -> %s", f"{df.height:,}", vintage, p.name)
    return p


def reset() -> None:
    """Delete all feature-store files."""
    if STORE_DIR.exists():
        for f in STORE_DIR.glob("*.parquet"):
            f.unlink()
    log.info("Feature store cleared.")


def scan():
    """Lazily scan the whole feature store (for Stage 5)."""
    return pl.scan_parquet(str(STORE_DIR / "*.parquet"))
