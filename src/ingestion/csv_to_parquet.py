"""
src/ingestion/csv_to_parquet.py
===============================
Converts one Fannie Mae quarterly file (pipe-delimited, no header, 108 cols)
into a clean Parquet file.

Key design choices (important on a CPU laptop):
- We use Polars in STREAMING / lazy mode so we never load the whole multi-GB
  file into memory at once.
- We sample WHOLE LOANS (not random rows) using a hash of the loan id, so each
  kept loan keeps its full monthly history intact. This is configurable via
  config.yaml -> data.sample_fraction.
- Column names are assigned positionally from the verified schema.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.settings import settings
from src.ingestion import schema
from src.ingestion.file_detector import QuarterlyFile
from src.utils.logger import get_logger

log = get_logger(__name__)


def _assign_column_names(n_cols: int) -> list[str]:
    """Give a name to each physical column, robust to 108 vs 110-field files."""
    names = list(schema.COLUMNS)
    if n_cols == len(names):
        return names
    if n_cols > len(names):
        extra = [f"extra_field_{i}" for i in range(len(names) + 1, n_cols + 1)]
        log.warning(
            "File has %d columns (expected %d). Padding %d extra column(s) with "
            "generic names. Check the latest file layout.", n_cols, len(names), len(extra)
        )
        return names + extra
    raise ValueError(
        f"File has only {n_cols} columns, fewer than the expected {len(names)}. "
        "Wrong delimiter or wrong file?"
    )


def convert_file(qf: QuarterlyFile, sample_fraction: float | None = None) -> Path:
    """Convert one quarterly file to Parquet. Returns the output path."""
    if sample_fraction is None:
        sample_fraction = float(settings.config["data"].get("sample_fraction", 1.0))

    raw_path = qf.path
    out_dir: Path = settings.paths.parquet / "loan_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{qf.vintage}.parquet"

    # 1) Peek at the first line to count physical columns.
    with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()
    n_cols = first_line.count("|") + 1
    col_names = _assign_column_names(n_cols)

    log.info("Converting %s (%d columns) | sample_fraction=%.3f",
             raw_path.name, n_cols, sample_fraction)

    # 2) Lazy scan: read every column as string first (safe), no header.
    lazy = pl.scan_csv(
        raw_path,
        separator="|",
        has_header=False,
        new_columns=col_names,
        infer_schema_length=0,         # treat all as Utf8; we cast ourselves
        truncate_ragged_lines=True,
        ignore_errors=True,
    )

    # 3) Add the vintage label.
    lazy = lazy.with_columns(pl.lit(qf.vintage).alias("vintage"))

    # 4) Loan-level sampling via a stable hash of the loan id.
    if sample_fraction < 1.0:
        keep_threshold = int(sample_fraction * 1_000_000)
        lazy = lazy.filter(
            (pl.col("Loan Identifier").hash(seed=settings.random_seed) % 1_000_000)
            < keep_threshold
        )

    # 5) Cast numeric and date columns.
    casts = []
    for c in schema.NUMERIC_COLUMNS:
        casts.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
    for c in schema.DATE_COLUMNS:
        # Fannie Mae dates are MMYYYY (e.g. 032018). strict=False -> bad values become null.
        casts.append(
            pl.col(c).str.strptime(pl.Date, format="%m%Y", strict=False).alias(c)
        )
    lazy = lazy.with_columns(casts)

    # 6) Stream straight to Parquet (low memory).
    try:
        lazy.sink_parquet(out_path)
    except Exception as exc:  # fallback if streaming engine rejects the plan
        log.warning("Streaming sink failed (%s); falling back to in-memory collect.", exc)
        lazy.collect().write_parquet(out_path)

    rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    log.info("Wrote %s (%s rows).", out_path.name, f"{rows:,}")
    return out_path
