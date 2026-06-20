"""
src/ingestion/run_ingestion.py
==============================
The master Phase-1 pipeline. Run it from the project root:

    # 1) Just convert raw files to Parquet (no database needed yet):
    python -m src.ingestion.run_ingestion --parquet-only

    # 2) Full run: Parquet -> PostgreSQL warehouse:
    python -m src.ingestion.run_ingestion

    # Useful flags:
    --sample 0.03      keep ~3% of loans (overrides config)
    --reset            drop & recreate tables before loading

What it does for every quarterly file in data/raw:
    detect file -> CSV to Parquet (sampled) -> split into master/monthly
    -> load into PostgreSQL -> write an audit row in ingestion_log
"""

from __future__ import annotations

import argparse

from sqlalchemy import text

from config.settings import settings
from src.ingestion import csv_to_parquet, db, metadata, transform
from src.ingestion.file_detector import detect_files
from src.utils.logger import get_logger

log = get_logger("ingestion.run")


def run(parquet_only: bool = False, sample: float | None = None,
        reset: bool = False, force: bool = False) -> None:
    raw_dir = settings.paths.raw_data
    files = detect_files(raw_dir)

    if not files:
        log.error("No quarterly files found in %s. "
                  "Expected names containing YYYYQn, e.g. 2018Q1.csv", raw_dir)
        return

    log.info("Found %d quarterly file(s): %s",
             len(files), ", ".join(f.vintage for f in files))

    effective_sample = (sample if sample is not None
                        else float(settings.config["data"].get("sample_fraction", 1.0)))

    engine = None
    done: set[str] = set()
    if not parquet_only:
        engine = db.get_engine()
        if reset:
            with engine.begin() as conn:
                for tbl in ["loan_monthly", "loan_master", "ingestion_log", "metadata_catalog"]:
                    conn.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE"))
            log.info("Dropped existing tables (--reset).")
        db.create_tables(engine)
        db.write_schema_sql()
        metadata.populate_catalog(engine)
        # Vintages already loaded successfully -> skip them (incremental mode).
        done = set() if (reset or force) else db.get_ingested_vintages(engine)
        if done:
            log.info("Already ingested (will skip): %s", ", ".join(sorted(done)))

    n_skipped = n_loaded = 0
    for qf in files:
        # ---- Incremental skip ----
        if qf.vintage in done:
            log.info("Skipping %s (already ingested). Use --force to redo.", qf.vintage)
            n_skipped += 1
            continue
        if parquet_only and not force:
            existing = settings.paths.parquet / "loan_level" / f"{qf.vintage}.parquet"
            if existing.exists():
                log.info("Skipping %s (Parquet already exists). Use --force to redo.", qf.vintage)
                n_skipped += 1
                continue

        try:
            parquet_path = csv_to_parquet.convert_file(qf, sample_fraction=sample)
            if parquet_only:
                n_loaded += 1
                continue
            # Idempotent: clear any partial rows from a previous failed attempt.
            db.delete_vintage(engine, qf.vintage)
            master_df, monthly_df = transform.split_parquet(parquet_path)
            rm = db.load_dataframe(master_df, "loan_master", engine)
            rmo = db.load_dataframe(monthly_df, "loan_monthly", engine)
            metadata.log_ingestion(engine, qf.vintage, qf.path.name, rm, rmo,
                                   "SUCCESS", sample_fraction=effective_sample)
            n_loaded += 1
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to ingest %s", qf.vintage)
            if engine is not None:
                metadata.log_ingestion(engine, qf.vintage, qf.path.name, 0, 0,
                                       "FAILED", str(exc)[:500], sample_fraction=effective_sample)

    log.info("Ingestion complete. Loaded %d vintage(s), skipped %d.", n_loaded, n_skipped)


def main() -> None:
    p = argparse.ArgumentParser(description="IFRS 9 Phase 1 ingestion")
    p.add_argument("--parquet-only", action="store_true",
                   help="Stop after writing Parquet (no database).")
    p.add_argument("--sample", type=float, default=None,
                   help="Fraction of loans to keep, e.g. 0.03 (overrides config).")
    p.add_argument("--reset", action="store_true",
                   help="Drop and recreate tables before loading (full re-ingest).")
    p.add_argument("--force", action="store_true",
                   help="Re-ingest even vintages that were already loaded.")
    args = p.parse_args()
    run(parquet_only=args.parquet_only, sample=args.sample,
        reset=args.reset, force=args.force)


if __name__ == "__main__":
    main()