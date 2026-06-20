"""
src/quality_checks/run_quality.py
=================================
The Phase-2 pipeline. Run from the project root:

    python -m src.quality_checks.run_quality

It reads loan_master + loan_monthly from PostgreSQL, runs every quality check,
builds a scorecard, writes an HTML report to reports/, and saves the detailed
results to the dq_results table.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import polars as pl

from config.settings import settings
from src.ingestion import db, schema
from src.quality_checks import checks, drift, report, scorecard
from src.utils.logger import get_logger

log = get_logger("quality.run")

# Only these (smaller) monthly columns are needed for checks -> saves memory.
_MONTHLY_COLS = ["loan_id", "reporting_period", "upb_current", "delq_status",
                 "loan_age", "total_principal_current"]


def _load(engine, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    cols = "*" if not columns else ", ".join(f'"{c}"' for c in columns)
    query = f"SELECT {cols} FROM {table}"
    with engine.connect() as conn:
        return pl.read_database(query, connection=conn)


def run(write_to_db: bool = True) -> dict:
    q = settings.config["quality"]
    engine = db.get_engine()

    log.info("Loading tables from PostgreSQL...")
    master = _load(engine, "loan_master")
    monthly = _load(engine, "loan_monthly", _MONTHLY_COLS)
    log.info("loan_master: %s rows | loan_monthly: %s rows",
             f"{master.height:,}", f"{monthly.height:,}")

    results: list[checks.CheckResult] = []

    # loan_master checks
    results += checks.validate_schema(master, "loan_master",
                                      ["loan_id"] + [schema.db_name(c) for c in schema.STATIC_COLUMNS] + ["vintage"])
    results += checks.check_completeness(master, "loan_master", q["core_columns"]["loan_master"], q["max_null_pct"])
    results += checks.check_uniqueness(master, "loan_master", ["loan_id"])
    results += checks.check_non_negative(master, "loan_master",
                                         [c for c in q["non_negative_columns"] if c in master.columns])
    results += checks.check_numeric_ranges(master, "loan_master", q["numeric_ranges"])
    results += checks.check_dates(master, "loan_master",
                                  {k: v for k, v in q["date_bounds"].items() if k in master.columns})
    results += checks.check_outliers(master, "loan_master", q["outlier_columns"])

    # loan_monthly checks
    results += checks.check_completeness(monthly, "loan_monthly", q["core_columns"]["loan_monthly"], q["max_null_pct"])
    results += checks.check_uniqueness(monthly, "loan_monthly", ["loan_id", "reporting_period"])
    results += checks.check_non_negative(monthly, "loan_monthly",
                                         [c for c in q["non_negative_columns"] if c in monthly.columns])
    results += checks.check_dates(monthly, "loan_monthly",
                                  {k: v for k, v in q["date_bounds"].items() if k in monthly.columns})

    # coverage + drift (on master)
    results += drift.check_vintage_coverage(master)
    if "vintage" in master.columns and master["vintage"].n_unique() > 1:
        d = q["drift"]
        results += drift.check_drift(master, d["features"],
                                     d["psi_warn"], d["psi_alert"], d["bins"])

    card = scorecard.build_scorecard(results)
    log.info("Overall quality: %s (%s/100) | PASS %d WARN %d FAIL %d",
             card["overall"]["grade"], card["overall"]["score"],
             card["overall"]["PASS"], card["overall"]["WARN"], card["overall"]["FAIL"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = settings.project_root / "reports" / f"data_quality_{run_id}.html"
    report.write_html(results, card, report_path, run_id)

    if write_to_db:
        try:
            report.write_db(report.results_to_df(results, run_id), engine)
        except Exception:
            log.exception("Could not write dq_results table (continuing).")

    return card


def main() -> None:
    p = argparse.ArgumentParser(description="IFRS 9 Phase 2 data quality")
    p.add_argument("--no-db", action="store_true", help="Skip writing dq_results table.")
    args = p.parse_args()
    run(write_to_db=not args.no_db)


if __name__ == "__main__":
    main()