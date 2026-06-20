"""
src/feature_engineering/run_features.py
=======================================
Phase-4 pipeline. Run from the project root:

    python -m src.feature_engineering.run_features              # all vintages
    python -m src.feature_engineering.run_features --vintages 2018Q1 2018Q2
    python -m src.feature_engineering.run_features --reset      # clear store first

Per vintage: load the loan-month panel (loan_monthly joined to loan_master) from
PostgreSQL, compute engineered + vintage + macro features, and write a Parquet
file to the feature store. Processing one vintage at a time keeps memory low.
"""

from __future__ import annotations

import argparse

import pandas as pd
import polars as pl

from config.settings import settings
from src.feature_engineering import feature_store, macro, transforms
from src.ingestion import db
from src.utils.logger import get_logger

log = get_logger("features.run")

# Final columns kept in the feature store.
_ID_COLS = ["loan_id", "reporting_period", "vintage", "state"]
_RAW_COLS = ["fico_orig", "ltv_orig", "dti_orig", "upb_orig", "int_rate_orig",
             "upb_current", "int_rate_current", "loan_age", "delq_num"]


def _load_panel(engine, vintage: str) -> pl.DataFrame:
    sql = f"""
        SELECT m.loan_id, m.reporting_period, m.loan_age, m.upb_current,
               m.int_rate_current, m.delq_status, m.total_principal_current, m.vintage,
               l.fico_orig, l.ltv_orig, l.dti_orig, l.upb_orig, l.int_rate_orig, l.state
        FROM loan_monthly m
        JOIN loan_master l ON m.loan_id = l.loan_id
        WHERE m.vintage = '{vintage}'
    """
    with engine.connect() as conn:
        # pandas read_sql infers column dtypes across all rows, avoiding the
        # Polars read_database first-N-rows inference bug (e.g. a column that
        # looks int/null early then hits a float like 1158.81 in crisis-era data).
        pdf = pd.read_sql(sql, conn)
    return pl.from_pandas(pdf)


def build_vintage_features(panel: pl.DataFrame, vstats: dict, engine, cfg: dict) -> pl.DataFrame:
    """Pure-ish builder: features + vintage metrics + (optional) macro. Testable."""
    feat = transforms.add_features(
        panel,
        balance_months=cfg["balance_change_months"],
        delq_months=cfg["delq_trend_months"],
        windows=cfg["rolling_windows"],
        dpd30_min=cfg["dpd30_min_months"],
    )
    # vintage metrics (constant for the vintage)
    feat = feat.with_columns([
        pl.lit(vstats["vintage_loan_count"]).cast(pl.Int64).alias("vintage_loan_count"),
        pl.lit(vstats["vintage_avg_fico"]).cast(pl.Float64).alias("vintage_avg_fico"),
    ])
    if engine is not None:
        feat = macro.join_macro(feat, engine)

    engineered = [c for c in feat.columns if c not in _ID_COLS + _RAW_COLS
                  + ["delq_status", "total_principal_current"]]
    keep = _ID_COLS + _RAW_COLS + sorted(set(engineered))
    keep = [c for c in keep if c in feat.columns]
    return feat.select(keep)


def run(vintages: list[str] | None = None, reset: bool = False) -> None:
    cfg = settings.config["features"]
    engine = db.get_engine()

    if reset:
        feature_store.reset()

    # vintage stats from loan_master
    with engine.connect() as conn:
        vstats_df = pl.read_database(
            "SELECT vintage, count(*) AS vintage_loan_count, avg(fico_orig) AS vintage_avg_fico "
            "FROM loan_master GROUP BY vintage", connection=conn)
    all_vintages = sorted(vstats_df["vintage"].to_list())
    targets = vintages if vintages else all_vintages
    log.info("Building features for %d vintage(s): %s", len(targets), ", ".join(targets))

    for v in targets:
        row = vstats_df.filter(pl.col("vintage") == v)
        if row.height == 0:
            log.warning("Vintage %s not found in loan_master; skipping.", v)
            continue
        vstats = {"vintage_loan_count": int(row["vintage_loan_count"][0]),
                  "vintage_avg_fico": float(row["vintage_avg_fico"][0] or 0.0)}
        log.info("Loading panel for %s ...", v)
        panel = _load_panel(engine, v)
        if panel.height == 0:
            log.warning("No rows for vintage %s; skipping.", v)
            continue
        feats = build_vintage_features(panel, vstats, engine, cfg)
        feature_store.write_vintage(feats, v)

    log.info("Feature engineering complete. Store: %s", feature_store.STORE_DIR)


def main() -> None:
    p = argparse.ArgumentParser(description="IFRS 9 Phase 4 feature engineering")
    p.add_argument("--vintages", nargs="*", default=None, help="Specific vintages (default: all).")
    p.add_argument("--reset", action="store_true", help="Clear the feature store first.")
    args = p.parse_args()
    run(vintages=args.vintages, reset=args.reset)


if __name__ == "__main__":
    main()