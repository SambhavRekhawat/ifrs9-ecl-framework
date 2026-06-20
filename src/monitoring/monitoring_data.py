"""
src/monitoring/monitoring_data.py
==============================
Assembles the panels the monitor needs:

  build_backtest_panel  — performing loan-months with the 12-month forward
                          default label (reusing the PD labelling) plus the
                          model's predicted PD; for back-testing and PD drift.
  delinquency_panel     — all loan-months' delinquency by period (cheap scan);
                          for the portfolio delinquency trend.
"""

from __future__ import annotations

import polars as pl

from config.settings import settings
from src.feature_engineering import feature_store
from src.pd_model import dataset as ds
from src.pit_calibration import ttc_pit
from src.staging import staging_data
from src.utils.logger import get_logger

log = get_logger(__name__)


def _add_period(df: pl.DataFrame, col: str = "reporting_period") -> pl.DataFrame:
    return df.with_columns(
        pl.date(pl.col(col).dt.year(), ((pl.col(col).dt.month() - 1) // 3) * 3 + 1, 1).alias("period"))


def build_backtest_panel() -> pl.DataFrame:
    pcfg = settings.config["pd"]
    feats = pcfg["features"]
    full = ds.build_labeled_frame(feats, pcfg["horizon_months"], pcfg["default_dpd"])
    full = ttc_pit.drop_incomplete_window(full, pcfg["horizon_months"])
    model, calib = staging_data._load_pd_model()
    full = full.with_columns(staging_data._score(full, model, calib).alias("pred_pd"))
    panel = _add_period(full).select(["period", "reporting_period", "pred_pd", "target", "delq_num"])
    log.info("Backtest panel: %s performing loan-months across %s periods",
             f"{panel.height:,}", panel["period"].n_unique())
    return panel


def delinquency_panel() -> pl.DataFrame:
    df = feature_store.scan().select(["reporting_period", "delq_num"]).collect()
    panel = _add_period(df).select(["period", "delq_num"])
    log.info("Delinquency panel: %s loan-months across %s periods",
             f"{panel.height:,}", panel["period"].n_unique())
    return panel
