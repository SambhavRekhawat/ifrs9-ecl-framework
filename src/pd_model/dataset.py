"""
src/pd_model/dataset.py
======================
Assembles the PD modelling dataset from the Parquet feature store:
labels each vintage, concatenates, splits out-of-time (train on earlier
reporting periods, test on later), and stratified-samples the training rows
to keep things fast on a laptop (always keeping ALL the rare default rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from config.settings import settings
from src.feature_engineering import feature_store
from src.pd_model import target as target_mod
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Dataset:
    X_train: pl.DataFrame
    y_train: pl.Series
    X_test: pl.DataFrame
    y_test: pl.Series
    features: list[str]


def build_labeled_frame(features: list[str], horizon: int, default_dpd: int) -> pl.DataFrame:
    """Label every vintage parquet and stack them."""
    frames = []
    files = sorted(feature_store.STORE_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature-store files in {feature_store.STORE_DIR}. Run Stage 4 first.")
    keep = features + ["target", "reporting_period", "vintage"]
    for f in files:
        df = pl.read_parquet(f)
        labeled = target_mod.build_target(df, horizon, default_dpd)
        frames.append(labeled.select([c for c in keep if c in labeled.columns]))
        log.info("Labeled %s: %s rows (%.2f%% default)",
                 f.stem, f"{labeled.height:,}",
                 100 * labeled["target"].mean() if labeled.height else 0)
    return pl.concat(frames, how="vertical_relaxed")


def _stratified_sample(train: pl.DataFrame, max_rows: int, seed: int) -> pl.DataFrame:
    if train.height <= max_rows:
        return train
    pos = train.filter(pl.col("target") == 1)
    neg = train.filter(pl.col("target") == 0)
    n_neg = max(max_rows - pos.height, 1000)
    neg_s = neg.sample(n=min(n_neg, neg.height), seed=seed)
    out = pl.concat([pos, neg_s]).sample(fraction=1.0, shuffle=True, seed=seed)
    log.info("Sampled train to %s rows (%s default, %s non-default)",
             f"{out.height:,}", f"{pos.height:,}", f"{neg_s.height:,}")
    return out


def make_dataset() -> Dataset:
    cfg = settings.config["pd"]
    features = cfg["features"]
    full = build_labeled_frame(features, cfg["horizon_months"], cfg["default_dpd"])

    split = datetime.strptime(str(cfg["oot_split_date"]), "%Y-%m-%d").date()
    train = full.filter(pl.col("reporting_period") < split)
    test = full.filter(pl.col("reporting_period") >= split)
    log.info("Out-of-time split at %s | train %s rows | test %s rows",
             split, f"{train.height:,}", f"{test.height:,}")

    train = _stratified_sample(train, cfg["max_train_rows"], cfg["random_seed"])

    return Dataset(
        X_train=train.select(features), y_train=train["target"],
        X_test=test.select(features), y_test=test["target"], features=features,
    )
