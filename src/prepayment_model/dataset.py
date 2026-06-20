"""
src/prepayment/dataset.py
========================
Assembles the prepayment modelling dataset: labels each vintage with the
competing-risk prepay target, optionally joins macro to add the rate-incentive
driver (refi_incentive = loan rate - 10y Treasury), splits out-of-time, and
stratified-samples the training rows. Reuses the PD sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from config.settings import settings
from src.feature_engineering import feature_store
from src.pd_model.dataset import Dataset
from src.prepayment_model import target as tgt
from src.prepayment_model import terminal_events
from src.utils.logger import get_logger

log = get_logger(__name__)


def _proportional_sample(frame: pl.DataFrame, max_rows: int, seed: int,
                         label: str = "train") -> pl.DataFrame:
    """Downsample to ~max_rows while PRESERVING the prepay rate.

    Unlike the PD sampler (which keeps all rare positives), prepayment's
    positive class is common (~15%), so we sample both classes by the same
    fraction. This keeps the natural rate and treats max_rows as a true cap.
    """
    if frame.height <= max_rows:
        return frame
    frac = max_rows / frame.height
    pos = frame.filter(pl.col("target") == 1).sample(fraction=frac, seed=seed)
    neg = frame.filter(pl.col("target") == 0).sample(fraction=frac, seed=seed)
    out = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed)
    log.info("Prepay %s sampled to %s rows (%.2f%% prepay) from %s",
             label, f"{out.height:,}", 100 * out["target"].mean(), f"{frame.height:,}")
    return out


def _load_macro() -> pl.DataFrame:
    path = settings.project_root / "data" / "macro" / "macro_data.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.pit_calibration.macro_data` (Stage 6) first.")
    m = pl.read_parquet(path)
    if m["reporting_period"].dtype != pl.Date:
        m = m.with_columns(pl.col("reporting_period").cast(pl.Date))
    return m


def make_dataset() -> Dataset:
    cfg = settings.config["prepay"]
    base = cfg["features"]
    use_macro = bool(cfg.get("use_macro", False))

    term = terminal_events.load_terminal_events()
    keep = list(dict.fromkeys(base + ["target", "reporting_period", "vintage",
                                       "int_rate_orig", "rate_spread_vs_orig", "months_on_book"]))
    frames = []
    for f in sorted(feature_store.STORE_DIR.glob("*.parquet")):
        df = pl.read_parquet(f)
        labeled = tgt.build_prepay_target(df, term, cfg["horizon_months"])
        frames.append(labeled.select([c for c in keep if c in labeled.columns]))
        log.info("Labeled %s: %s rows (%.2f%% prepay)", f.stem, f"{labeled.height:,}",
                 100 * labeled["target"].mean() if labeled.height else 0)
    full = pl.concat(frames, how="vertical_relaxed")

    feats = list(base)
    if use_macro:
        macro = _load_macro()
        if "mortgage_30y" not in macro.columns:
            raise KeyError("macro_data is missing 'mortgage_30y'. Re-run "
                           "`python -m src.pit_calibration.macro_data` after adding MORTGAGE30US to config.")
        full = full.join(macro.select(["reporting_period", "treasury_10y", "mortgage_30y"]),
                         on="reporting_period", how="left")
        # Reconstruct the loan's CURRENT note rate, then the dynamic refi incentive
        # against the prevailing 30-yr mortgage rate (in-the-money when positive).
        full = full.with_columns(
            (pl.col("int_rate_orig") + pl.col("rate_spread_vs_orig").fill_null(0.0)).alias("_rate_now"))
        full = full.with_columns(
            (pl.col("_rate_now") - pl.col("mortgage_30y")).alias("refi_incentive"))
        full = full.with_columns(
            (pl.col("refi_incentive") * (pl.col("months_on_book").clip(0, 60) / 12.0)).alias("incentive_burnout"))
        feats = base + cfg["macro_features"]

    split = datetime.strptime(str(cfg["oot_split_date"]), "%Y-%m-%d").date()
    train = full.filter(pl.col("reporting_period") < split)
    test = full.filter(pl.col("reporting_period") >= split)
    log.info("Out-of-time split at %s | train %s rows | test %s rows",
             split, f"{train.height:,}", f"{test.height:,}")
    train = _proportional_sample(train, cfg["max_train_rows"], cfg["random_seed"], "train")
    test = _proportional_sample(test, cfg.get("max_test_rows", cfg["max_train_rows"]),
                                cfg["random_seed"], "test")

    return Dataset(X_train=train.select(feats), y_train=train["target"],
                   X_test=test.select(feats), y_test=test["target"], features=feats)
