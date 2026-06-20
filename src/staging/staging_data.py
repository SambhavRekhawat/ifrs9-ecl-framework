"""
src/staging/staging_data.py
=========================
Builds the staged snapshot. For each loan we need PD at origination and PD now,
plus current delinquency. PD is scored with the trained PD model (best model
from Stage 5) on the feature-store panel.

  origination snapshot = each loan's earliest row (min loan_age)
  reporting snapshot    = each loan's row at `period` (or its latest row)
"""

from __future__ import annotations

import joblib
import polars as pl

from config.settings import settings
from src.feature_engineering import feature_store
from src.pd_model import models as M
from src.utils.logger import get_logger

log = get_logger(__name__)
MODELS_DIR = settings.project_root / "models"


def _features() -> list[str]:
    return settings.config["pd"]["features"]


def _load_pd_model():
    cfg = settings.config["staging"]
    model = joblib.load(MODELS_DIR / cfg["pd_model_file"])
    calib = None
    cpath = MODELS_DIR / "pd_calibrator_isotonic.joblib"
    if cpath.exists():
        calib = joblib.load(cpath)
    return model, calib


def _score(df: pl.DataFrame, model, calib) -> pl.Series:
    X = df.select(_features()).to_numpy()
    p = M.proba(model, X)
    if calib is not None:
        p = calib.predict(p)
    return pl.Series("pd", p)


def _uniq(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def _origination_pd(model, calib) -> pl.DataFrame:
    cols = _uniq(_features() + ["loan_id", "loan_age"])
    # Earliest row per loan (min loan_age). We aggregate WITHIN each loan group
    # (cheap, ~tens of rows per loan) instead of a global sort of the entire
    # feature store, which OOMs the process silently at full-history scale.
    df = (feature_store.scan().select(cols)
          .group_by("loan_id")
          .agg(pl.exclude("loan_id").sort_by("loan_age").first())
          .collect())
    log.info("Origination snapshot built: %s loans", f"{df.height:,}")
    return df.with_columns(_score(df, model, calib).alias("pd_orig")).select(["loan_id", "pd_orig"])


def _reporting_snapshot(period: str | None) -> pl.DataFrame:
    cols = _uniq(_features() + ["loan_id", "reporting_period", "delq_num"])
    lf = feature_store.scan().select(cols)
    if period:
        df = lf.filter(pl.col("reporting_period") == pl.lit(period).str.to_date()).collect()
    else:
        # Latest row per loan, via per-loan aggregation (no global sort).
        df = (lf.group_by("loan_id")
              .agg(pl.exclude("loan_id").sort_by("reporting_period").last())
              .collect())
    log.info("Reporting snapshot (%s) built: %s loans", period or "latest", f"{df.height:,}")
    return df


def staged_snapshot(period: str | None = None) -> pl.DataFrame:
    """Return loan_id, reporting_period, delq_num, pd_orig, pd_now for `period`."""
    model, calib = _load_pd_model()
    orig = _origination_pd(model, calib)
    rep = _reporting_snapshot(period)
    rep = rep.with_columns(_score(rep, model, calib).alias("pd_now"))
    out = rep.join(orig, on="loan_id", how="left").with_columns(
        pl.col("pd_orig").fill_null(pl.col("pd_now")))   # fallback if no origination row
    log.info("Staged snapshot %s: %s loans (mean PD_now %.4f, PD_orig %.4f)",
             period or "latest", f"{out.height:,}",
             out["pd_now"].mean(), out["pd_orig"].mean())
    return out.select(["loan_id", "reporting_period", "delq_num", "pd_orig", "pd_now"])