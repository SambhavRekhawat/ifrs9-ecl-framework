"""
src/pit_calibration/ttc_pit.py
=============================
Builds the TTC <-> PIT framework:

  1. TTC PD       = long-run average default rate (the through-the-cycle anchor).
  2. Observed DR  = default rate per reporting period (point-in-time reality).
  3. Z_t          = systematic factor implied by DR_t vs TTC PD (Vasicek inverse).
  4. Macro -> Z   = regression of Z_t on macro variables, so Z (and therefore
                    PIT PD) can be projected under any macro scenario (Stage 11).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from src.pit_calibration import vasicek
from src.utils.logger import get_logger

log = get_logger(__name__)


def drop_incomplete_window(labeled: pl.DataFrame, horizon_months: int) -> pl.DataFrame:
    """Remove the last `horizon_months` of observation periods.

    Their 12-month outcome window has not fully elapsed, so non-defaulters were
    censored out during labelling and the surviving rows are ~100% defaults.
    Including them would corrupt the per-period default rate (and thus Z).
    """
    mi = pl.col("reporting_period").dt.year() * 12 + pl.col("reporting_period").dt.month()
    max_mi = labeled.select(mi.max()).item()
    return (labeled.with_columns(mi.alias("_mi"))
            .filter(pl.col("_mi") <= max_mi - horizon_months)
            .drop("_mi"))


def default_rate_by_period(labeled: pl.DataFrame, min_n: int = 500) -> pl.DataFrame:
    """Observed 12-month default rate per reporting period."""
    dr = (labeled.group_by("reporting_period")
          .agg(pl.col("target").mean().alias("dr"), pl.len().alias("n"))
          .filter(pl.col("n") >= min_n)
          .sort("reporting_period"))
    return dr


def add_z_factor(dr: pl.DataFrame, ttc_pd: float, rho: float) -> pl.DataFrame:
    z = vasicek.implied_z(dr["dr"].to_numpy(), ttc_pd, rho)
    return dr.with_columns(pl.Series("z", z))


def fit_macro_to_z(dr_z: pl.DataFrame, macro: pl.DataFrame, features: list[str]) -> dict:
    """Regress the systematic factor Z on macro variables."""
    merged = dr_z.join(macro, on="reporting_period", how="inner").drop_nulls(["z"] + features)
    if merged.height < len(features) + 2:
        raise ValueError("Not enough overlapping macro/Z observations to fit the regression.")
    X = merged.select(features).to_numpy()
    y = merged["z"].to_numpy()
    model = LinearRegression().fit(X, y)
    r2 = float(model.score(X, y))
    coefs = dict(zip(features, [round(float(c), 4) for c in model.coef_]))
    log.info("Macro->Z regression R2=%.3f | coefs=%s", r2, coefs)
    return {"model": model, "r2": round(r2, 4), "coefficients": coefs,
            "intercept": round(float(model.intercept_), 4), "features": features,
            "n_obs": merged.height}
