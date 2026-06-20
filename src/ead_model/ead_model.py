"""
src/ead_model/ead_model.py
=========================
Validates the amortization engine against actual observed paydown, and exposes
the EAD projection used by the ECL engine.

Validation: for each loan-month, project the balance h months forward and
compare to the loan's ACTUAL balance h months later (within-loan shift). The
ratio actual/predicted reveals curtailments (ratio < 1, faster paydown) or
modifications/forbearance (ratio > 1).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.ead_model import amortization as A


def validate(panel: pl.DataFrame, horizons: list[int]) -> list[dict]:
    panel = panel.sort(["loan_id", "reporting_period"])
    upb = panel["upb_current"].to_numpy()
    rate = panel["note_rate"].to_numpy()
    term = panel["remaining_term"].to_numpy()
    rows = []
    for h in horizons:
        pred = A.remaining_balance(upb, rate, term, h)
        p = panel.with_columns([
            pl.Series("pred", pred),
            pl.col("upb_current").shift(-h).over("loan_id").alias("actual_future"),
        ]).filter(pl.col("actual_future").is_not_null() & (pl.col("pred") > 0))
        if p.height == 0:
            continue
        pr = p["pred"].to_numpy()
        ac = p["actual_future"].to_numpy()
        mae = float(np.mean(np.abs(pr - ac)))
        mape = float(np.median(np.abs(pr - ac) / np.maximum(ac, 1.0)))
        ratio = float(np.median(ac / np.maximum(pr, 1.0)))   # actual / scheduled
        rows.append({"horizon": h, "n": p.height,
                     "mean_pred": round(float(pr.mean()), 0),
                     "mean_actual": round(float(ac.mean()), 0),
                     "mae": round(mae, 0), "median_abs_pct_err": round(mape, 4),
                     "actual_to_scheduled": round(ratio, 4)})
    return rows


def curtailment_factor(validation: list[dict], at_horizon: int = 12) -> float:
    """Median actual/scheduled paydown ratio at a horizon (1.0 = matches schedule)."""
    for r in validation:
        if r["horizon"] == at_horizon:
            return r["actual_to_scheduled"]
    return validation[-1]["actual_to_scheduled"] if validation else 1.0


def project_ead(upb, note_rate, remaining_term, horizon: int, curtailment: float = 1.0):
    """Projected EAD path (months 1..horizon). `curtailment` optionally scales the
    scheduled balance toward observed paydown behaviour (<1 = faster paydown)."""
    path = A.project_balance(upb, note_rate, remaining_term, horizon)
    return path * curtailment
