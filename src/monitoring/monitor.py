"""
src/monitoring/monitor.py
=======================
Pure period-over-period monitoring metrics with RAG (red/amber/green) status.

  * backtest_by_period  — predicted PD vs realised 12-month default rate
  * psi_by_period       — predicted-PD distribution drift between consecutive periods
  * delinquency_trend_rag — RAG on the rise in 90+ DPD share

All functions operate on plain frames so they are fully testable.
"""

from __future__ import annotations

import polars as pl

from src.quality_checks import drift

_EPS = 1e-6


def rag(value: float | None, amber: float, red: float) -> str:
    if value is None:
        return "n/a"
    return "RED" if value >= red else ("AMBER" if value >= amber else "GREEN")


def backtest_by_period(panel: pl.DataFrame, min_n: int,
                       amber: float, red: float) -> list[dict]:
    """Per period: mean predicted PD vs realised default rate (target mean)."""
    g = (panel.group_by("period")
         .agg(pl.len().alias("n"),
              pl.col("pred_pd").mean().alias("mean_pred"),
              pl.col("target").mean().alias("realised_dr"))
         .filter(pl.col("n") >= min_n)
         .sort("period"))
    rows = []
    for r in g.iter_rows(named=True):
        dev = abs(r["mean_pred"] / max(r["realised_dr"], _EPS) - 1.0)
        rows.append({"period": str(r["period"]), "n": r["n"],
                     "mean_pred_pd": round(r["mean_pred"], 4),
                     "realised_dr": round(r["realised_dr"], 4),
                     "pred_over_realised": round(r["mean_pred"] / max(r["realised_dr"], _EPS), 3),
                     "rag": rag(dev, amber, red)})
    return rows


def psi_by_period(panel: pl.DataFrame, bins: int, amber: float, red: float) -> list[dict]:
    """PSI of predicted PD between each period and the one before it."""
    periods = panel.select("period").unique().sort("period")["period"].to_list()
    rows = []
    for prev, cur in zip(periods, periods[1:]):
        base = panel.filter(pl.col("period") == prev)["pred_pd"]
        comp = panel.filter(pl.col("period") == cur)["pred_pd"]
        val = drift.psi(base, comp, bins)
        rows.append({"from": str(prev), "to": str(cur),
                     "psi": round(val, 4) if val is not None else None,
                     "rag": rag(val, amber, red)})
    return rows


def delinquency_trend(panel: pl.DataFrame, default_dpd: int) -> list[dict]:
    """Per period: share of loans 30+ DPD (early) and 90+ DPD (default-grade)."""
    g = (panel.group_by("period")
         .agg(pl.len().alias("n"),
              (pl.col("delq_num") >= 1).mean().alias("share_30dpd"),
              (pl.col("delq_num") >= default_dpd).mean().alias("share_90dpd"))
         .sort("period"))
    return [{"period": str(r["period"]), "n": r["n"],
             "share_30dpd": round(r["share_30dpd"], 4),
             "share_90dpd": round(r["share_90dpd"], 4)} for r in g.iter_rows(named=True)]


def overall_status(backtest: list[dict], psi: list[dict]) -> str:
    """Worst RAG across the whole history (has anything ever breached)."""
    flags = [r["rag"] for r in backtest] + [r["rag"] for r in psi]
    if "RED" in flags:
        return "RED"
    if "AMBER" in flags:
        return "AMBER"
    return "GREEN" if flags else "n/a"


def latest_status(backtest: list[dict], psi: list[dict]) -> str:
    """RAG of the most recent period — is the model healthy NOW? This is the
    headline; historical breaches are context, not a permanent alarm."""
    flags = []
    if backtest:
        flags.append(backtest[-1]["rag"])
    if psi:
        flags.append(psi[-1]["rag"])
    if "RED" in flags:
        return "RED"
    if "AMBER" in flags:
        return "AMBER"
    return "GREEN" if flags else "n/a"


def breach_summary(backtest: list[dict], psi: list[dict]) -> dict:
    flags = [r["rag"] for r in backtest] + [r["rag"] for r in psi]
    return {"red": flags.count("RED"), "amber": flags.count("AMBER"),
            "green": flags.count("GREEN")}
