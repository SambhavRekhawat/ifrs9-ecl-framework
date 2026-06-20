"""Tests for the monitoring period metrics."""
import numpy as np
import polars as pl
from datetime import date

from src.monitoring import monitor as M


def _block(period, n, pred_mean, realised, seed):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "period": [period] * n,
        "pred_pd": np.clip(rng.lognormal(np.log(pred_mean), 0.4, n), 1e-4, 1),
        "target": rng.binomial(1, realised, n),
        "delq_num": rng.choice([0, 1, 3], n, p=[0.96, 0.03, 0.01])})


def test_rag_thresholds():
    assert M.rag(0.05, 0.10, 0.25) == "GREEN"
    assert M.rag(0.15, 0.10, 0.25) == "AMBER"
    assert M.rag(0.40, 0.10, 0.25) == "RED"
    assert M.rag(None, 0.10, 0.25) == "n/a"


def test_backtest_green_when_calibrated_red_when_off():
    panel = pl.concat([_block(date(2022, 1, 1), 5000, 0.010, 0.010, 1),
                       _block(date(2022, 4, 1), 5000, 0.060, 0.010, 2)])  # 6x over-predict
    bt = M.backtest_by_period(panel, min_n=500, amber=0.5, red=1.0)
    by = {r["period"]: r["rag"] for r in bt}
    assert by["2022-01-01"] == "GREEN"
    assert by["2022-04-01"] == "RED"


def test_backtest_respects_min_n():
    panel = _block(date(2022, 1, 1), 100, 0.01, 0.01, 3)
    assert M.backtest_by_period(panel, min_n=500, amber=0.5, red=1.0) == []


def test_psi_detects_distribution_shift():
    panel = pl.concat([_block(date(2022, 1, 1), 5000, 0.010, 0.01, 4),
                       _block(date(2022, 4, 1), 5000, 0.080, 0.01, 5)])  # big PD shift
    psi = M.psi_by_period(panel, bins=10, amber=0.10, red=0.25)
    assert len(psi) == 1 and psi[0]["rag"] == "RED"


def test_delinquency_trend_shares():
    rng = np.random.default_rng(7)
    panel = pl.DataFrame({"period": [date(2022, 1, 1)] * 1000,
                          "delq_num": rng.choice([0, 1, 3], 1000, p=[0.9, 0.06, 0.04])})
    out = M.delinquency_trend(panel, default_dpd=3)
    assert len(out) == 1
    assert 0.08 <= out[0]["share_30dpd"] <= 0.12   # ~10% are 30+ DPD
    assert 0.02 <= out[0]["share_90dpd"] <= 0.06   # ~4% are 90+ DPD


def test_overall_status_priority():
    g = [{"rag": "GREEN"}]
    a = [{"rag": "AMBER"}]
    r = [{"rag": "RED"}]
    assert M.overall_status(g, g) == "GREEN"
    assert M.overall_status(g, a) == "AMBER"
    assert M.overall_status(a, r) == "RED"
    assert M.overall_status([], []) == "n/a"


def test_latest_status_uses_most_recent_period():
    # historical RED but latest period GREEN -> latest_status GREEN, overall RED
    bt = [{"rag": "RED"}, {"rag": "AMBER"}, {"rag": "GREEN"}]
    psi = [{"rag": "RED"}, {"rag": "GREEN"}]
    assert M.latest_status(bt, psi) == "GREEN"
    assert M.overall_status(bt, psi) == "RED"


def test_breach_summary_counts():
    bt = [{"rag": "RED"}, {"rag": "GREEN"}]
    psi = [{"rag": "AMBER"}, {"rag": "GREEN"}]
    b = M.breach_summary(bt, psi)
    assert b == {"red": 1, "amber": 1, "green": 2}
