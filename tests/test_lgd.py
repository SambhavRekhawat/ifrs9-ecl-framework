"""
tests/test_lgd.py
================
Tests the LGD model math: severity stats, LTV-bucket monotonicity, a positive
MTM-LTV coefficient, the LGD floor, and the downturn HPI uplift.
"""

import numpy as np

from src.lgd_model import lgd_model as L


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    mtm = rng.uniform(50, 110, n)
    age = rng.uniform(1, 5, n)
    # LGD rises with MTM-LTV (collateral shortfall)
    lgd = np.clip(0.02 + 0.006 * (mtm - 60) + rng.normal(0, 0.03, n), 0, 1)
    X = np.column_stack([mtm, age])
    return X, lgd, ["mtm_ltv", "loan_age_years"]


def test_summary_stats():
    s = L.summary_stats(np.array([0.0, 0.1, 0.2, 0.3, 0.4]))
    assert s["n"] == 5 and 0.19 <= s["mean"] <= 0.21


def test_bucket_monotonic_in_ltv():
    X, lgd, _ = _synthetic()
    rows = L.lgd_by_bucket(X[:, 0], lgd, [0, 60, 70, 80, 90, 100, 999])
    means = [r["mean_lgd"] for r in rows]
    assert means == sorted(means)          # LGD increases with MTM-LTV bucket


def test_fit_positive_ltv_and_floor():
    X, lgd, feats = _synthetic()
    fit = L.fit_lgd(X, lgd, feats, eps=0.001, floor=0.05)
    assert fit["coefficients"]["mtm_ltv"] > 0
    preds = L.predict_lgd(fit, X)
    assert preds.min() >= 0.05 - 1e-9 and preds.max() <= 1.0   # floor + bound respected


def test_downturn_raises_lgd():
    X, lgd, feats = _synthetic()
    fit = L.fit_lgd(X, lgd, feats)
    dt = L.downturn_lgd(fit, X, feats, hpi_shock=-0.20)
    assert dt["uplift_ratio"] > 1.0        # falling house prices raise LGD


def test_empirical_downturn_finds_worst_window():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    # benign baseline (LGD ~0.12) with an embedded crisis (LGD ~0.50)
    benign_m = pd.period_range("2015-01", "2024-12", freq="M").to_timestamp()
    crisis_m = pd.period_range("2009-06", "2011-06", freq="M").to_timestamp()
    periods, lgd = [], []
    for m in benign_m:
        periods += [m] * 40; lgd += list(np.clip(rng.normal(0.12, 0.1, 40), 0, 1))
    for m in crisis_m:
        periods += [m] * 120; lgd += list(np.clip(rng.normal(0.50, 0.1, 120), 0, 1))
    emp = L.empirical_downturn_lgd(periods, np.array(lgd), window_months=24, min_obs=100)
    # the worst window isolates the crisis, not the blended ~0.27 mean
    assert emp is not None and 0.45 <= emp["lgd"] <= 0.55
    assert emp["window_start"][:4] in ("2009", "2010")


def test_empirical_drives_downturn_when_highest(monkeypatch):
    import numpy as np
    monkeypatch.setattr(L, "predict_lgd", lambda a, X: np.full(len(X), 0.33))
    emp = {"lgd": 0.50, "window_start": "2010-02", "window_end": "2012-01",
           "n_obs": 2040, "window_months": 24}
    dt = L.downturn_lgd({"model": None, "floor": 0.05}, np.zeros((10, 1)),
                        ["mtm_ltv"], -0.2, benchmark=0.35, empirical=emp)
    assert dt["driver"] == "empirical" and dt["downturn_lgd"] == 0.50


def test_empirical_returns_none_when_too_thin():
    import numpy as np
    import pandas as pd
    periods = list(pd.period_range("2020-01", "2020-06", freq="M").to_timestamp()) * 5
    emp = L.empirical_downturn_lgd(periods, np.full(len(periods), 0.3),
                                   window_months=24, min_obs=100)
    assert emp is None            # no 24-month window meets min_obs
