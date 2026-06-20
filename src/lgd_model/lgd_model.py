"""
src/lgd_model/lgd_model.py
=========================
Parsimonious LGD model for a thin default sample:
  - empirical severity stats + LGD by mark-to-market-LTV bucket (robust),
  - a smooth fractional-logit fit LGD ~ mtm_ltv (+ loan age) for a continuous
    function (kept simple given few defaults),
  - a downturn LGD via an HPI shock that raises MTM-LTV.

No statsmodels dependency: the fractional response is fit as OLS on the logit
of the (clipped) LGD, then mapped back through the logistic function.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def summary_stats(lgd: np.ndarray) -> dict:
    lgd = np.asarray(lgd, dtype=float)
    lgd = lgd[np.isfinite(lgd)]
    if lgd.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    return {"n": int(len(lgd)), "mean": round(float(np.mean(lgd)), 4),
            "median": round(float(np.median(lgd)), 4), "std": round(float(np.std(lgd)), 4),
            "p10": round(float(np.percentile(lgd, 10)), 4),
            "p90": round(float(np.percentile(lgd, 90)), 4)}


def lgd_by_bucket(mtm_ltv: np.ndarray, lgd: np.ndarray, buckets: list) -> list[dict]:
    mtm_ltv = np.asarray(mtm_ltv, dtype=float)
    lgd = np.asarray(lgd, dtype=float)
    rows = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        sel = (mtm_ltv >= lo) & (mtm_ltv < hi) & np.isfinite(mtm_ltv)
        if sel.sum() == 0:
            continue
        rows.append({"ltv_bucket": f"{lo}-{hi}", "n": int(sel.sum()),
                     "mean_lgd": round(float(lgd[sel].mean()), 4)})
    return rows


def fit_lgd(X: np.ndarray, lgd: np.ndarray, feature_names: list[str],
            eps: float = 0.001, floor: float = 0.05) -> dict:
    """Direct mean regression E[LGD|X] via OLS (clipped on predict).

    We model the conditional mean directly rather than logit(LGD): LGD here is
    zero-inflated (many full recoveries), which makes a logit-OLS collapse to the
    floor. OLS on the level targets the mean, which is what expected loss needs.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(lgd, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < 2:
        raise ValueError(f"Too few usable LGD rows after cleaning ({int(ok.sum())}). "
                         "Check that mtm_ltv / features are populated.")
    model = LinearRegression().fit(X[ok], y[ok])
    r2 = float(model.score(X[ok], y[ok]))
    return {"model": model, "features": feature_names, "floor": floor,
            "r2": round(r2, 4), "n": int(ok.sum()),
            "coefficients": dict(zip(feature_names, [round(float(c), 4) for c in model.coef_])),
            "intercept": round(float(model.intercept_), 4)}


def predict_lgd(artifact: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.clip(artifact["model"].predict(X), artifact["floor"], 1.0)


def empirical_downturn_lgd(periods, lgd, window_months: int = 24,
                           min_obs: int = 100) -> dict | None:
    """Worst rolling `window_months` mean LGD across disposition months.

    `periods` are disposition dates, `lgd` the matching realised losses. The
    monthly series is reindexed to a complete calendar range (so gaps don't
    shrink the window), then the highest-mean window meeting `min_obs`
    dispositions is returned. None if no window qualifies.
    """
    import pandas as pd

    m = pd.to_datetime(pd.Series(list(periods))).dt.to_period("M")
    df = pd.DataFrame({"m": m, "lgd": np.asarray(lgd, dtype=float)})
    g = df.groupby("m")["lgd"].agg(s="sum", c="count")
    if g.empty:
        return None
    idx = pd.period_range(g.index.min(), g.index.max(), freq="M")
    g = g.reindex(idx, fill_value=0.0)
    roll_s = g["s"].rolling(window_months).sum()
    roll_c = g["c"].rolling(window_months).sum()
    mean = (roll_s / roll_c.replace(0, np.nan)).where(roll_c >= min_obs)
    if not mean.notna().any():
        return None
    end = mean.idxmax()
    start = end - (window_months - 1)
    return {"lgd": round(float(mean.max()), 4), "window_start": str(start),
            "window_end": str(end), "n_obs": int(roll_c.loc[end]),
            "window_months": window_months}


def downturn_lgd(artifact: dict, X: np.ndarray, feature_names: list[str],
                 hpi_shock: float, benchmark: float = 0.0,
                 empirical: dict | None = None) -> dict:
    """Downturn LGD = max(empirical worst-window, model-stressed, benchmark).

    With crisis-era data the empirical worst-window LGD is the primary,
    data-driven anchor. The model-stressed path (HPI shock through the fitted
    MTM-LTV coefficient) and the benchmark floor remain as fallbacks/prudence
    anchors for when the data cannot reveal stress losses.
    """
    X = np.asarray(X, dtype=float).copy()
    base = predict_lgd(artifact, X)
    if "mtm_ltv" in feature_names:
        j = feature_names.index("mtm_ltv")
        X[:, j] = X[:, j] / (1.0 + hpi_shock)        # -20% HPI -> LTV / 0.8
    stressed = predict_lgd(artifact, X)
    base_m = float(np.nanmean(base))
    model_dt = float(np.nanmean(stressed))
    emp = float(empirical["lgd"]) if empirical else 0.0
    candidates = {"empirical": emp, "model": model_dt, "benchmark": benchmark}
    driver = max(candidates, key=candidates.get)
    final = candidates[driver]
    return {"hpi_shock": hpi_shock, "base_mean_lgd": round(base_m, 4),
            "model_downturn_lgd": round(model_dt, 4), "benchmark": benchmark,
            "empirical_downturn": empirical, "driver": driver,
            "downturn_lgd": round(final, 4),
            "uplift_ratio": round(final / max(base_m, 1e-6), 3)}
