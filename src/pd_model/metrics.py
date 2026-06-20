"""
src/pd_model/metrics.py
======================
Standard PD performance metrics.
  AUC   - ranking power (0.5 = random, 1.0 = perfect)
  Gini  - 2*AUC - 1
  KS    - max separation between cumulative default / non-default distributions
  precision/recall at a threshold
  Brier - mean squared error of probabilities (calibration)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             roc_auc_score)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    y = np.asarray(y_true)[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, y_prob), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 6),
        "default_rate": round(float(y_true.mean()), 4),
        "n": int(len(y_true)),
    }


def _downsample(idx_len: int, max_points: int) -> np.ndarray:
    if idx_len <= max_points:
        return np.arange(idx_len)
    return np.unique(np.linspace(0, idx_len - 1, max_points).round().astype(int))


def roc_curve_points(y_true, y_prob, max_points: int = 60) -> list[dict]:
    """Downsampled ROC curve (false-positive rate vs true-positive rate)."""
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    keep = _downsample(len(fpr), max_points)
    return [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in keep]


def ks_curve_points(y_true, y_prob, max_points: int = 60) -> list[dict]:
    """Cumulative bad/good distributions across the score-sorted population.

    x = fraction of population (lowest to highest predicted PD)
    cum_bad / cum_good = cumulative share of defaults / non-defaults captured.
    The KS statistic is the maximum vertical gap between the two curves.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    order = np.argsort(y_prob)
    y = y_true[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    pop = np.arange(1, len(y) + 1) / len(y)
    keep = _downsample(len(y), max_points)
    return [{"pop_pct": round(float(pop[i]), 4),
             "cum_bad": round(float(cum_bad[i]), 4),
             "cum_good": round(float(cum_good[i]), 4)} for i in keep]


def calibration_table(y_true, y_prob, bins: int = 10) -> list[dict]:
    """Decile table comparing predicted PD vs observed default rate."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(sel.sum()),
                     "pred_pd": round(float(y_prob[sel].mean()), 4),
                     "obs_default": round(float(y_true[sel].mean()), 4)})
    return rows
