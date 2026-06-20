"""
src/pd_model/models.py
=====================
Trains the two PD tracks:
  1. Traditional scorecard - Logistic Regression on WOE-transformed features.
  2. Machine learning      - XGBoost and LightGBM on raw features.

Class imbalance (defaults are rare) is handled with scale_pos_weight /
class_weight. NaNs are fine for XGBoost/LightGBM; the WOE features have none.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def pos_weight(y: np.ndarray) -> float:
    pos = max(int((y == 1).sum()), 1)
    neg = int((y == 0).sum())
    return neg / pos


def train_scorecard(Xw: np.ndarray, y: np.ndarray, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(Xw, y)
    return model


def train_xgb(X: np.ndarray, y: np.ndarray, seed: int = 42):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight(y), eval_metric="auc",
        tree_method="hist", random_state=seed, n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_lgbm(X: np.ndarray, y: np.ndarray, seed: int = 42):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight(y),
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)
    return model


def proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def cv_auc(model_fn, X: np.ndarray, y: np.ndarray, seed: int = 42, folds: int = 3) -> float:
    """Mean cross-validated AUC for a fresh model from model_fn()."""
    try:
        scores = cross_val_score(model_fn(), X, y, cv=folds, scoring="roc_auc", n_jobs=-1)
        return round(float(np.mean(scores)), 4)
    except Exception:
        return float("nan")


def scorecard_points(model: LogisticRegression, feature_names: list[str],
                     pdo: int = 20, base_score: int = 600, base_odds: float = 50.0) -> dict:
    """Convert logistic coefficients into classic scorecard scaling factors."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return {
        "factor": round(float(factor), 4),
        "offset": round(float(offset), 4),
        "coefficients": dict(zip(feature_names, [round(float(c), 4) for c in model.coef_[0]])),
        "intercept": round(float(model.intercept_[0]), 4),
    }
