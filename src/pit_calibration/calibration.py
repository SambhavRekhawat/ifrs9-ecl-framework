"""
src/pit_calibration/calibration.py
==================================
Recalibrates raw model scores to the true population base rate.

The Stage-5 models discriminate well but over-predict the absolute PD level
(class weighting). Isotonic regression maps the model's score to the observed
default frequency, so the calibrated PD's average matches reality - which is
what ECL needs (PD x LGD x EAD must use a true probability).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=int))
    return iso


def calibrate(iso: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    return iso.predict(np.asarray(scores, dtype=float))


def calibration_summary(scores: np.ndarray, y: np.ndarray, iso: IsotonicRegression) -> dict:
    cal = calibrate(iso, scores)
    return {
        "observed_default_rate": round(float(np.mean(y)), 5),
        "mean_raw_score": round(float(np.mean(scores)), 5),
        "mean_calibrated_pd": round(float(np.mean(cal)), 5),
    }
