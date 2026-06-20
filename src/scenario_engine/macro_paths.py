"""
src/scenario_engine/macro_paths.py
=================================
Builds forward macro paths per scenario. Baseline = the latest observed macro
values held flat. Each scenario adds a shock that ramps linearly from 0 to its
peak over `ramp_months`, then holds (a persistent-stress profile, prudent for
lifetime ECL).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def latest_macro(macro: pl.DataFrame, features: list[str]) -> dict:
    """Most recent non-null value for each macro feature (the baseline anchor)."""
    m = macro.sort("reporting_period")
    out = {}
    for f in features:
        s = m[f].drop_nulls()
        if s.len() == 0:
            raise ValueError(f"Macro feature '{f}' has no non-null values.")
        out[f] = float(s[-1])
    return out


def shock_profile(horizon: int, ramp_months: int) -> np.ndarray:
    """Multiplier path: linear ramp 0->1 over ramp_months, then held at 1."""
    t = np.arange(1, horizon + 1, dtype=float)
    return np.clip(t / max(ramp_months, 1), 0.0, 1.0)


def build_macro_path(baseline: dict, shock: dict, features: list[str],
                     horizon: int, ramp_months: int) -> np.ndarray:
    """Return an (horizon x n_features) array of macro values for one scenario."""
    prof = shock_profile(horizon, ramp_months)
    cols = []
    for f in features:
        base_v = baseline[f]
        peak = float(shock.get(f, 0.0))
        cols.append(base_v + peak * prof)
    return np.column_stack(cols)
