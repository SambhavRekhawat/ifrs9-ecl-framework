"""
src/ecl_engine/term_structure.py
==============================
Pure building blocks for the ECL term structure.

PD level calibration: rather than re-deriving each loan's PD from the Vasicek
mapping (which would discard its calibrated level), the scenario PIT-PD paths are
turned into RELATIVE multipliers anchored so the probability-weighted month-1
factor equals 1. Each loan's own calibrated 12-month PD is then shaped by these
multipliers — preserving the book's absolute level while imposing each
scenario's shape and severity.
"""

from __future__ import annotations

import numpy as np


def annual_to_monthly_hazard(pd_annual):
    """Constant monthly default hazard consistent with a 12-month PD."""
    pd_annual = np.clip(np.asarray(pd_annual, dtype=float), 0.0, 0.999999)
    return 1.0 - (1.0 - pd_annual) ** (1.0 / 12.0)


def cpr_to_smm(cpr_annual: float) -> float:
    """Annual CPR (conditional prepayment rate) -> single monthly mortality."""
    cpr_annual = min(max(cpr_annual, 0.0), 0.999999)
    return 1.0 - (1.0 - cpr_annual) ** (1.0 / 12.0)


def monthly_discount_rate(annual_rate_pct):
    return np.asarray(annual_rate_pct, dtype=float) / 100.0 / 12.0


def scenario_multipliers(pit_pd_paths: dict, weighted_pd_path: list) -> dict:
    """Relative PD multipliers per scenario, anchored to the weighted month-1 PD.

    mult[s][t] = pit_pd_path[s][t] / weighted_pd_path[0]
    so that sum_s w_s * mult[s][0] == 1 (book stays at its calibrated level).
    """
    anchor = float(weighted_pd_path[0])
    if anchor <= 0:
        raise ValueError("Weighted PD anchor is non-positive; cannot build multipliers.")
    return {s: np.asarray(path, dtype=float) / anchor for s, path in pit_pd_paths.items()}
