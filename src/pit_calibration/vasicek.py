"""
src/pit_calibration/vasicek.py
=============================
The Vasicek single-factor (ASRF) link between through-the-cycle (TTC) and
point-in-time (PIT) PD.

  PD_PIT = Phi( (Phi^-1(PD_TTC) - sqrt(rho) * Z) / sqrt(1 - rho) )

Z is the systematic economic factor (standard normal): Z > 0 = good economy
-> lower PD; Z < 0 = stress -> higher PD. rho is the asset correlation
(Basel residential-mortgage value = 0.15).

Given an observed default rate DR for a period, the implied Z is:
  Z = ( Phi^-1(PD_TTC) - sqrt(1-rho) * Phi^-1(DR) ) / sqrt(rho)
which is the exact inverse of the formula above.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-6


def _clip(p):
    return np.clip(p, _EPS, 1 - _EPS)


def vasicek_pit(pd_ttc, z, rho: float):
    """PIT PD given a TTC PD and systematic factor Z."""
    pd_ttc = _clip(np.asarray(pd_ttc, dtype=float))
    return norm.cdf((norm.ppf(pd_ttc) - np.sqrt(rho) * np.asarray(z, dtype=float)) / np.sqrt(1 - rho))


def implied_z(dr, pd_ttc, rho: float):
    """Systematic factor Z implied by an observed default rate DR vs the TTC PD."""
    dr = _clip(np.asarray(dr, dtype=float))
    pd_ttc = _clip(np.asarray(pd_ttc, dtype=float))
    return (norm.ppf(pd_ttc) - np.sqrt(1 - rho) * norm.ppf(dr)) / np.sqrt(rho)
