"""
src/ead_model/amortization.py
============================
Scheduled-balance amortization for EAD. For a fully-amortizing fixed-rate
mortgage, the outstanding balance k months ahead (with n months remaining) is:

    B_k = UPB * ((1+r)^n - (1+r)^k) / ((1+r)^n - 1)

where r is the monthly rate. For r ~ 0 this degenerates to the straight-line
B_k = UPB * (n - k) / n. All functions are numpy-vectorised so they work on a
single loan or a whole portfolio.

Mortgages have no undrawn commitment, so EAD = this scheduled balance at the
(future) month of default. Prepayment/survival weighting is applied later in
the ECL engine, not here.
"""

from __future__ import annotations

import numpy as np


def monthly_rate(annual_rate_pct):
    return np.asarray(annual_rate_pct, dtype=float) / 100.0 / 12.0


def monthly_payment(upb, annual_rate_pct, n_months):
    upb = np.asarray(upb, dtype=float)
    n = np.asarray(n_months, dtype=float)
    r = monthly_rate(annual_rate_pct)
    n_safe = np.where(n <= 0, np.nan, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        amort = upb * r / (1 - (1 + r) ** (-n_safe))
        pay = np.where(np.abs(r) < 1e-9, upb / n_safe, amort)
    return np.where((upb <= 0) | (n <= 0), 0.0, pay)


def remaining_balance(upb, annual_rate_pct, n_remaining, k):
    """Outstanding balance after k more scheduled payments (n_remaining left)."""
    upb = np.asarray(upb, dtype=float)
    n = np.asarray(n_remaining, dtype=float)
    k = np.minimum(np.asarray(k, dtype=float), n)
    r = monthly_rate(annual_rate_pct)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + r) ** n
        fk = (1 + r) ** k
        amort = upb * (f - fk) / np.where(np.abs(f - 1) < 1e-12, np.nan, f - 1)
        lin = upb * (n - k) / np.where(n <= 0, np.nan, n)
        bal = np.where(np.abs(r) < 1e-9, lin, amort)
    bal = np.where((upb <= 0) | (n <= 0), 0.0, bal)
    return np.clip(bal, 0.0, None)


def project_balance(upb, annual_rate_pct, n_remaining, horizon):
    """Balance path for months 1..horizon. Returns array shape (horizon,) for a
    scalar loan, or (horizon, n_loans) for arrays."""
    return np.array([remaining_balance(upb, annual_rate_pct, n_remaining, k)
                     for k in range(1, horizon + 1)])
