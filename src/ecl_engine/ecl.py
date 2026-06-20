"""
src/ecl_engine/ecl.py
===================
Core ECL computation. For each loan, across each scenario, walk the lifetime
month by month:

    marginal_default_t = survival_{t-1} * hazard_t
    EL_t               = marginal_default_t * LGD_s * EAD_t * discount_t
    survival_t         = survival_{t-1} * (1 - hazard_t) * (1 - SMM)   # default & prepay exits

12-month ECL sums t<=12; lifetime ECL sums the whole horizon. The per-loan ECL
is the stage-weighted choice (Stage 1 -> 12m, Stage 2/3 -> lifetime), and each
figure is probability-weighted across scenarios. Stage 3 (already credit-
impaired) is treated as default-certain: ECL = weighted LGD * current balance.

Vectorised across loans; loops only over months x scenarios.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.ead_model import amortization as A
from src.ecl_engine import term_structure as T


def compute_ecl(loans: pl.DataFrame, multipliers: dict, weights: dict,
                lgd_by_scenario: dict, smm: float, horizon: int) -> dict:
    """loans columns required: loan_id, pd12, upb, rate, term, stage."""
    pd12 = loans["pd12"].to_numpy()
    upb = loans["upb"].to_numpy()
    rate = loans["rate"].to_numpy()
    term = loans["term"].to_numpy()
    stage = loans["stage"].to_numpy()
    n = len(pd12)

    h_loan = T.annual_to_monthly_hazard(pd12)              # constant monthly hazard per loan
    r_m = T.monthly_discount_rate(rate)
    weighted_lgd = sum(weights[s] * lgd_by_scenario[s] for s in weights)

    ecl12 = np.zeros(n)
    ecl_life = np.zeros(n)

    for s, w in weights.items():
        mult = multipliers[s]
        lgd_s = lgd_by_scenario[s]
        survival = np.ones(n)
        e12_s = np.zeros(n)
        elife_s = np.zeros(n)
        for t in range(1, horizon + 1):
            h_t = np.clip(h_loan * mult[t - 1], 0.0, 1.0)
            ead_t = A.remaining_balance(upb, rate, term, t)
            disc_t = (1.0 + r_m) ** (-t)
            marginal = survival * h_t
            el_t = marginal * lgd_s * ead_t * disc_t
            elife_s += el_t
            if t <= 12:
                e12_s += el_t
            survival = survival * (1.0 - h_t) * (1.0 - smm)
        ecl12 += w * e12_s
        ecl_life += w * elife_s

    # Stage 3: default-certain -> ECL = weighted LGD * current balance
    stage3 = stage >= 3
    ecl_life = np.where(stage3, weighted_lgd * upb, ecl_life)
    ecl12 = np.where(stage3, weighted_lgd * upb, ecl12)

    selected = np.where(stage == 1, ecl12, ecl_life)       # Stage 1 -> 12m, else lifetime

    out = loans.select(["loan_id", "stage", "upb"]).with_columns([
        pl.Series("ecl_12m", ecl12),
        pl.Series("ecl_lifetime", ecl_life),
        pl.Series("ecl", selected),
    ])
    summary = _summarise(out)
    return {"per_loan": out, "summary": summary, "weighted_lgd": float(weighted_lgd)}


def _summarise(df: pl.DataFrame) -> dict:
    total_ead = float(df["upb"].sum())
    total_ecl = float(df["ecl"].sum())
    by_stage = (df.group_by("stage")
                .agg([pl.len().alias("n_loans"), pl.col("upb").sum().alias("ead"),
                      pl.col("ecl").sum().alias("ecl")])
                .sort("stage")
                .with_columns((pl.col("ecl") / pl.col("ead") * 100).round(3).alias("coverage_pct"))
                .to_dicts())
    return {"n_loans": df.height, "total_ead": round(total_ead, 2),
            "total_ecl": round(total_ecl, 2),
            "coverage_pct": round(total_ecl / max(total_ead, 1.0) * 100, 4),
            "ecl_12m_total": round(float(df["ecl_12m"].sum()), 2),
            "ecl_lifetime_total": round(float(df["ecl_lifetime"].sum()), 2),
            "by_stage": by_stage}
