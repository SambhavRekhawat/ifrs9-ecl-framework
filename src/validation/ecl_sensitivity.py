"""
src/validation/ecl_sensitivity.py
================================
Re-runs the ECL core under perturbed assumptions to show how the headline moves
with each lever (CPR, lifetime horizon, scenario weights, downturn LGD). The
loan inputs are assembled once; only the fast vectorised computation is repeated.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.ecl_engine import ecl as E, term_structure as T


def _weighted_path(pit_paths: dict, weights: dict) -> list:
    H = len(next(iter(pit_paths.values())))
    return [sum(weights[s] * pit_paths[s][t] for s in weights) for t in range(H)]


def _run(loans, pit_paths, weights, lgd_by_scenario, cpr, horizon):
    mult = T.scenario_multipliers(pit_paths, _weighted_path(pit_paths, weights))
    smm = T.cpr_to_smm(cpr)
    res = E.compute_ecl(loans, mult, weights, lgd_by_scenario, smm, horizon)
    return res["summary"]["total_ecl"], res["summary"]["coverage_pct"]


def run_grid(loans: pl.DataFrame, pit_paths: dict, base_weights: dict,
             lgd_base: float, lgd_down: float, downside_name: str,
             base_cpr: float, base_horizon: int, cfg: dict) -> dict:
    def lgd_map(down):
        return {s: (down if s == downside_name else lgd_base) for s in base_weights}

    base_lgd = lgd_map(lgd_down)
    base_ecl, base_cov = _run(loans, pit_paths, base_weights, base_lgd, base_cpr, base_horizon)

    rows = [{"lever": "BASE CASE", "value": f"cpr={base_cpr}, H={base_horizon}",
             "total_ecl": base_ecl, "coverage_pct": base_cov, "delta_vs_base_pct": 0.0}]

    def add(lever, value, ecl, cov):
        rows.append({"lever": lever, "value": str(value), "total_ecl": round(ecl, 0),
                     "coverage_pct": cov,
                     "delta_vs_base_pct": round((ecl / base_ecl - 1) * 100, 1)})

    for cpr in cfg["cpr_values"]:
        ecl, cov = _run(loans, pit_paths, base_weights, base_lgd, cpr, base_horizon)
        add("annual_cpr", cpr, ecl, cov)
    for H in cfg["horizon_values"]:
        H = min(H, len(next(iter(pit_paths.values()))))
        ecl, cov = _run(loans, pit_paths, base_weights, base_lgd, base_cpr, H)
        add("horizon_months", H, ecl, cov)
    for d in cfg["downturn_lgd_values"]:
        ecl, cov = _run(loans, pit_paths, base_weights, lgd_map(d), base_cpr, base_horizon)
        add("downturn_lgd", d, ecl, cov)
    for name, w in cfg["weight_variants"].items():
        ecl, cov = _run(loans, pit_paths, w, base_lgd, base_cpr, base_horizon)
        add("scenario_weights", name, ecl, cov)

    return {"base_ecl": base_ecl, "base_coverage_pct": base_cov, "grid": rows}
