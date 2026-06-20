"""
src/dashboard/loaders.py
======================
Pure loaders that turn the saved JSON artifacts into display-ready structures.
No Streamlit dependency, so they are fully unit-testable. Every loader degrades
gracefully (returns None / empty) when an artifact is missing, so the dashboard
can tell the user which stage to run.
"""

from __future__ import annotations

import json

from config.settings import settings

MODELS_DIR = settings.project_root / "models"


def load(name: str) -> dict | None:
    p = MODELS_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def overview() -> dict:
    ecl = load("ecl_results.json") or {}
    stg = load("staging_artifacts.json") or {}
    mon = load("monitoring_results.json") or {}
    s = ecl.get("summary", {})
    latest = mon.get("latest_status") or mon.get("overall_status")
    return {
        "total_ecl": s.get("total_ecl"),
        "total_ead": s.get("total_ead"),
        "coverage_pct": s.get("coverage_pct"),
        "n_loans": s.get("n_loans") or stg.get("n_loans"),
        "weighted_lgd": ecl.get("weighted_lgd"),
        "reporting_period": ecl.get("reporting_period"),
        "monitoring_status": latest,
        "stage_distribution": stg.get("distribution", []),
    }


def ecl_view() -> dict | None:
    ecl = load("ecl_results.json")
    if not ecl:
        return None
    s = ecl.get("summary", {})
    return {"summary": s, "by_stage": s.get("by_stage", []),
            "weighted_lgd": ecl.get("weighted_lgd"),
            "lgd_by_scenario": ecl.get("lgd_by_scenario", {}),
            "ecl_12m_total": s.get("ecl_12m_total"),
            "ecl_lifetime_total": s.get("ecl_lifetime_total")}


def scenario_view() -> dict | None:
    scn = load("scenario_artifacts.json")
    if not scn:
        return None
    return {"weights": scn.get("weights", {}), "pit_pd_12m": scn.get("pit_pd_12m", {}),
            "z_12m": scn.get("z_12m", {}),
            "weighted_pit_pd_12m": scn.get("weighted_pit_pd_12m"),
            "ordering_ok": scn.get("ordering_ok"),
            "pit_pd_paths": scn.get("pit_pd_paths", {}),
            "weighted_pit_pd_path": scn.get("weighted_pit_pd_path", []),
            "horizon_months": scn.get("horizon_months")}


def validation_view() -> dict | None:
    v = load("validation_results.json")
    if not v:
        return None
    return {"checks": v.get("checks", []),
            "n_error_passed": v.get("n_error_passed"), "n_error_checks": v.get("n_error_checks"),
            "n_advisory": v.get("n_advisory"),
            "discrimination": v.get("discrimination", {}),
            "calibrated_portfolio_pd": v.get("calibrated_portfolio_pd"),
            "pd_migration_psi": v.get("pd_migration_psi_orig_vs_now"),
            "sensitivity": (v.get("sensitivity") or {}).get("grid", [])}


def monitoring_view() -> dict | None:
    m = load("monitoring_results.json")
    if not m:
        return None
    return {"latest_status": m.get("latest_status") or m.get("overall_status"),
            "historical_worst": m.get("historical_worst") or m.get("overall_status"),
            "breach_summary": m.get("breach_summary", {}),
            "backtest": m.get("backtest", []), "psi": m.get("psi", []),
            "delinquency_trend": m.get("delinquency_trend", [])}


def model_view() -> dict:
    pd_m = load("pd_metrics.json") or {}
    best = pd_m.get("best_model")
    models = pd_m.get("models", {}) or {}
    disc = models.get(best, {}) if best else {}
    lgd = load("lgd_stats.json") or {}
    ead = load("ead_model.json") or {}
    pit = load("pit_artifacts.json") or {}
    scn = load("scenario_artifacts.json") or {}
    dt = lgd.get("downturn") or {}
    emp = dt.get("empirical_downturn") or {}
    curves = pd_m.get("curves", {}) or {}
    return {
        # --- PD (existing keys preserved) ---
        "pd_best_model": best,
        "pd_auc": disc.get("auc") or disc.get("AUC"),
        "pd_gini": disc.get("gini"), "pd_ks": disc.get("ks"),
        # --- PD (new) ---
        "pd_models": models,                       # all models: auc/gini/ks/precision/recall/brier
        "pd_iv": pd_m.get("iv", []),               # information value by feature
        "pd_calibration": pd_m.get("calibration", []),
        "pd_roc": curves.get("roc", []),           # [] until next run_pd populates it
        "pd_ks_curve": curves.get("ks", []),
        "pd_features": pd_m.get("features", []),
        # --- LGD (existing + new) ---
        "lgd_mean": (lgd.get("stats") or {}).get("mean"),
        "lgd_n": (lgd.get("stats") or {}).get("n"),
        "lgd_downturn": dt.get("downturn_lgd"),
        "lgd_uplift": dt.get("uplift_ratio"),
        "lgd_driver": dt.get("driver"),
        "lgd_empirical_lgd": emp.get("lgd"),
        "lgd_window": (f"{emp.get('window_start')} – {emp.get('window_end')}"
                       if emp.get("window_start") else None),
        "lgd_fit_r2": (lgd.get("fit") or {}).get("r2"),
        # --- EAD (existing + new) ---
        "ead_method": ead.get("method"),
        "ead_curtailment_12m": ead.get("curtailment_factor_12m"),
        "ead_apply_curtailment": ead.get("apply_curtailment_adjustment"),
        # --- TTC / correlation (existing) ---
        "ttc_pd": pit.get("ttc_pd") or scn.get("ttc_pd"),
        "asset_correlation": pit.get("asset_correlation") or scn.get("asset_correlation"),
    }
