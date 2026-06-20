"""Tests for the dashboard loaders (Streamlit-free)."""
import json

import pytest

from src.dashboard import loaders as L


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "MODELS_DIR", tmp_path)
    (tmp_path / "ecl_results.json").write_text(json.dumps({
        "reporting_period": "latest", "weighted_lgd": 0.1821,
        "lgd_by_scenario": {"base": 0.126, "downside": 0.35},
        "summary": {"n_loans": 464418, "total_ead": 1.14e11, "total_ecl": 3.3e8,
                    "coverage_pct": 0.289, "ecl_12m_total": 2.8e8, "ecl_lifetime_total": 8.2e8,
                    "by_stage": [{"stage": 1, "n_loans": 455150, "ead": 1.12e11, "ecl": 1.2e8, "coverage_pct": 0.11},
                                 {"stage": 3, "n_loans": 1970, "ead": 5e8, "ecl": 9e7, "coverage_pct": 18.2}]}}))
    (tmp_path / "staging_artifacts.json").write_text(json.dumps({
        "n_loans": 466725, "distribution": [{"stage": 1, "n": 457457, "pct": 98.01}]}))
    (tmp_path / "scenario_artifacts.json").write_text(json.dumps({
        "ordering_ok": True, "weights": {"base": 0.5, "downside": 0.25},
        "pit_pd_12m": {"base": 0.0074, "downside": 0.0406}, "weighted_pit_pd_12m": 0.0148,
        "pit_pd_paths": {"base": [0.0074] * 60}, "weighted_pit_pd_path": [0.0148] * 60}))
    (tmp_path / "validation_results.json").write_text(json.dumps({
        "checks": [{"check": "c1", "passed": True, "severity": "error", "detail": "ok"}],
        "n_error_passed": 6, "n_error_checks": 6, "n_advisory": 1,
        "discrimination": {"best_model": "lightgbm", "auc": 0.8914, "gini": 0.78, "ks": 0.61},
        "calibrated_portfolio_pd": 0.0085, "pd_migration_psi_orig_vs_now": 0.51,
        "sensitivity": {"grid": [{"lever": "BASE CASE", "value": "x", "total_ecl": 3.3e8,
                                  "coverage_pct": 0.289, "delta_vs_base_pct": 0.0}]}}))
    (tmp_path / "monitoring_results.json").write_text(json.dumps({
        "latest_status": "GREEN", "historical_worst": "RED",
        "breach_summary": {"red": 4, "amber": 7, "green": 46},
        "backtest": [{"period": "2023-01-01", "mean_pred_pd": 0.0085, "realised_dr": 0.0067, "rag": "GREEN"}],
        "psi": [{"from": "a", "to": "b", "psi": 0.002, "rag": "GREEN"}],
        "delinquency_trend": [{"period": "2023-01-01", "share_30dpd": 0.011, "share_90dpd": 0.004}]}))
    (tmp_path / "pd_metrics.json").write_text(json.dumps({
        "best_model": "lightgbm", "models": {"lightgbm": {"auc": 0.8914, "gini": 0.78, "ks": 0.61}}}))
    (tmp_path / "lgd_stats.json").write_text(json.dumps({"stats": {"mean": 0.126}, "downturn": {"downturn_lgd": 0.35}}))
    (tmp_path / "ead_model.json").write_text(json.dumps({"method": "scheduled_amortization", "curtailment_factor_12m": 1.0}))
    (tmp_path / "pit_artifacts.json").write_text(json.dumps({"ttc_pd": 0.0105, "asset_correlation": 0.15}))
    return tmp_path


def test_overview(artifacts):
    o = L.overview()
    assert o["total_ecl"] == 3.3e8 and o["coverage_pct"] == 0.289
    assert o["monitoring_status"] == "GREEN" and len(o["stage_distribution"]) == 1


def test_ecl_and_scenario_views(artifacts):
    assert len(L.ecl_view()["by_stage"]) == 2
    s = L.scenario_view()
    assert s["ordering_ok"] and len(s["weighted_pit_pd_path"]) == 60


def test_validation_and_model_views(artifacts):
    v = L.validation_view()
    assert v["n_error_passed"] == 6 and v["discrimination"]["auc"] == 0.8914
    assert len(v["sensitivity"]) == 1
    m = L.model_view()
    assert m["pd_auc"] == 0.8914 and m["lgd_downturn"] == 0.35


def test_monitoring_view_and_legacy_key(artifacts):
    assert L.monitoring_view()["latest_status"] == "GREEN"
    # legacy artifact with only overall_status still resolves
    (artifacts / "monitoring_results.json").write_text(json.dumps({"overall_status": "AMBER"}))
    assert L.monitoring_view()["latest_status"] == "AMBER"


def test_missing_artifact_returns_none(artifacts):
    (artifacts / "ecl_results.json").unlink()
    assert L.ecl_view() is None
    assert L.overview()["total_ecl"] is None


def test_model_view_enriched_fields(artifacts):
    # overwrite pd_metrics/lgd_stats/ead with richer artifacts
    import json as _j
    (artifacts / "pd_metrics.json").write_text(_j.dumps({
        "best_model": "scorecard",
        "models": {"scorecard": {"auc": 0.91, "gini": 0.82, "ks": 0.66, "brier": 0.006},
                   "xgboost": {"auc": 0.90, "gini": 0.80, "ks": 0.64, "brier": 0.0065}},
        "iv": [{"feature": "fico", "iv": 0.42, "strength": "strong"}],
        "calibration": [{"bin": 0, "n": 100, "pred_pd": 0.01, "obs_default": 0.012}],
        "curves": {"roc": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}],
                   "ks": [{"pop_pct": 0.0, "cum_bad": 0.0, "cum_good": 0.0}]},
        "features": ["fico"]}))
    (artifacts / "lgd_stats.json").write_text(_j.dumps({
        "stats": {"mean": 0.332, "n": 10281}, "fit": {"r2": 0.002},
        "downturn": {"downturn_lgd": 0.376, "uplift_ratio": 1.13, "driver": "empirical",
                     "empirical_downturn": {"lgd": 0.376, "window_start": "2014-12",
                                            "window_end": "2016-11", "n_obs": 1200}}}))
    (artifacts / "ead_model.json").write_text(_j.dumps({
        "method": "scheduled_amortization", "curtailment_factor_12m": 0.991,
        "apply_curtailment_adjustment": True}))
    m = L.model_view()
    # existing keys preserved
    assert m["pd_auc"] == 0.91 and m["lgd_downturn"] == 0.376
    # new PD keys
    assert len(m["pd_models"]) == 2 and len(m["pd_iv"]) == 1
    assert len(m["pd_calibration"]) == 1 and len(m["pd_roc"]) == 2 and len(m["pd_ks_curve"]) == 1
    # new LGD keys
    assert m["lgd_uplift"] == 1.13 and m["lgd_driver"] == "empirical"
    assert m["lgd_window"] == "2014-12 – 2016-11" and m["lgd_n"] == 10281
    # new EAD keys
    assert m["ead_apply_curtailment"] is True


def test_model_view_degrades_without_curves(artifacts):
    # the shipped minimal fixture has no iv/calibration/curves -> empty, not error
    m = L.model_view()
    assert m["pd_iv"] == [] and m["pd_roc"] == [] and m["pd_ks_curve"] == []
