"""
src/validation/run_validation.py
==============================
Phase-13 pipeline. Run from the project root:

    python -m src.validation.run_validation

Produces a validation report covering: PD discrimination & calibration (from the
saved metrics), reconciliation pass/fail checks over the ECL/scenario artifacts,
PD score drift since origination (PSI), and an ECL sensitivity/attribution grid.
"""

from __future__ import annotations

import json
from datetime import datetime

from config.settings import settings
from src.ecl_engine import ecl_data, run_ecl
from src.quality_checks import drift
from src.validation import checks, ecl_sensitivity
from src.utils.logger import get_logger

log = get_logger("validation.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def _load(name):
    return json.loads((MODELS_DIR / name).read_text())


def run() -> dict:
    cfg = settings.config["validation"]
    ecfg = settings.config["ecl"]
    REPORTS_DIR.mkdir(exist_ok=True)

    pd_metrics = _load("pd_metrics.json")
    ecl_results = _load("ecl_results.json")
    scn = _load("scenario_artifacts.json")

    # 1. reconciliation checks over saved artifacts
    results = checks.run_all(ecl_results, scn, pd_metrics, cfg)

    # 2. discrimination summary
    best = pd_metrics["best_model"]
    disc = pd_metrics["models"][best]

    # 3. assemble loans once -> effective calibration + PD migration + sensitivity
    staged = ecl_data.stage_snapshot(ecfg["reporting_period"])
    cal_mean = float(staged["pd_now"].mean())
    ttc = scn.get("ttc_pd")
    if ttc:
        results.append(checks.effective_pd_calibration(cal_mean, ttc, cfg["calibration_ratio_band"]))

    # PD migration since origination (informational; expected from multi-year seasoning,
    # NOT a same-population data-drift alarm — true temporal drift monitoring is Stage 14)
    pd_migration_psi = drift.psi(staged["pd_orig"], staged["pd_now"])
    log.info("PD migration since origination (PSI pd_orig vs pd_now): %s "
             "[informational - reflects seasoning, not data drift]", pd_migration_psi)

    # headline tally counts ERROR-severity checks; WARN checks are advisories
    err = [r for r in results if r["severity"] == "error"]
    warns = [r for r in results if r["severity"] == "warn"]
    n_pass = sum(r["passed"] for r in err)
    for r in results:
        lvl = "PASS" if r["passed"] else ("WARN" if r["severity"] == "warn" else "FAIL")
        log.info("  [%s] %s — %s", lvl, r["check"], r["detail"])
    log.info("Checks: %d/%d error-level passed (+ %d advisory)", n_pass, len(err), len(warns))

    loans = ecl_data.attach_ead(staged, ecfg["reporting_period"])
    pit_paths, weighted_path, weights, _ = run_ecl._load_scenarios()
    lgd_map = run_ecl._load_lgd(list(weights.keys()), ecfg["lgd_downside_scenario"])
    lgd_base = lgd_map[[s for s in weights if s != ecfg["lgd_downside_scenario"]][0]]
    lgd_down = lgd_map[ecfg["lgd_downside_scenario"]]
    sens = ecl_sensitivity.run_grid(
        loans, pit_paths, weights, lgd_base, lgd_down, ecfg["lgd_downside_scenario"],
        ecfg["annual_cpr"], min(ecfg["horizon_months"], len(weighted_path)), cfg["sensitivity"])
    log.info("Sensitivity grid: %d scenarios around base ECL %s",
             len(sens["grid"]), f"${sens['base_ecl']:,.0f}")
    for g in sens["grid"]:
        log.info("    %-18s %-14s ECL=%s  cov=%.4f%%  (%+.1f%%)",
                 g["lever"], g["value"], f"${g['total_ecl']:,.0f}",
                 g["coverage_pct"], g["delta_vs_base_pct"])

    artifact = {
        "checks": results, "n_error_passed": n_pass, "n_error_checks": len(err),
        "n_advisory": len(warns),
        "discrimination": {"best_model": best,
                           "auc": disc.get("auc") or disc.get("AUC"),
                           "gini": disc.get("gini"), "ks": disc.get("ks")},
        "calibration_table": pd_metrics.get("calibration", []),
        "calibrated_portfolio_pd": round(cal_mean, 5),
        "pd_migration_psi_orig_vs_now": pd_migration_psi,
        "sensitivity": sens,
    }
    (MODELS_DIR / "validation_results.json").write_text(json.dumps(artifact, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(artifact, REPORTS_DIR / f"validation_report_{run_id}.html", run_id)
    log.info("Validation done. %d/%d error-level checks passed (+%d advisory).",
             n_pass, len(err), len(warns))
    return artifact


def _write_html(art: dict, path, run_id: str) -> None:
    crows = "".join(
        f"<tr><td>{c['check']}</td><td style='color:{'#1a7f37' if c['passed'] else '#cf222e'}'>"
        f"{'PASS' if c['passed'] else 'FAIL'}</td><td>{c['severity']}</td><td>{c['detail']}</td></tr>"
        for c in art["checks"])
    srows = "".join(
        f"<tr><td>{g['lever']}</td><td>{g['value']}</td><td>${g['total_ecl']:,.0f}</td>"
        f"<td>{g['coverage_pct']}%</td><td>{g['delta_vs_base_pct']:+.1f}%</td></tr>"
        for g in art["sensitivity"]["grid"])
    d = art["discrimination"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Validation {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — Model Validation</h1><div>Run {run_id} · {art['n_error_passed']}/{art['n_error_checks']} error-level checks passed · {art['n_advisory']} advisory</div>
<h2>Reconciliation checks</h2>
<table><tr><th>Check</th><th>Result</th><th>Severity</th><th>Detail</th></tr>{crows}</table>
<h2>PD discrimination</h2>
<table><tr><th>Best model</th><th>AUC</th><th>Gini</th><th>KS</th></tr>
<tr><td>{d['best_model']}</td><td>{d['auc']}</td><td>{d['gini']}</td><td>{d['ks']}</td></tr></table>
<p>Calibrated portfolio PD (feeds ECL): <b>{art['calibrated_portfolio_pd']}</b></p>
<p>PD migration since origination (PSI pd_orig vs pd_now): <b>{art['pd_migration_psi_orig_vs_now']}</b>
— informational; reflects multi-year seasoning, <i>not</i> a same-population data-drift alarm
(temporal drift monitoring is Stage 14).</p>
<h2>ECL sensitivity / attribution</h2>
<table><tr><th>Lever</th><th>Value</th><th>Total ECL</th><th>Coverage</th><th>Δ vs base</th></tr>{srows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote validation report to %s", path)


if __name__ == "__main__":
    run()
