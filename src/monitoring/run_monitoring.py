"""
src/monitoring/run_monitoring.py
=============================
Phase-14 pipeline. Run from the project root:

    python -m src.monitoring.run_monitoring

Period-over-period surveillance: PD back-testing (predicted vs realised default
rate), PD score-distribution drift (PSI), and the portfolio delinquency trend,
each with RAG status, plus an overall traffic-light.
"""

from __future__ import annotations

import json
from datetime import datetime

from config.settings import settings
from src.monitoring import monitor, monitoring_data
from src.utils.logger import get_logger

log = get_logger("monitoring.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def run() -> dict:
    mcfg = settings.config["monitoring"]
    default_dpd = settings.config["pd"]["default_dpd"]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    bt_panel = monitoring_data.build_backtest_panel()
    backtest = monitor.backtest_by_period(bt_panel, mcfg["min_n_per_period"],
                                          mcfg["backtest_amber_dev"], mcfg["backtest_red_dev"])
    psi = monitor.psi_by_period(bt_panel, mcfg["psi_bins"], mcfg["psi_amber"], mcfg["psi_red"])

    dq_panel = monitoring_data.delinquency_panel()
    delinquency = monitor.delinquency_trend(dq_panel, default_dpd)

    latest = monitor.latest_status(backtest, psi)
    historical = monitor.overall_status(backtest, psi)
    breaches = monitor.breach_summary(backtest, psi)
    log.info("=== PD back-test (predicted vs realised default rate) ===")
    for r in backtest:
        log.info("  %s  n=%s  pred=%.4f  realised=%.4f  ratio=%.2f  [%s]",
                 r["period"], f"{r['n']:,}", r["mean_pred_pd"], r["realised_dr"],
                 r["pred_over_realised"], r["rag"])
    log.info("=== PD distribution drift (PSI, period-over-period) ===")
    for r in psi:
        log.info("  %s -> %s  PSI=%s  [%s]", r["from"], r["to"], r["psi"], r["rag"])
    log.info("=== Delinquency trend ===")
    for r in delinquency:
        log.info("  %s  30+DPD=%.3f%%  90+DPD=%.3f%%",
                 r["period"], r["share_30dpd"] * 100, r["share_90dpd"] * 100)
    log.info("LATEST-PERIOD STATUS: %s  |  historical worst: %s  (breaches red=%d amber=%d green=%d)",
             latest, historical, breaches["red"], breaches["amber"], breaches["green"])

    artifact = {"latest_status": latest, "historical_worst": historical,
                "breach_summary": breaches, "backtest": backtest, "psi": psi,
                "delinquency_trend": delinquency,
                "thresholds": {k: mcfg[k] for k in
                               ["psi_amber", "psi_red", "backtest_amber_dev", "backtest_red_dev"]}}
    (MODELS_DIR / "monitoring_results.json").write_text(json.dumps(artifact, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(artifact, REPORTS_DIR / f"monitoring_report_{run_id}.html", run_id)
    log.info("Monitoring done. Latest-period status %s (historical worst %s).", latest, historical)
    return artifact


def _color(rag):
    return {"GREEN": "#1a7f37", "AMBER": "#9a6700", "RED": "#cf222e"}.get(rag, "#57606a")


def _write_html(art: dict, path, run_id: str) -> None:
    btr = "".join(
        f"<tr><td>{r['period']}</td><td>{r['n']:,}</td><td>{r['mean_pred_pd']}</td>"
        f"<td>{r['realised_dr']}</td><td>{r['pred_over_realised']}</td>"
        f"<td style='color:{_color(r['rag'])};font-weight:600'>{r['rag']}</td></tr>" for r in art["backtest"])
    psir = "".join(
        f"<tr><td>{r['from']} → {r['to']}</td><td>{r['psi']}</td>"
        f"<td style='color:{_color(r['rag'])};font-weight:600'>{r['rag']}</td></tr>" for r in art["psi"])
    dqr = "".join(
        f"<tr><td>{r['period']}</td><td>{r['n']:,}</td><td>{r['share_30dpd']*100:.3f}%</td>"
        f"<td>{r['share_90dpd']*100:.3f}%</td></tr>" for r in art["delinquency_trend"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Monitoring {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}.status{{font-size:20px;font-weight:700;color:{_color(art['latest_status'])}}}</style></head><body>
<h1>IFRS 9 — Model Monitoring</h1><div>Run {run_id}</div>
<p class="status">Latest-period status: {art['latest_status']}</p>
<p>Historical worst (incl. COVID / regime shifts): <b style="color:{_color(art['historical_worst'])}">{art['historical_worst']}</b>
&nbsp;·&nbsp; breaches: {art['breach_summary']['red']} red, {art['breach_summary']['amber']} amber,
{art['breach_summary']['green']} green</p>
<h2>PD back-test — predicted vs realised default rate</h2>
<table><tr><th>Period</th><th>N</th><th>Mean pred PD</th><th>Realised DR</th><th>Pred/Realised</th><th>RAG</th></tr>{btr}</table>
<h2>PD distribution drift (PSI, period-over-period)</h2>
<table><tr><th>Transition</th><th>PSI</th><th>RAG</th></tr>{psir}</table>
<p>PSI &lt;{art['thresholds']['psi_amber']} green · &lt;{art['thresholds']['psi_red']} amber · ≥ red</p>
<h2>Delinquency trend</h2>
<table><tr><th>Period</th><th>N</th><th>30+ DPD</th><th>90+ DPD</th></tr>{dqr}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote monitoring report to %s", path)


if __name__ == "__main__":
    run()
