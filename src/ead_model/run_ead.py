"""
src/ead_model/run_ead.py
=======================
Phase-9 pipeline. Run from the project root:

    python -m src.ead_model.run_ead

Validates the amortization-based EAD engine against actual observed paydown,
reports projection accuracy by horizon, derives the observed curtailment factor,
and saves the EAD model artifact for the ECL engine.
"""

from __future__ import annotations

import json
from datetime import datetime

from config.settings import settings
from src.ead_model import ead_data, ead_model as E
from src.utils.logger import get_logger

log = get_logger("ead.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def run() -> dict:
    cfg = settings.config["ead"]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    panel = ead_data.load_ead_panel()
    validation = E.validate(panel, cfg["horizons"])
    curt = E.curtailment_factor(validation, at_horizon=12)
    for r in validation:
        log.info("  h=%-3d n=%s  pred=%s actual=%s  MAE=%s  actual/sched=%s",
                 r["horizon"], f"{r['n']:,}", f"{r['mean_pred']:,.0f}",
                 f"{r['mean_actual']:,.0f}", f"{r['mae']:,.0f}", r["actual_to_scheduled"])
    log.info("Curtailment factor @12mo (actual/scheduled) = %.4f", curt)

    artifact = {
        "method": "scheduled_amortization",
        "curtailment_factor_12m": curt,
        "apply_curtailment_adjustment": bool(cfg["apply_curtailment_adjustment"]),
        "validation": validation,
    }
    (MODELS_DIR / "ead_model.json").write_text(json.dumps(artifact, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(artifact, REPORTS_DIR / f"ead_report_{run_id}.html", run_id)
    log.info("EAD done. Validated on %s loan-months; curtailment x%.3f",
             f"{panel.height:,}", curt)
    return artifact


def _write_html(art: dict, path, run_id: str) -> None:
    rows = "".join(
        f"<tr><td>{r['horizon']}</td><td>{r['n']:,}</td><td>{r['mean_pred']:,.0f}</td>"
        f"<td>{r['mean_actual']:,.0f}</td><td>{r['mae']:,.0f}</td>"
        f"<td>{r['median_abs_pct_err']}</td><td>{r['actual_to_scheduled']}</td></tr>"
        for r in art["validation"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>EAD {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — EAD Model Report</h1><div>Run {run_id} · method: scheduled amortization</div>
<h2>Projected vs actual balance, by horizon</h2>
<table><tr><th>Horizon (mo)</th><th>N</th><th>Mean predicted</th><th>Mean actual</th>
<th>MAE ($)</th><th>Median abs % err</th><th>Actual / scheduled</th></tr>{rows}</table>
<p>Curtailment factor @12mo (actual/scheduled paydown): <b>{art['curtailment_factor_12m']}</b>
&mdash; below 1.0 means borrowers pay down faster than the schedule (curtailments).</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote EAD report to %s", path)


if __name__ == "__main__":
    run()
