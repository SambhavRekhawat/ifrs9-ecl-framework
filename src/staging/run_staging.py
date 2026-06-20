"""
src/staging/run_staging.py
=========================
Phase-10 pipeline. Run from the project root:

    python -m src.staging.run_staging

Assigns IFRS 9 stages (1/2/3) to the current book, reports the stage
distribution and SICR trigger breakdown, and — if migration periods are
configured — a stage-migration matrix between two reporting dates.
"""

from __future__ import annotations

import json
from datetime import datetime

from config.settings import settings
from src.staging import sicr, staging_data
from src.utils.logger import get_logger

log = get_logger("staging.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def _stage(period):
    cfg = settings.config["staging"]
    snap = staging_data.staged_snapshot(period)
    return sicr.assign_stage(
        snap, default_dpd=cfg["default_dpd"], backstop_dpd=cfg["backstop_dpd"],
        pd_rel=cfg["sicr_pd_rel_threshold"], pd_abs=cfg["sicr_pd_abs_threshold"])


def run() -> dict:
    cfg = settings.config["staging"]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    staged = _stage(cfg["reporting_period"])
    dist = sicr.stage_distribution(staged)
    n_default = int(staged["is_default"].sum())
    n_backstop = int(staged["sicr_backstop_30dpd"].sum())
    n_pd = int(staged["sicr_pd_deterioration"].sum())
    for r in dist.iter_rows(named=True):
        log.info("  Stage %d: %s loans (%.2f%%)", r["stage"], f"{r['n']:,}", r["pct"])
    log.info("Triggers: default(90+DPD)=%s  backstop(30+DPD)=%s  PD-deterioration=%s",
             f"{n_default:,}", f"{n_backstop:,}", f"{n_pd:,}")

    migration = None
    if cfg["migration_from"] and cfg["migration_to"]:
        prev = _stage(cfg["migration_from"])
        curr = _stage(cfg["migration_to"])
        migration = sicr.migration_matrix(prev, curr).to_dicts()
        log.info("Migration matrix %s -> %s computed (%d cells).",
                 cfg["migration_from"], cfg["migration_to"], len(migration))

    artifact = {
        "reporting_period": cfg["reporting_period"] or "latest",
        "n_loans": staged.height,
        "distribution": dist.to_dicts(),
        "triggers": {"default_90dpd": n_default, "backstop_30dpd": n_backstop,
                     "pd_deterioration": n_pd},
        "thresholds": {k: cfg[k] for k in
                       ["default_dpd", "backstop_dpd", "sicr_pd_rel_threshold",
                        "sicr_pd_abs_threshold"]},
        "migration": migration,
    }
    (MODELS_DIR / "staging_artifacts.json").write_text(json.dumps(artifact, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(artifact, REPORTS_DIR / f"staging_report_{run_id}.html", run_id)
    log.info("Staging done. %s loans staged.", f"{staged.height:,}")
    return artifact


def _write_html(art: dict, path, run_id: str) -> None:
    drows = "".join(f"<tr><td>Stage {d['stage']}</td><td>{d['n']:,}</td><td>{d['pct']}%</td></tr>"
                    for d in art["distribution"])
    mig = ""
    if art["migration"]:
        mrows = "".join(f"<tr><td>{m['stage_from']}</td><td>{m['stage_to']}</td><td>{m['n']:,}</td></tr>"
                        for m in art["migration"])
        mig = (f"<h2>Stage migration</h2><table><tr><th>From</th><th>To</th><th>Loans</th></tr>{mrows}</table>")
    t = art["triggers"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Staging {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — Staging Report</h1><div>Run {run_id} · book: {art['reporting_period']} · {art['n_loans']:,} loans</div>
<h2>Stage distribution</h2>
<table><tr><th>Stage</th><th>Loans</th><th>Share</th></tr>{drows}</table>
<h2>SICR / default triggers (loan counts)</h2>
<table><tr><th>Default (90+ DPD)</th><th>Backstop (30+ DPD)</th><th>PD deterioration</th></tr>
<tr><td>{t['default_90dpd']:,}</td><td>{t['backstop_30dpd']:,}</td><td>{t['pd_deterioration']:,}</td></tr></table>
{mig}
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote staging report to %s", path)


if __name__ == "__main__":
    run()
