"""
src/ecl_engine/run_ecl.py
=======================
Phase-12 pipeline. Run from the project root:

    python -m src.ecl_engine.run_ecl

Assembles per-loan inputs, builds scenario PD multipliers, and computes the
probability-weighted, scenario-conditioned IFRS 9 ECL (12-month vs lifetime by
stage), combining PD term structure x LGD x EAD x prepayment survival, discounted
at the note rate.
"""

from __future__ import annotations

import json
from datetime import datetime

from config.settings import settings
from src.ecl_engine import ecl as E, ecl_data, term_structure as T
from src.utils.logger import get_logger

log = get_logger("ecl.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def _load_scenarios():
    art = json.loads((MODELS_DIR / "scenario_artifacts.json").read_text())
    return art["pit_pd_paths"], art["weighted_pit_pd_path"], art["weights"], art["horizon_months"]


def _load_lgd(scenario_names, downside_name):
    art = json.loads((MODELS_DIR / "lgd_stats.json").read_text())
    base = float(art["stats"]["mean"])
    down = float(art["downturn"]["downturn_lgd"])
    return {s: (down if s == downside_name else base) for s in scenario_names}


def run() -> dict:
    cfg = settings.config["ecl"]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    pit_paths, weighted_path, weights, scen_horizon = _load_scenarios()
    horizon = min(cfg["horizon_months"], scen_horizon)
    multipliers = T.scenario_multipliers(pit_paths, weighted_path)
    lgd_by_scenario = _load_lgd(list(weights.keys()), cfg["lgd_downside_scenario"])
    smm = T.cpr_to_smm(cfg["annual_cpr"])

    loans = ecl_data.load_loans(cfg["reporting_period"])
    res = E.compute_ecl(loans, multipliers, weights, lgd_by_scenario, smm, horizon)
    s = res["summary"]

    log.info("ECL portfolio: EAD %s | ECL %s | coverage %.4f%%",
             f"${s['total_ead']:,.0f}", f"${s['total_ecl']:,.0f}", s["coverage_pct"])
    log.info("  12m total %s | lifetime total %s | weighted LGD %.4f",
             f"${s['ecl_12m_total']:,.0f}", f"${s['ecl_lifetime_total']:,.0f}", res["weighted_lgd"])
    for r in s["by_stage"]:
        log.info("  Stage %d: %s loans | EAD %s | ECL %s | coverage %.3f%%",
                 r["stage"], f"{r['n_loans']:,}", f"${r['ead']:,.0f}",
                 f"${r['ecl']:,.0f}", r["coverage_pct"])

    artifact = {
        "reporting_period": cfg["reporting_period"] or "latest",
        "horizon_months": horizon, "annual_cpr": cfg["annual_cpr"],
        "weighted_lgd": round(res["weighted_lgd"], 4),
        "lgd_by_scenario": {k: round(v, 4) for k, v in lgd_by_scenario.items()},
        "weights": weights, "summary": s,
    }
    (MODELS_DIR / "ecl_results.json").write_text(json.dumps(artifact, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(artifact, REPORTS_DIR / f"ecl_report_{run_id}.html", run_id)
    log.info("ECL engine done. Saved ecl_results.json")
    return artifact


def _write_html(art: dict, path, run_id: str) -> None:
    s = art["summary"]
    srows = "".join(
        f"<tr><td>Stage {r['stage']}</td><td>{r['n_loans']:,}</td><td>${r['ead']:,.0f}</td>"
        f"<td>${r['ecl']:,.0f}</td><td>{r['coverage_pct']}%</td></tr>" for r in s["by_stage"])
    lrows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in art["lgd_by_scenario"].items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ECL {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}.big{{font-size:22px;font-weight:600}}</style></head><body>
<h1>IFRS 9 — Expected Credit Loss</h1>
<div>Run {run_id} · book: {art['reporting_period']} · horizon {art['horizon_months']}m · CPR {art['annual_cpr']}</div>
<h2>Portfolio</h2>
<div class="big">ECL ${s['total_ecl']:,.0f} &nbsp; on EAD ${s['total_ead']:,.0f} &nbsp; = {s['coverage_pct']}% coverage</div>
<p>12-month basis total: ${s['ecl_12m_total']:,.0f} · lifetime basis total: ${s['ecl_lifetime_total']:,.0f}
· probability-weighted LGD: {art['weighted_lgd']}</p>
<h2>By stage</h2>
<table><tr><th>Stage</th><th>Loans</th><th>EAD</th><th>ECL</th><th>Coverage</th></tr>{srows}</table>
<h2>LGD by scenario</h2>
<table><tr><th>Scenario</th><th>LGD</th></tr>{lrows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote ECL report to %s", path)


if __name__ == "__main__":
    run()
