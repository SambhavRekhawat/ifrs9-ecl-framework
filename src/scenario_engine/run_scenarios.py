"""
src/scenario_engine/run_scenarios.py
==================================
Phase-11 pipeline. Run from the project root:

    python -m src.scenario_engine.run_scenarios

Builds base / upside / downside macro scenarios, maps them to systematic-factor
Z paths via the Stage-6 macro->Z model, applies the Vasicek PIT mapping, and
reports the probability-weighted PIT PD term path for the ECL engine (Stage 12).
"""

from __future__ import annotations

import json
from datetime import datetime

import polars as pl

from config.settings import settings
from src.scenario_engine import scenarios as S
from src.utils.logger import get_logger

log = get_logger("scenario.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"
MACRO_PARQUET = settings.project_root / "data" / "macro" / "macro_data.parquet"


def run() -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    macro = pl.read_parquet(MACRO_PARQUET)
    res = S.build_scenarios(macro)

    horizon = len(res["weighted_pit_pd"])
    log.info("Scenarios (%s mode) over %d months | TTC PD %.4f | rho %.2f",
             res["mode"], horizon, res["ttc_pd"], res["rho"])
    for s, d in res["scenarios"].items():
        log.info("  %-9s w=%.2f  Z[12m]=%+.3f  PIT_PD[12m]=%.4f  PIT_PD[end]=%.4f",
                 s, d["weight"], d["z_path"][min(11, horizon - 1)],
                 d["pit_pd_path"][min(11, horizon - 1)], d["pit_pd_path"][-1])
    log.info("Weighted PIT PD: 12m=%.4f  end=%.4f",
             res["weighted_pit_pd"][min(11, horizon - 1)], res["weighted_pit_pd"][-1])
    if res["monotonic_adjusted"]:
        log.warning("Monotonicity enforced: macro->Z gave a non-monotone ordering "
                    "(expected, given the unemployment-coefficient artifact); Z paths were clamped.")

    # sanity ordering check on the 12m PIT PD
    pds = {s: float(res["scenarios"][s]["pit_pd_path"][min(11, horizon - 1)])
           for s in res["scenarios"]}
    ordering_ok = pds.get("downside", 1) >= pds.get("base", 0) >= pds.get("upside", 0)
    log.info("PIT PD ordering downside>=base>=upside at 12m: %s (%s)",
             ordering_ok, {k: round(v, 4) for k, v in pds.items()})

    artifact = {
        "mode": res["mode"], "horizon_months": horizon, "ttc_pd": round(res["ttc_pd"], 5),
        "rho": res["rho"], "features": res["features"],
        "baseline": {k: round(v, 4) for k, v in res["baseline"].items()},
        "monotonic_adjusted": res["monotonic_adjusted"],
        "weights": {s: res["scenarios"][s]["weight"] for s in res["scenarios"]},
        "z_12m": {s: round(float(res["scenarios"][s]["z_path"][min(11, horizon - 1)]), 4)
                  for s in res["scenarios"]},
        "pit_pd_12m": {s: round(v, 5) for s, v in pds.items()},
        "weighted_pit_pd_12m": round(float(res["weighted_pit_pd"][min(11, horizon - 1)]), 5),
        "ordering_ok": bool(ordering_ok),
        # full paths for the ECL engine
        "z_paths": {s: [round(float(x), 5) for x in res["scenarios"][s]["z_path"]]
                    for s in res["scenarios"]},
        "pit_pd_paths": {s: [round(float(x), 6) for x in res["scenarios"][s]["pit_pd_path"]]
                         for s in res["scenarios"]},
        "weighted_pit_pd_path": [round(float(x), 6) for x in res["weighted_pit_pd"]],
    }
    (MODELS_DIR / "scenario_artifacts.json").write_text(json.dumps(artifact, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(artifact, REPORTS_DIR / f"scenario_report_{run_id}.html", run_id)
    log.info("Scenario engine done. Saved scenario_artifacts.json")
    return artifact


def _write_html(art: dict, path, run_id: str) -> None:
    srows = "".join(
        f"<tr><td>{s}</td><td>{art['weights'][s]:.2f}</td><td>{art['z_12m'][s]:+.3f}</td>"
        f"<td>{art['pit_pd_12m'][s]:.4f}</td></tr>" for s in art["weights"])
    brows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in art["baseline"].items())
    warn = ("<p style='color:#9a6700'><b>Note:</b> macro&rarr;Z produced a non-monotone "
            "ordering and Z paths were clamped (expected given the unemployment-coefficient "
            "artifact documented in Stage 6).</p>") if art["monotonic_adjusted"] else ""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Scenarios {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — Scenario Engine</h1>
<div>Run {run_id} · mode: {art['mode']} · horizon {art['horizon_months']}m · TTC PD {art['ttc_pd']} · &rho; {art['rho']}</div>
<h2>Scenarios (at 12 months)</h2>
<table><tr><th>Scenario</th><th>Weight</th><th>Z (12m)</th><th>PIT PD (12m)</th></tr>{srows}</table>
<p>Probability-weighted PIT PD at 12m: <b>{art['weighted_pit_pd_12m']}</b>
&nbsp;|&nbsp; ordering downside&ge;base&ge;upside: <b>{art['ordering_ok']}</b></p>
{warn}
<h2>Baseline macro anchor (latest observed)</h2>
<table><tr><th>Feature</th><th>Value</th></tr>{brows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote scenario report to %s", path)


if __name__ == "__main__":
    run()
