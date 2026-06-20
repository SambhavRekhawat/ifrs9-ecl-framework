"""
src/lgd_model/run_lgd.py
=======================
Phase-8 pipeline. Run from the project root:

    python -m src.lgd_model.run_lgd

Builds the LGD model from realised liquidations: observed severity stats,
LGD-by-MTM-LTV buckets, a fractional-logit LGD(LTV, age) fit, and a downturn
LGD via an HPI shock. Saves the model + stats and writes an HTML report.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import polars as pl

from config.settings import settings
from src.lgd_model import lgd_data, lgd_model as L
from src.utils.logger import get_logger

log = get_logger("lgd.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def _prepare_features(df, feature_names):
    """Drop features that are entirely missing; median-impute the rest. Keeps all rows."""
    keep, cols = [], []
    for f in feature_names:
        col = df[f].to_numpy().astype(float)
        finite = np.isfinite(col)
        if finite.sum() == 0:
            log.warning("LGD feature '%s' is entirely missing in this sample - dropping it.", f)
            continue
        med = float(np.median(col[finite]))
        keep.append(f)
        cols.append(np.where(finite, col, med))
    if not keep:
        raise ValueError("No usable LGD features (all missing). Check ltv/upb/loan_age population.")
    return keep, np.column_stack(cols)


def run() -> dict:
    cfg = settings.config["lgd"]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = lgd_data.load_lgd_frame()
    if df.height < 30:
        log.warning("Only %d LGD observations - results are indicative only.", df.height)

    lgd = df["lgd"].to_numpy()
    stats = L.summary_stats(lgd)
    buckets = L.lgd_by_bucket(df["mtm_ltv"].to_numpy(), lgd, cfg["ltv_buckets"])

    feats, X = _prepare_features(df, cfg["features"])
    fit = L.fit_lgd(X, lgd, feats, cfg["eps"], cfg["lgd_floor"])
    emp = L.empirical_downturn_lgd(df["default_period"].to_list(), lgd,
                                   cfg.get("downturn_window_months", 24),
                                   cfg.get("downturn_min_obs", 100))
    dt = L.downturn_lgd(fit, X, feats, cfg["downturn_hpi_shock"],
                        cfg.get("downturn_lgd_benchmark", 0.0), empirical=emp)
    if emp:
        log.info("Empirical downturn: worst %dm window %s..%s mean LGD %.3f (n=%d)",
                 emp["window_months"], emp["window_start"], emp["window_end"],
                 emp["lgd"], emp["n_obs"])
    log.info("LGD fit r2=%.3f | coefs=%s | base LGD %.3f -> downturn %.3f (x%.2f, driver=%s)",
             fit["r2"], fit["coefficients"], dt["base_mean_lgd"],
             dt["downturn_lgd"], dt["uplift_ratio"], dt["driver"])

    artifact = {"features": feats, "model": fit["model"], "floor": fit["floor"]}
    joblib.dump(artifact, MODELS_DIR / "lgd_model.joblib")
    out = {"stats": stats, "buckets": buckets, "fit": {k: fit[k] for k in
           ("r2", "coefficients", "intercept", "n")}, "downturn": dt}
    (MODELS_DIR / "lgd_stats.json").write_text(json.dumps(out, indent=2))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(out, REPORTS_DIR / f"lgd_report_{run_id}.html", run_id)
    log.info("LGD done. Mean LGD %.3f over %d defaults | downturn x%.2f",
             stats["mean"], stats["n"], dt["uplift_ratio"])
    return out


def _write_html(out: dict, path, run_id: str) -> None:
    s = out["stats"]
    b_rows = "".join(f"<tr><td>{r['ltv_bucket']}</td><td>{r['n']}</td><td>{r['mean_lgd']}</td></tr>"
                     for r in out["buckets"])
    c_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in out["fit"]["coefficients"].items())
    dt = out["downturn"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>LGD {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — LGD Model Report</h1><div>Run {run_id} · {s['n']} realised defaults</div>
<h2>Observed loss severity (through-the-cycle anchor)</h2>
<table><tr><th>N</th><th>Mean</th><th>Median</th><th>Std</th><th>P10</th><th>P90</th></tr>
<tr><td>{s['n']}</td><td>{s['mean']}</td><td>{s['median']}</td><td>{s['std']}</td><td>{s['p10']}</td><td>{s['p90']}</td></tr></table>
<h2>LGD by mark-to-market LTV bucket</h2>
<table><tr><th>MTM-LTV</th><th>N</th><th>Mean LGD</th></tr>{b_rows}</table>
<h2>Mean regression (r&sup2; = {out['fit']['r2']}, n = {out['fit']['n']})</h2>
<table><tr><th>Feature</th><th>Coefficient</th></tr>{c_rows}
<tr><td><i>intercept</i></td><td>{out['fit']['intercept']}</td></tr></table>
<h2>Downturn LGD ({int(dt['hpi_shock']*100)}% HPI shock; max of empirical / model / benchmark — driver: {dt['driver']})</h2>
<table><tr><th>Base mean LGD</th><th>Empirical worst-window</th><th>Model stressed</th><th>Benchmark</th><th>Downturn LGD (max)</th></tr>
<tr><td>{dt['base_mean_lgd']}</td><td>{(dt['empirical_downturn'] or {}).get('lgd', '—')}</td><td>{dt['model_downturn_lgd']}</td><td>{dt['benchmark']}</td><td>{dt['downturn_lgd']}</td></tr></table>
{f"<p>Worst {dt['empirical_downturn']['window_months']}-month window: {dt['empirical_downturn']['window_start']} to {dt['empirical_downturn']['window_end']} · {dt['empirical_downturn']['n_obs']} dispositions</p>" if dt.get('empirical_downturn') else ""}
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote LGD report to %s", path)


if __name__ == "__main__":
    run()
