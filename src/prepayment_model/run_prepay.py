"""
src/prepayment/run_prepay.py
===========================
Phase-7 pipeline. Run from the project root:

    python -m src.prepayment_model.run_prepay

Mirrors the PD pipeline (reusing its WOE / metrics / model code) but for the
competing-risk PREPAYMENT target. Saves models with a `pp_` prefix.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime

import joblib

from config.settings import settings
from src.pd_model import metrics as M
from src.pd_model import models as mdl
from src.pd_model import woe
from src.prepayment_model import dataset as ds
from src.utils.logger import get_logger

log = get_logger("prepay.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def run() -> dict:
    cfg = settings.config["prepay"]
    seed = cfg["random_seed"]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    log.info("Building prepayment dataset...")
    data = ds.make_dataset()
    feats = data.features
    ytr = data.y_train.to_numpy().astype(int)
    yte = data.y_test.to_numpy().astype(int)
    log.info("Train prepay rate %.2f%% | Test prepay rate %.2f%%", 100 * ytr.mean(), 100 * yte.mean())

    results = {}

    # ---- Track 1: WOE scorecard ----
    log.info("Fitting WOE/IV + logistic scorecard...")
    iv_tbl, maps = woe.iv_table(data.X_train, data.y_train, feats, cfg["woe_bins"])
    Xtr_woe = woe.transform_frame(data.X_train, maps).to_numpy()
    Xte_woe = woe.transform_frame(data.X_test, maps).to_numpy()
    sc = mdl.train_scorecard(Xtr_woe, ytr, seed)
    results["scorecard"] = M.evaluate(yte, mdl.proba(sc, Xte_woe))
    joblib.dump(sc, MODELS_DIR / "pp_scorecard_logreg.joblib")
    with open(MODELS_DIR / "pp_woe_maps.pkl", "wb") as f:
        pickle.dump(maps, f)

    # free the TRAIN WOE array before the ML track — it is not needed again and
    # holding it alongside the raw matrices + XGBoost DMatrix can exhaust RAM.
    # (Xte_woe is kept: it is reused below to score the scorecard for best_prob.)
    import gc
    del Xtr_woe
    gc.collect()

    # ---- Track 2: ML ----
    Xtr, Xte = data.X_train.to_numpy(), data.X_test.to_numpy()
    log.info("Training XGBoost...")
    xgbm = mdl.train_xgb(Xtr, ytr, seed)
    results["xgboost"] = M.evaluate(yte, mdl.proba(xgbm, Xte))
    joblib.dump(xgbm, MODELS_DIR / "pp_xgboost.joblib")

    log.info("Training LightGBM...")
    lgbm = mdl.train_lgbm(Xtr, ytr, seed)
    p_lgb = mdl.proba(lgbm, Xte)
    results["lightgbm"] = M.evaluate(yte, p_lgb)
    joblib.dump(lgbm, MODELS_DIR / "pp_lightgbm.joblib")

    best = max(results, key=lambda k: results[k]["auc"])
    best_prob = {"scorecard": mdl.proba(sc, Xte_woe), "xgboost": mdl.proba(xgbm, Xte), "lightgbm": p_lgb}[best]
    calib = M.calibration_table(yte, best_prob)

    out = {"models": results, "iv": iv_tbl.to_dicts(), "best_model": best,
           "calibration": calib, "features": feats}
    (MODELS_DIR / "pp_metrics.json").write_text(json.dumps(out, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(out, REPORTS_DIR / f"prepay_report_{run_id}.html", run_id)

    log.info("Prepay models done. Best: %s (AUC %.4f, Gini %.4f, KS %.4f)",
             best, results[best]["auc"], results[best]["gini"], results[best]["ks"])
    for name, m in results.items():
        log.info("  %-10s AUC=%.4f Gini=%.4f KS=%.4f", name, m["auc"], m["gini"], m["ks"])
    return out


def _write_html(out: dict, path, run_id: str) -> None:
    def row(n, m):
        return (f"<tr><td>{n}</td><td>{m['auc']}</td><td>{m['gini']}</td><td>{m['ks']}</td>"
                f"<td>{m['precision']}</td><td>{m['recall']}</td><td>{m['brier']}</td></tr>")
    model_rows = "".join(row(n, m) for n, m in out["models"].items())
    iv_rows = "".join(f"<tr><td>{r['feature']}</td><td>{r['iv']}</td><td>{r['strength']}</td></tr>"
                      for r in out["iv"])
    cal_rows = "".join(f"<tr><td>{c['bin']}</td><td>{c['n']}</td><td>{c['pred_pd']}</td>"
                       f"<td>{c['obs_default']}</td></tr>" for c in out["calibration"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Prepay {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:12px 0 24px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — Prepayment Model Report</h1><div>Run {run_id} · best model: <b>{out['best_model']}</b></div>
<h2>Model performance (out-of-time test)</h2>
<table><tr><th>Model</th><th>AUC</th><th>Gini</th><th>KS</th><th>Precision</th><th>Recall</th><th>Brier</th></tr>{model_rows}</table>
<h2>Information Value (prepayment drivers)</h2>
<table><tr><th>Feature</th><th>IV</th><th>Strength</th></tr>{iv_rows}</table>
<h2>Calibration — best model (predicted vs observed prepay rate)</h2>
<table><tr><th>Bin</th><th>N</th><th>Predicted</th><th>Observed</th></tr>{cal_rows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote prepayment report to %s", path)


if __name__ == "__main__":
    run()
