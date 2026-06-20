"""
src/pd_model/run_pd.py
=====================
Phase-5 pipeline. Run from the project root:

    python -m src.pd_model.run_pd

Steps:
  1. Build the labelled, out-of-time-split dataset from the feature store.
  2. Fit WOE/IV and a logistic-regression scorecard.
  3. Train XGBoost and LightGBM on raw features.
  4. Evaluate all three on the out-of-time test set (AUC, Gini, KS, etc.).
  5. Save models + WOE maps to models/, write metrics JSON and an HTML report.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime

import joblib
import numpy as np

from config.settings import settings
from src.pd_model import dataset as ds
from src.pd_model import metrics as M
from src.pd_model import models as mdl
from src.pd_model import woe
from src.utils.logger import get_logger

log = get_logger("pd.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def run() -> dict:
    cfg = settings.config["pd"]
    seed = cfg["random_seed"]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    log.info("Building dataset...")
    data = ds.make_dataset()
    feats = data.features
    ytr = data.y_train.to_numpy().astype(int)
    yte = data.y_test.to_numpy().astype(int)
    log.info("Train default rate %.3f%% | Test default rate %.3f%%",
             100 * ytr.mean(), 100 * yte.mean())

    results = {}

    # ---- Track 1: WOE scorecard ----
    log.info("Fitting WOE/IV + logistic scorecard...")
    iv_tbl, maps = woe.iv_table(data.X_train, data.y_train, feats, cfg["woe_bins"])
    Xtr_woe = woe.transform_frame(data.X_train, maps).to_numpy()
    Xte_woe = woe.transform_frame(data.X_test, maps).to_numpy()
    sc = mdl.train_scorecard(Xtr_woe, ytr, seed)
    p_sc = mdl.proba(sc, Xte_woe)
    results["scorecard"] = M.evaluate(yte, p_sc)
    joblib.dump(sc, MODELS_DIR / "pd_scorecard_logreg.joblib")
    with open(MODELS_DIR / "woe_maps.pkl", "wb") as f:
        pickle.dump(maps, f)

    # ---- Track 2: ML ----
    Xtr = data.X_train.to_numpy()
    Xte = data.X_test.to_numpy()
    log.info("Training XGBoost...")
    xgbm = mdl.train_xgb(Xtr, ytr, seed)
    results["xgboost"] = M.evaluate(yte, mdl.proba(xgbm, Xte))
    joblib.dump(xgbm, MODELS_DIR / "pd_xgboost.joblib")

    log.info("Training LightGBM...")
    lgbm = mdl.train_lgbm(Xtr, ytr, seed)
    p_lgb = mdl.proba(lgbm, Xte)
    results["lightgbm"] = M.evaluate(yte, p_lgb)
    joblib.dump(lgbm, MODELS_DIR / "pd_lightgbm.joblib")

    # Calibration for the best AUC model
    best = max(results, key=lambda k: results[k]["auc"])
    best_prob = {"scorecard": p_sc, "xgboost": mdl.proba(xgbm, Xte), "lightgbm": p_lgb}[best]
    calib = M.calibration_table(yte, best_prob)
    curves = {"roc": M.roc_curve_points(yte, best_prob),
              "ks": M.ks_curve_points(yte, best_prob)}

    # ---- Persist + report ----
    out = {"models": results, "iv": iv_tbl.to_dicts(), "best_model": best,
           "calibration": calib, "curves": curves, "features": feats}
    (MODELS_DIR / "pd_metrics.json").write_text(json.dumps(out, indent=2))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_html(out, REPORTS_DIR / f"pd_report_{run_id}.html", run_id)

    log.info("PD models done. Best: %s (AUC %.4f, Gini %.4f, KS %.4f)",
             best, results[best]["auc"], results[best]["gini"], results[best]["ks"])
    for name, m in results.items():
        log.info("  %-10s AUC=%.4f Gini=%.4f KS=%.4f", name, m["auc"], m["gini"], m["ks"])
    return out


def _write_html(out: dict, path, run_id: str) -> None:
    def row(name, m):
        return (f"<tr><td>{name}</td><td>{m['auc']}</td><td>{m['gini']}</td><td>{m['ks']}</td>"
                f"<td>{m['precision']}</td><td>{m['recall']}</td><td>{m['brier']}</td></tr>")
    model_rows = "".join(row(n, m) for n, m in out["models"].items())
    iv_rows = "".join(f"<tr><td>{r['feature']}</td><td>{r['iv']}</td><td>{r['strength']}</td></tr>"
                      for r in out["iv"])
    cal_rows = "".join(f"<tr><td>{c['bin']}</td><td>{c['n']}</td><td>{c['pred_pd']}</td>"
                       f"<td>{c['obs_default']}</td></tr>" for c in out["calibration"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PD Models {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:12px 0 24px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — PD Model Report</h1><div>Run {run_id} · best model: <b>{out['best_model']}</b></div>
<h2>Model performance (out-of-time test)</h2>
<table><tr><th>Model</th><th>AUC</th><th>Gini</th><th>KS</th><th>Precision</th><th>Recall</th><th>Brier</th></tr>{model_rows}</table>
<h2>Information Value (feature strength)</h2>
<table><tr><th>Feature</th><th>IV</th><th>Strength</th></tr>{iv_rows}</table>
<h2>Calibration — best model (predicted PD vs observed default)</h2>
<table><tr><th>Bin</th><th>N</th><th>Predicted PD</th><th>Observed default</th></tr>{cal_rows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote PD report to %s", path)


if __name__ == "__main__":
    run()
