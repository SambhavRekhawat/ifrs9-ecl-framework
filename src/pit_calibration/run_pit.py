"""
src/pit_calibration/run_pit.py
=============================
Phase-6 pipeline. Run from the project root:

    # 1. (once) pull macro data from FRED into the warehouse:
    python -m src.pit_calibration.macro_data

    # 2. build calibration + TTC/PIT:
    python -m src.pit_calibration.run_pit

Steps:
  1. Re-label the panel and load the trained PD model.
  2. Recalibrate model scores to the true base rate (isotonic).
  3. TTC PD = long-run default rate; Z_t implied per period (Vasicek).
  4. Regress Z_t on macro variables -> macro->Z model.
  5. Save calibrator, PIT params, macro->Z model; write an HTML report.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import polars as pl

from config.settings import settings
from src.ingestion import db
from src.pit_calibration import calibration, ttc_pit, vasicek
from src.pd_model import dataset as ds
from src.utils.logger import get_logger

log = get_logger("pit.run")
MODELS_DIR = settings.project_root / "models"
REPORTS_DIR = settings.project_root / "reports"


def _load_macro() -> pl.DataFrame:
    engine = db.get_engine()
    try:
        with engine.connect() as conn:
            m = pl.read_database("SELECT * FROM macro_data", connection=conn)
        # ensure reporting_period is a date
        if m["reporting_period"].dtype != pl.Date:
            m = m.with_columns(pl.col("reporting_period").cast(pl.Date))
        return m
    except Exception as exc:
        raise RuntimeError("Could not read macro_data. Run "
                           "`python -m src.pit_calibration.macro_data` first.") from exc


def run() -> dict:
    pcfg = settings.config["pd"]
    pit_cfg = settings.config["pit"]
    rho = float(pit_cfg["asset_correlation"])
    feats = pcfg["features"]

    log.info("Re-labelling panel...")
    full = ds.build_labeled_frame(feats, pcfg["horizon_months"], pcfg["default_dpd"])
    n0 = full.height
    full = ttc_pit.drop_incomplete_window(full, pcfg["horizon_months"])
    log.info("Dropped %s rows in the last %d months (outcome window not yet complete)",
             f"{n0 - full.height:,}", pcfg["horizon_months"])

    # ---- 1. score with the trained model ----
    model_path = MODELS_DIR / "pd_lightgbm.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run Stage 5 (src.pd_model.run_pd) first.")
    model = joblib.load(model_path)

    split = datetime.strptime(str(pcfg["oot_split_date"]), "%Y-%m-%d").date()
    test = full.filter(pl.col("reporting_period") >= split)
    Xte = test.select(feats).to_numpy()
    s_test = model.predict_proba(Xte)[:, 1]
    yte = test["target"].to_numpy().astype(int)

    # ---- 2. recalibrate to base rate ----
    iso = calibration.fit_calibrator(s_test, yte)
    cal_summary = calibration.calibration_summary(s_test, yte, iso)
    log.info("Calibration: observed=%.4f | raw mean=%.4f | calibrated mean=%.4f",
             cal_summary["observed_default_rate"], cal_summary["mean_raw_score"],
             cal_summary["mean_calibrated_pd"])
    joblib.dump(iso, MODELS_DIR / "pd_calibrator_isotonic.joblib")

    # ---- 3. TTC PD + Z factor ----
    ttc_pd = float(full["target"].mean())
    dr = ttc_pit.default_rate_by_period(full)
    dr = ttc_pit.add_z_factor(dr, ttc_pd, rho)
    log.info("TTC PD (long-run) = %.4f over %d periods", ttc_pd, dr.height)

    # ---- 4. macro -> Z ----
    macro = _load_macro()
    macro_feats = pit_cfg["macro_features"]
    reg = ttc_pit.fit_macro_to_z(dr, macro, macro_feats)

    # ---- 5. persist + report ----
    artifacts = {
        "ttc_pd": round(ttc_pd, 5),
        "asset_correlation": rho,
        "calibration": cal_summary,
        "macro_to_z": {k: reg[k] for k in ("r2", "coefficients", "intercept", "features", "n_obs")},
    }
    joblib.dump({"model": reg["model"], "features": macro_feats},
                MODELS_DIR / "macro_to_z.joblib")
    (MODELS_DIR / "pit_artifacts.json").write_text(json.dumps(artifacts, indent=2))

    dr_out = dr.join(macro.select(["reporting_period"] + macro_feats), on="reporting_period", how="left")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dr_out.write_parquet(REPORTS_DIR / f"pit_z_series_{run_id}.parquet")
    _write_html(artifacts, dr_out, REPORTS_DIR / f"pit_report_{run_id}.html", run_id)

    log.info("PIT done. TTC PD=%.4f | macro->Z R2=%.3f", ttc_pd, reg["r2"])
    return artifacts


def _write_html(art: dict, dr: pl.DataFrame, path, run_id: str) -> None:
    coef_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                        for k, v in art["macro_to_z"]["coefficients"].items())
    z_rows = "".join(
        f"<tr><td>{r['reporting_period']}</td><td>{round(r['dr'],4)}</td><td>{round(r['z'],3)}</td></tr>"
        for r in dr.sort("reporting_period").tail(18).to_dicts())
    c = art["calibration"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PIT {run_id}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:28px;color:#1f2328}}
table{{border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid #d0d7de;padding:6px 12px;font-size:13px}}
th{{background:#f6f8fa}}h2{{font-size:17px}}</style></head><body>
<h1>IFRS 9 — TTC / PIT Calibration Report</h1><div>Run {run_id}</div>
<h2>PD recalibration (out-of-time test)</h2>
<table><tr><th>Observed default rate</th><th>Mean RAW model PD</th><th>Mean CALIBRATED PD</th></tr>
<tr><td>{c['observed_default_rate']}</td><td>{c['mean_raw_score']}</td><td>{c['mean_calibrated_pd']}</td></tr></table>
<h2>TTC anchor &amp; correlation</h2>
<table><tr><th>TTC PD (long-run)</th><th>Asset correlation &rho;</th></tr>
<tr><td>{art['ttc_pd']}</td><td>{art['asset_correlation']}</td></tr></table>
<h2>Macro &rarr; Z regression (R&sup2; = {art['macro_to_z']['r2']}, n = {art['macro_to_z']['n_obs']})</h2>
<table><tr><th>Macro feature</th><th>Coefficient</th></tr>{coef_rows}
<tr><td><i>intercept</i></td><td>{art['macro_to_z']['intercept']}</td></tr></table>
<h2>Recent observed default rate &amp; implied Z</h2>
<table><tr><th>Period</th><th>Observed DR</th><th>Z factor</th></tr>{z_rows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote PIT report to %s", path)


if __name__ == "__main__":
    run()
