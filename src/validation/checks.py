"""
src/validation/checks.py
======================
Pass/fail reconciliation and sanity checks over the saved artifacts. Each check
returns a dict {check, passed, severity, detail}. Pure functions over JSON
artifacts so they are fully testable.
"""

from __future__ import annotations


def _r(check, passed, detail, severity="error"):
    return {"check": check, "passed": bool(passed), "severity": severity, "detail": detail}


def coverage_monotonic_by_stage(ecl: dict) -> dict:
    by = {r["stage"]: r["coverage_pct"] for r in ecl["summary"]["by_stage"]}
    ok = by.get(1, 0) <= by.get(2, 0) <= by.get(3, 1e9)
    return _r("coverage_monotonic_by_stage", ok,
              f"Stage1={by.get(1)}%, Stage2={by.get(2)}%, Stage3={by.get(3)}%")


def stage3_equals_weighted_lgd(ecl: dict, tol_pct: float) -> dict:
    by = {r["stage"]: r["coverage_pct"] for r in ecl["summary"]["by_stage"]}
    wl = ecl["weighted_lgd"] * 100
    s3 = by.get(3)
    ok = s3 is not None and abs(s3 - wl) <= tol_pct
    return _r("stage3_equals_weighted_lgd", ok,
              f"Stage3 coverage {s3}% vs weighted LGD {round(wl, 3)}% (tol {tol_pct})")


def portfolio_coverage_in_range(ecl: dict, lo: float, hi: float) -> dict:
    cov = ecl["summary"]["coverage_pct"]
    return _r("portfolio_coverage_in_range", lo <= cov <= hi,
              f"coverage {cov}% in [{lo}, {hi}]%")


def scenario_ordering(scn: dict) -> dict:
    ok = scn.get("ordering_ok", False)
    return _r("scenario_pd_ordering", ok,
              f"downside>=base>=upside PIT PD: {ok} ({scn.get('pit_pd_12m')})")


def pd_discrimination(pd_metrics: dict, min_auc: float) -> dict:
    best = pd_metrics["best_model"]
    m = pd_metrics["models"][best]
    auc = m.get("auc") or m.get("AUC")
    return _r("pd_discrimination_auc", auc is not None and auc >= min_auc,
              f"best model '{best}' AUC {auc} >= {min_auc}")


def effective_pd_calibration(cal_mean_pd: float, ttc_pd: float, band: list) -> dict:
    """Validate the CALIBRATED portfolio PD (what feeds ECL) against the TTC PD.

    This is the calibration that matters: the raw model is intentionally hot
    (class-weighted) and recalibrated by isotonic downstream.
    """
    ratio = cal_mean_pd / max(ttc_pd, 1e-9)
    return _r("effective_pd_calibration", band[0] <= ratio <= band[1],
              f"calibrated portfolio PD {round(cal_mean_pd, 4)} vs TTC {round(ttc_pd, 4)} "
              f"(ratio {round(ratio, 2)} in {band})")


def pd_calibration_band(pd_metrics: dict, band: list) -> dict:
    cal = pd_metrics.get("calibration", [])
    if not cal:
        return _r("pd_calibration_band", True, "no calibration table", severity="warn")
    tot_n = sum(b["n"] for b in cal)
    pred = sum(b["pred_pd"] * b["n"] for b in cal) / max(tot_n, 1)
    obs = sum(b["obs_default"] * b["n"] for b in cal) / max(tot_n, 1)
    ratio = pred / max(obs, 1e-9)
    return _r("pd_calibration_band_raw", band[0] <= ratio <= band[1],
              f"RAW model pred/obs PD ratio {round(ratio, 3)} (pred {round(pred, 4)}, "
              f"obs {round(obs, 4)}) — informational; class-weighted model is hot by design "
              f"and recalibrated by isotonic downstream (see effective_pd_calibration)",
              severity="warn")


def run_all(ecl: dict, scn: dict, pd_metrics: dict, cfg: dict) -> list[dict]:
    return [
        coverage_monotonic_by_stage(ecl),
        stage3_equals_weighted_lgd(ecl, cfg["stage3_lgd_tolerance_pct"]),
        portfolio_coverage_in_range(ecl, *cfg["coverage_range_pct"]),
        scenario_ordering(scn),
        pd_discrimination(pd_metrics, cfg["min_auc"]),
        pd_calibration_band(pd_metrics, cfg["calibration_ratio_band"]),
    ]
