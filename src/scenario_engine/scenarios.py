"""
src/scenario_engine/scenarios.py
==============================
Turns macro scenarios into systematic-factor Z paths and probability-weighted
PIT PD, using the Stage-6 macro->Z model and Vasicek mapping.

Z convention (ASRF): higher Z = stronger economy = LOWER PD. So a downside
scenario must produce a LOWER Z than baseline. Because the fitted unemployment
coefficient is unreliable (COVID-forbearance artifact), two safeguards exist:
  * scenarios lean on HPI (correct sign) as the dominant downside driver;
  * `enforce_monotonic` clamps Z so downside <= base <= upside at every month;
  * `mode = z_overlay` bypasses macro->Z entirely and shifts Z directly.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl

from config.settings import settings
from src.pit_calibration import vasicek
from src.scenario_engine import macro_paths as MP
from src.utils.logger import get_logger

log = get_logger(__name__)
MODELS_DIR = settings.project_root / "models"


def load_macro_to_z():
    art = joblib.load(MODELS_DIR / "macro_to_z.joblib")
    return art["model"], art["features"]


def load_pit_params() -> tuple[float, float]:
    art = json.loads((MODELS_DIR / "pit_artifacts.json").read_text())
    ttc_pd = float(art["ttc_pd"])
    rho = float(art.get("asset_correlation", settings.config["pit"]["asset_correlation"]))
    return ttc_pd, rho


def _baseline_z(model, features, baseline: dict) -> float:
    X = np.array([[baseline[f] for f in features]])
    return float(model.predict(X)[0])


def enforce_monotonic_z(z_paths: dict) -> tuple[dict, bool]:
    """Clamp so downside <= base <= upside at every month. Returns (paths, adjusted?)."""
    if not ({"base", "downside", "upside"} <= set(z_paths)):
        return z_paths, False
    b = z_paths["base"]
    new_down = np.minimum(z_paths["downside"], b)
    new_up = np.maximum(z_paths["upside"], b)
    adjusted = not (np.allclose(new_down, z_paths["downside"]) and np.allclose(new_up, z_paths["upside"]))
    out = dict(z_paths)
    out["downside"], out["upside"] = new_down, new_up
    return out, adjusted


def build_scenarios(macro: pl.DataFrame) -> dict:
    cfg = settings.config["scenario"]
    horizon, ramp = cfg["horizon_months"], cfg["ramp_months"]
    weights = cfg["weights"]
    names = list(weights.keys())

    model, features = load_macro_to_z()
    ttc_pd, rho = load_pit_params()
    baseline = MP.latest_macro(macro, features)

    z_paths: dict[str, np.ndarray] = {}
    macro_path_store: dict[str, np.ndarray] = {}

    if cfg["mode"] == "z_overlay":
        base_z = _baseline_z(model, features, baseline)
        for s in names:
            z_paths[s] = np.full(horizon, base_z + float(cfg["z_overlay"][s]))
    else:  # macro mode
        for s in names:
            path = MP.build_macro_path(baseline, cfg["shocks"][s], features, horizon, ramp)
            macro_path_store[s] = path
            z_paths[s] = model.predict(path)

    monotonic_adjusted = False
    if cfg["enforce_monotonic"]:
        z_paths, monotonic_adjusted = enforce_monotonic_z(z_paths)

    scenarios = {}
    for s in names:
        pit_pd = vasicek.vasicek_pit(np.full(horizon, ttc_pd), z_paths[s], rho)
        scenarios[s] = {"weight": float(weights[s]), "z_path": z_paths[s],
                        "pit_pd_path": np.asarray(pit_pd, dtype=float)}

    weighted = np.zeros(horizon)
    for s in names:
        weighted += scenarios[s]["weight"] * scenarios[s]["pit_pd_path"]

    return {"scenarios": scenarios, "weighted_pit_pd": weighted, "ttc_pd": ttc_pd,
            "rho": rho, "features": features, "baseline": baseline,
            "monotonic_adjusted": monotonic_adjusted, "mode": cfg["mode"],
            "macro_paths": macro_path_store}
