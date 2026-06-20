"""
src/pd_model/woe.py
==================
Weight of Evidence (WOE) binning and Information Value (IV) - the traditional
credit-scorecard toolkit.

  WOE_bin = ln( (% of non-defaults in bin) / (% of defaults in bin) )
  IV      = sum_bins ( (%non-default - %default) * WOE )

IV rule of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong.
Nulls form their own bin, so WOE-transformed features never contain NaN.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_EPS = 0.5  # Laplace smoothing so empty cells don't blow up the log


def fit_woe(x: pl.Series, y: pl.Series, bins: int = 8, max_discrete: int = 20) -> dict:
    """Return a WOE mapping for one feature.

    Discrete / sparse features (few distinct values, e.g. delinquency counts that
    are mostly 0) are binned by their DISTINCT VALUES; continuous features use
    quantile bins. This stops sparse features collapsing into one bin (IV=0).
    """
    xv = x.to_numpy().astype(float)
    yv = y.to_numpy().astype(int)
    mask = ~np.isnan(xv)

    total_bad = max(yv.sum(), 1)
    total_good = max((yv == 0).sum(), 1)
    woe_map, iv = {}, 0.0

    # Null values always form their own bin.
    if (~mask).any():
        w, c = _bin_woe(yv[~mask], total_good, total_bad)
        woe_map["__null__"] = w
        iv += c

    uniques = np.unique(xv[mask]) if mask.any() else np.array([])

    # ---- Discrete mode: one bin per distinct value ----
    if 0 < len(uniques) <= max_discrete:
        for v in uniques:
            sel = xv[mask] == v
            w, c = _bin_woe(yv[mask][sel], total_good, total_bad)
            woe_map[float(v)] = w
            iv += c
        return {"kind": "discrete", "woe": woe_map, "iv": float(iv)}

    # ---- Numeric mode: quantile bins ----
    if mask.sum() == 0:
        edges = np.array([-np.inf, np.inf])
    else:
        edges = np.unique(np.nanquantile(xv[mask], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(xv[mask], edges[1:-1], right=False)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() == 0:
            woe_map[b] = 0.0
            continue
        w, c = _bin_woe(yv[mask][sel], total_good, total_bad)
        woe_map[b] = w
        iv += c
    return {"kind": "numeric", "edges": edges, "woe": woe_map, "iv": float(iv)}


def _bin_woe(y_bin: np.ndarray, total_good: int, total_bad: int) -> tuple[float, float]:
    bad = (y_bin == 1).sum()
    good = (y_bin == 0).sum()
    pct_bad = (bad + _EPS) / (total_bad + _EPS)
    pct_good = (good + _EPS) / (total_good + _EPS)
    woe = float(np.log(pct_good / pct_bad))
    contrib = (pct_good - pct_bad) * woe
    return woe, contrib


def transform_woe(x: pl.Series, mapping: dict) -> pl.Series:
    """Map raw feature values to their WOE values (discrete or numeric)."""
    xv = x.to_numpy().astype(float)
    woe = mapping["woe"]
    out = np.empty(len(xv), dtype=float)
    null_woe = woe.get("__null__", 0.0)
    nanmask = np.isnan(xv)
    out[nanmask] = null_woe

    if mapping["kind"] == "discrete":
        out[~nanmask] = [woe.get(float(v), 0.0) for v in xv[~nanmask]]
    else:
        edges = mapping["edges"]
        idx = np.digitize(xv[~nanmask], edges[1:-1], right=False)
        out[~nanmask] = [woe.get(int(b), 0.0) for b in idx]
    return pl.Series(x.name + "_woe", out)


def iv_table(X: pl.DataFrame, y: pl.Series, features: list[str], bins: int = 8) -> pl.DataFrame:
    rows = []
    maps = {}
    for f in features:
        m = fit_woe(X[f], y, bins)
        maps[f] = m
        rows.append({"feature": f, "iv": round(m["iv"], 4),
                     "strength": _iv_strength(m["iv"])})
    table = pl.DataFrame(rows).sort("iv", descending=True)
    return table, maps


def _iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspiciously strong"


def transform_frame(X: pl.DataFrame, maps: dict) -> pl.DataFrame:
    return pl.DataFrame({f + "_woe": transform_woe(X[f], maps[f]) for f in maps})
