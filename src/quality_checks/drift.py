"""
src/quality_checks/drift.py
===========================
Population Stability Index (PSI) drift monitoring + vintage coverage.

PSI measures how much a feature's distribution shifted between a baseline
population and a comparison population. Common rule of thumb:
  PSI < 0.10  -> stable
  0.10-0.25   -> moderate shift (WARN)
  > 0.25      -> significant shift (ALERT)
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from src.quality_checks.checks import CheckResult


def psi(baseline: pl.Series, compare: pl.Series, bins: int = 10) -> float | None:
    """Compute PSI of `compare` against `baseline` using baseline quantile bins."""
    b = baseline.drop_nulls().to_numpy()
    c = compare.drop_nulls().to_numpy()
    if len(b) < 20 or len(c) < 20:
        return None
    edges = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    b_cnt, _ = np.histogram(b, bins=edges)
    c_cnt, _ = np.histogram(c, bins=edges)
    eps = 1e-6
    b_prop = np.clip(b_cnt / b_cnt.sum(), eps, None)
    c_prop = np.clip(c_cnt / c_cnt.sum(), eps, None)
    return float(np.sum((c_prop - b_prop) * np.log(c_prop / b_prop)))


def check_drift(master: pl.DataFrame, features: list[str],
                warn: float, alert: float, bins: int) -> list[CheckResult]:
    """PSI of each vintage vs the IMMEDIATELY PRECEDING vintage (period-over-period).

    This is the standard approach: it catches sudden shifts between consecutive
    quarters, instead of comparing everything to the oldest vintage (which makes
    drift look huge simply because a lot of time has passed).
    """
    out: list[CheckResult] = []
    if "vintage" not in master.columns:
        return out
    # "YYYYQn" sorts chronologically as plain text.
    vintages = sorted(v for v in master["vintage"].unique().to_list() if v)
    if len(vintages) < 2:
        return out

    for prev_v, cur_v in zip(vintages[:-1], vintages[1:]):
        prev = master.filter(pl.col("vintage") == prev_v)
        cur = master.filter(pl.col("vintage") == cur_v)
        for feat in features:
            if feat not in master.columns:
                continue
            val = psi(prev[feat], cur[feat], bins=bins)
            if val is None:
                continue
            status = "FAIL" if val > alert else ("WARN" if val > warn else "PASS")
            out.append(CheckResult("loan_master", "drift", feat, f"psi_{cur_v}_vs_{prev_v}",
                                   round(val, 4), warn, status,
                                   f"{feat} {cur_v} vs {prev_v}: PSI={round(val, 4)}"))
    return out


_VINTAGE_RE = re.compile(r"(20\d{2})Q([1-4])")


def check_vintage_coverage(master: pl.DataFrame) -> list[CheckResult]:
    """Detect gaps in the quarterly vintage sequence (your missing-2018 concern)."""
    if "vintage" not in master.columns:
        return []
    present = sorted({v for v in master["vintage"].unique().to_list() if v and _VINTAGE_RE.match(v)})
    if not present:
        return []

    def to_idx(v: str) -> int:
        y, q = _VINTAGE_RE.match(v).groups()
        return int(y) * 4 + (int(q) - 1)

    def to_str(idx: int) -> str:
        return f"{idx // 4}Q{idx % 4 + 1}"

    idxs = [to_idx(v) for v in present]
    full = list(range(min(idxs), max(idxs) + 1))
    missing = [to_str(i) for i in full if i not in idxs]
    status = "WARN" if missing else "PASS"
    msg = (f"{len(present)} vintages present ({present[0]}..{present[-1]}); "
           f"missing: {missing}" if missing else f"All {len(present)} quarters present, no gaps")
    return [CheckResult("loan_master", "coverage", "vintage", "missing_quarters",
                        float(len(missing)), 0, status, msg)]