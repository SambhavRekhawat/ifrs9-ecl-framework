"""
src/quality_checks/scorecard.py
==============================
Turns a flat list of CheckResult objects into a scorecard: counts by status,
a 0-100 quality score, and a letter grade, per table and overall.

Scoring: start at 100, subtract a penalty for each FAIL and each WARN
(scored checks only; INFO is ignored). Score is floored at 0.
"""

from __future__ import annotations

from src.quality_checks.checks import CheckResult

# WARN counts as a half-pass; INFO is not scored. This keeps the score bounded
# 0-100 and proportional, so a handful of expected cross-vintage drift FAILs
# over a 6-year span don't tank the grade to zero.
WARN_WEIGHT = 0.5


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _score_group(results: list[CheckResult]) -> dict:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    scored = counts["PASS"] + counts["WARN"] + counts["FAIL"]
    if scored == 0:
        score = 100.0
    else:
        score = 100.0 * (counts["PASS"] + WARN_WEIGHT * counts["WARN"]) / scored
    return {**counts, "score": round(score, 1), "grade": _grade(score),
            "total_checks": sum(counts.values())}


def build_scorecard(results: list[CheckResult]) -> dict:
    """Return {'overall': {...}, 'by_table': {table: {...}}}."""
    by_table: dict[str, list[CheckResult]] = {}
    for r in results:
        by_table.setdefault(r.table, []).append(r)
    return {
        "overall": _score_group(results),
        "by_table": {t: _score_group(rs) for t, rs in by_table.items()},
    }