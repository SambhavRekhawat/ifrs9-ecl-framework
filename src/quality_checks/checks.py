"""
src/quality_checks/checks.py
============================
The individual data-quality checks. Each returns a list of CheckResult objects
so the orchestrator can aggregate, score, and report them uniformly.

Status meanings:
  PASS  - check satisfied
  WARN  - a data oddity worth noting (e.g. out-of-range values) but not fatal
  FAIL  - a hard error (missing key data, duplicates, negative balances)
  INFO  - report-only metric, not scored (e.g. outlier counts)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime

import polars as pl


@dataclass
class CheckResult:
    table: str
    check_type: str
    column: str | None
    metric: str
    value: float | None
    threshold: float | None
    status: str          # PASS / WARN / FAIL / INFO
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 4) if total else 0.0


# --------------------------------------------------------------------------- #
# 1. Completeness (missing values)
# --------------------------------------------------------------------------- #
def check_completeness(df: pl.DataFrame, table: str, core_cols: list[str],
                       max_null_pct: float) -> list[CheckResult]:
    out: list[CheckResult] = []
    total = df.height
    for col in core_cols:
        if col not in df.columns:
            out.append(CheckResult(table, "completeness", col, "exists", 0, None,
                                   "FAIL", f"Core column '{col}' is missing from {table}"))
            continue
        nulls = df[col].null_count()
        pct = _pct(nulls, total)
        status = "FAIL" if pct > max_null_pct else "PASS"
        out.append(CheckResult(table, "completeness", col, "null_pct", pct, max_null_pct,
                               status, f"{col}: {pct}% null (limit {max_null_pct}%)"))
    # Report-only: how many columns are very sparse (>50% null)
    sparse = sum(1 for c in df.columns if _pct(df[c].null_count(), total) > 50)
    out.append(CheckResult(table, "completeness", None, "sparse_columns", sparse, None,
                           "INFO", f"{sparse} column(s) are >50% null (often legitimate)"))
    return out


# --------------------------------------------------------------------------- #
# 2. Uniqueness (duplicates)
# --------------------------------------------------------------------------- #
def check_uniqueness(df: pl.DataFrame, table: str, key_cols: list[str]) -> list[CheckResult]:
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols:
        return []
    dupes = df.height - df.select(key_cols).unique().height
    status = "FAIL" if dupes > 0 else "PASS"
    return [CheckResult(table, "uniqueness", "+".join(key_cols), "duplicate_rows",
                        float(dupes), 0, status,
                        f"{dupes} duplicate row(s) on key {key_cols}")]


# --------------------------------------------------------------------------- #
# 3. Non-negative balances
# --------------------------------------------------------------------------- #
def check_non_negative(df: pl.DataFrame, table: str, cols: list[str]) -> list[CheckResult]:
    out: list[CheckResult] = []
    for col in cols:
        if col not in df.columns:
            continue
        neg = df.filter(pl.col(col) < 0).height
        status = "FAIL" if neg > 0 else "PASS"
        out.append(CheckResult(table, "non_negative", col, "negative_count",
                               float(neg), 0, status, f"{col}: {neg} negative value(s)"))
    return out


# --------------------------------------------------------------------------- #
# 4. Numeric ranges (validity)
# --------------------------------------------------------------------------- #
def check_numeric_ranges(df: pl.DataFrame, table: str,
                         ranges: dict[str, list]) -> list[CheckResult]:
    out: list[CheckResult] = []
    total = df.height
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        bad = df.filter(
            (pl.col(col).is_not_null()) & ((pl.col(col) < lo) | (pl.col(col) > hi))
        ).height
        pct = _pct(bad, total)
        status = "WARN" if bad > 0 else "PASS"
        out.append(CheckResult(table, "range", col, "out_of_range_pct", pct, None,
                               status, f"{col}: {bad} value(s) outside [{lo}, {hi}] ({pct}%)"))
    return out


# --------------------------------------------------------------------------- #
# 5. Date validity (unparseable or out of plausible range)
# --------------------------------------------------------------------------- #
def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def check_dates(df: pl.DataFrame, table: str, bounds: dict[str, list]) -> list[CheckResult]:
    out: list[CheckResult] = []
    total = df.height
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        lo_d, hi_d = _to_date(str(lo)), _to_date(str(hi))
        nulls = df[col].null_count()
        oob = df.filter(
            (pl.col(col).is_not_null()) & ((pl.col(col) < lo_d) | (pl.col(col) > hi_d))
        ).height
        bad = nulls + oob
        pct = _pct(bad, total)
        status = "WARN" if bad > 0 else "PASS"
        out.append(CheckResult(table, "date_validity", col, "invalid_date_pct", pct, None,
                               status, f"{col}: {nulls} unparseable + {oob} out-of-range "
                                       f"[{lo_d}..{hi_d}] = {pct}%"))
    return out


# --------------------------------------------------------------------------- #
# 6. Outliers (IQR method, report-only)
# --------------------------------------------------------------------------- #
def check_outliers(df: pl.DataFrame, table: str, cols: list[str]) -> list[CheckResult]:
    out: list[CheckResult] = []
    total = df.height
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].drop_nulls()
        if s.len() < 20:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = df.filter(
            (pl.col(col).is_not_null()) & ((pl.col(col) < lo) | (pl.col(col) > hi))
        ).height
        out.append(CheckResult(table, "outlier", col, "outlier_pct", _pct(n_out, total),
                               None, "INFO",
                               f"{col}: {n_out} IQR-outlier(s) outside [{round(lo,2)}, {round(hi,2)}]"))
    return out


# --------------------------------------------------------------------------- #
# 7. Schema validation
# --------------------------------------------------------------------------- #
def validate_schema(df: pl.DataFrame, table: str, expected_cols: list[str]) -> list[CheckResult]:
    actual = set(df.columns)
    expected = set(expected_cols)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    out = [CheckResult(table, "schema", None, "missing_columns", float(len(missing)),
                       0, "FAIL" if missing else "PASS",
                       f"Missing expected columns: {missing}" if missing else "All expected columns present")]
    if extra:
        out.append(CheckResult(table, "schema", None, "extra_columns", float(len(extra)),
                               None, "WARN", f"Unexpected extra columns: {extra}"))
    return out
