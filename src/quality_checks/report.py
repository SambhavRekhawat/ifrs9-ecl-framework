"""
src/quality_checks/report.py
============================
Writes quality results two ways:
  1. An HTML scorecard report (for humans / your dashboard).
  2. A dq_results table in PostgreSQL (for tracking quality over time).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.quality_checks.checks import CheckResult
from src.utils.logger import get_logger

log = get_logger(__name__)

_STATUS_COLOR = {"PASS": "#1a7f37", "WARN": "#b35900", "FAIL": "#cf222e", "INFO": "#57606a"}


def results_to_df(results: list[CheckResult], run_id: str) -> pl.DataFrame:
    rows = [{**r.as_dict(), "run_id": run_id} for r in results]
    return pl.DataFrame(rows)


def write_db(df: pl.DataFrame, engine: Engine) -> None:
    """Create dq_results if needed and append this run's rows."""
    ddl = (
        "CREATE TABLE IF NOT EXISTS dq_results ("
        " run_id TEXT, run_ts TIMESTAMP DEFAULT now(), table_name TEXT, check_type TEXT,"
        " column_name TEXT, metric TEXT, value DOUBLE PRECISION, threshold DOUBLE PRECISION,"
        " status TEXT, message TEXT)"
    )
    with engine.begin() as conn:
        conn.execute(text(ddl))
    out = df.rename({"table": "table_name", "column": "column_name"})
    out.write_database("dq_results", connection=engine,
                       if_table_exists="append", engine="sqlalchemy")
    log.info("Wrote %d rows to dq_results.", out.height)


def write_html(results: list[CheckResult], scorecard: dict, path: Path, run_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ov = scorecard["overall"]

    def badge(status: str) -> str:
        return (f'<span style="color:#fff;background:{_STATUS_COLOR.get(status,"#888")};'
                f'padding:2px 8px;border-radius:10px;font-size:12px">{status}</span>')

    rows_html = "".join(
        f"<tr><td>{r.table}</td><td>{r.check_type}</td><td>{r.column or ''}</td>"
        f"<td>{r.metric}</td><td style='text-align:right'>{'' if r.value is None else r.value}</td>"
        f"<td>{badge(r.status)}</td><td>{r.message}</td></tr>"
        for r in sorted(results, key=lambda x: {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}[x.status])
    )

    table_cards = "".join(
        f'<div class="card"><h3>{t}</h3><div class="grade">{s["grade"]}</div>'
        f'<div class="score">{s["score"]}/100</div>'
        f'<div class="counts">PASS {s["PASS"]} · WARN {s["WARN"]} · FAIL {s["FAIL"]}</div></div>'
        for t, s in scorecard["by_table"].items()
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Data Quality Report - {run_id}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#1f2328;background:#f6f8fa}}
 h1{{margin-bottom:4px}} .sub{{color:#57606a;margin-bottom:24px}}
 .cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px}}
 .card{{background:#fff;border:1px solid #d0d7de;border-radius:12px;padding:18px 24px;min-width:150px}}
 .grade{{font-size:42px;font-weight:700}} .score{{font-size:15px;color:#57606a}}
 .counts{{font-size:12px;color:#57606a;margin-top:6px}}
 table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #d0d7de;border-radius:8px;overflow:hidden}}
 th,td{{padding:8px 12px;border-bottom:1px solid #eaeef2;font-size:13px;text-align:left}}
 th{{background:#f6f8fa}}
 .overall{{font-size:54px;font-weight:800}}
</style></head><body>
<h1>IFRS 9 — Data Quality Report</h1>
<div class="sub">Run {run_id} · {datetime.now():%Y-%m-%d %H:%M}</div>
<div class="cards">
 <div class="card"><h3>Overall</h3><div class="overall">{ov['grade']}</div>
   <div class="score">{ov['score']}/100 · {ov['total_checks']} checks</div>
   <div class="counts">PASS {ov['PASS']} · WARN {ov['WARN']} · FAIL {ov['FAIL']}</div></div>
 {table_cards}
</div>
<h2>All checks</h2>
<table><thead><tr><th>Table</th><th>Check</th><th>Column</th><th>Metric</th>
<th>Value</th><th>Status</th><th>Detail</th></tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("Wrote HTML report to %s", path)
    return path