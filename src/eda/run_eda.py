"""
src/eda/run_eda.py
=================
Phase-3 EDA. Run from the project root:

    python -m src.eda.run_eda

Reads from PostgreSQL, builds publication-quality Plotly charts, and writes a
single self-contained HTML report to reports/eda_<timestamp>.html.
"""

from __future__ import annotations

from datetime import datetime

import plotly.io as pio

from config.settings import settings
from src.eda import plots, queries
from src.ingestion import db
from src.utils.logger import get_logger

log = get_logger("eda.run")


def _section(title: str, fig, first: bool) -> str:
    inner = pio.to_html(fig, full_html=False,
                        include_plotlyjs=("cdn" if first else False))
    return f'<section><h2>{title}</h2>{inner}</section>'


def run() -> str:
    engine = db.get_engine()

    log.info("Loading loan_master for distributions...")
    master = queries.master_sample(engine)
    log.info("loan_master rows: %s", f"{master.height:,}")

    sections: list[tuple[str, object]] = []

    for label, fig in plots.distribution_figures(master):
        sections.append((label, fig))

    log.info("Aggregating delinquency time series...")
    sections.append(("Delinquency over time", plots.delinquency_timeseries_fig(queries.delinquency_timeseries(engine))))

    log.info("Aggregating vintage outcomes...")
    sections.append(("Default & prepayment by vintage", plots.vintage_outcomes_fig(queries.vintage_outcomes(engine))))

    log.info("Aggregating seasoning curve...")
    sections.append(("Seasoning curve", plots.seasoning_fig(queries.seasoning_curve(engine))))

    log.info("Aggregating cohort matrix...")
    sections.append(("Cohort analysis", plots.cohort_heatmap_fig(queries.cohort_matrix(engine))))

    sections.append(("State-level performance", plots.state_choropleth_fig(master)))

    # Assemble HTML
    body = "".join(_section(t, f, i == 0) for i, (t, f) in enumerate(sections))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>IFRS 9 — EDA Report {run_id}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:28px;color:#1f2328;background:#fff}}
 h1{{margin-bottom:2px}} .sub{{color:#57606a;margin-bottom:20px}}
 section{{margin:26px 0;padding:14px;border:1px solid #eaeef2;border-radius:10px}}
 h2{{font-size:18px;margin:4px 0 8px}}
</style></head><body>
<h1>IFRS 9 — Exploratory Data Analysis</h1>
<div class="sub">Run {run_id} · portfolio overview</div>
{body}
</body></html>"""

    out = settings.project_root / "reports" / f"eda_{run_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("Wrote EDA report to %s", out)
    return str(out)


if __name__ == "__main__":
    run()
