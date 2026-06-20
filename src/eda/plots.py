"""
src/eda/plots.py
===============
Builds Plotly figures from the small summary frames produced by queries.py.
Each function returns a plotly Figure so they can be embedded in a report
or shown in a notebook.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

_TEMPLATE = "plotly_white"
_DIST_COLS = {
    "fico_orig": "FICO at origination",
    "ltv_orig": "Original LTV (%)",
    "dti_orig": "DTI (%)",
    "upb_orig": "Original UPB ($)",
    "int_rate_orig": "Original interest rate (%)",
}


def distribution_figures(master: pl.DataFrame) -> list[tuple[str, go.Figure]]:
    """One histogram per key numeric origination feature."""
    figs = []
    for col, label in _DIST_COLS.items():
        if col not in master.columns:
            continue
        vals = master[col].drop_nulls().to_list()
        if not vals:
            continue
        fig = px.histogram(x=vals, nbins=50, template=_TEMPLATE,
                           labels={"x": label}, title=f"Distribution of {label}")
        fig.update_layout(bargap=0.02, showlegend=False, height=360)
        figs.append((label, fig))
    return figs


def delinquency_timeseries_fig(df: pl.DataFrame) -> go.Figure:
    d = df.with_columns([
        (100 * pl.col("dpd30") / pl.col("active")).alias("dpd30_rate"),
        (100 * pl.col("dpd90") / pl.col("active")).alias("dpd90_rate"),
    ])
    fig = go.Figure()
    fig.add_scatter(x=d["reporting_period"], y=d["dpd30_rate"], name="30+ DPD %", mode="lines")
    fig.add_scatter(x=d["reporting_period"], y=d["dpd90_rate"], name="90+ DPD %", mode="lines")
    fig.update_layout(title="Delinquency rate over time", height=380, template=_TEMPLATE,
                      yaxis_title="% of active loans", xaxis_title="Reporting period")
    return fig


def vintage_outcomes_fig(df: pl.DataFrame) -> go.Figure:
    d = df.with_columns([
        (100 * pl.col("defaults") / pl.col("loans")).alias("default_rate"),
        (100 * pl.col("prepaid") / pl.col("loans")).alias("prepay_rate"),
    ])
    fig = go.Figure()
    fig.add_bar(x=d["vintage"], y=d["default_rate"], name="Default rate %")
    fig.add_bar(x=d["vintage"], y=d["prepay_rate"], name="Prepayment rate %")
    fig.update_layout(title="Default & prepayment rate by vintage", barmode="group",
                      height=400, template=_TEMPLATE, yaxis_title="% of loans", xaxis_title="Vintage")
    return fig


def seasoning_fig(df: pl.DataFrame) -> go.Figure:
    d = df.with_columns((100 * pl.col("dpd90") / pl.col("n")).alias("rate"))
    fig = go.Figure()
    fig.add_scatter(x=d["loan_age"], y=d["rate"], mode="lines+markers", name="90+ DPD %")
    fig.update_layout(title="Seasoning curve: serious delinquency by loan age",
                      height=380, template=_TEMPLATE, xaxis_title="Loan age (months)", yaxis_title="90+ DPD %")
    return fig


def cohort_heatmap_fig(df: pl.DataFrame) -> go.Figure:
    d = df.with_columns((100 * pl.col("dpd90") / pl.col("n")).alias("rate"))
    pivot = d.pivot(values="rate", index="vintage", on="age", aggregate_function="first").sort("vintage")
    ages = [c for c in pivot.columns if c != "vintage"]
    z = pivot.select(ages).to_numpy()
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[int(a) for a in ages], y=pivot["vintage"].to_list(),
        colorscale="OrRd", colorbar_title="90+ DPD %"))
    fig.update_layout(title="Cohort heatmap: 90+ DPD by vintage and loan age",
                      height=480, xaxis_title="Loan age (months)", yaxis_title="Vintage",
                      template=_TEMPLATE)
    return fig


def state_choropleth_fig(master: pl.DataFrame) -> go.Figure:
    by_state = (master.group_by("state").agg(pl.len().alias("loans"))
                .filter(pl.col("state").is_not_null()))
    fig = go.Figure(data=go.Choropleth(
        locations=by_state["state"].to_list(), z=by_state["loans"].to_list(),
        locationmode="USA-states", colorscale="Blues", colorbar_title="Loans"))
    fig.update_layout(title="Loan count by state", geo_scope="usa", height=420,
                      template=_TEMPLATE)
    return fig
