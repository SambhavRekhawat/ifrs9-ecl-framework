"""
tests/test_eda.py
================
Smoke tests for the EDA chart builders (no database needed).
"""

import polars as pl
from datetime import date

from src.eda import plots


def _master():
    return pl.DataFrame(dict(
        fico_orig=[700.0, 720.0, 680.0, 750.0],
        ltv_orig=[80.0, 90.0, 75.0, 60.0],
        cltv_orig=[80.0, 95.0, 75.0, 60.0],
        dti_orig=[35.0, 40.0, 28.0, 22.0],
        upb_orig=[200000.0, 350000.0, 180000.0, 420000.0],
        int_rate_orig=[3.5, 4.2, 3.1, 5.0],
        state=["CA", "TX", "CA", "FL"],
        vintage=["2018Q1", "2018Q1", "2019Q1", "2019Q1"]))


def test_distribution_figures():
    figs = plots.distribution_figures(_master())
    assert len(figs) == 5
    assert all(len(f.data) >= 1 for _, f in figs)


def test_delinquency_fig():
    ts = pl.DataFrame(dict(reporting_period=[date(2019, 1, 1), date(2019, 2, 1)],
                           active=[1000, 1000], dpd30=[50, 60], dpd90=[10, 12]))
    fig = plots.delinquency_timeseries_fig(ts)
    assert len(fig.data) == 2  # 30+ and 90+ lines


def test_vintage_outcomes_fig():
    vo = pl.DataFrame(dict(vintage=["2018Q1", "2019Q1"], loans=[1000, 1200],
                           defaults=[20, 15], prepaid=[300, 400]))
    fig = plots.vintage_outcomes_fig(vo)
    assert len(fig.data) == 2  # default + prepay bars


def test_state_choropleth_fig():
    fig = plots.state_choropleth_fig(_master())
    assert len(fig.data) == 1
