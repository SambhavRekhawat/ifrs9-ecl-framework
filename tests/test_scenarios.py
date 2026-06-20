"""Tests for the scenario engine: macro paths, Vasicek ordering, monotonic clamp."""
import numpy as np
import polars as pl
from datetime import date

from src.scenario_engine import macro_paths as MP, scenarios as S
from src.pit_calibration import vasicek


def test_shock_profile_ramps_then_holds():
    p = MP.shock_profile(horizon=24, ramp_months=12)
    assert p[0] < p[5] < p[11]            # ramping up
    assert abs(p[11] - 1.0) < 1e-9        # reaches peak at ramp_months
    assert np.allclose(p[12:], 1.0)       # held after


def test_build_macro_path_applies_shock_to_baseline():
    base = {"unemployment": 4.0, "hpi_yoy": 5.0}
    shock = {"unemployment": 4.0, "hpi_yoy": -12.0}
    path = MP.build_macro_path(base, shock, ["unemployment", "hpi_yoy"], horizon=24, ramp_months=12)
    assert abs(path[0, 0] - (4.0 + 4.0 / 12)) < 1e-6      # first month, partial ramp
    assert abs(path[-1, 0] - 8.0) < 1e-6                  # unemployment peaks at +4
    assert abs(path[-1, 1] - (-7.0)) < 1e-6               # hpi_yoy 5 - 12 = -7


def test_latest_macro_picks_last_nonnull():
    m = pl.DataFrame({"reporting_period": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
                      "unemployment": [4.5, 4.2, None]})
    assert MP.latest_macro(m, ["unemployment"])["unemployment"] == 4.2


def test_vasicek_pit_decreasing_in_z():
    # higher Z (stronger economy) -> lower PD
    pd_down = float(vasicek.vasicek_pit(0.0105, -1.0, 0.15))
    pd_base = float(vasicek.vasicek_pit(0.0105, 0.0, 0.15))
    pd_up = float(vasicek.vasicek_pit(0.0105, 1.0, 0.15))
    assert pd_down > pd_base > pd_up


def test_enforce_monotonic_clamps_when_violated():
    z = {"base": np.array([0.5, 0.5]),
         "downside": np.array([0.9, 0.9]),   # wrongly higher than base
         "upside": np.array([0.1, 0.1])}      # wrongly lower than base
    out, adjusted = S.enforce_monotonic_z(z)
    assert adjusted
    assert np.all(out["downside"] <= out["base"]) and np.all(out["upside"] >= out["base"])


def test_enforce_monotonic_noop_when_already_ordered():
    z = {"base": np.array([0.5]), "downside": np.array([-0.5]), "upside": np.array([1.5])}
    out, adjusted = S.enforce_monotonic_z(z)
    assert not adjusted
