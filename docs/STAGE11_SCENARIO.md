# Stage 11 — Scenario Engine

## Purpose

IFRS 9 requires expected credit loss to be **probability-weighted across
forward-looking macroeconomic scenarios**. This stage builds base / upside /
downside scenarios, maps each to a systematic-factor (`Z`) path, and produces
the probability-weighted point-in-time (PIT) PD path that the ECL engine
(Stage 12) applies to the PD term structure.

## The Z convention (important)

Using the Stage-6 Vasicek / ASRF mapping `vasicek_pit(pd_ttc, z, rho)`:

> **higher Z = stronger economy = LOWER PD.**

So a **downside** scenario must produce a **lower** Z than baseline, and upside a
higher Z. All ordering checks in this stage are built around that.

## How scenarios are built

1. **Baseline macro anchor** — the latest observed values of
   `[unemployment, hpi_yoy, gdp_yoy, treasury_10y]` from `macro_data.parquet`,
   held flat over the horizon.
2. **Shock paths** — each scenario adds a peak deviation (config `shocks`) that
   ramps linearly from 0 to its peak over `ramp_months`, then holds (a
   persistent-stress profile, prudent for lifetime ECL).
3. **Macro → Z** — each scenario's macro path is mapped to a `Z` path by the
   Stage-6 `macro_to_z` model.
4. **Vasicek PIT** — `Z` is turned into a PIT PD path at the portfolio TTC PD
   for reporting; the ECL engine will apply the same `Z` to each loan's PD term
   structure.
5. **Probability weighting** — scenario PIT PDs are combined with the configured
   `weights` (default 50/25/25).

## Handling the unemployment-coefficient artifact (from Stage 6)

Stage 6 found the unemployment coefficient was near-zero / wrong-signed because
COVID forbearance broke the usual unemployment–default link (2020 had peak
unemployment but record-low mortgage defaults). Three deliberate safeguards keep
the scenarios sensible regardless:

1. **HPI-led downside.** The downside shock leans hardest on `hpi_yoy`
   (−12pp), whose coefficient has the correct sign, so it dominates the
   (mis-signed) unemployment term. In practice this alone produces a correctly
   ordered downside.
2. **Monotonic enforcement** (`enforce_monotonic: true`). After mapping, `Z`
   paths are clamped so `downside ≤ base ≤ upside` at every month. If the
   macro→Z mapping ever produces a non-monotone ordering, it is corrected and a
   warning is logged. (`enforce_monotonic_z` is a standalone, tested function.)
3. **Z-overlay mode** (`mode: z_overlay`). Bypasses macro→Z entirely and shifts
   `Z` directly by configured amounts — a transparent "management overlay" used
   when the regression mapping is not trusted.

The run reports `ordering_ok` (downside ≥ base ≥ upside on PIT PD) and
`monotonic_adjusted` so the behaviour is always visible and auditable.

## Outputs

- `models/scenario_artifacts.json` — per-scenario weights, `Z` paths, PIT PD
  paths, the probability-weighted PIT PD path, baseline anchor, mode, and the
  `ordering_ok` / `monotonic_adjusted` flags. **This is the input to Stage 12.**
- `reports/scenario_report_<ts>.html` — scenario table and baseline anchor.

## Config (`config.yaml → scenario`)

```yaml
scenario:
  horizon_months: 60
  ramp_months: 12
  mode: macro                 # macro | z_overlay
  enforce_monotonic: true
  weights: {base: 0.50, upside: 0.25, downside: 0.25}
  shocks:
    base:     {unemployment: 0.0,  hpi_yoy: 0.0,   gdp_yoy: 0.0,  treasury_10y: 0.0}
    upside:   {unemployment: -1.0, hpi_yoy: 3.0,   gdp_yoy: 1.0,  treasury_10y: 0.5}
    downside: {unemployment: 4.0,  hpi_yoy: -12.0, gdp_yoy: -3.0, treasury_10y: -1.0}
  z_overlay: {base: 0.0, upside: 0.5, downside: -1.0}
```

The shocks and weights are policy levers; the −12pp HPI downside is a moderate
recession analogue. Tighten/loosen and watch the weighted PIT PD.

## Tests (`tests/test_scenarios.py`)

Shock profile ramps then holds; macro path applies shocks to the baseline;
latest-macro picks the last non-null; Vasicek PIT is decreasing in `Z`; the
monotonic clamp fires when violated and is a no-op when already ordered. Suite:
60 passing.

## Feeds

`scenario_artifacts.json` (the `Z` paths and weights) drives the
scenario-conditioned, probability-weighted ECL in Stage 12.
