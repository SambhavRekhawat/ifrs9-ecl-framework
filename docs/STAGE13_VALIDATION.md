# Stage 13 — Validation Framework

## Purpose

Put the whole ECL under scrutiny: confirm the models discriminate and calibrate,
the artifacts reconcile, the scores are stable, and — most usefully — quantify
how the headline ECL moves with each assumption.

## What it checks

### 1. Reconciliation pass/fail (`checks.py`)

Pure checks over the saved artifacts:

| Check | Passes when |
|-------|-------------|
| `coverage_monotonic_by_stage` | Stage 1 ≤ Stage 2 ≤ Stage 3 coverage |
| `stage3_equals_weighted_lgd` | Stage 3 coverage ≈ probability-weighted LGD (within tol) |
| `portfolio_coverage_in_range` | total coverage within a plausible band |
| `scenario_pd_ordering` | downside ≥ base ≥ upside PIT PD (`ordering_ok`) |
| `pd_discrimination_auc` | best-model AUC ≥ `min_auc` |
| `effective_pd_calibration` | **calibrated** portfolio PD (what feeds ECL) ≈ TTC PD — the calibration that matters |
| `pd_calibration_band_raw` | raw predicted/observed PD ratio (warn/advisory only; the class-weighted model is intentionally hot and recalibrated by isotonic downstream) |

### 2. Discrimination & calibration

Read from `pd_metrics.json` (computed in Stage 5): best-model AUC / Gini / KS and
the predicted-vs-observed calibration table — no recompute.

### 3. PD migration since origination (PSI) — informational

Population Stability Index of each loan's PD **now** vs **at origination**. This
is a *migration* metric, not a data-drift alarm: PDs are expected to move
substantially over multi-year seasoning (this is the same signal that drives
staging), so the usual <0.10/0.25 drift thresholds do **not** apply here. True
same-population temporal drift monitoring is Stage 14.

Headline pass/fail counts **error-level** checks only; advisory (warn) checks are
reported separately.

### 4. ECL sensitivity / attribution (`ecl_sensitivity.py`)

The high-value piece. The loan inputs are assembled once, then the fast ECL core
is re-run across a grid of assumptions, reporting total ECL, coverage, and the
delta vs the base case for each:

- **`annual_cpr`** — prepayment speed (higher CPR shortens life → lower ECL).
- **`horizon_months`** — lifetime cap (longer → higher ECL).
- **`downturn_lgd`** — downside LGD severity (higher → higher ECL).
- **`scenario_weights`** — base / downside-heavy / severe (more downside → higher ECL).

This turns "Stage 2 is 6.9% — is that right?" into a table showing exactly how
much each lever drives the number, so the assumptions can be challenged and
calibrated rather than taken on faith.

When changing scenario weights, the PD multipliers are re-anchored to the new
probability-weighted path, so the comparison is internally consistent.

## Outputs

- `models/validation_results.json` — checks, discrimination, PSI, sensitivity grid.
- `reports/validation_report_<ts>.html` — the same, rendered.

## Config (`config.yaml → validation`)

```yaml
validation:
  min_auc: 0.70
  coverage_range_pct: [0.05, 2.0]
  stage3_lgd_tolerance_pct: 0.5
  calibration_ratio_band: [0.3, 3.0]
  sensitivity:
    cpr_values: [0.0, 0.05, 0.10, 0.20]
    horizon_values: [36, 48, 60]
    downturn_lgd_values: [0.30, 0.35, 0.45]
    weight_variants:
      base:           {base: 0.50, upside: 0.25, downside: 0.25}
      downside_heavy: {base: 0.40, upside: 0.20, downside: 0.40}
      severe:         {base: 0.30, upside: 0.10, downside: 0.60}
```

## Tests (`tests/test_validation.py`)

Monotonicity check passes/fails correctly; Stage 3 ≈ weighted LGD; coverage band;
discrimination threshold; `run_all` returns the full suite; sensitivity moves in
the right direction for CPR, downturn LGD, and scenario weights. Suite: 72 passing.

## Note on dependence

`run_validation` depends on the saved artifacts from Stages 5/11/12
(`pd_metrics.json`, `scenario_artifacts.json`, `ecl_results.json`, `lgd_stats.json`)
and re-assembles the loan inputs once (PD scoring + EAD pull) for the PSI and
sensitivity steps. Run the upstream stages first.
