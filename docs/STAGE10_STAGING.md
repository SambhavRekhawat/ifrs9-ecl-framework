# Stage 10 — IFRS 9 Staging Engine

## What this stage does

Assigns every loan to an IFRS 9 **stage** at a reporting date. The stage decides
the ECL horizon the engine (Stage 12) will use:

| Stage | Meaning | ECL horizon |
|-------|---------|-------------|
| 1 | Performing, no significant increase in credit risk since origination | 12-month ECL |
| 2 | Significant increase in credit risk (SICR), not yet credit-impaired | Lifetime ECL |
| 3 | Credit-impaired / in default | Lifetime ECL |

This is a **rules engine**, not a statistical model — but it *consumes* the PD
model from Stage 5 to measure credit deterioration.

## The rules (`sicr.assign_stage`)

Evaluated in precedence order:

1. **Stage 3 (default)** — `delq_num >= default_dpd` (90+ DPD; `delq_num` counts
   30-day buckets, so 3 = 90 days). A credit-event flag can also force Stage 3.
2. **Stage 2 (SICR)** — any of:
   - **Quantitative / PD deterioration**: `PD_now / PD_orig >= rel_threshold`
     **or** `PD_now - PD_orig >= abs_threshold`. PD is scored with the trained
     PD model at the loan's origination and at the reporting date.
   - **Backstop**: `delq_num >= backstop_dpd` (30+ DPD — the IFRS 9 rebuttable
     presumption).
   - **Qualitative** (optional): a watchlist / forbearance flag column.
3. **Stage 1** — none of the above.

Each output row carries transparent flags (`is_default`,
`sicr_backstop_30dpd`, `sicr_pd_deterioration`) so the trigger is auditable.

## How PD deterioration is measured (`staging_data.py`)

- **Origination PD** — each loan's earliest row (min `loan_age`) scored by the
  PD model.
- **Reporting PD** — the loan's row at the reporting period (or its latest row)
  scored by the same model.

Both use the Stage 5 best model (LightGBM by default, `pd_model_file`) and, if
present, the Stage 6 isotonic calibrator. The relative-PD test is robust to
calibration (a monotonic calibrator preserves the ordering), so the comparison
is stable.

**Methodology note.** Full IFRS 9 quantitative SICR compares *residual lifetime*
PD at the reporting date to the residual-lifetime PD anticipated at origination.
Until the PD term structure exists (Stage 12), we use the widely-accepted
practical expedient of comparing **12-month PD at origination vs now** (relative
and absolute). When Stage 12 builds the term structure, this trigger can be
upgraded to the lifetime comparison without changing the staging interface.

## Validation

- **Stage distribution** — the bulk should sit in Stage 1; Stage 3 should track
  the 90+ DPD population.
- **Trigger breakdown** — counts of loans caught by default / backstop / PD
  deterioration.
- **Migration matrix** — set `migration_from` / `migration_to` to two reporting
  dates to get a stage-transition count matrix (sanity-checks that loans move
  between stages sensibly and that cures/transfers behave).

## Outputs

- `models/staging_artifacts.json` — distribution, trigger counts, thresholds,
  and (optional) migration matrix.
- `reports/staging_report_<ts>.html` — the same, rendered.

## Config (`config.yaml → staging`)

```yaml
staging:
  default_dpd: 3
  backstop_dpd: 1
  sicr_pd_rel_threshold: 2.0
  sicr_pd_abs_threshold: 0.01
  reporting_period: null        # null = latest row per loan (current book)
  pd_model_file: pd_lightgbm.joblib
  migration_from: null
  migration_to: null
```

The thresholds are policy levers. `rel=2.0` (PD doubling) with `abs=0.01` is a
common starting calibration; tighten/loosen them and watch the Stage 2 share.

## Tests (`tests/test_staging.py`)

Performing → Stage 1; 30+ DPD backstop → Stage 2; relative and absolute PD
deterioration → Stage 2; 90+ DPD → Stage 3 with precedence; distribution sums to
N; migration matrix counts common loans. Suite: 54 passing.

## Feeds

The staged snapshot (loan → stage) is the switch the ECL engine (Stage 12) uses
to apply 12-month vs lifetime ECL per loan.
