# Stage 14 — Monitoring

## Purpose

Where Stage 13 asked "is the model sound *now*?", Stage 14 is the ongoing
surveillance that asks "is it *staying* sound?" — the period-over-period checks a
model-risk function runs each reporting cycle to catch degradation before it
misstates a provision. Every metric carries a RAG (red/amber/green) status, and
an overall traffic-light summarises the run.

## What it monitors

### 1. PD back-testing (`backtest_by_period`)

For each reporting period, the **mean predicted PD** (calibrated model) vs the
**realised 12-month default rate** (the forward label). Reuses the PD labelling
(`build_labeled_frame`) and `drop_incomplete_window` so censored recent months
are excluded. RAG is on the deviation `|pred / realised − 1|`:

- GREEN < `backtest_amber_dev` · AMBER < `backtest_red_dev` · RED ≥ that.

This is the core PD-model monitor: it catches calibration drifting over time.

### 2. PD distribution drift (`psi_by_period`)

Population Stability Index of the **predicted-PD distribution** between
consecutive periods (reusing `quality_checks.drift.psi`). This is the *true*
temporal drift check (distinct from the origination-vs-now migration metric in
Stage 13). RAG on the usual scale: < 0.10 green · < 0.25 amber · ≥ 0.25 red.

### 3. Delinquency trend (`delinquency_trend`)

Per period, the share of loans **30+ DPD** and **90+ DPD** across the whole book
(a cheap scan, no scoring). A rising 90+ DPD share is an early warning that
realised losses — and Stage 3 — are about to climb.

### Status: latest vs historical

- **Latest-period status** (the headline) — the RAG of the most recent period.
  Answers "is the model healthy *now*?" so a one-off historical shock does not
  leave the dashboard permanently red.
- **Historical worst** — the worst RAG ever seen, with a breach count, shown as
  context. On this book the historical worst is RED, driven entirely by the 2019
  refi-boom composition shift and the 2020 COVID forbearance surge — both
  correctly flagged regime shifts, not model failures.

## Outputs

- `models/monitoring_results.json` — overall status, per-period back-test, PSI
  transitions, delinquency trend, thresholds.
- `reports/monitoring_report_<ts>.html` — colour-coded RAG tables.

## What "good" looks like on this book

The 2018–2023 data is a benign, stable regime, so expect:

- **Back-test**: predicted ≈ realised within tolerance each period (GREEN), with
  the model running slightly conservative (pred ≥ realised) being acceptable.
- **PSI**: large period-over-period shifts in 2019 (refi boom) and 2020 (COVID)
  correctly flag RED; stable (GREEN) from late 2021 onward.
- **Delinquency**: the 2020 COVID spike (90+ DPD ~0.15% -> 2.22%) is clearly
  visible, recovering thereafter, with a mild upward creep into 2024-2025.

Observed signal on this book: recent back-test ratios drift toward
under-prediction (~0.70) as post-COVID defaults normalise upward while the PD
model carries the suppressed 2020-21 era in its calibration. Still GREEN, but a
PD-level recalibration on recent data (or a small overlay) is the prudent
follow-up the framework is designed to surface.

## Config (`config.yaml → monitoring`)

```yaml
monitoring:
  min_n_per_period: 500
  psi_bins: 10
  psi_amber: 0.10
  psi_red: 0.25
  backtest_amber_dev: 0.50
  backtest_red_dev: 1.00
```

## Tests (`tests/test_monitoring.py`)

RAG thresholds; back-test green-when-calibrated / red-when-off; min-N filter;
PSI detects a distribution shift; delinquency shares; overall-status priority.
Suite: 79 passing.

## Note on cost

`build_backtest_panel` re-labels and scores the panel (≈ the PD-training data
pass), so a monitoring run takes a few minutes; the delinquency panel is a cheap
scan. Intended to be run each reporting cycle.
