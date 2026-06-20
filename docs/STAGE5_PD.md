# Stage 5 — PD Model

Builds 12-month Probability of Default models from the feature store, in two
tracks (a traditional WOE scorecard and machine-learning models), validated
out-of-time.

## Run it

```bash
python -m src.pd_model.run_pd
```

Outputs:
- `models/pd_scorecard_logreg.joblib`, `models/pd_xgboost.joblib`, `models/pd_lightgbm.joblib`
- `models/woe_maps.pkl`, `models/pd_metrics.json`
- `reports/pd_report_<timestamp>.html` (scorecard, IV table, calibration)

## The target (default within 12 months)

For each **performing** loan-month (current delinquency < 90 DPD), the label is
1 if the loan reaches **90+ DPD (`delq_num ≥ 3`) within the next 12 months**,
else 0. Rows whose 12-month forward window runs past the end of the data with
no event are **dropped** (right-censored) so we never mislabel "no data" as
"no default". This is verified by unit tests.

> Note: foreclosure/charge-off credit events are almost always preceded by 90+
> DPD, so the delinquency-based definition captures them. A stricter definition
> folding in `zero_balance_code` ∈ {02,03,09,15} can be added if desired (it
> would require carrying that column into the feature store).

## Out-of-time validation

Train on observations **before** `pd.oot_split_date` (default 2022-01-01), test
on observations **after**. This mirrors real deployment and is how regulators
expect PD models to be validated — far more honest than a random split.

## Two tracks

1. **Scorecard** — WOE-binned features + Information Value ranking + Logistic
   Regression (`class_weight="balanced"`). WOE handles nulls (each gets its own
   bin), so there are no NaN issues. Scorecard scaling factors (PDO/offset) are
   produced for a points-based card.
2. **Machine learning** — XGBoost and LightGBM on raw features (they handle NaN
   natively), with `scale_pos_weight` for the rare-default imbalance.

## Metrics (out-of-time test)

AUC, Gini (`2·AUC−1`), KS, precision, recall, Brier score, plus a 10-bin
**calibration table** (predicted PD vs observed default) for the best model.

## Laptop-friendliness

Defaults are rare, so training rows are stratified-sampled to
`pd.max_train_rows` (default 500k) — **keeping every default row** and sampling
non-defaults. Tune in `config.yaml -> pd`.

## What to expect on real data

Behavioural delinquency features are highly predictive, so real AUCs are
typically higher than a noisy synthetic test. If the scorecard and ML AUCs are
close, the simpler scorecard is often preferred for explainability; if ML is
materially better, it captures non-linear interactions.
