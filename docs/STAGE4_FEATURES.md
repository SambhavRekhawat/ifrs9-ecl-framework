# Stage 4 — Feature Engineering

Turns the raw loan-month panel into model-ready features and stores them in a
Parquet feature store (one file per vintage under `data/feature_store/`).

## Run it

```bash
python -m src.feature_engineering.run_features            # all vintages
python -m src.feature_engineering.run_features --reset    # clear store first
python -m src.feature_engineering.run_features --vintages 2018Q1 2018Q2   # subset (good for a first test)
```

Output: `data/feature_store/<vintage>.parquet`.

## Why a Parquet feature store (not a database table)

Writing ~21M feature rows back into PostgreSQL row-by-row is slow on a laptop.
Parquet is columnar, compresses well, writes in seconds, and Stage 5 reads it
back instantly with `pl.scan_parquet`. This is a standard feature-store pattern.
Processing one vintage at a time also keeps memory low.

## Features produced (Phase 4 checklist)

| Feature | Meaning |
|---|---|
| `loan_age` | months since origination (from source) |
| `months_on_book` | sequential month index per loan |
| `current_ltv` | balance-based current LTV proxy (`ltv_orig × upb_current/upb_orig`) |
| `upb_paydown_ratio` | share of original balance repaid |
| `cum_principal_paid` | cumulative principal paid (payment behaviour) |
| `rate_spread_vs_orig` | current rate − original rate |
| `delq_num` | delinquency level (months past due) |
| `balance_change_3m` | balance trend over 3 months |
| `delq_change_3m` | delinquency trend over 3 months |
| `max_delq_6m`, `max_delq_12m` | rolling max delinquency |
| `count_30dpd_12m` | rolling count of 30+ DPD months |
| `ever_30dpd_12m` | rolling flag: any 30+ DPD in last 12 months |
| `vintage_loan_count`, `vintage_avg_fico` | vintage metrics |
| macro columns | joined from `macro_data` **if present** (else skipped) |

Windows and definitions are configurable in `config.yaml -> features`.

## Two deliberate deferrals

1. **Macroeconomic features.** The pipeline left-joins a `macro_data` table
   (FRED unemployment/GDP/rates + FHFA HPI) keyed on `reporting_period` *if it
   exists*. We haven't ingested that yet — it's a focused step we'll do before
   Stage 6 (PIT calibration), where macro variables are first essential. Until
   then this safely no-ops, so the loan-level features build now.
2. **The PD target** (default within 12 months) is built in Stage 5, where it
   belongs, so the feature store stays a clean set of *inputs*.

## HPI-adjusted current LTV (note)

`current_ltv` here assumes the property value is unchanged (balance-based). The
mark-to-market version that inflates the original value by FHFA HPI will be
added with the macro step.
