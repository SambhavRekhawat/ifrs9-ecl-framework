# Stage 7 — Prepayment Model

Models the **competing risk** to default: a 12-month-forward probability that an
active, performing loan prepays (`zero_balance_code = 01`, Prepaid/Matured).
Prepayment determines how long a loan stays on the books, which drives the
lifetime/EAD profile in the ECL engine (Stage 12).

## Run it

```bash
python -m src.prepayment_model.run_prepay
```

Outputs: `models/pp_scorecard_logreg.joblib`, `pp_xgboost.joblib`,
`pp_lightgbm.joblib`, `pp_woe_maps.pkl`, `pp_metrics.json`, and
`reports/prepay_report_<ts>.html`.

## The target (competing risk)

Prepayment competes with default — a loan that defaults can no longer prepay.
Exit reasons aren't in the feature store, so `terminal_events.py` pulls each
loan's first prepay month, first credit-default month, and last observed month
from `loan_monthly`. Then for each active, performing loan-month:

| Outcome in next 12 months | Label |
|---|---|
| Prepays (and not after defaulting) | 1 |
| Survives the full 12 months, no prepay | 0 |
| Defaults first (competing exit) | censored (dropped) |
| Window cut off by end of data | censored (dropped) |

The risk set is the same active, performing population the PD model uses, so
default and prepayment are proper competing risks from one population.

## Drivers

Prepayment is dominated by the **rate incentive** to refinance. The feature
store's `rate_spread_vs_orig` is ~0 for fixed-rate loans, so we join the Stage-6
macro table to add the real signal:

- `mortgage_30y` — prevailing 30-yr fixed rate (Freddie Mac `MORTGAGE30US` via FRED)
- `refi_incentive` = `(int_rate_orig + rate_spread_vs_orig) − mortgage_30y` — the
  **dynamic** in-the-money amount: the loan's *current* rate vs today's market
  rate. Correctly goes negative when market rates rise above the loan rate.
- `incentive_burnout` = `refi_incentive × seasoning` — captures burnout (loans
  in-the-money for a long time without refinancing are less likely to)
- `treasury_10y` — broader rate environment (controlled by `prepay.use_macro: true`)

plus loan seasoning (`loan_age`, `months_on_book`), credit (`fico_orig`),
equity (`current_ltv`), and paydown. When rates fall, high-rate loans refinance
- so `refi_incentive` and `treasury_10y` are typically top IV features.

## Reuse

The WOE/IV, metrics (AUC/Gini/KS/calibration) and model training code are
imported directly from `src.pd_model` — the prepayment stage only adds its
competing-risk target, terminal-event extraction, and dataset assembly. Models
are saved with a `pp_` prefix to sit alongside the PD models.

## Notes

- Prepayment rates are high (often 20-40% within 12 months), so class imbalance
  is mild and `scale_pos_weight` is near 1.
- Like the PD model, the last 12 months of observations are right-censored and
  excluded from labels.
- A harmless LightGBM feature-name warning may appear.
