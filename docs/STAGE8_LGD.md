# Stage 8 — LGD (Loss Given Default)

Estimates the fraction of exposure lost when a loan defaults, from Fannie's
realised loss/proceeds fields, with a house-price (HPI) link so LGD responds to
scenarios.

## Run it

```bash
python -m src.lgd_model.run_lgd
```

Outputs: `models/lgd_model.joblib`, `models/lgd_stats.json`,
`reports/lgd_report_<ts>.html`.

## Observed loss severity

For each realised default (credit-event `zero_balance_code` in 02/03/09/15):

```
net_loss = default_UPB + costs - proceeds
LGD      = net_loss / default_UPB                      (clipped to [0, 1])
```

- **costs** = foreclosure + property-preservation/repair + asset-recovery +
  misc holding + property taxes
- **proceeds** = net sales + credit-enhancement + other-foreclosure +
  repurchase-make-whole
- **default_UPB** = UPB at removal (fallbacks: last current UPB, written-off
  principal)
- Fannie's own `cumulative_credit_event_net_gain_or_loss` is used as a fallback
  when the component fields are empty.

## Mark-to-market LTV (the economic driver)

`current_ltv` in the feature store is amortisation-only. Loss severity is really
driven by **equity**, so we compute an HPI-adjusted LTV:

```
MTM_LTV = ltv_orig * (default_UPB / upb_orig) * HPI_orig / HPI_default
```

Lower house prices -> higher MTM-LTV -> less recovery -> higher LGD. This is the
hook that makes LGD scenario-sensitive in Stages 11-12.

## Model

Given the **thin realised-default sample** (2018-2023 was benign + COVID
forbearance suppressed liquidations), the model is deliberately parsimonious:

1. **Empirical** severity stats + **LGD by MTM-LTV bucket** (robust, the primary
   reference).
2. A **direct mean regression** `E[LGD] ~ mtm_ltv (+ loan age)` (OLS on the LGD
   level, not logit). LGD here is *zero-inflated* (many full recoveries), which
   makes a logit fit collapse to the floor; OLS on the level targets the mean,
   which is what expected loss needs. An `lgd_floor` (5%) bounds predictions.
3. **Downturn LGD = max(model-stressed LGD, benchmark)**. An HPI shock
   (`downturn_hpi_shock`, default -20%) raises MTM-LTV; but because benign data
   shows little stress, the downturn is anchored to a prudent benchmark
   (`downturn_lgd_benchmark`, default 0.35) so it never understates stress loss.

## Important caveat — thin data

Realised liquidations are few (hundreds, not thousands) and occurred in a benign
price environment, so observed LGDs are low and the model can't "see" severe
downturns directly. The HPI-shock downturn LGD and the floor are how we inject
prudence. On real 2018-2023 data the LGD-vs-LTV signal is essentially flat (home-price
appreciation meant almost everyone recovered), so the regression r2 is ~0 and
the **downturn LGD is driven by the benchmark anchor, not the data** - the
correct, prudent outcome. Revisit (raise/calibrate the benchmark, refit) if the
realised-default sample grows. Tracked.
