# IFRS 9 Expected Credit Loss (ECL) Framework

An end-to-end, production-style IFRS 9 ECL engine built on the **Fannie Mae
Single-Family Loan Performance** dataset (2006-2025, ~1.1 M loans, ~67 M
monthly records) with FRED / FHFA macroeconomic data. It models every
component of the expected-loss calculation - Probability of Default (PD),
Loss Given Default (LGD), Exposure at Default (EAD), staging, forward-looking
macro scenarios - and surfaces the results in an interactive dashboard.

**Live demo:** _<[add your Streamlit Cloud URL here once deployed](https://ifrs9-ecl-framework-f8adumh8vmi7spwoajwhqd.streamlit.app/)>_

> The live dashboard reads small pre-computed JSON result files committed to
> this repo. It does **not** require a database, the raw loan data, or any
> local compute - so it runs entirely in the cloud.

---

## What it does

**ECL = PD x LGD x EAD**, discounted and probability-weighted across macro
scenarios. Stage 1 loans use a 12-month horizon; Stages 2-3 use lifetime.

| Component | Approach |
|---|---|
| **PD** | WOE scorecard + gradient-boosted trees, isotonic-calibrated, anchored to a through-the-cycle level and flexed point-in-time by a macro->Z regression. Best model AUC ~ **0.91**, TTC PD ~ **1.3%**. |
| **LGD** | Empirical loss severity with a downturn LGD = max(empirical worst-window, model-stressed, regulatory benchmark). |
| **EAD** | Scheduled amortization with an observed curtailment factor. |
| **Staging** | IFRS 9 stage allocation (90+ DPD default, 30+ DPD backstop, PD-deterioration SICR). |
| **Scenarios** | Base / upside / downside macro paths driving a Vasicek single-factor PIT PD. |
| **Validation** | Reconciliation checks, discrimination (AUC/Gini/KS), calibration curve, ECL sensitivity grid. |
| **Monitoring** | PD back-test vs realised default rate, PSI drift, delinquency trends across a full credit cycle (GFC + COVID). |

## Architecture

```
PostgreSQL + Parquet feature store  --->  modelling pipeline (PD, LGD, EAD,
   (local, heavy, NOT deployed)             staging, scenarios, ECL,
                                            validation, monitoring)
                                                    | writes
                                                    v
                                       models/*.json  (small result artifacts)
                                                    | read by
                                                    v
                                    Streamlit dashboard  --->  deployed to web
```

The dashboard is a **thin presentation layer over the JSON artifacts**, which
is why it deploys with only `streamlit`, `pandas`, `pyyaml`, `python-dotenv`.

## Run the dashboard locally

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

## Re-run the full modelling pipeline (needs the data + database)

```bash
pip install -r requirements-dev.txt
python -m src.pd_model.run_pd
python -m src.pit_calibration.run_pit
python -m src.staging.run_staging
python -m src.scenario_engine.run_scenarios
python -m src.ecl_engine.run_ecl
python -m src.validation.run_validation
python -m src.monitoring.run_monitoring
```

Each stage writes its result JSON into `models/`; commit those files and the
live dashboard updates on the next `git push`.

## Key assumptions & limitations

Stated openly, because every credit-risk model has them:

- **Agency loans only.** Fannie Mae conforming loans exclude subprime and
  private-label exposure, so absolute default levels are lower than a
  whole-market book.
- **National macro factors.** HPI and unemployment are national, not regional,
  which weakens the mark-to-market LTV signal in LGD (mortgage insurance and
  loss caps further flatten observed severity).
- **Downturn LGD is dated by resolution, not origination**, so the worst
  empirical window can land in the recovery era and understate true
  peak-crisis severity.
- **Era blend.** Pre-2008 (looser, pre-QM underwriting) and modern vintages are
  pooled for a first-pass PD; an era control is a planned refinement.
- **Point-in-time PD under-predicts crisis acceleration.** Loan-level features
  can't foresee a macro collapse; the forward-looking downside scenario is what
  supplies that stress in the ECL.

## Tech stack

Python - PostgreSQL - Polars - scikit-learn / XGBoost / LightGBM - Streamlit -
FRED & FHFA macro data.

---

_Built by Sambhav Rekhawat. The modelling pipeline runs locally; only the
result artifacts and code are deployed._
