# Stage 3 — Exploratory Data Analysis (EDA)

Reads `loan_master` + `loan_monthly` from PostgreSQL and produces a single,
self-contained HTML report of publication-quality Plotly charts.

## Run it

```bash
python -m src.eda.run_eda
```

Output: `reports/eda_<timestamp>.html` — open it in any browser.

## What it shows

| Chart | Source | What it tells you |
|---|---|---|
| FICO / LTV / DTI / UPB / rate distributions | loan_master | credit quality & loan mix at origination |
| Delinquency over time (30+ & 90+ DPD) | loan_monthly | portfolio stress through calendar time |
| Default & prepayment by vintage | loan_monthly | which origination cohorts performed best |
| Seasoning curve (90+ DPD by loan age) | loan_monthly | when in a loan's life defaults emerge |
| Cohort heatmap (vintage × loan age) | loan_monthly | vintage quality vs. seasoning at a glance |
| State choropleth | loan_master | geographic concentration |

## Definitions (verified against Fannie Mae docs, configurable in `config.yaml -> eda`)

- **Default** = `zero_balance_code` in {02, 03, 09, 15} — completed credit-event
  dispositions (Third-Party Sale, Short Sale, REO/Deed-in-Lieu, Note Sale).
- **Prepayment** = `zero_balance_code` = 01 (Prepaid or Matured).
- **30+ DPD** = `Current Loan Delinquency Status` >= 1 month.
- **90+ DPD (serious)** = `Current Loan Delinquency Status` >= 3 months.

## Performance note

The monthly aggregations (delinquency, vintage, seasoning, cohort) run as SQL
`GROUP BY` queries inside PostgreSQL, so only small summary tables come back to
Python — this keeps memory low even on your ~21M-row monthly table. The full
run may take up to a minute depending on your machine.
