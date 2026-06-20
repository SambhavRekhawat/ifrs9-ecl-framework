# Stage 2 — Data Quality

Reads `loan_master` + `loan_monthly` from PostgreSQL, runs automated quality
checks, scores them, and produces both an HTML report and a `dq_results` table.

## Run it

```bash
python -m src.quality_checks.run_quality
```

Outputs:
- `reports/data_quality_<timestamp>.html` — a scorecard report (open in a browser).
- `dq_results` table in PostgreSQL — every check result, with a run_id + timestamp,
  so you can track quality over time.

Flags: `--no-db` runs the checks and writes the HTML report without touching the database.

## What it checks

| Check | What it does | Status on issue |
|---|---|---|
| Schema validation | expected columns present, flags missing/extra | FAIL / WARN |
| Completeness | % null on *core* columns (others report-only) | FAIL |
| Uniqueness | duplicate loans / duplicate (loan, month) rows | FAIL |
| Non-negative | balances that are below zero | FAIL |
| Numeric ranges | FICO, LTV, DTI, rate, units outside plausible bounds | WARN |
| Date validity | unparseable or out-of-range dates | WARN |
| Outliers | IQR-based outlier counts (report-only) | INFO |
| Vintage coverage | **missing quarters in the vintage sequence** | WARN |
| Drift (PSI) | distribution shift of key features across vintages | WARN / FAIL |

## Scoring

Each table and the portfolio overall get a 0–100 score and a letter grade.
Start at 100, subtract 10 per FAIL and 2 per WARN (INFO is not scored).
A < 90 score simply means there are items worth reviewing — for real mortgage
data, some WARNs (range edges, mild drift) are completely normal.

## Configuring the rules

All thresholds live in `config/config.yaml` under `quality:` — core columns,
max null %, numeric ranges, non-negative columns, date bounds, outlier columns,
and PSI thresholds. Change them there; no code edits needed.

## Note on your missing 2018 quarters

The **vintage coverage** check is designed exactly for this: it lists the
missing quarters as a WARN (not a FAIL), documenting the gap without blocking
the pipeline. Missing origination cohorts are an acceptable, recorded condition
— the loans you do have keep their full performance history.
