# Stage 1 — Data Ingestion

Turns raw Fannie Mae quarterly files into a clean PostgreSQL warehouse.

```
data/raw/*.csv  ->  Parquet (sampled)  ->  loan_master + loan_monthly  (PostgreSQL)
                                            + ingestion_log + metadata_catalog
```

## Key facts about the data (verified)

- Since Oct 2020 the data is a **single file per acquisition quarter**, with
  **108 fields** (110 in the 2023+ release). It already contains BOTH the
  origination/"acquisition" data and the monthly performance rows — there is no
  separate acquisition file anymore.
- Files are **pipe (`|`) delimited** with **no column headers**.
- Access requires a **free Data Dynamics account** (registration required).
- `Loan Identifier` + `Monthly Reporting Period` together are the unique key.

## Step 1 — Test the pipeline with FAKE data first (no download needed)

```bash
python tests/make_synthetic_fannie.py          # writes data/raw/2018Q1.csv, 2019Q2.csv
python -m src.ingestion.run_ingestion --parquet-only --sample 1.0
```

You should see Parquet files appear in `data/parquet/loan_level/`. If that works,
your code is healthy and the only remaining variable is the real data + database.

## Step 2 — Download the real data

1. Register / log in at Data Dynamics:
   https://capitalmarkets.fanniemae.com/tools-applications/data-dynamics
2. Open the **Single-Family Loan Performance** historical dataset.
3. Download the quarterly files you want (start with just **2018Q1** to test,
   then add 2018Q2 ... 2023Q4).
4. Put them in `data/raw/`. Make sure each filename contains its vintage, e.g.
   `2018Q1.csv`. Rename if needed.

> The files are large. Do ONE quarter first, end to end, before downloading all 24.

## Step 3 — Create the database (one time)

In `psql` or pgAdmin:

```sql
CREATE DATABASE ifrs9;
```

Then copy `.env.example` to `.env` and set `DB_USER` / `DB_PASSWORD`.

## Step 4 — Run the full ingestion

```bash
# keep ~3% of loans across all vintages (good for a laptop)
python -m src.ingestion.run_ingestion --reset --sample 0.03
```

This will: create the 4 tables, write `sql/schema.sql`, convert each file to
Parquet, split into master/monthly, load PostgreSQL, and record an audit row in
`ingestion_log`.

## Verify it worked

```sql
SELECT vintage, rows_master, rows_monthly, status FROM ingestion_log;
SELECT count(*) FROM loan_master;
SELECT count(*) FROM loan_monthly;
SELECT * FROM metadata_catalog LIMIT 10;
```

## Tuning the sample size

Edit `config/config.yaml` -> `data.sample_fraction`, or pass `--sample 0.05`.
`1.0` keeps everything (only do this for a single small quarter on a laptop).
