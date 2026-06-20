"""
src/pit_calibration/macro_data.py
=================================
Fetches the macro series from FRED (the FHFA HPI is hosted on FRED too),
builds a monthly table keyed by reporting_period, derives year-on-year
transforms, and writes both a Parquet copy and the `macro_data` table that
the feature pipeline + PIT model join onto.

Requires a free FRED API key in .env:  FRED_API_KEY=...
(get one at https://fredaccount.stlouisfed.org/apikeys)
"""

from __future__ import annotations

import os

import pandas as pd

from config.settings import settings
from src.ingestion import db
from src.utils.logger import get_logger

log = get_logger(__name__)


def fetch_macro() -> pd.DataFrame:
    from fredapi import Fred

    key = os.getenv("FRED_API_KEY")
    if not key or key == "your_fred_api_key_here":
        raise RuntimeError(
            "FRED_API_KEY not set in .env. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikeys and add it to .env.")
    cfg = settings.config["macro"]
    start = str(cfg["start_date"])
    series = cfg["fred_series"]

    fred = Fred(api_key=key)
    raw = {}
    for name, sid in series.items():
        s = fred.get_series(sid, observation_start=start)
        s.index = pd.to_datetime(s.index)
        raw[name] = s
        log.info("Fetched %s (%s): %d observations", name, sid, len(s))

    # Monthly grid; daily -> monthly mean, quarterly -> forward-filled.
    idx = pd.date_range(start=start, end=pd.Timestamp.today().normalize(), freq="MS")
    df = pd.DataFrame(index=idx)
    for name, s in raw.items():
        df[name] = s.resample("MS").mean().reindex(idx).ffill()
    df.index.name = "reporting_period"

    # Year-on-year transforms.
    if "hpi" in df:
        df["hpi_yoy"] = df["hpi"].pct_change(12) * 100
    if "real_gdp" in df:
        df["gdp_yoy"] = df["real_gdp"].pct_change(12) * 100
    if "cpi" in df:
        df["cpi_yoy"] = df["cpi"].pct_change(12) * 100

    out = df.reset_index()
    out["reporting_period"] = out["reporting_period"].dt.date  # store as DATE
    return out


def build_macro_table() -> pd.DataFrame:
    df = fetch_macro()
    out_dir = settings.project_root / "data" / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "macro_data.parquet", index=False)

    engine = db.get_engine()
    df.to_sql("macro_data", engine, if_exists="replace", index=False)
    log.info("Wrote macro_data: %d monthly rows, columns: %s", len(df), list(df.columns))
    return df


if __name__ == "__main__":
    build_macro_table()
