"""
tests/make_synthetic_fannie.py
==============================
Creates a tiny FAKE Fannie Mae quarterly file (pipe-delimited, no header,
108 columns) so you can test the whole ingestion pipeline WITHOUT downloading
real data first.

Run:  python tests/make_synthetic_fannie.py
It writes data/raw/2018Q1.csv and data/raw/2019Q2.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Make the project root importable even when run as `python tests/this_file.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from src.ingestion import schema

random.seed(42)
N_COLS = len(schema.COLUMNS)  # 108

# Column index lookups so we fill the RIGHT positions.
IDX = {name: i for i, name in enumerate(schema.COLUMNS)}


def make_row(loan_id: str, period: str, age: int, vintage_year: int) -> str:
    cells = [""] * N_COLS
    cells[IDX["Loan Identifier"]] = loan_id
    cells[IDX["Monthly Reporting Period"]] = period
    cells[IDX["Channel"]] = random.choice(["R", "C", "B"])
    cells[IDX["Seller Name"]] = "SOME BANK"
    cells[IDX["Servicer Name"]] = "SOME SERVICER"
    cells[IDX["Original Interest Rate"]] = f"{random.uniform(3.0, 6.5):.3f}"
    cells[IDX["Current Interest Rate"]] = f"{random.uniform(3.0, 6.5):.3f}"
    cells[IDX["Original UPB"]] = f"{random.randint(80, 500) * 1000}"
    cells[IDX["Current Actual UPB"]] = f"{random.randint(60, 480) * 1000}"
    cells[IDX["Original Loan Term"]] = "360"
    cells[IDX["Origination Date"]] = f"01{vintage_year}"
    cells[IDX["First Payment Date"]] = f"03{vintage_year}"
    cells[IDX["Loan Age"]] = str(age)
    cells[IDX["Original Loan to Value Ratio (LTV)"]] = f"{random.randint(60, 97)}"
    cells[IDX["Original Combined Loan to Value Ratio (CLTV)"]] = f"{random.randint(60, 97)}"
    cells[IDX["Number of Borrowers"]] = str(random.choice([1, 2]))
    cells[IDX["Debt-To-Income (DTI)"]] = f"{random.randint(20, 50)}"
    cells[IDX["Borrower Credit Score at Origination"]] = str(random.randint(620, 820))
    cells[IDX["Loan Purpose"]] = random.choice(["P", "C", "R"])
    cells[IDX["Property State"]] = random.choice(["CA", "TX", "FL", "NY", "IL"])
    cells[IDX["Occupancy Status"]] = random.choice(["P", "S", "I"])
    cells[IDX["Property Type"]] = random.choice(["SF", "CO", "PU"])
    cells[IDX["Number of Units"]] = "1"
    cells[IDX["Current Loan Delinquency Status"]] = random.choice(
        ["00", "00", "00", "01", "02", "03"]
    )
    return "|".join(cells)


def make_file(vintage_year: int, quarter: int, n_loans: int = 50, n_months: int = 8) -> str:
    lines = []
    for i in range(n_loans):
        loan_id = f"{vintage_year}{quarter:02d}{i:06d}"
        for m in range(n_months):
            month = (quarter - 1) * 3 + 1 + m
            yr = vintage_year + (month - 1) // 12
            mm = ((month - 1) % 12) + 1
            period = f"{mm:02d}{yr}"
            lines.append(make_row(loan_id, period, m, vintage_year))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raw = settings.paths.raw_data
    for (yr, q) in [(2018, 1), (2019, 2)]:
        content = make_file(yr, q)
        path = raw / f"{yr}Q{q}.csv"
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path} ({content.count(chr(10))} rows)")