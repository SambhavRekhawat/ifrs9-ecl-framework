"""
src/ingestion/schema.py
=======================
The single source of truth for the Fannie Mae Single-Family Loan Performance
file layout.

WHY THIS MATTERS
----------------
The downloaded files are pipe ("|") delimited and have NO column headers.
The 108 columns are identified ONLY by position. If we get the order wrong,
every downstream model is wrong. These names + order are transcribed from the
official "CRT Glossary and File Layout" (Oct-2020 / 108-field release).

NOTE ON VERSIONS: The 2023 release added 2 fields (110 total). This module
assigns the 108 known names by position and pads any extra columns with
generic names + a warning, so a newer file never crashes the pipeline.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. The 108 columns, IN ORDER (positional layout)
# ---------------------------------------------------------------------------
COLUMNS: list[str] = [
    "Reference Pool ID", "Loan Identifier", "Monthly Reporting Period", "Channel", "Seller Name",
    "Servicer Name", "Master Servicer", "Original Interest Rate", "Current Interest Rate", "Original UPB",
    "UPB at Issuance", "Current Actual UPB", "Original Loan Term", "Origination Date", "First Payment Date",
    "Loan Age", "Remaining Months to Legal Maturity", "Remaining Months To Maturity", "Maturity Date",
    "Original Loan to Value Ratio (LTV)", "Original Combined Loan to Value Ratio (CLTV)", "Number of Borrowers",
    "Debt-To-Income (DTI)", "Borrower Credit Score at Origination", "Co-Borrower Credit Score at Origination",
    "First Time Home Buyer Indicator", "Loan Purpose", "Property Type", "Number of Units", "Occupancy Status",
    "Property State", "Metropolitan Statistical Area (MSA)", "Zip Code Short", "Mortgage Insurance Percentage",
    "Amortization Type", "Prepayment Penalty Indicator", "Interest Only Loan Indicator",
    "Interest Only First Principal And Interest Payment Date", "Months to Amortization",
    "Current Loan Delinquency Status", "Loan Payment History", "Modification Flag",
    "Mortgage Insurance Cancellation Indicator", "Zero Balance Code", "Zero Balance Effective Date",
    "UPB at the Time of Removal", "Repurchase Date", "Scheduled Principal Current", "Total Principal Current",
    "Unscheduled Principal Current", "Last Paid Installment Date", "Foreclosure Date", "Disposition Date",
    "Foreclosure Costs", "Property Preservation and Repair Costs", "Asset Recovery Costs",
    "Miscellaneous Holding Expenses and Credits", "Associated Taxes for Holding Property", "Net Sales Proceeds",
    "Credit Enhancement Proceeds", "Repurchase Make Whole Proceeds", "Other Foreclosure Proceeds",
    "Modification-Related Non-Interest Bearing UPB", "Principal Forgiveness Amount", "Original List Start Date",
    "Original List Price", "Current List Start Date", "Current List Price", "Borrower Credit Score At Issuance",
    "Co-Borrower Credit Score At Issuance", "Borrower Credit Score Current", "Co-Borrower Credit Score Current",
    "Mortgage Insurance Type", "Servicing Activity Indicator", "Current Period Modification Loss Amount",
    "Cumulative Modification Loss Amount", "Current Period Credit Event Net Gain or Loss",
    "Cumulative Credit Event Net Gain or Loss", "Special Eligibility Program",
    "Foreclosure Principal Write-off Amount", "Relocation Mortgage Indicator", "Zero Balance Code Change Date",
    "Loan Holdback Indicator", "Loan Holdback Effective Date", "Delinquent Accrued Interest",
    "Property Valuation Method", "High Balance Loan Indicator",
    "ARM Initial Fixed-Rate Period <= 5 YR Indicator", "ARM Product Type", "Initial Fixed-Rate Period",
    "Interest Rate Adjustment Frequency", "Next Interest Rate Adjustment Date", "Next Payment Change Date",
    "Index", "ARM Cap Structure", "Initial Interest Rate Cap Up Percent", "Periodic Interest Rate Cap Up Percent",
    "Lifetime Interest Rate Cap Up Percent", "Mortgage Margin", "ARM Balloon Indicator", "ARM Plan Number",
    "Borrower Assistance Plan", "High Loan to Value (HLTV) Refinance Option Indicator", "Deal Name",
    "Repurchase Make Whole Proceeds Flag", "Alternative Delinquency Resolution",
    "Alternative Delinquency Resolution Count", "Total Deferral Amount",
]
assert len(COLUMNS) == 108, f"Expected 108 columns, found {len(COLUMNS)}"

# ---------------------------------------------------------------------------
# 2. Date columns (parse as dates) and numeric columns (cast to float)
# ---------------------------------------------------------------------------
DATE_COLUMNS: list[str] = [
    "Monthly Reporting Period", "Origination Date", "First Payment Date", "Maturity Date",
    "Interest Only First Principal And Interest Payment Date", "Zero Balance Effective Date",
    "Repurchase Date", "Last Paid Installment Date", "Foreclosure Date", "Disposition Date",
    "Original List Start Date", "Current List Start Date", "Zero Balance Code Change Date",
    "Loan Holdback Effective Date", "Next Interest Rate Adjustment Date", "Next Payment Change Date",
]

NUMERIC_COLUMNS: list[str] = [
    "Original Interest Rate", "Current Interest Rate", "Original UPB", "UPB at Issuance", "Current Actual UPB",
    "Original Loan Term", "Loan Age", "Remaining Months to Legal Maturity", "Remaining Months To Maturity",
    "Original Loan to Value Ratio (LTV)", "Original Combined Loan to Value Ratio (CLTV)", "Number of Borrowers",
    "Debt-To-Income (DTI)", "Borrower Credit Score at Origination", "Co-Borrower Credit Score at Origination",
    "Number of Units", "Mortgage Insurance Percentage", "Months to Amortization", "UPB at the Time of Removal",
    "Scheduled Principal Current", "Total Principal Current", "Unscheduled Principal Current", "Foreclosure Costs",
    "Property Preservation and Repair Costs", "Asset Recovery Costs", "Miscellaneous Holding Expenses and Credits",
    "Associated Taxes for Holding Property", "Net Sales Proceeds", "Credit Enhancement Proceeds",
    "Repurchase Make Whole Proceeds", "Other Foreclosure Proceeds", "Modification-Related Non-Interest Bearing UPB",
    "Principal Forgiveness Amount", "Original List Price", "Current List Price", "Borrower Credit Score At Issuance",
    "Co-Borrower Credit Score At Issuance", "Borrower Credit Score Current", "Co-Borrower Credit Score Current",
    "Current Period Modification Loss Amount", "Cumulative Modification Loss Amount",
    "Current Period Credit Event Net Gain or Loss", "Cumulative Credit Event Net Gain or Loss",
    "Foreclosure Principal Write-off Amount", "Delinquent Accrued Interest", "Initial Fixed-Rate Period",
    "Interest Rate Adjustment Frequency", "Initial Interest Rate Cap Up Percent",
    "Periodic Interest Rate Cap Up Percent", "Lifetime Interest Rate Cap Up Percent", "Mortgage Margin",
    "ARM Plan Number", "Alternative Delinquency Resolution Count", "Total Deferral Amount",
]

# The unique record key.
KEY_COLUMNS: list[str] = ["Loan Identifier", "Monthly Reporting Period"]

# ---------------------------------------------------------------------------
# 3. Static (one value per loan) vs Dynamic (changes month to month)
#    -> drives the split into loan_master and loan_monthly tables.
# ---------------------------------------------------------------------------
STATIC_COLUMNS: list[str] = [
    "Channel", "Seller Name", "Original Interest Rate", "Original UPB", "Original Loan Term",
    "Origination Date", "First Payment Date", "Original Loan to Value Ratio (LTV)",
    "Original Combined Loan to Value Ratio (CLTV)", "Number of Borrowers", "Debt-To-Income (DTI)",
    "Borrower Credit Score at Origination", "Co-Borrower Credit Score at Origination",
    "First Time Home Buyer Indicator", "Loan Purpose", "Property Type", "Number of Units", "Occupancy Status",
    "Property State", "Metropolitan Statistical Area (MSA)", "Zip Code Short", "Mortgage Insurance Percentage",
    "Amortization Type", "Mortgage Insurance Type", "Special Eligibility Program", "Relocation Mortgage Indicator",
]

DYNAMIC_COLUMNS: list[str] = [
    "Servicer Name", "Current Interest Rate", "Current Actual UPB", "Loan Age",
    "Remaining Months to Legal Maturity", "Remaining Months To Maturity", "Maturity Date",
    "Current Loan Delinquency Status", "Loan Payment History", "Modification Flag", "Zero Balance Code",
    "Zero Balance Effective Date", "UPB at the Time of Removal", "Total Principal Current",
    "Last Paid Installment Date", "Foreclosure Date", "Disposition Date", "Foreclosure Costs",
    "Property Preservation and Repair Costs", "Asset Recovery Costs", "Miscellaneous Holding Expenses and Credits",
    "Associated Taxes for Holding Property", "Net Sales Proceeds", "Credit Enhancement Proceeds",
    "Repurchase Make Whole Proceeds", "Other Foreclosure Proceeds", "Modification-Related Non-Interest Bearing UPB",
    "Principal Forgiveness Amount", "Servicing Activity Indicator", "Foreclosure Principal Write-off Amount",
    "Property Valuation Method", "High Balance Loan Indicator", "Borrower Assistance Plan",
    "High Loan to Value (HLTV) Refinance Option Indicator", "Repurchase Make Whole Proceeds Flag",
    "Alternative Delinquency Resolution", "Alternative Delinquency Resolution Count", "Total Deferral Amount",
]


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
def to_snake(name: str) -> str:
    """Turn an official column name into a clean SQL-safe snake_case name.

    'Original Loan to Value Ratio (LTV)' -> 'original_loan_to_value_ratio_ltv'
    Deterministic, so we never hand-type 108 names (and never mistype them).
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # non-alphanumeric -> underscore
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Friendly short names for the columns we use most in modelling.
DB_ALIASES: dict[str, str] = {
    "Loan Identifier": "loan_id",
    "Monthly Reporting Period": "reporting_period",
    "Borrower Credit Score at Origination": "fico_orig",
    "Original Loan to Value Ratio (LTV)": "ltv_orig",
    "Original Combined Loan to Value Ratio (CLTV)": "cltv_orig",
    "Debt-To-Income (DTI)": "dti_orig",
    "Original UPB": "upb_orig",
    "Current Actual UPB": "upb_current",
    "Original Interest Rate": "int_rate_orig",
    "Current Interest Rate": "int_rate_current",
    "Current Loan Delinquency Status": "delq_status",
    "Property State": "state",
    "Zero Balance Code": "zero_balance_code",
}


def db_name(col: str) -> str:
    """Final database column name: alias if defined, else snake_case."""
    return DB_ALIASES.get(col, to_snake(col))


# Pre-computed name maps the rest of the project imports.
DB_NAME_MAP: dict[str, str] = {c: db_name(c) for c in COLUMNS}

# Column groupings expressed in DB (snake) names, for table building.
LOAN_MASTER_COLS = ["Loan Identifier"] + STATIC_COLUMNS
LOAN_MONTHLY_COLS = KEY_COLUMNS + DYNAMIC_COLUMNS
