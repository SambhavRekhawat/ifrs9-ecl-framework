"""
src/lgd_model/lgd_data.py
========================
Pulls realised defaults (credit-event dispositions) and computes observed loss
severity (LGD) from Fannie's loss/proceeds fields, plus a mark-to-market LTV
(HPI-adjusted) which is the economic driver of mortgage loss.

  net_loss = default_UPB + costs - proceeds
  LGD      = net_loss / default_UPB                      (clipped to [0, 1])
  MTM_LTV  = ltv_orig * (default_UPB / upb_orig) * HPI_orig / HPI_default

Note: realised liquidations are rare in 2018-2023 (benign credit + COVID
forbearance), so this is a thin sample - the model is deliberately parsimonious.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from config.settings import settings
from src.ingestion import db
from src.utils.logger import get_logger

log = get_logger(__name__)

_COST_FIELDS = ["foreclosure_costs", "property_preservation_and_repair_costs",
                "asset_recovery_costs", "miscellaneous_holding_expenses_and_credits",
                "associated_taxes_for_holding_property"]
_PROCEED_FIELDS = ["net_sales_proceeds", "credit_enhancement_proceeds",
                   "other_foreclosure_proceeds", "repurchase_make_whole_proceeds"]


def load_lgd_frame() -> pl.DataFrame:
    from sqlalchemy import inspect
    cfg = settings.config["lgd"]
    codes = ", ".join(f"'{c}'" for c in cfg["default_zbc"])
    engine = db.get_engine()
    insp = inspect(engine)
    mcols = {c["name"] for c in insp.get_columns("loan_monthly")}
    lcols = {c["name"] for c in insp.get_columns("loan_master")}

    monthly_fields = (["upb_at_the_time_of_removal", "upb_current",
                       "foreclosure_principal_write_off_amount", "loan_age"]
                      + _COST_FIELDS + _PROCEED_FIELDS)
    master_fields = ["ltv_orig", "fico_orig", "upb_orig", "origination_date", "state"]
    optional_fields = ["cumulative_credit_event_net_gain_or_loss",
                       "current_period_credit_event_net_gain_or_loss"]

    select = ["m.loan_id", "m.reporting_period AS default_period"]
    absent = []
    for f in monthly_fields:
        select.append(f"m.{f}" if f in mcols else f"NULL AS {f}")
        if f not in mcols:
            absent.append(f)
    for f in master_fields:
        select.append(f"l.{f}" if f in lcols else (f"m.{f}" if f in mcols else f"NULL AS {f}"))
    for f in optional_fields:               # only include if the column truly exists
        if f in mcols:
            select.append(f"m.{f}")

    if absent:
        log.warning("LGD: these loss fields are not in loan_monthly and will be treated as 0/missing: %s", absent)

    sql = (f"SELECT {', '.join(select)} FROM loan_monthly m "
           f"JOIN loan_master l USING (loan_id) "
           f"WHERE trim(CAST(m.zero_balance_code AS VARCHAR)) IN ({codes})")
    with engine.connect() as conn:
        pdf = pd.read_sql(sql, conn)
    log.info("Pulled %d credit-event dispositions", len(pdf))
    return _compute(pdf)


def _month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _compute(pdf: pd.DataFrame) -> pl.DataFrame:
    import numpy as np
    cfg = settings.config["lgd"]
    fannie = "cumulative_credit_event_net_gain_or_loss"
    numeric = (_COST_FIELDS + _PROCEED_FIELDS
               + ["upb_at_the_time_of_removal", "upb_current", "foreclosure_principal_write_off_amount"])
    for c in numeric:
        if c not in pdf.columns:
            pdf[c] = np.nan
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    has_fannie = fannie in pdf.columns
    if has_fannie:
        pdf[fannie] = pd.to_numeric(pdf[fannie], errors="coerce")

    # Default UPB: prefer UPB at removal, else last current UPB, else written-off principal.
    default_upb = (pdf["upb_at_the_time_of_removal"].fillna(0)
                   .where(lambda x: x > 0, pdf["upb_current"].fillna(0))
                   .where(lambda x: x > 0, pdf["foreclosure_principal_write_off_amount"].fillna(0)))
    pdf["default_upb"] = default_upb

    costs = pdf[_COST_FIELDS].fillna(0).sum(axis=1)
    proceeds = pdf[_PROCEED_FIELDS].fillna(0).sum(axis=1)
    net_loss_component = pdf["default_upb"] + costs - proceeds
    has_components = (costs + proceeds) > 0
    if has_fannie:
        net_loss_fannie = -pdf[fannie]
        pdf["net_loss"] = net_loss_component.where(has_components, net_loss_fannie)
    else:
        # No Fannie net-loss field in this DB: use components; if a loan reports
        # neither costs nor proceeds, its loss is unknown -> NaN (dropped below).
        pdf["net_loss"] = net_loss_component.where(has_components, np.nan)

    pdf["lgd_raw"] = pdf["net_loss"] / pdf["default_upb"].replace(0, pd.NA)
    pdf["lgd"] = pdf["lgd_raw"].clip(lower=0, upper=1)

    # Mark-to-market LTV via HPI. Use an integer month-key map (year*12+month) so
    # it is robust to date vs datetime dtype mismatches that break a date merge.
    macro = _load_hpi()

    def _mk(s):
        d = pd.to_datetime(s, errors="coerce")
        return d.dt.year * 12 + d.dt.month

    hpi_map = dict(zip(_mk(macro["reporting_period"]),
                       pd.to_numeric(macro["hpi"], errors="coerce")))
    hpi_orig = _mk(pdf["origination_date"]).map(hpi_map)
    hpi_def = _mk(pdf["default_period"]).map(hpi_map)
    cur_ltv = (pd.to_numeric(pdf["ltv_orig"], errors="coerce") * pdf["default_upb"]
               / pd.to_numeric(pdf["upb_orig"], errors="coerce").replace(0, np.nan))
    mtm = cur_ltv * hpi_orig / hpi_def
    # --- diagnostic: where does the HPI mark-to-market break? ---
    mk_macro = _mk(macro["reporting_period"])
    log.info("LGD HPI join | n=%d | origination_date non-null=%d | default_period non-null=%d "
             "| hpi_orig mapped=%d | hpi_def mapped=%d | mtm via HPI=%d, fell back to amortised=%d",
             len(pdf), int(pdf["origination_date"].notna().sum()),
             int(pdf["default_period"].notna().sum()),
             int(hpi_orig.notna().sum()), int(hpi_def.notna().sum()),
             int(mtm.notna().sum()), int(mtm.isna().sum()))
    log.info("LGD HPI key ranges | macro months %s..%s | origination %s..%s | default %s..%s",
             mk_macro.min(), mk_macro.max(),
             _mk(pdf["origination_date"]).min(), _mk(pdf["origination_date"]).max(),
             _mk(pdf["default_period"]).min(), _mk(pdf["default_period"]).max())
    pdf["mtm_ltv"] = mtm.fillna(cur_ltv)        # fall back to amortisation LTV if HPI missing
    pdf["loan_age_years"] = pd.to_numeric(pdf["loan_age"], errors="coerce") / 12.0

    keep = ["loan_id", "default_period", "default_upb", "net_loss", "lgd_raw", "lgd",
            "mtm_ltv", "loan_age_years", "ltv_orig", "fico_orig", "state"]
    out = pdf[keep].dropna(subset=["lgd", "default_upb"])
    out = out[out["default_upb"] > 0]
    log.info("Usable LGD observations: %d (mean LGD %.3f)", len(out), out["lgd"].mean())
    return pl.from_pandas(out)


def _load_hpi() -> pd.DataFrame:
    cfg = settings.config["lgd"]
    path = settings.project_root / "data" / "macro" / "macro_data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run Stage 6 macro_data first.")
    m = pd.read_parquet(path)[["reporting_period", cfg["hpi_series"]]].copy()
    m.columns = ["reporting_period", "hpi"]
    m["reporting_period"] = _month_start(m["reporting_period"])
    return m