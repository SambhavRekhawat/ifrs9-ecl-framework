"""
src/dashboard/app.py
==================
IFRS 9 ECL dashboard — a thin presentation layer over the saved artifacts.

Run from the project root:

    streamlit run src/dashboard/app.py

Reads models/*.json (no recompute). Run the upstream stages first so the
artifacts exist (ECL, scenarios, validation, monitoring).
"""

from __future__ import annotations

import pathlib
import sys

# `streamlit run` does not put the project root on sys.path (unlike `python -m`),
# so add it here before importing the `src` / `config` packages.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.dashboard import loaders as L

st.set_page_config(page_title="IFRS 9 ECL Dashboard", layout="wide")

_RAG = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴", "n/a": "⚪"}


def _money(x):
    return f"${x:,.0f}" if isinstance(x, (int, float)) else "—"


def _num(x, fmt="{:.4f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "—"


def glossary(expanded: bool = False):
    """Plain-language explainer for the three ECL building blocks."""
    with st.expander("ℹ️ What do PD, LGD and EAD mean?", expanded=expanded):
        st.markdown(
            "**Expected Credit Loss (ECL) = PD × LGD × EAD**, discounted to today and "
            "probability-weighted across macro scenarios. Stage 1 loans use a **12-month** "
            "ECL; Stage 2 and Stage 3 use a **lifetime** ECL.\n\n"
            "- **PD — Probability of Default.** How likely a borrower is to default. The model "
            "ranks loans by risk (a scorecard plus gradient-boosted trees), is calibrated to a "
            "long-run *through-the-cycle* (TTC) average, then flexed *point-in-time* by the macro "
            "scenarios. Discrimination is judged by AUC / Gini / KS; trustworthiness of the level "
            "by the calibration curve.\n"
            "- **LGD — Loss Given Default.** The share of exposure you *don't* recover after a "
            "default (1 − recovery rate). The **downturn LGD** is the stressed value applied in the "
            "adverse scenario and to credit-impaired (Stage 3) loans.\n"
            "- **EAD — Exposure at Default.** The balance expected to be outstanding when a default "
            "happens. Here it's projected from the loan's scheduled amortization, adjusted by an "
            "observed *curtailment factor* (borrowers paying down faster than schedule).")


def page_overview():
    st.title("IFRS 9 — Expected Credit Loss")
    glossary()
    o = L.overview()
    if o["total_ecl"] is None:
        st.warning("No ECL results yet. Run `python -m src.ecl_engine.run_ecl` first.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total ECL", _money(o["total_ecl"]))
    c2.metric("Exposure (EAD)", _money(o["total_ead"]))
    c3.metric("Coverage", f"{o['coverage_pct']}%")
    c4.metric("Loans", f"{o['n_loans']:,}" if o["n_loans"] else "—")
    st.caption("Coverage = ECL ÷ EAD — the provision as a share of exposure.")
    c5, c6 = st.columns(2)
    c5.metric("Probability-weighted LGD", o["weighted_lgd"])
    if o["monitoring_status"]:
        c6.metric("Monitoring (latest)", f"{_RAG.get(o['monitoring_status'],'')} {o['monitoring_status']}")
    st.caption(f"Reporting book: {o['reporting_period']}")

    if o["stage_distribution"]:
        st.subheader("Stage distribution")
        st.caption("Stage 1 = performing (12-month ECL) · Stage 2 = significant increase in "
                   "credit risk (lifetime ECL) · Stage 3 = credit-impaired / in default (lifetime ECL).")
        df = pd.DataFrame(o["stage_distribution"])
        df["Stage"] = "Stage " + df["stage"].astype(str)
        st.bar_chart(df.set_index("Stage")["pct"], y_label="% of loans")


def page_ecl():
    st.title("ECL breakdown")
    glossary()
    v = L.ecl_view()
    if not v:
        st.warning("Run `python -m src.ecl_engine.run_ecl` first.")
        return
    st.caption("ECL = PD × LGD × EAD, discounted and probability-weighted across scenarios. "
               "Stage 1 = 12-month horizon; Stages 2–3 = lifetime.")
    c1, c2, c3 = st.columns(3)
    c1.metric("12-month basis total", _money(v["ecl_12m_total"]))
    c2.metric("Lifetime basis total", _money(v["ecl_lifetime_total"]))
    c3.metric("Weighted LGD", v["weighted_lgd"])
    if v["by_stage"]:
        df = pd.DataFrame(v["by_stage"])
        df["Stage"] = "Stage " + df["stage"].astype(str)
        disp = df[["Stage", "n_loans", "ead", "ecl", "coverage_pct"]].rename(
            columns={"n_loans": "Loans", "ead": "EAD", "ecl": "ECL", "coverage_pct": "Coverage %"})
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.subheader("Coverage by stage")
        st.bar_chart(df.set_index("Stage")["coverage_pct"], y_label="Coverage %")
    if v["lgd_by_scenario"]:
        st.caption("LGD by scenario: " + ", ".join(f"{k}={x}" for k, x in v["lgd_by_scenario"].items()))


def page_scenarios():
    st.title("Macro scenarios")
    v = L.scenario_view()
    if not v:
        st.warning("Run `python -m src.scenario_engine.run_scenarios` first.")
        return
    st.metric("Probability-weighted PIT PD (12m)", v["weighted_pit_pd_12m"])
    st.write(f"Scenario ordering valid (downside ≥ base ≥ upside): "
             f"{'✅' if v['ordering_ok'] else '⚠️'}")
    rows = [{"Scenario": s, "Weight": v["weights"].get(s),
             "PIT PD 12m": v["pit_pd_12m"].get(s)} for s in v["weights"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if v["pit_pd_paths"]:
        st.subheader("PIT PD path by scenario")
        st.line_chart(pd.DataFrame(v["pit_pd_paths"]), y_label="PIT PD")


def page_validation():
    st.title("Validation")
    v = L.validation_view()
    if not v:
        st.warning("Run `python -m src.validation.run_validation` first.")
        return
    st.subheader(f"Checks: {v['n_error_passed']}/{v['n_error_checks']} error-level passed "
                 f"(+{v['n_advisory']} advisory)")
    for c in v["checks"]:
        icon = "✅" if c["passed"] else ("🟡" if c["severity"] == "warn" else "❌")
        st.write(f"{icon} **{c['check']}** — {c['detail']}")
    d = v["discrimination"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PD AUC", d.get("auc"))
    c2.metric("Gini", d.get("gini"))
    c3.metric("KS", d.get("ks"))
    c4.metric("Calibrated PD", v["calibrated_portfolio_pd"])
    if v["sensitivity"]:
        st.subheader("ECL sensitivity / attribution")
        df = pd.DataFrame(v["sensitivity"])
        st.dataframe(df.rename(columns={"lever": "Lever", "value": "Value",
                     "total_ecl": "Total ECL", "coverage_pct": "Coverage %",
                     "delta_vs_base_pct": "Δ vs base %"}), hide_index=True, use_container_width=True)
        plot = df[df["lever"] != "BASE CASE"].copy()
        plot["label"] = plot["lever"] + "=" + plot["value"].astype(str)
        st.bar_chart(plot.set_index("label")["delta_vs_base_pct"], y_label="Δ ECL vs base %")


def page_monitoring():
    st.title("Monitoring")
    v = L.monitoring_view()
    if not v:
        st.warning("Run `python -m src.monitoring.run_monitoring` first.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Latest-period status", f"{_RAG.get(v['latest_status'],'')} {v['latest_status']}")
    c2.metric("Historical worst", f"{_RAG.get(v['historical_worst'],'')} {v['historical_worst']}")
    b = v["breach_summary"]
    if b:
        st.caption(f"Breaches across history — 🔴 {b.get('red',0)} · 🟡 {b.get('amber',0)} · 🟢 {b.get('green',0)}")
    if v["backtest"]:
        st.subheader("PD back-test — predicted vs realised default rate")
        df = pd.DataFrame(v["backtest"])
        st.line_chart(df.set_index("period")[["mean_pred_pd", "realised_dr"]])
    if v["psi"]:
        st.subheader("PD distribution drift (PSI)")
        df = pd.DataFrame(v["psi"])
        st.line_chart(df.set_index("to")["psi"], y_label="PSI")
    if v["delinquency_trend"]:
        st.subheader("Delinquency trend")
        df = pd.DataFrame(v["delinquency_trend"])
        st.line_chart(df.set_index("period")[["share_30dpd", "share_90dpd"]])


def page_models():
    st.title("Model diagnostics")
    glossary(expanded=True)
    m = L.model_view()

    # ---------------- PD ----------------
    st.header("PD — Probability of Default")
    st.markdown(
        "Ranks every loan by default risk, then is calibrated to a long-run level and flexed by "
        "the macro scenarios. **AUC / Gini / KS** measure *discrimination* (can it tell good from "
        "bad?); the **calibration curve** measures whether the predicted *level* matches reality.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", _num(m["pd_auc"], "{:.4f}"))
    c2.metric("Gini", _num(m["pd_gini"], "{:.4f}"))
    c3.metric("KS", _num(m["pd_ks"], "{:.4f}"))
    c4.metric("TTC PD", _num(m["ttc_pd"], "{:.4f}"))
    st.caption(f"Best model: **{m['pd_best_model'] or '—'}** · asset correlation ρ "
               f"{_num(m['asset_correlation'], '{:.2f}')} · "
               "AUC 0.5 = coin-flip, 1.0 = perfect · Gini = 2·AUC−1 · "
               "KS = widest gap between good/bad score distributions.")

    models = m["pd_models"] or {}
    if models:
        st.subheader("Model comparison")
        rows = [{"model": name, "AUC": v.get("auc"), "Gini": v.get("gini"), "KS": v.get("ks"),
                 "Brier": v.get("brier")} for name, v in models.items()]
        mdf = pd.DataFrame(rows)
        st.dataframe(mdf, hide_index=True, use_container_width=True)
        st.bar_chart(mdf.set_index("model")[["AUC", "Gini", "KS"]])
        st.caption("Brier score (lower = better) measures probability accuracy; AUC/Gini/KS measure ranking.")

    roc = m["pd_roc"]
    if roc:
        st.subheader("ROC curve")
        rdf = pd.DataFrame(roc)
        rdf["random (AUC 0.5)"] = rdf["fpr"]
        st.line_chart(rdf.set_index("fpr").rename(columns={"tpr": "model"})[["model", "random (AUC 0.5)"]],
                      x_label="False-positive rate", y_label="True-positive rate")
        st.caption("The further the curve bows toward the top-left (away from the diagonal), the stronger the model.")
    else:
        st.caption("ROC / KS curves populate after the next `python -m src.pd_model.run_pd` "
                   "(this run's metrics file predates the curve export).")

    ksc = m["pd_ks_curve"]
    if ksc:
        st.subheader("KS curve — cumulative good vs bad capture")
        kdf = pd.DataFrame(ksc)
        st.line_chart(kdf.set_index("pop_pct")[["cum_bad", "cum_good"]],
                      x_label="Population (sorted by predicted PD)", y_label="Cumulative share")
        st.caption("KS is the maximum vertical gap between the two curves — bigger gap = sharper separation.")

    cal = m["pd_calibration"]
    if cal:
        st.subheader("Calibration curve — predicted vs observed PD")
        cdf = pd.DataFrame(cal)
        cdf["perfect"] = cdf["pred_pd"]
        st.line_chart(cdf.set_index("pred_pd").rename(columns={"obs_default": "observed"})
                      [["observed", "perfect"]], x_label="Predicted PD (decile mean)",
                      y_label="Observed default rate")
        st.caption("Points on the diagonal mean predicted PD matches the realised default rate. "
                   "Above the line = under-prediction; below = over-prediction.")

    iv = m["pd_iv"]
    if iv:
        st.subheader("Feature strength (Information Value)")
        idf = pd.DataFrame(iv).head(15)
        st.bar_chart(idf.set_index("feature")["iv"], y_label="Information Value")
        st.caption("IV ranks how much each feature separates defaulters from non-defaulters "
                   "(<0.02 useless · 0.1–0.3 medium · >0.3 strong).")

    # ---------------- LGD ----------------
    st.header("LGD — Loss Given Default")
    st.markdown(
        "The fraction of exposure not recovered after default (1 − recovery). The **downturn LGD** "
        "is the stressed value used for the adverse scenario and Stage 3 loans.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LGD (through-the-cycle)", _num(m["lgd_mean"], "{:.3f}"))
    c2.metric("LGD (downturn)", _num(m["lgd_downturn"], "{:.3f}"))
    c3.metric("Downturn uplift", _num(m["lgd_uplift"], "×{:.2f}"))
    c4.metric("Defaults observed", f"{m['lgd_n']:,}" if isinstance(m["lgd_n"], (int, float)) else "—")
    bits = []
    if m["lgd_driver"]:
        bits.append(f"downturn driver: **{m['lgd_driver']}**")
    if m["lgd_window"]:
        bits.append(f"worst empirical window: {m['lgd_window']}")
    if m["lgd_fit_r2"] is not None:
        bits.append(f"model fit R² {_num(m['lgd_fit_r2'], '{:.3f}')}")
    if bits:
        st.caption(" · ".join(bits))

    # ---------------- EAD ----------------
    st.header("EAD — Exposure at Default")
    st.markdown(
        "The balance expected outstanding at default, projected from scheduled amortization and "
        "nudged by an observed curtailment factor (a factor below 1.0 means borrowers pay down "
        "faster than the contractual schedule).")
    c1, c2 = st.columns(2)
    c1.metric("Curtailment factor (12m)", _num(m["ead_curtailment_12m"], "{:.3f}"))
    c2.metric("Curtailment applied", "Yes" if m["ead_apply_curtailment"] else "No")
    st.caption(f"Method: {m['ead_method'] or '—'}")


PAGES = {
    "Overview": page_overview,
    "ECL breakdown": page_ecl,
    "Scenarios": page_scenarios,
    "Validation": page_validation,
    "Monitoring": page_monitoring,
    "Model diagnostics": page_models,
}

st.sidebar.title("IFRS 9 ECL")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
st.sidebar.caption("Thin view over models/*.json — run the pipeline stages to refresh.")
PAGES[choice]()