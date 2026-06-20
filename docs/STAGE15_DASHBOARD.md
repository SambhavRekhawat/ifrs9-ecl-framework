# Stage 15 — Streamlit Dashboard

## Purpose

The presentation layer that ties the whole framework together. It is a **thin
view over the saved artifacts** (`models/*.json`) — it performs no computation,
so it loads instantly and always reflects the last pipeline run.

## Running it

```bash
pip install streamlit          # one-time, if not already installed
streamlit run src/dashboard/app.py
```

Run the pipeline stages first so the artifacts exist; the dashboard tells you
which stage to run if any are missing.

## Pages

| Page | Shows | Source artifact |
|------|-------|-----------------|
| **Overview** | Total ECL, EAD, coverage, loan count, weighted LGD, latest monitoring status, stage distribution | `ecl_results`, `staging_artifacts`, `monitoring_results` |
| **ECL breakdown** | 12-month vs lifetime totals, by-stage table, coverage-by-stage chart, LGD by scenario | `ecl_results` |
| **Scenarios** | Weighted PIT PD, ordering check, per-scenario weights/PD, PIT-PD paths | `scenario_artifacts` |
| **Validation** | Pass/fail checks, discrimination (AUC/Gini/KS), calibrated PD, sensitivity/attribution grid | `validation_results` |
| **Monitoring** | Latest vs historical RAG, breach counts, back-test, PSI, delinquency trends | `monitoring_results` |
| **Model parameters** | PD AUC/Gini/KS, TTC PD, ρ, LGD (TTC/downturn), EAD method/curtailment | `pd_metrics`, `lgd_stats`, `ead_model`, `pit_artifacts` |

## Design

- **`loaders.py`** — pure, Streamlit-free functions that read each artifact into
  display-ready structures. Every loader degrades gracefully (returns `None` /
  empty) when an artifact is missing, and tolerates both the new
  (`latest_status`) and legacy (`overall_status`) monitoring keys. Fully
  unit-tested.
- **`app.py`** — the Streamlit UI: sidebar navigation across the six pages,
  `st.metric` cards, `st.dataframe` tables, and native `st.bar_chart` /
  `st.line_chart` (no extra chart dependency). RAG status shown with 🟢🟡🔴.

Keeping all data access in `loaders.py` means the dashboard logic is testable
without a running Streamlit server (which is how it is validated in CI).

## Tests (`tests/test_dashboard.py`)

Each loader returns the right shape from representative artifacts; missing files
return `None`; the legacy monitoring key still resolves. Suite: 86 passing.

## Note

The dashboard is read-only and local. It is the human-facing endpoint of the
pipeline: PD → PIT/macro → prepayment → LGD → EAD → staging → scenarios → ECL →
validation → monitoring, each stage writing the artifact this view renders.
