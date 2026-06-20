# Stage 12 — ECL Engine

## Purpose

The keystone. Combines every prior component into the IFRS 9 Expected Credit
Loss: PD term structure × LGD × EAD × prepayment survival, scenario-conditioned,
probability-weighted, discounted, and split 12-month vs lifetime by stage.

## The calculation

For each loan, under each scenario, walk the lifetime month by month:

```
hazard_t            = monthly_hazard(pd12) * scenario_multiplier_t      (clipped 0..1)
EAD_t               = scheduled amortized balance at month t            (Stage 9)
discount_t          = (1 + note_rate/12)^(-t)                           (EIR proxy)
marginal_default_t  = survival_{t-1} * hazard_t
EL_t                = marginal_default_t * LGD_scenario * EAD_t * discount_t
survival_t          = survival_{t-1} * (1 - hazard_t) * (1 - SMM)       (default & prepay exits)
```

- **12-month ECL** sums `t <= 12`; **lifetime ECL** sums the full horizon.
- **Stage selection**: Stage 1 → 12-month ECL; Stage 2/3 → lifetime ECL (Stage 10).
- **Probability weighting**: each loan's ECL is `Σ_s w_s · ECL_s` (Stage 11 weights).
- **Stage 3** (already credit-impaired) is treated as default-certain:
  `ECL = weighted-LGD × current balance`.

## PD level calibration (the loop closed here)

The scenario engine's PIT-PD paths are turned into **relative multipliers**
anchored so the probability-weighted month-1 factor equals 1
(`term_structure.scenario_multipliers`). Each loan's *own* calibrated 12-month
PD (from the Stage 5/6 model) is shaped by these multipliers. This preserves the
book's absolute PD level while imposing each scenario's shape and severity —
resolving the `Z=0 ≠ TTC` anchoring point noted in Stage 11. The monthly hazard
is the constant-hazard equivalent of the 12-month PD
(`1 - (1-PD)^(1/12)`), which by construction reproduces the 12-month PD.

## Inputs (where each comes from)

| Input | Source |
|-------|--------|
| `pd12`, `stage` | staging pipeline (Stage 10) — PD scored, SICR applied |
| `upb`, `note_rate`, `remaining_term` | `loan_monthly` latest row per loan |
| EAD path | amortization engine (Stage 9) |
| LGD (base / downturn) | `lgd_stats.json` (Stage 8): `stats.mean`, `downturn.downturn_lgd` |
| scenario Z / PD paths, weights | `scenario_artifacts.json` (Stage 11) |
| prepayment | constant CPR assumption (`annual_cpr`) → SMM |

### Prepayment as a CPR assumption

The survival curve uses a constant **CPR** (conditional prepayment rate)
converted to a monthly SMM, rather than loan-level prepay-model scores. This is a
standard ECL simplification and is the prudent choice given the prepay model's
limited discriminatory power (AUC ≈ 0.59, Stage 7): for ECL the prepay model's
value is portfolio-level *speed*, which the CPR captures. Loan-level prepay
scoring can replace this later without changing the interface.

### LGD by scenario

Base and upside use the through-the-cycle LGD (`stats.mean`); the downside uses
the downturn LGD (the Stage 8 benchmark, 0.35). The probability-weighted LGD is
what Stage 3 exposures are charged.

## Outputs

- `models/ecl_results.json` — total EAD, total ECL, coverage %, 12m vs lifetime
  totals, weighted LGD, LGD-by-scenario, and a by-stage breakdown.
- `reports/ecl_report_<ts>.html` — the headline ECL and the by-stage table.

## What "good" looks like

- **Coverage rises by stage**: Stage 1 (12-month) < Stage 2 (lifetime) <
  Stage 3 (≈ weighted LGD). This monotonicity is the core sanity check.
- **Portfolio coverage** for a benign prime mortgage book is small — on the
  order of tenths of a percent.
- Stage 3 coverage ≈ the probability-weighted LGD.

## Config (`config.yaml → ecl`)

```yaml
ecl:
  horizon_months: 60
  annual_cpr: 0.10
  reporting_period: null            # null = latest row per loan
  lgd_downside_scenario: downside
  discount_at_note_rate: true
```

## Tests (`tests/test_ecl.py`)

Monthly hazard reproduces the annual PD; CPR↔SMM; multipliers anchored to 1;
single-loan ECL matches a hand value; Stage 3 = weighted-LGD × balance; coverage
rises by stage. Suite: 66 passing.

## Performance

Vectorised across loans; loops only over months × scenarios (≈ 60 × 3). Memory is
`O(n_loans)`, so the full ~466k-loan book runs in seconds once inputs are
assembled (the assembly — PD scoring + EAD pull — is the slower part).
