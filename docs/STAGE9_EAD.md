# Stage 9 — EAD (Exposure at Default)

## What EAD is for a mortgage

Exposure at Default is the outstanding balance we expect to be exposed to at the
moment a loan defaults. For revolving products (credit cards, lines of credit)
this requires modelling undrawn commitments via a credit-conversion factor. A
**fixed-rate amortizing mortgage has no undrawn commitment** — the borrower
cannot draw more — so EAD is simply the **scheduled outstanding balance at the
future month of default**. No machine learning is required; EAD is deterministic
cash-flow mechanics.

## The amortization engine (`amortization.py`)

For a loan with current balance `UPB`, monthly rate `r = annual%/100/12`, and
`n` months remaining, the balance after `k` further scheduled payments is:

```
B_k = UPB * ((1+r)^n - (1+r)^k) / ((1+r)^n - 1)
```

For `r ≈ 0` this degenerates to the straight line `B_k = UPB * (n-k)/n`. The
functions are numpy-vectorised (one loan or a whole portfolio), and guard the
edge cases `UPB ≤ 0`, `n ≤ 0`, and `k > n` (balance floored at 0). Verified
against hand-computed values: a $100k / 6% / 360-month loan has a $599.55
payment, a $98,772 balance after 12 months, and $0 at maturity.

## Validation against actual paydown (`ead_model.validate`)

We do not take the formula on faith — we check it reproduces reality. For each
sampled loan-month we project the balance `h` months forward and compare to the
loan's **actual** balance `h` months later (a within-loan shift). Reported per
horizon (3/6/12/24 months): N, mean predicted, mean actual, MAE, median absolute
percent error, and the **actual/scheduled ratio**.

- ratio ≈ 1.0 → loans track the schedule (expected for standard fixed-rate loans)
- ratio < 1.0 → **curtailments** (borrowers pay extra principal, faster paydown)
- ratio > 1.0 → modifications / forbearance / interest-only periods

The **curtailment factor** is the median actual/scheduled ratio at 12 months. It
is reported and saved; applying it to scale EAD is optional
(`apply_curtailment_adjustment`, default off — scheduled balance is the prudent,
slightly conservative choice since curtailments only reduce exposure).

## Sampling (`ead_data.load_ead_panel`)

Validation samples ~1 in `sample_loan_mod` loans (default 50 ≈ 2%) via
`abs(hashtext(loan_id)) % mod = 0`, keeping whole loan histories intact. The
rate and remaining-term columns are chosen by introspection
(`int_rate_current` else `int_rate_orig`; `remaining_months_to_maturity` else
`remaining_months_to_legal_maturity`) so the pull is robust to schema variation.

## Outputs

- `models/ead_model.json` — method, 12-month curtailment factor, and the full
  per-horizon validation table.
- `reports/ead_report_<ts>.html` — the validation table and curtailment note.

## How this feeds the ECL engine (Stage 12)

`project_ead(upb, note_rate, remaining_term, horizon, curtailment)` returns the
EAD path for months 1..horizon. The ECL engine projects each loan's balance over
its life and applies `ECL = Σ marginal_PD_t × LGD × EAD_t`, discounted, with the
prepayment survival curve weighting the marginal default probabilities. EAD
supplies the `EAD_t` term.

## Thin-data note

Unlike PD/LGD, EAD here has no statistical-power problem: it is a deterministic
identity validated on millions of loan-months, not estimated from the 388 credit
events. Curtailment behaviour is the only empirical input, and it is measured,
reported, and (by default) left unapplied for prudence.

## Config (`config.yaml → ead`)

```yaml
ead:
  horizons: [3, 6, 12, 24]
  sample_loan_mod: 50
  apply_curtailment_adjustment: false
```

## Tests (`tests/test_ead.py`)

Payment and balance against known values; zero-rate straight-line; zero-UPB /
zero-term guards; validation reproduces a clean amortizing loan to <0.1% error;
curtailment scaling monotonicity. All green (suite: 48 passing).
