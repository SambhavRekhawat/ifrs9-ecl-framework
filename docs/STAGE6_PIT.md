# Stage 6 — TTC/PIT Calibration + Macro

Turns the Stage-5 PD into something ECL-ready: recalibrated to the true base
rate, anchored through-the-cycle, and linked to the economy point-in-time.

## Run it

```bash
# 1. (once) pull macro data from FRED into the warehouse
python -m src.pit_calibration.macro_data

# 2. build calibration + TTC/PIT framework
python -m src.pit_calibration.run_pit
```

Outputs:
- `data/macro/macro_data.parquet` + `macro_data` table in Postgres
- `models/pd_calibrator_isotonic.joblib` — maps raw model score -> calibrated PD
- `models/macro_to_z.joblib` + `models/pit_artifacts.json`
- `reports/pit_report_<ts>.html`, `reports/pit_z_series_<ts>.parquet`

## 1. Recalibration to the true base rate

The Stage-5 models *rank* well but over-predict the PD *level* (class
weighting). We fit an **isotonic regression** mapping model score -> observed
default frequency on the out-of-time test set, so the calibrated PD's average
matches reality. ECL needs a true probability (PD x LGD x EAD), not just a
ranking.

## 2. TTC <-> PIT (Vasicek single-factor / ASRF)

```
PD_PIT = Phi( (Phi^-1(PD_TTC) - sqrt(rho) * Z) / sqrt(1 - rho) )
```

- **TTC PD** — long-run average default rate (through-the-cycle anchor).
- **Z** — systematic economic factor (standard normal). Z > 0 = good economy
  -> lower PD; Z < 0 = stress -> higher PD.
- **rho** — asset correlation, set to **0.15** (the Basel residential-mortgage
  value), in `config.yaml -> pit.asset_correlation`.

Note (standard ASRF subtlety): the TTC PD is the *average over* Z, so Z = 0 is
**not** exactly the TTC PD; the cycle averages to TTC as Z averages to 0.

## 3. Linking PD to the economy

For each reporting period we compute the observed default rate `DR_t`, invert
the Vasicek formula to get the implied factor `Z_t`, then regress `Z_t` on macro
variables (`pit.macro_features`: unemployment, HPI YoY, GDP YoY, 10y Treasury).
This **macro -> Z** model lets Stage 11 project Z (and therefore PIT PD) under
base / upside / downside scenarios.

## Macro data — where it comes from

All series come from **FRED** via one free API key (the FHFA HPI is hosted on
FRED, so no separate FHFA download is needed):

| Field | FRED series | Notes |
|---|---|---|
| unemployment | `UNRATE` | monthly % |
| fed_funds | `FEDFUNDS` | monthly % |
| cpi | `CPIAUCSL` | monthly index (SA) |
| treasury_10y | `DGS10` | daily -> monthly average |
| real_gdp | `GDPC1` | quarterly -> forward-filled monthly |
| hpi | `HPIPONM226S` | **FHFA** purchase-only HPI, monthly (SA) |

Daily series are averaged to month-start; quarterly series are forward-filled;
year-on-year transforms (`hpi_yoy`, `gdp_yoy`, `cpi_yoy`) are derived.

## Known modelling considerations — revisit in Stage 11 (scenario engine)

**Macro -> Z: unemployment coefficient is ~0 / slightly positive** (real run:
unemployment +0.02, hpi_yoy +0.10, gdp_yoy +0.02, treasury_10y +0.52,
R2 ~ 0.51). Economically we would normally expect unemployment to be *negative*
(more unemployment -> lower Z -> higher PD).

**Why:** this is a genuine feature of the 2018-2024 data, not a code bug. During
the COVID unemployment spike (2020, ~14%) mortgage defaults did **not** rise —
forbearance programmes plus the house-price boom protected borrowers. The
2020 vintages carry the *lowest* 12-month default rates (0.45-0.75%) in the
whole panel despite peak unemployment, which cancels out the usual
unemployment->default relationship. HPI (the dominant mortgage-credit driver)
retains the correct positive sign.

**Decision (Sambhav, by design):** keep the model as-is. We deliberately do
**not** dummy-out or exclude the COVID period, because crashes of this kind
recur across market history and represent real tail dynamics that the model
should reflect — artificially "correcting" the coefficient would erase
legitimate crisis behaviour.

**Flagged as a future anomaly to watch:** if downside/stress scenarios in
Stage 11 behave counter-intuitively (e.g. a high-unemployment downside fails to
raise PD enough), revisit the macro->Z specification then. Options to consider
*without* discarding crisis data: lagged macro variables (defaults lag the
economy), sign-constrained regression to enforce economic priors, adding a
forbearance/policy proxy, or a richer/again-validated feature set.