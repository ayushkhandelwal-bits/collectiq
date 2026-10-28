# Known Limitations

## Data
* **All data is synthetic.** The LendingClub source could not be accessed through an
  authorised route in the build environment (network policy 403, no Kaggle credentials),
  so no public observations are present. Numbers describe a simulated book.
* The generator is a stylised mechanism (latent risk index, logistic hazards, seasonality,
  a 2025–26 stress ramp). Real portfolios have richer behaviour, macro shocks and
  operational noise.
* Payment timing within a month is not simulated: each account has at most one
  collection event per month and the month's cash is attributed to it.
* Geography is a five-zone label with no simulated effect; currency is INR by convention.

## Collections and causality
* Synthetic collection events are **not evidence of real-world collection effectiveness**.
* Historical actions were assigned at random within stage, which makes action effects
  identifiable in this data. Real collection logs are confounded (collectors choose
  actions based on information the data may not hold); incremental estimates would need
  randomised tests or causal adjustment.
* Contact restrictions, cooling-off periods and caps are configurable demonstration
  assumptions, not a statement of any jurisdiction's law or regulator's guidance.

## Models
* Model A covers only accounts at 1–89 DPD; 90+ and charged-off accounts show observed
  status instead of a prediction.
* Model C (amount) is modest at account level; aggregate predictions are better
  calibrated than individual ones.
* Drift in `months_on_book` reflects a seasoning synthetic book; real retraining cadence
  must be set on real data.

## Forecasting
* About 3 years of monthly history; the validated forecast beats a naive baseline only
  narrowly. Treat projections as indicative.

## Engineering
* SQLite is single-writer; adequate for a prototype, not for concurrent multi-user writes.
* On Streamlit Community Cloud the filesystem is ephemeral: optimisation runs saved from
  the UI persist only until the app restarts.
* Model artifacts are pickles (joblib) and are loaded only after SHA-256 verification
  against the registry written at training time.
