# CollectIQ — Project Blueprint

## Business problem
A lender has thousands of overdue accounts but finite collector hours and an operating
budget. Working every account the same way spends effort on accounts that would pay
anyway and misses accounts where an action changes the outcome. CollectIQ is a
decision-support prototype that connects five steps: understand the book, identify
accounts likely to roll into serious delinquency, estimate what each action could
recover, allocate resources under constraints, and monitor outcomes and models.

It complements **RiskLens** (origination credit-risk scoring) by focusing on the
*collections* decision after an account is already delinquent.

## Target users
Collections and recovery managers, credit-risk and portfolio analysts, lending
operations, data scientists / ML engineers, and senior management.

## Dataset strategy
1. Preferred: LendingClub data on Kaggle. Probed at runtime (`src/ingestion/source_check.py`).
   In the build environment Kaggle was blocked by network policy (HTTP 403) and no
   authorised credentials existed, so the prescribed fallback applied.
2. Fallback (used): a deterministic month-by-month simulation (`seed = 42`) of 50,000
   loans with internally consistent repayment histories, collection events and
   capped recoveries. Every row carries `data_origin = 'SYNTHETIC'`; the schema keeps
   public rows distinguishable if they are added later.

## Data architecture
`raw (data/synthetic/, immutable)` → `staging (data/staging/)` → `data-quality gate`
→ `SQLite (data/processed/collectiq.db)` → `features (data/processed/*.parquet)` →
`models (models/artifacts/, SHA-256 registered)` → `scoring & prioritisation` →
`optimisation` → `Streamlit`. See ARCHITECTURE.md.

## Data schema
Ten tables (DATA_DICTIONARY.md): `loan_accounts`, `repayment_snapshots`,
`collection_events`, `recovery_outcomes`, `portfolio_snapshots`, `pipeline_runs`,
`data_quality_results`, `model_runs`, `optimization_runs`, `optimization_assignments`.

## Modelling objectives
| Model | Question | Unit | Target |
|---|---|---|---|
| A | Which delinquent accounts will roll into serious delinquency? | account-month at 1-89 DPD | 90+ DPD or charge-off within 3 month-ends |
| B | How likely is a payment after each possible action (incl. none)? | collection opportunity | any cash next month |
| C | How much is paid when a payment happens? | paying opportunity | log(1+amount) |

## Evaluation methodology
Chronological train → validation → test splits. Model and threshold selection on
validation only; test reported once. ROC-AUC, PR-AUC, precision, recall, F1, Brier,
calibration, segment-level performance (product, grade, region), monthly stability
and PSI drift. Model C: MAE/RMSE given payment and MAE of the combined expected cash.

## Optimisation objective
Maximise Σ estimated net recovery (incremental expected cash − action cost) over
account-action pairs subject to budget, agent hours, action-specific limits, actions
per account and optional stage coverage. MILP solved with HiGHS (SciPy). Compared with
four baseline rules under identical constraints.

## Dashboard structure
Executive Overview · Portfolio Explorer · Account Intelligence · Collections Workbench ·
Recovery Optimization · Model & Monitoring · Data & Governance · Scenario Lab.

## Quality controls
42 data-quality checks per run (keys, FKs, ranges, dates, reconciliation, leakage);
critical failures block loading and training. Leakage guard on feature lists.
SHA-256 verification before any model load. 82 automated tests.

## Deployment strategy
Streamlit Community Cloud. No secrets required. On first start the app offers to build
the `demo` profile (20,000 loans, ~1 minute); nothing large is committed.

## Assumptions and limitations
See LIMITATIONS.md. In short: all data is synthetic; collection effects are
identifiable only because the simulated historical policy was randomised within stage;
contact rules are demonstration assumptions, not legal guidance.
