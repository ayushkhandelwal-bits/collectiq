# Data Dictionary

Generated from the SQLAlchemy schema metadata in `src/database/schema.py` (the same metadata that creates the database), so it cannot drift from the implementation. The Data & Governance page verifies it against the live database.

**Conventions.** Dates: ISO `YYYY-MM-DD`; timestamps `YYYY-MM-DDTHH:MM:SS` (UTC for audit tables). Money: synthetic Indian Rupees (INR) stored as REAL rupees. Ratios (DTI, utilisation, payment-to-income) are fractions (0.35 = 35%). `interest_rate` is percent per annum (14.5 = 14.5% p.a.). `data_origin` is PUBLIC or SYNTHETIC on every data row.

## `loan_accounts`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `loan_id` | TEXT | no | ✓ |  | Fictional loan identifier |  | Synthetic generator (src/data_generation) |
| `customer_id` | TEXT | no |  |  | Fictional pseudonymous customer identifier |  | Synthetic generator (src/data_generation) |
| `source_dataset` | TEXT | no |  |  | Dataset the record came from |  | Synthetic generator (src/data_generation) |
| `data_origin` | TEXT | no |  |  | Record origin | PUBLIC \| SYNTHETIC | Synthetic generator (src/data_generation) |
| `loan_product` | TEXT | no |  |  | Lending product | Personal, Two-Wheeler, Consumer Durable, Business, Used Car Loan | Synthetic generator (src/data_generation) |
| `loan_purpose` | TEXT | no |  |  | Stated loan purpose |  | Synthetic generator (src/data_generation) |
| `region` | TEXT | yes |  |  | Broad geographic zone (reporting/fairness only; not a model feature) | North \| West \| South \| East \| Central | Synthetic generator (src/data_generation) |
| `risk_grade` | TEXT | yes |  |  | Origination risk grade from bureau-style score | A (best) .. E (worst) | Synthetic generator (src/data_generation) |
| `bureau_score` | FLOAT | yes |  |  | Synthetic bureau-style score at origination | 300-900 | Synthetic generator (src/data_generation) |
| `loan_amount` | FLOAT | no |  |  | Sanctioned loan amount (INR) | > 0 | Synthetic generator (src/data_generation) |
| `annual_income` | FLOAT | yes |  |  | Declared annual income (INR) | > 0 | Synthetic generator (src/data_generation) |
| `employment_length_years` | FLOAT | yes |  |  | Years in current employment (≈4% missing) | 0-35 | Synthetic generator (src/data_generation) |
| `debt_to_income_ratio` | FLOAT | yes |  |  | Total monthly debt obligations incl. this EMI / monthly income (fraction) | 0-0.85 | Synthetic generator (src/data_generation) |
| `payment_to_income_ratio` | FLOAT | yes |  |  | This loan's annual EMI / annual income (fraction) | 0-0.45 | Synthetic generator (src/data_generation) |
| `interest_rate` | FLOAT | no |  |  | Contract interest rate, percent per annum | 9.5-34 | Synthetic generator (src/data_generation) |
| `loan_term_months` | INTEGER | no |  |  | Contractual term in months | 6-60 | Synthetic generator (src/data_generation) |
| `emi_amount` | FLOAT | yes |  |  | Equated monthly instalment (INR) | > 0 | Synthetic generator (src/data_generation) |
| `credit_history_length_months` | INTEGER | yes |  |  | Age of oldest credit line (months) |  | Synthetic generator (src/data_generation) |
| `credit_utilization_ratio` | FLOAT | yes |  |  | Revolving utilisation (fraction) | 0-1 | Synthetic generator (src/data_generation) |
| `prior_delinquency_count` | INTEGER | yes |  |  | Delinquencies on bureau before origination | >= 0 | Synthetic generator (src/data_generation) |
| `open_credit_accounts` | INTEGER | yes |  |  | Open credit lines at origination | >= 1 | Synthetic generator (src/data_generation) |
| `revolving_balance` | FLOAT | yes |  |  | Revolving balance at origination (INR) | >= 0 | Synthetic generator (src/data_generation) |
| `origination_date` | TEXT | no |  |  | Disbursement date |  | Synthetic generator (src/data_generation) |
| `maturity_date` | TEXT | yes |  |  | Contractual maturity date |  | Synthetic generator (src/data_generation) |
| `loan_status` | TEXT | no |  |  | Account status at the as-of date (OUTCOME field) | Current \| Delinquent 1-29/30-59/60-89/90+ \| Charged Off \| Fully Paid | Synthetic generator (src/data_generation) |
| `original_principal` | FLOAT | no |  |  | Principal at origination (INR) |  | Synthetic generator (src/data_generation) |
| `current_outstanding_principal` | FLOAT | yes |  |  | Outstanding principal at as-of date; for charged-off loans the remaining collectible balance (INR) |  | Synthetic generator (src/data_generation) |
| `charge_off_date` | TEXT | yes |  |  | Month-end on which the loan was charged off (OUTCOME) |  | Synthetic generator (src/data_generation) |
| `charged_off_principal` | FLOAT | yes |  |  | Principal balance written off (OUTCOME, INR) |  | Synthetic generator (src/data_generation) |
| `final_recovery_amount` | FLOAT | yes |  |  | Cumulative post-charge-off recoveries to as-of date (OUTCOME - never a predictor) |  | Synthetic generator (src/data_generation) |
| `final_recovery_date` | TEXT | yes |  |  | Month of most recent recovery (OUTCOME) |  | Synthetic generator (src/data_generation) |
| `contact_restriction` | TEXT | yes |  |  | Contact-restriction state at the as-of date | NULL \| Dispute under review \| Hardship assessment hold \| Do-not-call request | Synthetic generator (src/data_generation) |
| `source_record_id` | TEXT | yes |  |  | Identifier in the original public source (NULL for synthetic) |  | Synthetic generator (src/data_generation) |
| `created_at` | TEXT | no |  |  | Record creation timestamp |  | ETL pipeline (src/pipeline) |

## `repayment_snapshots`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `snapshot_id` | TEXT | no | ✓ |  | Snapshot key (loan_id-YYYYMM) |  | Synthetic generator (src/data_generation) |
| `loan_id` | TEXT | no |  | loan_accounts.loan_id | Loan |  | Synthetic generator (src/data_generation) |
| `snapshot_date` | TEXT | no |  |  | Month-end observation date |  | Synthetic generator (src/data_generation) |
| `scheduled_payment` | FLOAT | yes |  |  | Instalment falling due in the month (INR) |  | Synthetic generator (src/data_generation) |
| `payment_received` | FLOAT | yes |  |  | Cash received in the month (INR); recoveries for charged-off accounts |  | Synthetic generator (src/data_generation) |
| `outstanding_principal` | FLOAT | yes |  |  | Principal outstanding at month-end; remaining collectible for charged-off accounts (INR) |  | Synthetic generator (src/data_generation) |
| `days_past_due` | INTEGER | yes |  |  | Days past due at month-end | >= 0 | Synthetic generator (src/data_generation) |
| `delinquency_bucket` | TEXT | yes |  |  | Delinquency bucket at month-end | Current \| 1-29 DPD \| 30-59 DPD \| 60-89 DPD \| 90+ DPD \| Charged Off \| Closed | Synthetic generator (src/data_generation) |
| `missed_payment_indicator` | INTEGER | yes |  |  | 1 if the instalment due in the month was not paid in full | 0 \| 1 | Synthetic generator (src/data_generation) |
| `cumulative_principal_paid` | FLOAT | yes |  |  | Cumulative scheduled principal repaid (INR) |  | Synthetic generator (src/data_generation) |
| `account_status` | TEXT | yes |  |  | Account status at month-end |  | Synthetic generator (src/data_generation) |
| `data_origin` | TEXT | no |  |  | Record origin | PUBLIC \| SYNTHETIC | Synthetic generator (src/data_generation) |

## `collection_events`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `event_id` | TEXT | no | ✓ |  | Collection-event identifier |  | Synthetic generator (src/data_generation) |
| `loan_id` | TEXT | no |  | loan_accounts.loan_id | Loan |  | Synthetic generator (src/data_generation) |
| `event_date` | TEXT | no |  |  | Date the action was taken |  | Synthetic generator (src/data_generation) |
| `prediction_timestamp` | TEXT | yes |  |  | Information cut-off for pre-action features (previous month-end) |  | Synthetic generator (src/data_generation) |
| `delinquency_stage_at_event` | TEXT | yes |  |  | Bucket at the prior month-end |  | Synthetic generator (src/data_generation) |
| `action_type` | TEXT | yes |  |  | Collection action | Reminder notification \| Outbound call \| Follow-up call \| Payment-plan review \| Specialist review | Synthetic generator (src/data_generation) |
| `contact_channel` | TEXT | yes |  |  | Channel | SMS \| Phone | Synthetic generator (src/data_generation) |
| `attempt_number` | INTEGER | yes |  |  | Attempt number in the current delinquency episode |  | Synthetic generator (src/data_generation) |
| `estimated_action_cost` | FLOAT | yes |  |  | Operational cost of the action (INR) | >= 0 | Synthetic generator (src/data_generation) |
| `contact_outcome` | TEXT | yes |  |  | Contact result (POST-ACTION) | Delivered \| Not delivered \| Right-party contact \| Promise to pay \| No contact | Synthetic generator (src/data_generation) |
| `promise_to_pay_indicator` | INTEGER | yes |  |  | Promise to pay obtained (POST-ACTION) |  | Synthetic generator (src/data_generation) |
| `promise_to_pay_amount` | FLOAT | yes |  |  | Promised amount (POST-ACTION, INR) |  | Synthetic generator (src/data_generation) |
| `promise_fulfilled_indicator` | INTEGER | yes |  |  | Promise honoured within window (OUTCOME) |  | Synthetic generator (src/data_generation) |
| `subsequent_payment_amount` | FLOAT | yes |  |  | Cash received in the 30-day outcome window (OUTCOME, INR) |  | Synthetic generator (src/data_generation) |
| `outcome_window_days` | INTEGER | yes |  |  | Outcome window length in days |  | Synthetic generator (src/data_generation) |
| `agent_capacity_category` | TEXT | yes |  |  | Work queue that performs the action |  | Synthetic generator (src/data_generation) |
| `data_origin` | TEXT | no |  |  | Record origin |  | Synthetic generator (src/data_generation) |
| `generation_version` | TEXT | yes |  |  | Synthetic generator version |  | Synthetic generator (src/data_generation) |

## `recovery_outcomes`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `recovery_outcome_id` | TEXT | no | ✓ |  | Outcome identifier |  | ETL pipeline (src/pipeline) |
| `loan_id` | TEXT | no |  | loan_accounts.loan_id | Loan |  | Synthetic generator (src/data_generation) |
| `event_id` | TEXT | yes |  | collection_events.event_id | Collection event whose window this row covers |  | Synthetic generator (src/data_generation) |
| `observation_start_date` | TEXT | no |  |  | Window start (event date) |  | Synthetic generator (src/data_generation) |
| `observation_end_date` | TEXT | no |  |  | Window end (event date + 29 days) |  | Synthetic generator (src/data_generation) |
| `opening_collectible_balance` | FLOAT | yes |  |  | Contractual receivable (EMI x remaining instalments) or remaining charged-off balance at window start (INR) |  | Synthetic generator (src/data_generation) |
| `observed_recovery_amount` | FLOAT | yes |  |  | Cash received in the window (INR) |  | Synthetic generator (src/data_generation) |
| `observed_recovery_cost` | FLOAT | yes |  |  | Cost of the action that opened the window (INR) |  | Synthetic generator (src/data_generation) |
| `net_observed_recovery` | FLOAT | yes |  |  | Recovery minus cost (INR) |  | Synthetic generator (src/data_generation) |
| `recovery_success_indicator` | INTEGER | yes |  |  | 1 if any cash received in window |  | Synthetic generator (src/data_generation) |
| `outcome_definition` | TEXT | yes |  |  | Text definition of the outcome |  | Synthetic generator (src/data_generation) |
| `data_origin` | TEXT | no |  |  | Record origin |  | Synthetic generator (src/data_generation) |

## `portfolio_snapshots`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `portfolio_snapshot_id` | TEXT | no | ✓ |  | Segment snapshot key |  | ETL pipeline (src/pipeline) |
| `snapshot_date` | TEXT | no |  |  | Month-end |  | ETL pipeline (src/pipeline) |
| `loan_product` | TEXT | yes |  |  | Product segment |  | ETL pipeline (src/pipeline) |
| `risk_grade` | TEXT | yes |  |  | Risk-grade segment |  | ETL pipeline (src/pipeline) |
| `delinquency_bucket` | TEXT | yes |  |  | Bucket segment |  | ETL pipeline (src/pipeline) |
| `account_count` | INTEGER | yes |  |  | Accounts in segment |  | ETL pipeline (src/pipeline) |
| `original_principal_total` | FLOAT | yes |  |  | Sum of original principal (INR) |  | ETL pipeline (src/pipeline) |
| `outstanding_principal_total` | FLOAT | yes |  |  | Sum of outstanding / collectible (INR) |  | ETL pipeline (src/pipeline) |
| `delinquent_exposure` | FLOAT | yes |  |  | Outstanding principal in 1+ DPD open buckets (INR) |  | ETL pipeline (src/pipeline) |
| `observed_recovery_total` | FLOAT | yes |  |  | Cash received in month (INR); recoveries for the Charged Off bucket, collections otherwise |  | ETL pipeline (src/pipeline) |
| `defaulted_account_count` | INTEGER | yes |  |  | Accounts charged off during the month |  | ETL pipeline (src/pipeline) |
| `data_origin` | TEXT | yes |  |  | Record origin |  | ETL pipeline (src/pipeline) |

## `pipeline_runs`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `pipeline_run_id` | TEXT | no | ✓ |  | Run identifier |  | ETL pipeline (src/pipeline) |
| `started_at` | TEXT | no |  |  | UTC start |  | ETL pipeline (src/pipeline) |
| `completed_at` | TEXT | yes |  |  | UTC completion |  | ETL pipeline (src/pipeline) |
| `pipeline_status` | TEXT | yes |  |  | RUNNING \| SUCCESS \| FAILED \| BLOCKED |  | ETL pipeline (src/pipeline) |
| `source_dataset` | TEXT | yes |  |  | Source dataset name |  | ETL pipeline (src/pipeline) |
| `source_checksum` | TEXT | yes |  |  | Checksum of raw input files |  | ETL pipeline (src/pipeline) |
| `input_record_count` | INTEGER | yes |  |  | Records read |  | ETL pipeline (src/pipeline) |
| `output_record_count` | INTEGER | yes |  |  | Records loaded |  | ETL pipeline (src/pipeline) |
| `error_summary` | TEXT | yes |  |  | Errors, if any |  | ETL pipeline (src/pipeline) |

## `data_quality_results`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `quality_result_id` | TEXT | no | ✓ |  | Check result identifier |  | ETL pipeline (src/pipeline) |
| `pipeline_run_id` | TEXT | yes |  | pipeline_runs.pipeline_run_id | Pipeline run |  | ETL pipeline (src/pipeline) |
| `table_name` | TEXT | yes |  |  | Table checked |  | ETL pipeline (src/pipeline) |
| `check_name` | TEXT | yes |  |  | Check name |  | ETL pipeline (src/pipeline) |
| `check_status` | TEXT | yes |  |  | PASS \| WARN \| FAIL |  | ETL pipeline (src/pipeline) |
| `affected_record_count` | INTEGER | yes |  |  | Records failing the check |  | ETL pipeline (src/pipeline) |
| `severity` | TEXT | yes |  |  | CRITICAL \| WARNING \| INFO |  | ETL pipeline (src/pipeline) |
| `details` | TEXT | yes |  |  | Human-readable detail |  | ETL pipeline (src/pipeline) |
| `executed_at` | TEXT | yes |  |  | UTC timestamp |  | ETL pipeline (src/pipeline) |

## `model_runs`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `model_run_id` | TEXT | no | ✓ |  | Model run identifier |  | Model training (src/models) |
| `model_name` | TEXT | yes |  |  | Model name |  | Model training (src/models) |
| `model_version` | TEXT | yes |  |  | Version string |  | Model training (src/models) |
| `target_definition` | TEXT | yes |  |  | Target definition |  | Model training (src/models) |
| `training_data_origin` | TEXT | yes |  |  | PUBLIC \| SYNTHETIC \| MIXED |  | Model training (src/models) |
| `training_start_date` | TEXT | yes |  |  | Earliest training observation |  | Model training (src/models) |
| `training_end_date` | TEXT | yes |  |  | Latest training observation |  | Model training (src/models) |
| `training_record_count` | INTEGER | yes |  |  | Training rows |  | Model training (src/models) |
| `validation_record_count` | INTEGER | yes |  |  | Validation rows |  | Model training (src/models) |
| `test_record_count` | INTEGER | yes |  |  | Held-out test rows |  | Model training (src/models) |
| `model_parameters_json` | TEXT | yes |  |  | Hyper-parameters (JSON) |  | Model training (src/models) |
| `metrics_json` | TEXT | yes |  |  | Validation and test metrics (JSON) |  | Model training (src/models) |
| `dataset_version` | TEXT | yes |  |  | Dataset fingerprint |  | Model training (src/models) |
| `artifact_path` | TEXT | yes |  |  | Serialized pipeline path |  | Model training (src/models) |
| `trained_at` | TEXT | yes |  |  | UTC timestamp |  | Model training (src/models) |
| `model_status` | TEXT | yes |  |  | SELECTED \| CHALLENGER |  | Model training (src/models) |

## `optimization_runs`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `optimization_run_id` | TEXT | no | ✓ |  | Run identifier |  | Optimization engine (src/optimization) |
| `run_timestamp` | TEXT | yes |  |  | UTC timestamp |  | Optimization engine (src/optimization) |
| `planning_period_start` | TEXT | yes |  |  | Planning period start |  | Optimization engine (src/optimization) |
| `planning_period_end` | TEXT | yes |  |  | Planning period end |  | Optimization engine (src/optimization) |
| `budget_limit` | FLOAT | yes |  |  | Budget limit (INR) |  | Optimization engine (src/optimization) |
| `available_capacity` | FLOAT | yes |  |  | Agent hours available |  | Optimization engine (src/optimization) |
| `selected_account_count` | INTEGER | yes |  |  | Accounts selected |  | Optimization engine (src/optimization) |
| `estimated_gross_recovery` | FLOAT | yes |  |  | Model-estimated gross recovery (INR) |  | Optimization engine (src/optimization) |
| `estimated_operating_cost` | FLOAT | yes |  |  | Estimated cost (INR) |  | Optimization engine (src/optimization) |
| `estimated_net_recovery` | FLOAT | yes |  |  | Estimated net recovery (INR) |  | Optimization engine (src/optimization) |
| `solver_name` | TEXT | yes |  |  | Solver used |  | Optimization engine (src/optimization) |
| `solver_status` | TEXT | yes |  |  | OPTIMAL \| FEASIBLE_TIME_LIMIT \| HEURISTIC \| INFEASIBLE \| EMPTY \| FAILED |  | Optimization engine (src/optimization) |
| `objective_value` | FLOAT | yes |  |  | Objective value (INR) |  | Optimization engine (src/optimization) |
| `baseline_comparison_json` | TEXT | yes |  |  | Baseline results (JSON) |  | Optimization engine (src/optimization) |
| `assumptions_json` | TEXT | yes |  |  | Constraints and assumptions (JSON) |  | Optimization engine (src/optimization) |

## `optimization_assignments`

| Column | Type | Null | PK | FK | Definition | Allowed values | Provenance |
|---|---|---|---|---|---|---|---|
| `assignment_id` | TEXT | no | ✓ |  | Assignment identifier |  | Optimization engine (src/optimization) |
| `optimization_run_id` | TEXT | no |  | optimization_runs.optimization_run_id | Run |  | Optimization engine (src/optimization) |
| `loan_id` | TEXT | no |  | loan_accounts.loan_id | Loan |  | Optimization engine (src/optimization) |
| `action_type` | TEXT | yes |  |  | Selected action |  | Optimization engine (src/optimization) |
| `estimated_gross_recovery` | FLOAT | yes |  |  | Estimated gross recovery (INR) |  | Optimization engine (src/optimization) |
| `estimated_action_cost` | FLOAT | yes |  |  | Action cost (INR) |  | Optimization engine (src/optimization) |
| `estimated_net_recovery` | FLOAT | yes |  |  | Estimated net (INR) |  | Optimization engine (src/optimization) |
| `required_capacity` | FLOAT | yes |  |  | Agent hours |  | Optimization engine (src/optimization) |
| `assigned_priority` | INTEGER | yes |  |  | 1 = highest priority |  | Optimization engine (src/optimization) |
| `eligibility_status` | TEXT | yes |  |  | Eligibility at planning time |  | Optimization engine (src/optimization) |
| `recommendation_reason` | TEXT | yes |  |  | Rationale |  | Optimization engine (src/optimization) |
