"""SQLAlchemy schema for the CollectIQ SQLite database.

Every column carries ``info`` metadata (definition, allowed values, provenance) so
that the data dictionary is generated from the schema itself and cannot drift
from the implementation.

Conventions
-----------
* Dates are ISO-8601 text ``YYYY-MM-DD``; timestamps ``YYYY-MM-DDTHH:MM:SS``.
* Money: synthetic Indian Rupees (INR), stored as REAL rupees (2 dp).
* Ratios (DTI, utilisation, payment-to-income) are stored as fractions (0.35 = 35%).
* ``interest_rate`` is stored in percent per annum (14.5 = 14.5% p.a.).
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()

SYN = "Synthetic generator (src/data_generation)"
PIPE = "ETL pipeline (src/pipeline)"
MODEL = "Model training (src/models)"
OPT = "Optimization engine (src/optimization)"


def col(name: str, type_, definition: str, *args, nullable: bool = True,
        allowed: str = "", provenance: str = SYN, **kwargs) -> Column:
    """Build a column with dictionary metadata attached."""
    return Column(name, type_, *args, nullable=nullable, comment=definition,
                  info={"definition": definition, "allowed": allowed,
                        "provenance": provenance}, **kwargs)


ORIGINS = "data_origin IN ('PUBLIC','SYNTHETIC')"

loan_accounts = Table(
    "loan_accounts", metadata,
    col("loan_id", Text, "Fictional loan identifier", primary_key=True, nullable=False),
    col("customer_id", Text, "Fictional pseudonymous customer identifier", nullable=False),
    col("source_dataset", Text, "Dataset the record came from", nullable=False),
    col("data_origin", Text, "Record origin", nullable=False, allowed="PUBLIC | SYNTHETIC"),
    col("loan_product", Text, "Lending product", nullable=False,
        allowed="Personal, Two-Wheeler, Consumer Durable, Business, Used Car Loan"),
    col("loan_purpose", Text, "Stated loan purpose", nullable=False),
    col("region", Text, "Broad geographic zone (reporting/fairness only; not a model feature)",
        allowed="North | West | South | East | Central"),
    col("risk_grade", Text, "Origination risk grade from bureau-style score",
        allowed="A (best) .. E (worst)"),
    col("bureau_score", Float, "Synthetic bureau-style score at origination", allowed="300-900"),
    col("loan_amount", Float, "Sanctioned loan amount (INR)", nullable=False, allowed="> 0"),
    col("annual_income", Float, "Declared annual income (INR)", allowed="> 0"),
    col("employment_length_years", Float, "Years in current employment (≈4% missing)",
        allowed="0-35"),
    col("debt_to_income_ratio", Float, "Total monthly debt obligations incl. this EMI / "
        "monthly income (fraction)", allowed="0-0.85"),
    col("payment_to_income_ratio", Float, "This loan's annual EMI / annual income (fraction)",
        allowed="0-0.45"),
    col("interest_rate", Float, "Contract interest rate, percent per annum", nullable=False,
        allowed="9.5-34"),
    col("loan_term_months", Integer, "Contractual term in months", nullable=False,
        allowed="6-60"),
    col("emi_amount", Float, "Equated monthly instalment (INR)", allowed="> 0"),
    col("credit_history_length_months", Integer, "Age of oldest credit line (months)"),
    col("credit_utilization_ratio", Float, "Revolving utilisation (fraction)", allowed="0-1"),
    col("prior_delinquency_count", Integer, "Delinquencies on bureau before origination",
        allowed=">= 0"),
    col("open_credit_accounts", Integer, "Open credit lines at origination", allowed=">= 1"),
    col("revolving_balance", Float, "Revolving balance at origination (INR)", allowed=">= 0"),
    col("origination_date", Text, "Disbursement date", nullable=False),
    col("maturity_date", Text, "Contractual maturity date"),
    col("loan_status", Text, "Account status at the as-of date (OUTCOME field)", nullable=False,
        allowed="Current | Delinquent 1-29/30-59/60-89/90+ | Charged Off | Fully Paid"),
    col("original_principal", Float, "Principal at origination (INR)", nullable=False),
    col("current_outstanding_principal", Float, "Outstanding principal at as-of date; for "
        "charged-off loans the remaining collectible balance (INR)"),
    col("charge_off_date", Text, "Month-end on which the loan was charged off (OUTCOME)"),
    col("charged_off_principal", Float, "Principal balance written off (OUTCOME, INR)"),
    col("final_recovery_amount", Float, "Cumulative post-charge-off recoveries to as-of date "
        "(OUTCOME - never a predictor)"),
    col("final_recovery_date", Text, "Month of most recent recovery (OUTCOME)"),
    col("contact_restriction", Text, "Contact-restriction state at the as-of date",
        allowed="NULL | Dispute under review | Hardship assessment hold | Do-not-call request"),
    col("source_record_id", Text, "Identifier in the original public source (NULL for synthetic)"),
    col("created_at", Text, "Record creation timestamp", nullable=False, provenance=PIPE),
    CheckConstraint(ORIGINS, name="ck_loan_origin"),
    CheckConstraint("loan_amount > 0 AND original_principal > 0", name="ck_loan_amount_pos"),
    CheckConstraint("current_outstanding_principal IS NULL OR current_outstanding_principal >= 0",
                    name="ck_loan_outstanding_nonneg"),
    CheckConstraint("final_recovery_amount IS NULL OR final_recovery_amount >= 0",
                    name="ck_loan_recovery_nonneg"),
)
Index("ix_loan_status", loan_accounts.c.loan_status)
Index("ix_loan_origination", loan_accounts.c.origination_date)
Index("ix_loan_product", loan_accounts.c.loan_product)

repayment_snapshots = Table(
    "repayment_snapshots", metadata,
    col("snapshot_id", Text, "Snapshot key (loan_id-YYYYMM)", primary_key=True, nullable=False),
    col("loan_id", Text, "Loan", ForeignKey("loan_accounts.loan_id"), nullable=False),
    col("snapshot_date", Text, "Month-end observation date", nullable=False),
    col("scheduled_payment", Float, "Instalment falling due in the month (INR)"),
    col("payment_received", Float, "Cash received in the month (INR); recoveries for "
        "charged-off accounts"),
    col("outstanding_principal", Float, "Principal outstanding at month-end; remaining "
        "collectible for charged-off accounts (INR)"),
    col("days_past_due", Integer, "Days past due at month-end", allowed=">= 0"),
    col("delinquency_bucket", Text, "Delinquency bucket at month-end",
        allowed="Current | 1-29 DPD | 30-59 DPD | 60-89 DPD | 90+ DPD | Charged Off | Closed"),
    col("missed_payment_indicator", Integer, "1 if the instalment due in the month was not "
        "paid in full", allowed="0 | 1"),
    col("cumulative_principal_paid", Float, "Cumulative scheduled principal repaid (INR)"),
    col("account_status", Text, "Account status at month-end"),
    col("data_origin", Text, "Record origin", nullable=False, allowed="PUBLIC | SYNTHETIC"),
    UniqueConstraint("loan_id", "snapshot_date", name="uq_snapshot_loan_date"),
    CheckConstraint(ORIGINS, name="ck_snap_origin"),
    CheckConstraint("days_past_due >= 0", name="ck_snap_dpd"),
    CheckConstraint("payment_received >= 0 AND outstanding_principal >= 0", name="ck_snap_amt"),
)
Index("ix_snap_loan", repayment_snapshots.c.loan_id)
Index("ix_snap_date", repayment_snapshots.c.snapshot_date)
Index("ix_snap_bucket", repayment_snapshots.c.delinquency_bucket)

collection_events = Table(
    "collection_events", metadata,
    col("event_id", Text, "Collection-event identifier", primary_key=True, nullable=False),
    col("loan_id", Text, "Loan", ForeignKey("loan_accounts.loan_id"), nullable=False),
    col("event_date", Text, "Date the action was taken", nullable=False),
    col("prediction_timestamp", Text, "Information cut-off for pre-action features "
        "(previous month-end)"),
    col("delinquency_stage_at_event", Text, "Bucket at the prior month-end"),
    col("action_type", Text, "Collection action",
        allowed="Reminder notification | Outbound call | Follow-up call | "
                "Payment-plan review | Specialist review"),
    col("contact_channel", Text, "Channel", allowed="SMS | Phone"),
    col("attempt_number", Integer, "Attempt number in the current delinquency episode"),
    col("estimated_action_cost", Float, "Operational cost of the action (INR)", allowed=">= 0"),
    col("contact_outcome", Text, "Contact result (POST-ACTION)",
        allowed="Delivered | Not delivered | Right-party contact | Promise to pay | No contact"),
    col("promise_to_pay_indicator", Integer, "Promise to pay obtained (POST-ACTION)"),
    col("promise_to_pay_amount", Float, "Promised amount (POST-ACTION, INR)"),
    col("promise_fulfilled_indicator", Integer, "Promise honoured within window (OUTCOME)"),
    col("subsequent_payment_amount", Float, "Cash received in the 30-day outcome window "
        "(OUTCOME, INR)"),
    col("outcome_window_days", Integer, "Outcome window length in days"),
    col("agent_capacity_category", Text, "Work queue that performs the action"),
    col("data_origin", Text, "Record origin", nullable=False),
    col("generation_version", Text, "Synthetic generator version"),
    CheckConstraint(ORIGINS, name="ck_event_origin"),
    CheckConstraint("estimated_action_cost >= 0", name="ck_event_cost"),
    CheckConstraint("subsequent_payment_amount IS NULL OR subsequent_payment_amount >= 0",
                    name="ck_event_payment"),
)
Index("ix_event_loan", collection_events.c.loan_id)
Index("ix_event_date", collection_events.c.event_date)

recovery_outcomes = Table(
    "recovery_outcomes", metadata,
    col("recovery_outcome_id", Text, "Outcome identifier", primary_key=True, nullable=False,
        provenance=PIPE),
    col("loan_id", Text, "Loan", ForeignKey("loan_accounts.loan_id"), nullable=False),
    col("event_id", Text, "Collection event whose window this row covers",
        ForeignKey("collection_events.event_id")),
    col("observation_start_date", Text, "Window start (event date)", nullable=False),
    col("observation_end_date", Text, "Window end (event date + 29 days)", nullable=False),
    col("opening_collectible_balance", Float, "Contractual receivable (EMI x remaining "
        "instalments) or remaining charged-off balance at window start (INR)"),
    col("observed_recovery_amount", Float, "Cash received in the window (INR)"),
    col("observed_recovery_cost", Float, "Cost of the action that opened the window (INR)"),
    col("net_observed_recovery", Float, "Recovery minus cost (INR)"),
    col("recovery_success_indicator", Integer, "1 if any cash received in window"),
    col("outcome_definition", Text, "Text definition of the outcome"),
    col("data_origin", Text, "Record origin", nullable=False),
    CheckConstraint("observed_recovery_amount >= 0", name="ck_rec_amount"),
    CheckConstraint("observed_recovery_amount <= opening_collectible_balance + 0.01",
                    name="ck_rec_cap"),
)
Index("ix_rec_loan", recovery_outcomes.c.loan_id)

portfolio_snapshots = Table(
    "portfolio_snapshots", metadata,
    col("portfolio_snapshot_id", Text, "Segment snapshot key", primary_key=True, nullable=False,
        provenance=PIPE),
    col("snapshot_date", Text, "Month-end", nullable=False, provenance=PIPE),
    col("loan_product", Text, "Product segment", provenance=PIPE),
    col("risk_grade", Text, "Risk-grade segment", provenance=PIPE),
    col("delinquency_bucket", Text, "Bucket segment", provenance=PIPE),
    col("account_count", Integer, "Accounts in segment", provenance=PIPE),
    col("original_principal_total", Float, "Sum of original principal (INR)", provenance=PIPE),
    col("outstanding_principal_total", Float, "Sum of outstanding / collectible (INR)",
        provenance=PIPE),
    col("delinquent_exposure", Float, "Outstanding principal in 1+ DPD open buckets (INR)",
        provenance=PIPE),
    col("observed_recovery_total", Float, "Cash received in month (INR); recoveries for the "
        "Charged Off bucket, collections otherwise", provenance=PIPE),
    col("defaulted_account_count", Integer, "Accounts charged off during the month",
        provenance=PIPE),
    col("data_origin", Text, "Record origin", provenance=PIPE),
)
Index("ix_port_date", portfolio_snapshots.c.snapshot_date)

pipeline_runs = Table(
    "pipeline_runs", metadata,
    col("pipeline_run_id", Text, "Run identifier", primary_key=True, nullable=False,
        provenance=PIPE),
    col("started_at", Text, "UTC start", nullable=False, provenance=PIPE),
    col("completed_at", Text, "UTC completion", provenance=PIPE),
    col("pipeline_status", Text, "RUNNING | SUCCESS | FAILED | BLOCKED", provenance=PIPE),
    col("source_dataset", Text, "Source dataset name", provenance=PIPE),
    col("source_checksum", Text, "Checksum of raw input files", provenance=PIPE),
    col("input_record_count", Integer, "Records read", provenance=PIPE),
    col("output_record_count", Integer, "Records loaded", provenance=PIPE),
    col("error_summary", Text, "Errors, if any", provenance=PIPE),
)

data_quality_results = Table(
    "data_quality_results", metadata,
    col("quality_result_id", Text, "Check result identifier", primary_key=True, nullable=False,
        provenance=PIPE),
    col("pipeline_run_id", Text, "Pipeline run", ForeignKey("pipeline_runs.pipeline_run_id"),
        provenance=PIPE),
    col("table_name", Text, "Table checked", provenance=PIPE),
    col("check_name", Text, "Check name", provenance=PIPE),
    col("check_status", Text, "PASS | WARN | FAIL", provenance=PIPE),
    col("affected_record_count", Integer, "Records failing the check", provenance=PIPE),
    col("severity", Text, "CRITICAL | WARNING | INFO", provenance=PIPE),
    col("details", Text, "Human-readable detail", provenance=PIPE),
    col("executed_at", Text, "UTC timestamp", provenance=PIPE),
)
Index("ix_dq_run", data_quality_results.c.pipeline_run_id)

model_runs = Table(
    "model_runs", metadata,
    col("model_run_id", Text, "Model run identifier", primary_key=True, nullable=False,
        provenance=MODEL),
    col("model_name", Text, "Model name", provenance=MODEL),
    col("model_version", Text, "Version string", provenance=MODEL),
    col("target_definition", Text, "Target definition", provenance=MODEL),
    col("training_data_origin", Text, "PUBLIC | SYNTHETIC | MIXED", provenance=MODEL),
    col("training_start_date", Text, "Earliest training observation", provenance=MODEL),
    col("training_end_date", Text, "Latest training observation", provenance=MODEL),
    col("training_record_count", Integer, "Training rows", provenance=MODEL),
    col("validation_record_count", Integer, "Validation rows", provenance=MODEL),
    col("test_record_count", Integer, "Held-out test rows", provenance=MODEL),
    col("model_parameters_json", Text, "Hyper-parameters (JSON)", provenance=MODEL),
    col("metrics_json", Text, "Validation and test metrics (JSON)", provenance=MODEL),
    col("dataset_version", Text, "Dataset fingerprint", provenance=MODEL),
    col("artifact_path", Text, "Serialized pipeline path", provenance=MODEL),
    col("trained_at", Text, "UTC timestamp", provenance=MODEL),
    col("model_status", Text, "SELECTED | CHALLENGER", provenance=MODEL),
)

optimization_runs = Table(
    "optimization_runs", metadata,
    col("optimization_run_id", Text, "Run identifier", primary_key=True, nullable=False,
        provenance=OPT),
    col("run_timestamp", Text, "UTC timestamp", provenance=OPT),
    col("planning_period_start", Text, "Planning period start", provenance=OPT),
    col("planning_period_end", Text, "Planning period end", provenance=OPT),
    col("budget_limit", Float, "Budget limit (INR)", provenance=OPT),
    col("available_capacity", Float, "Agent hours available", provenance=OPT),
    col("selected_account_count", Integer, "Accounts selected", provenance=OPT),
    col("estimated_gross_recovery", Float, "Model-estimated gross recovery (INR)",
        provenance=OPT),
    col("estimated_operating_cost", Float, "Estimated cost (INR)", provenance=OPT),
    col("estimated_net_recovery", Float, "Estimated net recovery (INR)", provenance=OPT),
    col("solver_name", Text, "Solver used", provenance=OPT),
    col("solver_status", Text, "OPTIMAL | FEASIBLE_TIME_LIMIT | HEURISTIC | INFEASIBLE | "
        "EMPTY | FAILED", provenance=OPT),
    col("objective_value", Float, "Objective value (INR)", provenance=OPT),
    col("baseline_comparison_json", Text, "Baseline results (JSON)", provenance=OPT),
    col("assumptions_json", Text, "Constraints and assumptions (JSON)", provenance=OPT),
)

optimization_assignments = Table(
    "optimization_assignments", metadata,
    col("assignment_id", Text, "Assignment identifier", primary_key=True, nullable=False,
        provenance=OPT),
    col("optimization_run_id", Text, "Run",
        ForeignKey("optimization_runs.optimization_run_id"), nullable=False, provenance=OPT),
    col("loan_id", Text, "Loan", ForeignKey("loan_accounts.loan_id"), nullable=False,
        provenance=OPT),
    col("action_type", Text, "Selected action", provenance=OPT),
    col("estimated_gross_recovery", Float, "Estimated gross recovery (INR)", provenance=OPT),
    col("estimated_action_cost", Float, "Action cost (INR)", provenance=OPT),
    col("estimated_net_recovery", Float, "Estimated net (INR)", provenance=OPT),
    col("required_capacity", Float, "Agent hours", provenance=OPT),
    col("assigned_priority", Integer, "1 = highest priority", provenance=OPT),
    col("eligibility_status", Text, "Eligibility at planning time", provenance=OPT),
    col("recommendation_reason", Text, "Rationale", provenance=OPT),
    UniqueConstraint("optimization_run_id", "loan_id", "action_type",
                     name="uq_assignment_run_loan_action"),
)
Index("ix_assign_run", optimization_assignments.c.optimization_run_id)

CORE_TABLES = ["loan_accounts", "repayment_snapshots", "collection_events",
               "recovery_outcomes", "portfolio_snapshots"]
