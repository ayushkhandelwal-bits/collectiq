"""Source-to-target field mapping (addendum Part B).

For the synthetic source the generator emits standardised names, so source and
target names coincide; the mapping still records role, timing and whether each
field may be used by the deployed prediction pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.features.build_features import PROHIBITED_FEATURES, RECOVERY_NUMERIC, RISK_NUMERIC

# (table, column, unit / valid range, role, available_at_prediction, transformation)
FIELDS: list[tuple[str, str, str, str, str, str]] = [
    ("loan_accounts", "loan_id", "LN#######", "identifier", "yes", "none"),
    ("loan_accounts", "customer_id", "CU#######", "identifier", "yes", "none (fictional)"),
    ("loan_accounts", "loan_product", "category", "feature", "yes", "one-hot"),
    ("loan_accounts", "loan_purpose", "category", "reporting", "yes", "none"),
    ("loan_accounts", "region", "category", "reporting / fairness", "yes",
     "excluded from models by policy"),
    ("loan_accounts", "risk_grade", "A-E", "feature", "yes", "one-hot"),
    ("loan_accounts", "bureau_score", "300-900", "feature", "yes", "median impute"),
    ("loan_accounts", "loan_amount", "INR > 0", "feature", "yes", "log"),
    ("loan_accounts", "annual_income", "INR > 0", "feature", "yes", "log"),
    ("loan_accounts", "employment_length_years", "0-35 years", "feature", "yes",
     "median impute + missing indicator"),
    ("loan_accounts", "debt_to_income_ratio", "fraction 0-0.85", "feature", "yes", "none"),
    ("loan_accounts", "payment_to_income_ratio", "fraction 0-0.45", "feature", "yes", "none"),
    ("loan_accounts", "interest_rate", "% p.a. 9.5-34", "feature", "yes", "none"),
    ("loan_accounts", "loan_term_months", "6-60", "feature", "yes", "none"),
    ("loan_accounts", "emi_amount", "INR > 0", "feature", "yes", "log"),
    ("loan_accounts", "credit_history_length_months", "6-360", "feature", "yes", "none"),
    ("loan_accounts", "credit_utilization_ratio", "fraction 0-1", "feature", "yes", "none"),
    ("loan_accounts", "prior_delinquency_count", ">= 0", "feature", "yes", "none"),
    ("loan_accounts", "open_credit_accounts", ">= 1", "feature", "yes", "none"),
    ("loan_accounts", "revolving_balance", "INR >= 0", "reporting", "yes", "none"),
    ("loan_accounts", "origination_date", "date", "reporting / feature source", "yes",
     "months on book"),
    ("loan_accounts", "loan_status", "category", "outcome (as-of)", "NO", "never a feature"),
    ("loan_accounts", "current_outstanding_principal", "INR", "outcome (as-of)", "NO",
     "never a feature"),
    ("loan_accounts", "charged_off_principal", "INR", "outcome", "NO", "never a feature"),
    ("loan_accounts", "final_recovery_amount", "INR", "outcome", "NO", "never a feature"),
    ("loan_accounts", "contact_restriction", "category", "policy / eligibility", "yes",
     "eligibility rule only"),
    ("repayment_snapshots", "days_past_due", ">= 0", "feature (at t)", "yes", "none"),
    ("repayment_snapshots", "delinquency_bucket", "category", "feature (at t) / target (t+1..3)",
     "at t only", "ordinal; future values form the Model A target"),
    ("repayment_snapshots", "missed_payment_indicator", "0/1", "feature source", "yes",
     "rolling 3m/6m sums"),
    ("repayment_snapshots", "payment_received", "INR", "feature source", "yes (<= t)",
     "3m payment ratio; months since last payment"),
    ("repayment_snapshots", "scheduled_payment", "INR", "feature source", "yes", "ratio"),
    ("repayment_snapshots", "outstanding_principal", "INR", "feature (at t)", "yes",
     "ratio to original; log"),
    ("collection_events", "action_type", "category", "feature (decision)", "yes", "one-hot"),
    ("collection_events", "attempt_number", ">= 1", "feature", "yes", "none"),
    ("collection_events", "delinquency_stage_at_event", "category", "feature", "yes", "one-hot"),
    ("collection_events", "estimated_action_cost", "INR >= 0", "cost input", "yes", "none"),
    ("collection_events", "contact_outcome", "category", "post-action", "NO", "never a feature"),
    ("collection_events", "promise_to_pay_indicator", "0/1", "post-action / history",
     "only prior closed windows", "prior PTP count"),
    ("collection_events", "promise_fulfilled_indicator", "0/1", "outcome / history",
     "only prior closed windows", "prior fulfilment rate"),
    ("collection_events", "subsequent_payment_amount", "INR >= 0", "target (Models B, C)", "NO",
     "binary target and log1p amount target"),
]


def mapping_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mapping with measured missing-value rates and pipeline permission."""
    rows = []
    model_feats = set(RISK_NUMERIC) | set(RECOVERY_NUMERIC) | {
        "loan_product", "risk_grade", "action_type", "stage"}
    for table, col, unit, role, avail, transform in FIELDS:
        df = frames.get(table)
        miss = float(df[col].isna().mean()) if df is not None and col in df.columns else None
        permitted = "no" if (col in PROHIBITED_FEATURES or avail == "NO") else "yes"
        rows.append({"source_table": table, "source_column": col, "standardized_column": col,
                     "dtype": str(df[col].dtype) if df is not None and col in df.columns else "",
                     "unit_valid_range": unit, "missing_rate": miss, "provenance": "SYNTHETIC",
                     "role": role, "available_at_prediction_time": avail,
                     "transformation": transform,
                     "permitted_in_deployed_pipeline": permitted,
                     "direct_model_feature": col in model_feats})
    return pd.DataFrame(rows)


def mapping_markdown(df: pd.DataFrame) -> str:
    """Render the mapping as Markdown."""
    lines = [
        "# Source-to-Target Mapping", "",
        "Generated by the pipeline from `src/preprocessing/mapping.py` with measured "
        "missing-value rates. Source: synthetic generator (no public data was available; "
        "see DATA_SOURCES.md). Source and standardised names coincide because the "
        "generator emits the standard schema.", "",
        "| Table | Column | Type | Unit / range | Missing | Role | Available at prediction "
        "| Transformation | Permitted in deployed pipeline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples(index=False):
        miss = "" if r.missing_rate is None else f"{r.missing_rate:.1%}"
        lines.append(f"| {r.source_table} | `{r.source_column}` | {r.dtype} | "
                     f"{r.unit_valid_range} | {miss} | {r.role} | "
                     f"{r.available_at_prediction_time} | {r.transformation} | "
                     f"{r.permitted_in_deployed_pipeline} |")
    lines += ["", "Engineered features are defined in FEATURE_DICTIONARY.md."]
    return "\n".join(lines) + "\n"
