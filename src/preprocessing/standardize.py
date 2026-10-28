"""Staging and standardisation: type coercion plus construction of derived tables
(``recovery_outcomes`` and ``portfolio_snapshots``) from the source tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

OPEN_DLQ = ["1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD"]

_INT_COLS = {
    "loan_accounts": ["loan_term_months", "credit_history_length_months",
                      "prior_delinquency_count", "open_credit_accounts"],
    "repayment_snapshots": ["days_past_due", "missed_payment_indicator"],
    "collection_events": ["attempt_number", "promise_to_pay_indicator", "outcome_window_days"],
}


def standardize(loans: pd.DataFrame, snapshots: pd.DataFrame,
                events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Standardise column names / types and return staged frames."""
    frames = {"loan_accounts": loans.copy(), "repayment_snapshots": snapshots.copy(),
              "collection_events": events.copy()}
    for name, df in frames.items():
        df.columns = [c.strip().lower() for c in df.columns]
        for c in _INT_COLS.get(name, []):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in df.columns:
            if c.endswith("_date") or c in ("created_at", "prediction_timestamp"):
                df[c] = df[c].astype("string")
    ev = frames["collection_events"]
    ev["promise_fulfilled_indicator"] = ev["promise_fulfilled_indicator"].astype("Int64")
    return frames


def _remaining_installments(balance: np.ndarray, emi: np.ndarray, annual_rate: np.ndarray
                            ) -> np.ndarray:
    """Number of unpaid instalments implied by an annuity balance."""
    r = annual_rate / 1200.0
    ratio = np.clip(balance * r / emi, 0, 0.999999)
    return np.ceil(-np.log(1 - ratio) / np.log(1 + r) - 1e-6)


def build_recovery_outcomes(events: pd.DataFrame, snapshots: pd.DataFrame,
                            loans: pd.DataFrame) -> pd.DataFrame:
    """One row per collection event and its 30-day outcome window.

    Each simulated month has at most one event per loan and the month's cash is
    attributed to that single event, so event-level amounts sum to account-level
    cash without double counting.
    """
    snap = snapshots[["loan_id", "snapshot_date", "outstanding_principal",
                      "delinquency_bucket"]]
    ev = events.merge(snap, left_on=["loan_id", events["prediction_timestamp"].str[:10]],
                      right_on=["loan_id", "snapshot_date"], how="left")
    ev = ev.merge(loans[["loan_id", "emi_amount", "interest_rate"]], on="loan_id", how="left")
    is_co = ev["delinquency_stage_at_event"] == "Charged Off"
    n_rem = _remaining_installments(ev["outstanding_principal"].to_numpy(float),
                                    ev["emi_amount"].to_numpy(float),
                                    ev["interest_rate"].to_numpy(float))
    contractual = np.round(n_rem * ev["emi_amount"].to_numpy(float), 2)
    opening = np.where(is_co, ev["outstanding_principal"], contractual)
    opening = np.maximum(opening, ev["subsequent_payment_amount"].to_numpy(float))
    start = pd.to_datetime(ev["event_date"])
    out = pd.DataFrame({
        "recovery_outcome_id": "RO" + ev["event_id"].str[2:],
        "loan_id": ev["loan_id"],
        "event_id": ev["event_id"],
        "observation_start_date": start.dt.strftime("%Y-%m-%d"),
        "observation_end_date": (start + pd.Timedelta(days=29)).dt.strftime("%Y-%m-%d"),
        "opening_collectible_balance": np.round(opening, 2),
        "observed_recovery_amount": ev["subsequent_payment_amount"].astype(float),
        "observed_recovery_cost": ev["estimated_action_cost"].astype(float),
        "net_observed_recovery": np.round(ev["subsequent_payment_amount"].astype(float)
                                          - ev["estimated_action_cost"].astype(float), 2),
        "recovery_success_indicator": (ev["subsequent_payment_amount"] > 0).astype(int),
        "outcome_definition": np.where(
            is_co, "Recovery cash on charged-off balance within 30 days of action",
            "Any instalment payment within 30 days of action"),
        "data_origin": ev["data_origin"],
    })
    return out


def build_portfolio_snapshots(snapshots: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly snapshots to date x product x grade x bucket segments.

    Totals reconcile exactly with ``repayment_snapshots`` by construction.
    """
    s = snapshots.merge(loans[["loan_id", "loan_product", "risk_grade", "original_principal"]],
                        on="loan_id", how="left")
    s["is_dlq"] = s["delinquency_bucket"].isin(OPEN_DLQ)
    # A loan is "defaulted in month" on the first Charged Off snapshot.
    first_co = (s[s["delinquency_bucket"] == "Charged Off"]
                .groupby("loan_id")["snapshot_date"].min().rename("first_co_date"))
    s = s.merge(first_co, on="loan_id", how="left")
    s["defaulted_now"] = s["snapshot_date"] == s["first_co_date"]
    s["dlq_exposure"] = np.where(s["is_dlq"], s["outstanding_principal"], 0.0)
    g = (s.groupby(["snapshot_date", "loan_product", "risk_grade", "delinquency_bucket"],
                   observed=True)
         .agg(account_count=("loan_id", "size"),
              original_principal_total=("original_principal", "sum"),
              outstanding_principal_total=("outstanding_principal", "sum"),
              delinquent_exposure=("dlq_exposure", "sum"),
              observed_recovery_total=("payment_received", "sum"),
              defaulted_account_count=("defaulted_now", "sum"))
         .reset_index())
    g["portfolio_snapshot_id"] = (
        "PS-" + g["snapshot_date"].str.replace("-", "").str[:6] + "-"
        + g.index.astype(str).str.zfill(6))
    g["data_origin"] = "SYNTHETIC"
    for c in ["original_principal_total", "outstanding_principal_total", "delinquent_exposure",
              "observed_recovery_total"]:
        g[c] = g[c].round(2)
    g["defaulted_account_count"] = g["defaulted_account_count"].astype(int)
    return g
