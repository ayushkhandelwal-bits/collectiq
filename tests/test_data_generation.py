"""Synthetic generator: keys, constraints, chronology, prevalence and reproducibility."""

from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data_generation.generate_data import (
    ACTIONS,
    BUCKETS,
    LOAN_STATUSES,
    annuity_payment,
    generate,
    remaining_principal,
)


def test_reproducible_with_same_seed():
    a = generate(n_loans=800, seed=11)
    b = generate(n_loans=800, seed=11)
    assert_frame_equal(a.loans, b.loans)
    assert_frame_equal(a.snapshots, b.snapshots)
    assert_frame_equal(a.events, b.events)


def test_different_seed_differs():
    a = generate(n_loans=500, seed=1)
    b = generate(n_loans=500, seed=2)
    assert not a.loans["loan_amount"].equals(b.loans["loan_amount"])


def test_primary_and_foreign_keys(small_generation):
    g = small_generation
    assert g.loans["loan_id"].is_unique
    assert g.snapshots["snapshot_id"].is_unique
    assert g.events["event_id"].is_unique
    assert not g.snapshots.duplicated(["loan_id", "snapshot_date"]).any()
    ids = set(g.loans["loan_id"])
    assert g.snapshots["loan_id"].isin(ids).all()
    assert g.events["loan_id"].isin(ids).all()


def test_no_personal_identifiers(small_generation):
    cols = set(small_generation.loans.columns)
    for forbidden in ["name", "address", "phone", "email", "account_number"]:
        assert not any(forbidden in c for c in cols)
    assert small_generation.loans["loan_id"].str.match(r"^LN\d{7}$").all()


def test_financial_amounts_non_negative(small_generation):
    g = small_generation
    assert (g.loans[["loan_amount", "annual_income", "emi_amount"]] > 0).all().all()
    assert (g.loans["current_outstanding_principal"] >= 0).all()
    assert (g.snapshots[["payment_received", "outstanding_principal", "scheduled_payment"]]
            >= 0).all().all()
    assert (g.events["estimated_action_cost"] >= 0).all()
    assert (g.events["subsequent_payment_amount"] >= 0).all()


def test_ratios_bounded(small_generation):
    L = small_generation.loans
    assert L["debt_to_income_ratio"].between(0, 0.85).all()
    assert L["payment_to_income_ratio"].between(0, 0.45 + 1e-6).all()
    assert L["credit_utilization_ratio"].between(0, 1).all()


def test_valid_categorical_states(small_generation):
    g = small_generation
    assert g.loans["loan_status"].isin(LOAN_STATUSES).all()
    assert g.snapshots["delinquency_bucket"].isin(BUCKETS).all()
    assert g.events["action_type"].isin(ACTIONS).all()
    assert set(g.loans["risk_grade"]) <= set("ABCDE")


def test_plausible_target_prevalence(small_generation):
    L = small_generation.loans
    co = (L["loan_status"] == "Charged Off").mean()
    assert 0.01 < co < 0.20
    succ = (small_generation.events["subsequent_payment_amount"] > 0).mean()
    assert 0.10 < succ < 0.90


def test_missing_values_only_where_documented(small_generation):
    L = small_generation.loans
    assert L["employment_length_years"].isna().mean() < 0.08
    required = ["loan_id", "loan_amount", "interest_rate", "origination_date", "loan_status"]
    assert not L[required].isna().any().any()
    co = L["loan_status"] == "Charged Off"
    assert L.loc[co, "charged_off_principal"].notna().all()
    assert L.loc[~co, "final_recovery_amount"].isna().all()


def test_recovery_constraints(small_generation):
    L = small_generation.loans
    co = L[L["loan_status"] == "Charged Off"]
    assert (co["final_recovery_amount"] <= co["charged_off_principal"] + 0.01).all()
    assert (co["current_outstanding_principal"]
            - (co["charged_off_principal"] - co["final_recovery_amount"])).abs().max() < 0.05


def test_chronological_consistency(small_generation):
    g = small_generation
    s = g.snapshots.merge(g.loans[["loan_id", "origination_date"]], on="loan_id")
    assert (s["snapshot_date"] > s["origination_date"]).all()
    e = g.events.merge(g.loans[["loan_id", "origination_date"]], on="loan_id")
    assert (e["event_date"] > e["origination_date"]).all()
    assert (g.events["prediction_timestamp"].str[:10] < g.events["event_date"]).all()
    assert (g.loans["maturity_date"] > g.loans["origination_date"]).all()


def test_history_consistent_with_final_status(small_generation):
    g = small_generation
    last = g.snapshots.sort_values("snapshot_date").groupby("loan_id").tail(1)
    m = last.merge(g.loans[["loan_id", "loan_status"]], on="loan_id")
    assert (m["account_status"] == m["loan_status"]).all()
    paid = g.loans[g.loans["loan_status"] == "Fully Paid"]
    assert (paid["current_outstanding_principal"] == 0).all()


def test_event_payments_match_snapshot_cash(small_generation):
    g = small_generation
    s = g.snapshots.copy()
    s["month"] = s["snapshot_date"].str[:7]
    e = g.events.copy()
    e["month"] = e["event_date"].str[:7]
    m = e.merge(s[["loan_id", "month", "payment_received"]], on=["loan_id", "month"])
    assert len(m) == len(e)
    assert (m["subsequent_payment_amount"] - m["payment_received"]).abs().max() < 0.01


def test_annuity_helpers():
    import numpy as np

    p, r, n = np.array([100000.0]), np.array([0.01]), np.array([12])
    emi = annuity_payment(p, r, n)
    assert abs(remaining_principal(p, r, n, np.array([0]))[0] - 100000) < 1e-6
    assert remaining_principal(p, r, n, np.array([12]))[0] < 1e-6
    assert abs(emi[0] * 12 - (100000 + 6618.55)) < 5


def test_dates_parse(small_generation):
    for col in ["origination_date", "maturity_date"]:
        assert pd.to_datetime(small_generation.loans[col], errors="coerce").notna().all()
