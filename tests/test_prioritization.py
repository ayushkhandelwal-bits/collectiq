"""Prioritisation engine: eligibility rules, value definition and ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import path
from src.prioritization.engine import Policy, apply_policy, eligible_candidates


@pytest.fixture(scope="module")
def scored(built):
    pop = pd.read_parquet(path("processed_dir") / "collections_population.parquet")
    cand = pd.read_parquet(path("processed_dir") / "action_candidates.parquet")
    return pop, cand


def _pop_row(**kw):
    base = {"loan_id": "LX1", "delinquency_bucket": "30-59 DPD", "days_past_due": 40,
            "outstanding_principal": 10000.0, "contact_restriction": None,
            "days_since_last_contact": 20.0, "contacts_last_30d": 0, "risk_probability": 0.3,
            "risk_note": "Model estimate", "risk_drivers": "", "p_payment_no_action": 0.4,
            "expected_cash_no_action": 1000.0}
    base.update(kw)
    return base


def _cand(loan_id, action, net):
    return {"loan_id": loan_id, "stage": "30-59 DPD", "action_type": action, "p_payment": 0.6,
            "amount_if_paid": 5000.0, "expected_cash_with_action": 3000.0,
            "p_payment_no_action": 0.4, "expected_cash_no_action": 1000.0,
            "expected_gross_recovery": net + 100, "action_cost": 100.0, "agent_hours": 0.25,
            "capacity_category": "x", "expected_net_recovery": net}


def test_eligibility_rules():
    pol = Policy(cooling_off_days=7, max_contacts_30d=3,
                 restricted_states=["Dispute under review"],
                 digital_only_states=["Do-not-call request"])
    pop = pd.DataFrame([
        _pop_row(loan_id="A"),
        _pop_row(loan_id="B", contact_restriction="Dispute under review"),
        _pop_row(loan_id="C", days_since_last_contact=3.0),
        _pop_row(loan_id="D", contacts_last_30d=3),
        _pop_row(loan_id="E", contact_restriction="Do-not-call request"),
        _pop_row(loan_id="F", outstanding_principal=0.0),
    ])
    cand = pd.DataFrame([_cand(lid, a, 500.0) for lid in "ABCDEF"
                         for a in ["Reminder notification", "Outbound call"]])
    q = apply_policy(pop, cand, pol).set_index("loan_id")
    assert q.loc["A", "eligible"]
    assert q.loc["B", "eligibility_status"].startswith("Restricted")
    assert q.loc["C", "eligibility_status"].startswith("Cooling-off")
    assert q.loc["D", "eligibility_status"].startswith("Contact cap")
    assert q.loc["E", "eligibility_status"] == "Eligible: digital reminders only"
    assert q.loc["E", "recommended_action"] == "Reminder notification"
    assert q.loc["F", "eligibility_status"].startswith("Ineligible")
    for lid in "BCDF":
        assert q.loc[lid, "recommended_action"] == "No additional contact until eligible"
        assert np.isnan(q.loc[lid, "priority_score"])


def test_no_contact_when_value_not_positive():
    pop = pd.DataFrame([_pop_row(loan_id="A")])
    cand = pd.DataFrame([_cand("A", "Outbound call", -50.0)])
    q = apply_policy(pop, cand, Policy())
    assert q.iloc[0]["recommended_action"] == "No additional contact this period"
    assert q.iloc[0]["priority_category"] == "Low value"


def test_value_definition_is_incremental(scored):
    _, cand = scored
    gross = cand["expected_cash_with_action"] - cand["expected_cash_no_action"]
    assert np.allclose(cand["expected_gross_recovery"], gross)
    assert np.allclose(cand["expected_net_recovery"], gross - cand["action_cost"])
    assert ((cand["p_payment"] >= 0) & (cand["p_payment"] <= 1)).all()


def test_queue_ranked_by_value_not_risk(scored):
    pop, cand = scored
    q = apply_policy(pop, cand, Policy.from_settings())
    ranked = q[q["priority_score"].notna()]
    assert ranked["priority_score"].is_monotonic_decreasing
    assert (ranked["priority_rank"].diff().dropna() == 1).all()
    for col in ["loan_id", "outstanding_principal", "delinquency_bucket", "risk_probability",
                "expected_net_recovery", "priority_category", "recommended_action",
                "recommendation_explanation", "model_version", "data_timestamp"]:
        assert col in q.columns


def test_risk_not_applied_to_observed_serious(scored):
    pop, _ = scored
    serious = pop[pop["delinquency_bucket"].isin(["90+ DPD", "Charged Off"])]
    assert serious["risk_probability"].isna().all()


def test_eligible_candidates_respect_permitted_actions(scored):
    pop, cand = scored
    ec = eligible_candidates(pop, cand, Policy.from_settings())
    co = ec[ec["stage"] == "Charged Off"]
    assert not co["action_type"].isin(["Reminder notification", "Outbound call"]).any()
