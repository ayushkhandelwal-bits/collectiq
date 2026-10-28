"""Collection prioritisation engine.

Two steps, kept separate so business users can change policy without re-scoring:

1. ``score_population`` - run the saved models once for every account in the
   collections population at the as-of date and every action permitted for its
   stage (account x action candidates).
2. ``apply_policy`` - apply configurable eligibility / contact rules and derive a
   recommended action, an expected-net-recovery priority score and a rationale.

Definitions
-----------
* Roll-forward risk (Model A): probability a 1-89 DPD account reaches 90+ DPD or
  charge-off within 3 months. It describes *how bad* things may get.
* Expected cash with action (Models B x C): P(payment next month | action) x
  E[amount | payment, action].
* Expected cash with no action: the same quantity with action = "No action".
* Expected gross recovery = expected cash with action - expected cash with no
  action, i.e. the estimated *incremental* cash associated with the action.
* Expected net recovery value (ENRV) = expected gross recovery - action cost.

ENRV ranks *where an action is expected to add value*, which differs from risk: a
high-risk account may have little recoverable value, and a low-risk account may
pay without any contact. ENRV is a model estimate, not a guaranteed amount. The
incremental reading relies on the historical (simulated) action assignment being
random within stage; on observational real-world data it is an association.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_settings
from src.features.build_features import (
    NO_ACTION,
    RECOVERY_CATEGORICAL,
    RECOVERY_NUMERIC,
    RISK_CATEGORICAL,
    RISK_NUMERIC,
)
from src.models.explain import explain, top_reasons

COLLECTION_STAGES = ["1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD", "Charged Off"]
FEATURE_LABELS = {
    "days_past_due": "days past due", "remaining_term_ratio": "remaining term",
    "bureau_score": "bureau score", "debt_to_income_ratio": "debt-to-income",
    "prior_delinquency_count": "prior delinquencies", "bucket_ordinal": "delinquency stage",
    "payment_to_income_ratio": "payment-to-income", "credit_utilization_ratio": "utilisation",
    "credit_history_length_months": "credit history length",
    "outstanding_ratio": "outstanding vs original", "missed_last_6m": "missed payments (6m)",
    "missed_last_3m": "missed payments (3m)", "max_dpd_6m": "max DPD (6m)",
    "payment_ratio_3m": "payment ratio (3m)", "interest_rate": "interest rate",
    "months_on_book": "months on book", "log_annual_income": "income",
    "times_30plus_12m": "30+ DPD episodes (12m)", "risk_grade": "risk grade",
    "loan_product": "product", "months_since_last_missed": "months since last miss",
    "log_loan_amount": "loan amount", "open_credit_accounts": "open credit lines",
    "employment_length_years": "employment length", "loan_term_months": "loan term",
}


@dataclass
class Policy:
    """Configurable contact / eligibility policy (demonstration assumptions)."""

    cooling_off_days: int = 7
    max_contacts_30d: int = 3
    include_charged_off: bool = True
    min_days_past_due: int = 1
    restricted_states: list[str] = field(default_factory=list)
    digital_only_states: list[str] = field(default_factory=list)
    excluded_actions: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls) -> Policy:
        c = load_settings()["collections"]
        return cls(cooling_off_days=c["cooling_off_days"], max_contacts_30d=c["max_contacts_30d"],
                   restricted_states=list(c["restricted_states"]),
                   digital_only_states=list(c["digital_only_states"]))


def actions_by_stage() -> dict[str, list[str]]:
    """Permitted actions per delinquency stage from configuration."""
    acts = load_settings()["collections"]["actions"]
    out: dict[str, list[str]] = {s: [] for s in COLLECTION_STAGES}
    for name, spec in acts.items():
        for st in spec["stages"]:
            out[st].append(name)
    return out


def collections_population(state: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    """Accounts in a collections stage at the as-of date, with loan attributes."""
    pop = state[state["delinquency_bucket"].isin(COLLECTION_STAGES)].copy()
    extra = loans[["loan_id", "contact_restriction", "loan_status", "original_principal",
                   "loan_purpose"]]
    return pop.merge(extra, on="loan_id", how="left")


def score_population(state: pd.DataFrame, loans: pd.DataFrame, risk_pipe, risk_meta: dict,
                     prop_pipe, amount_pipe, amount_meta: dict
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score accounts (risk) and account x action candidates (recovery)."""
    pop = collections_population(state, loans)
    pop["risk_model_applicable"] = pop["delinquency_bucket"].isin(
        ["1-29 DPD", "30-59 DPD", "60-89 DPD"])
    pop["risk_probability"] = np.nan
    pop["risk_drivers"] = ""
    app = pop["risk_model_applicable"]
    if app.any():
        X = pop.loc[app]
        pop.loc[app, "risk_probability"] = risk_pipe.predict_proba(
            X[RISK_NUMERIC + RISK_CATEGORICAL])[:, 1]
        contribs, _ = explain(risk_pipe, X, RISK_NUMERIC, RISK_CATEGORICAL)
        drivers = []
        for i in range(len(contribs)):
            reasons = top_reasons(contribs.iloc[i], 3)
            drivers.append("; ".join(
                f"{FEATURE_LABELS.get(f, f)} {'raises' if v > 0 else 'lowers'} risk"
                for f, v in reasons))
        pop.loc[app, "risk_drivers"] = drivers
    pop["risk_note"] = np.select(
        [app, pop["delinquency_bucket"] == "90+ DPD"],
        ["Model estimate", "Observed: already 90+ DPD (model not applicable)"],
        default="Observed: charged off (model not applicable)")

    # Account x action candidates.
    acfg = load_settings()["collections"]["actions"]
    stage_actions = actions_by_stage()
    rows = []
    for stage, acts in stage_actions.items():
        sub = pop[pop["delinquency_bucket"] == stage]
        for a in acts:
            if sub.empty:
                continue
            c = sub.copy()
            c["action_type"] = a
            rows.append(c)
    if not rows:
        return pop, pd.DataFrame()
    cand = pd.concat(rows, ignore_index=True)
    cand["stage"] = cand["delinquency_bucket"]
    cols = RECOVERY_NUMERIC + RECOVERY_CATEGORICAL
    smear = amount_meta.get("smearing_factor", getattr(amount_pipe, "_collectiq_smearing", 1.0))

    def expected_cash(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = prop_pipe.predict_proba(frame[cols])[:, 1]
        amt = np.maximum(np.expm1(amount_pipe.predict(frame[cols])) * smear, 0.0)
        # Cash can never exceed what is collectible.
        cap = np.where(frame["stage"] == "Charged Off", frame["outstanding_principal"],
                       frame["outstanding_principal"] + frame["amount_due"])
        return p, np.minimum(amt, cap)

    cand["p_payment"], cand["amount_if_paid"] = expected_cash(cand)
    cand["expected_cash_with_action"] = cand["p_payment"] * cand["amount_if_paid"]
    # Counterfactual baseline: the same account with "No action" this period.
    base = pop.copy()
    base["stage"] = base["delinquency_bucket"]
    base["action_type"] = NO_ACTION
    p0, a0 = expected_cash(base)
    base_tbl = pd.DataFrame({"loan_id": base["loan_id"], "p_payment_no_action": p0,
                             "expected_cash_no_action": p0 * a0})
    pop = pop.merge(base_tbl, on="loan_id", how="left")
    cand = cand.merge(base_tbl, on="loan_id", how="left")
    cand["expected_gross_recovery"] = (cand["expected_cash_with_action"]
                                       - cand["expected_cash_no_action"])
    cand["action_cost"] = cand["action_type"].map(lambda a: acfg[a]["cost"]).astype(float)
    cand["agent_hours"] = cand["action_type"].map(lambda a: acfg[a]["agent_hours"]).astype(float)
    cand["capacity_category"] = cand["action_type"].map(lambda a: acfg[a]["capacity_category"])
    cand["expected_net_recovery"] = cand["expected_gross_recovery"] - cand["action_cost"]
    keep = ["loan_id", "stage", "action_type", "p_payment", "amount_if_paid",
            "expected_cash_with_action", "p_payment_no_action", "expected_cash_no_action",
            "expected_gross_recovery", "action_cost", "agent_hours", "capacity_category",
            "expected_net_recovery"]
    return pop, cand[keep]


def _eligibility(pop: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """Eligibility status and the set of permitted actions per account."""
    stage_actions = actions_by_stage()
    status, permitted = [], []
    for r in pop.itertuples(index=False):
        acts = [a for a in stage_actions.get(r.delinquency_bucket, [])
                if a not in policy.excluded_actions]
        restr = r.contact_restriction if isinstance(r.contact_restriction, str) else None
        dsl = r.days_since_last_contact
        if r.outstanding_principal is None or not r.outstanding_principal > 0:
            s, acts = "Ineligible: no collectible balance", []
        elif r.delinquency_bucket == "Charged Off" and not policy.include_charged_off:
            s, acts = "Ineligible: charged-off accounts excluded by policy", []
        elif r.delinquency_bucket != "Charged Off" and r.days_past_due < policy.min_days_past_due:
            s, acts = f"Ineligible: below minimum {policy.min_days_past_due} DPD", []
        elif restr in policy.restricted_states:
            s, acts = f"Restricted: {restr}", []
        elif pd.notna(dsl) and dsl < policy.cooling_off_days:
            s, acts = f"Cooling-off: contacted {int(dsl)} day(s) ago", []
        elif r.contacts_last_30d >= policy.max_contacts_30d:
            s, acts = f"Contact cap reached ({int(r.contacts_last_30d)} in 30 days)", []
        elif restr in policy.digital_only_states:
            acts = [a for a in acts if a == "Reminder notification"]
            s = "Eligible: digital reminders only" if acts else f"Restricted: {restr}"
        elif not acts:
            s = "Ineligible: no permitted action"
        else:
            s = "Eligible"
        status.append(s)
        permitted.append(acts)
    out = pop[["loan_id"]].copy()
    out["eligibility_status"] = status
    out["permitted_actions"] = permitted
    out["eligible"] = [len(a) > 0 for a in permitted]
    return out


def eligible_candidates(pop: pd.DataFrame, cand: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """Candidates restricted to eligible accounts and their permitted actions."""
    el = _eligibility(pop, policy)
    perm = el.explode("permitted_actions").dropna(subset=["permitted_actions"])
    perm = perm.rename(columns={"permitted_actions": "action_type"})[["loan_id", "action_type"]]
    return cand.merge(perm, on=["loan_id", "action_type"], how="inner")


def _fmt_inr(x: float) -> str:
    return f"₹{x:,.0f}"


def apply_policy(pop: pd.DataFrame, cand: pd.DataFrame, policy: Policy | None = None,
                 model_version: str = "", data_timestamp: str = "") -> pd.DataFrame:
    """Account-level prioritisation queue with recommendations and rationale."""
    policy = policy or Policy.from_settings()
    pcfg = load_settings()["priority"]
    el = _eligibility(pop, policy)
    q = pop.merge(el, on="loan_id", how="left")
    ec = eligible_candidates(pop, cand, policy)
    best = (ec.sort_values("expected_net_recovery", ascending=False)
            .groupby("loan_id").head(1).set_index("loan_id"))
    q = q.merge(best[["action_type", "p_payment", "amount_if_paid", "expected_cash_with_action",
                      "expected_gross_recovery", "action_cost", "agent_hours",
                      "expected_net_recovery"]],
                left_on="loan_id", right_index=True, how="left")
    pos = q["expected_net_recovery"] > 0
    hi = q.loc[pos, "expected_net_recovery"].quantile(pcfg["high_quantile"]) if pos.any() else 0
    md = q.loc[pos, "expected_net_recovery"].quantile(pcfg["medium_quantile"]) if pos.any() else 0
    q["priority_category"] = np.select(
        [~q["eligible"], ~pos, q["expected_net_recovery"] >= hi,
         q["expected_net_recovery"] >= md],
        ["Not eligible", "Low value", "High", "Medium"], default="Standard")
    q["recommended_action"] = np.where(
        ~q["eligible"], "No additional contact until eligible",
        np.where(pos, q["action_type"], "No additional contact this period"))
    rationale = []
    for r in q.itertuples(index=False):
        if not r.eligible:
            rationale.append(f"{r.eligibility_status}. No contact recommended under the "
                             "configured policy.")
            continue
        risk_txt = (f"roll-forward risk {r.risk_probability:.0%}"
                    if pd.notna(r.risk_probability) else r.risk_note.lower())
        if r.expected_net_recovery <= 0:
            rationale.append(
                f"Best permitted action ({r.action_type}) adds an estimated "
                f"{_fmt_inr(r.expected_gross_recovery)} over no action, below its cost "
                f"{_fmt_inr(r.action_cost)}. Expected cash without contact "
                f"{_fmt_inr(r.expected_cash_no_action)}. Stage {r.delinquency_bucket}, "
                f"{risk_txt}.")
            continue
        drivers = f" Main risk drivers: {r.risk_drivers}." if r.risk_drivers else ""
        rationale.append(
            f"{r.action_type} has the highest estimated net value "
            f"({_fmt_inr(r.expected_net_recovery)}): payment probability {r.p_payment:.0%} vs "
            f"{r.p_payment_no_action:.0%} with no action; expected cash "
            f"{_fmt_inr(r.expected_cash_with_action)} vs "
            f"{_fmt_inr(r.expected_cash_no_action)}; cost {_fmt_inr(r.action_cost)}. "
            f"Stage {r.delinquency_bucket}, {risk_txt}.{drivers}")
    q["recommendation_explanation"] = rationale
    q["priority_score"] = np.where(q["eligible"] & pos, q["expected_net_recovery"], np.nan)
    q = q.sort_values(["priority_score", "outstanding_principal"], ascending=[False, False],
                      na_position="last").reset_index(drop=True)
    q["priority_rank"] = np.where(q["priority_score"].notna(),
                                  np.arange(1, len(q) + 1), np.nan)
    q["model_version"] = model_version
    q["data_timestamp"] = data_timestamp
    return q


QUEUE_COLUMNS = [
    "priority_rank", "loan_id", "loan_product", "risk_grade", "delinquency_bucket",
    "days_past_due", "outstanding_principal", "amount_due", "risk_probability", "risk_note",
    "p_payment", "p_payment_no_action", "amount_if_paid", "expected_cash_with_action",
    "expected_cash_no_action", "expected_gross_recovery", "action_cost",
    "expected_net_recovery", "recommended_action", "priority_category", "eligibility_status",
    "eligible", "recommendation_explanation", "risk_drivers", "contact_restriction",
    "days_since_last_contact", "contacts_last_30d", "model_version", "data_timestamp",
]


def queue_view(q: pd.DataFrame) -> pd.DataFrame:
    """Columns used by the UI and exports."""
    return q[[c for c in QUEUE_COLUMNS if c in q.columns]]


def summarise_queue(q: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers for the workbench."""
    el = q[q["eligible"]]
    return {
        "population": int(len(q)), "eligible": int(len(el)),
        "eligible_exposure": float(el["outstanding_principal"].sum()),
        "positive_enrv_total": float(q["priority_score"].sum(skipna=True)),
        "high_priority": int((q["priority_category"] == "High").sum()),
    }
