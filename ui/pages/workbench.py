"""Collections Workbench: the actionable prioritisation queue."""

from __future__ import annotations

import streamlit as st

from src.config import load_settings
from src.prioritization.engine import COLLECTION_STAGES
from ui import data, filters
from ui.components.core import (
    download_csv,
    empty_state,
    inr,
    kpi_strip,
    methodology,
    note,
    num,
    page_header,
    pct,
    provenance_band,
    uncertainty_warning,
)

PRIORITIES = ["High", "Medium", "Standard", "Low value", "Not eligible"]


def render() -> None:
    cfg = load_settings()["collections"]
    page_header("Collections Workbench",
                "Eligible accounts ranked by estimated net recovery value, with the recommended "
                "action and the reasoning behind it.")
    with st.expander("Contact policy (demonstration assumptions — adjust to recompute the "
                     "queue)"):
        p1, p2, p3, p4 = st.columns(4)
        cool = p1.number_input("Cooling-off after a contact (days)", 0, 30,
                               cfg["cooling_off_days"])
        cap = p2.number_input("Max contacts in 30 days", 1, 10, cfg["max_contacts_30d"])
        min_dpd = p3.number_input("Minimum days past due", 1, 90, 1)
        inc_co = p4.toggle("Include charged-off accounts", True)
        excl = st.multiselect("Exclude actions", list(cfg["actions"]), placeholder="None")
        st.caption("Restricted states (no contact): " + ", ".join(cfg["restricted_states"])
                   + ". Digital reminders only: " + ", ".join(cfg["digital_only_states"])
                   + ". These are configurable demonstration rules, not legal guidance.")
    q = data.queue_for_policy(int(cool), int(cap), bool(inc_co), int(min_dpd), tuple(excl))
    q = filters.apply(q)

    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1.1, 1])
    stages = f1.multiselect("Delinquency stage", COLLECTION_STAGES, placeholder="All stages")
    prios = f2.multiselect("Priority", PRIORITIES, placeholder="All priorities")
    elig = f3.selectbox("Eligibility", ["All accounts", "Eligible only", "Not eligible only"])
    search = f4.text_input("Search loan ID", placeholder="e.g. LN00012")
    view = q
    if stages:
        view = view[view["delinquency_bucket"].isin(stages)]
    if prios:
        view = view[view["priority_category"].isin(prios)]
    if elig == "Eligible only":
        view = view[view["eligible"]]
    elif elig == "Not eligible only":
        view = view[~view["eligible"]]
    if search.strip():
        view = view[view["loan_id"].str.contains(search.strip().upper(), regex=False)]

    el = view[view["eligible"]]
    provenance_band([("Population", f"{num(len(q))} accounts in collections stages"),
                     ("In view", num(len(view))), ("Global filters", filters.describe())])
    kpi_strip([
        {"label": "Eligible accounts", "value": num(len(el)),
         "detail": f"of {num(len(view))} in view"},
        {"label": "Eligible exposure", "value": inr(el["outstanding_principal"].sum(), True),
         "detail": "outstanding / collectible (observed)"},
        {"label": "Estimated net recovery opportunity",
         "value": inr(view["priority_score"].sum(skipna=True), True),
         "detail": "best action per account, unconstrained", "estimate": True},
        {"label": "High-priority accounts",
         "value": num((view["priority_category"] == "High").sum()),
         "detail": "top 20% of positive net value"},
    ])
    methodology("How the queue is ranked", (
        "* **Roll-forward risk** (Model A) estimates how likely a 1-89 DPD account is to reach "
        "90+ DPD or charge-off in 3 months. It describes severity, not value.\n"
        "* **Expected cash with action** = P(payment next month | action) × E[amount | payment, "
        "action] (Models B and C). The same is estimated with **no action**.\n"
        "* **Estimated net value** = (cash with action − cash with no action) − action cost. "
        "The queue is ranked by the best permitted action's net value, so an account that "
        "will probably pay without contact ranks low even if its balance is large.\n"
        "* Priority bands: High = top 20% of positive net value, Medium = next 30%, Standard = "
        "remaining positive, Low value = no positive net value this period.\n"
        "* The incremental reading holds in this simulation because past actions were "
        "assigned at random within stage. On real data it would need A/B-test evidence."))
    if view.empty:
        empty_state("No accounts match these filters. Clear the stage, priority or eligibility "
                    "filters.")
        return
    cols = ["priority_rank", "loan_id", "delinquency_bucket", "days_past_due",
            "outstanding_principal", "risk_probability", "p_payment", "amount_if_paid",
            "action_cost", "expected_net_recovery", "recommended_action", "priority_category",
            "eligibility_status", "recommendation_explanation"]
    st.markdown("**Prioritised queue** — observed facts in the first columns; model estimates "
                "marked *est.* Click a row for details.")
    event = st.dataframe(
        view[cols], hide_index=True, width="stretch", height=440,
        on_select="rerun", selection_mode="single-row", key="wb_table",
        column_config={
            "priority_rank": st.column_config.NumberColumn("Rank", format="%d"),
            "loan_id": "Loan", "delinquency_bucket": "Stage",
            "days_past_due": st.column_config.NumberColumn("DPD"),
            "outstanding_principal": st.column_config.NumberColumn("Outstanding (₹)",
                                                                   format="localized"),
            "risk_probability": st.column_config.NumberColumn(
                "Roll-forward risk (est.)", format="percent",
                help="Blank for 90+ DPD and charged-off accounts (already observed)."),
            "p_payment": st.column_config.NumberColumn("Payment prob. (est.)", format="percent"),
            "amount_if_paid": st.column_config.NumberColumn("Amount if paid (est., ₹)",
                                                            format="localized"),
            "action_cost": st.column_config.NumberColumn("Action cost (₹)", format="localized"),
            "expected_net_recovery": st.column_config.NumberColumn("Net value (est., ₹)",
                                                                   format="localized"),
            "recommended_action": "Recommended action", "priority_category": "Priority",
            "eligibility_status": "Eligibility",
            "recommendation_explanation": st.column_config.TextColumn("Explanation",
                                                                      width="large")})
    download_csv(view.drop(columns=["permitted_actions"], errors="ignore"),
                 f"Download {num(len(view))} rows (CSV)", "collectiq_priority_queue.csv",
                 "dl_queue")
    rows = event.selection.rows if event and hasattr(event, "selection") else []
    if rows:
        r = view.iloc[rows[0]]
        st.subheader(f"Selected account {r['loan_id']}")
        d1, d2, d3 = st.columns(3)
        d1.markdown(f"**Observed**  \nStage: {r['delinquency_bucket']} ({int(r['days_past_due'])} "
                    f"DPD)  \nOutstanding: {inr(r['outstanding_principal'])}  \nAmount due: "
                    f"{inr(r['amount_due'])}  \nRestriction: "
                    f"{r['contact_restriction'] or 'none'}  \nContacts in last 30 days: "
                    f"{int(r['contacts_last_30d'])}")
        d2.markdown(f"**Model estimates**  \nRoll-forward risk: {pct(r['risk_probability'])}  \n"
                    f"Payment prob. with action: {pct(r['p_payment'])}  \n"
                    f"Payment prob. without: {pct(r['p_payment_no_action'])}  \n"
                    f"Cash with / without action: {inr(r['expected_cash_with_action'])} / "
                    f"{inr(r['expected_cash_no_action'])}")
        d3.markdown(f"**Decision support**  \nRecommended: {r['recommended_action']}  \n"
                    f"Net value (est.): {inr(r['expected_net_recovery'])}  \n"
                    f"Priority: {r['priority_category']}  \nEligibility: "
                    f"{r['eligibility_status']}")
        note(r["recommendation_explanation"])
        st.caption("Open Account Intelligence for full history and the risk explanation.")
    uncertainty_warning("Recommendations support, not replace, a collector's judgement and "
                        "never authorise adverse action on their own.")
