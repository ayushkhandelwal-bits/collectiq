"""Account Intelligence: one account's history, model estimates and recommendation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.build_features import RISK_CATEGORICAL, RISK_NUMERIC
from src.models.explain import explain
from src.prioritization.engine import FEATURE_LABELS
from ui import data
from ui.charts.plots import ACCENT, ADVERSE, BUCKET_COLORS, INK, style
from ui.components.core import (
    empty_state,
    fmt_date,
    inr,
    kpi_strip,
    note,
    num,
    page_header,
    pct,
    provenance_band,
    tag,
    uncertainty_warning,
)


def render() -> None:
    page_header("Account Intelligence",
                "Look up one account: its repayment history, model estimates, eligibility and "
                "recommended next action.")
    q = data.default_queue()
    example = q[q["priority_score"].notna()]["loan_id"].head(1)
    hint = example.iloc[0] if len(example) else "LN0000001"
    term = st.text_input("Search by loan ID", value=st.session_state.get("acct_term", hint),
                         help="Partial IDs work, e.g. 'LN00123'. IDs are fictional.")
    st.session_state["acct_term"] = term
    if not term.strip():
        empty_state("Enter a loan ID to begin.")
        return
    res = data.search_loans(term)
    if res.empty:
        empty_state(f"No loan ID contains '{term}'. Loan IDs look like LN0001234.")
        return
    if len(res) > 1:
        st.dataframe(res, hide_index=True, width="stretch", height=180,
                     column_config={"current_outstanding_principal": st.column_config.NumberColumn(
                         "Outstanding (₹)", format="localized")})
    loan_id = st.selectbox("Account", res["loan_id"].tolist(), index=0)
    loan = data.loans().set_index("loan_id").loc[loan_id]
    provenance_band([("Loan", loan_id), ("Product", loan["loan_product"]),
                     ("Grade", loan["risk_grade"]), ("Status", loan["loan_status"])])

    kpi_strip([
        {"label": "Original principal", "value": inr(loan["original_principal"]),
         "detail": f"{int(loan['loan_term_months'])} months at {loan['interest_rate']:.2f}% p.a."},
        {"label": "Outstanding / collectible", "value": inr(loan["current_outstanding_principal"]),
         "detail": "observed at as-of"},
        {"label": "Monthly instalment", "value": inr(loan["emi_amount"]),
         "detail": f"originated {fmt_date(loan['origination_date'])}"},
        {"label": "Bureau score", "value": num(loan["bureau_score"]),
         "detail": f"grade {loan['risk_grade']} at origination"},
    ])

    hist = data.account_history(loan_id)
    ev = data.account_events(loan_id)
    c1, c2 = st.columns([3, 2])
    with c1:
        if hist.empty:
            empty_state("No monthly snapshots for this account yet.")
        else:
            fig = go.Figure()
            fig.add_bar(x=hist["snapshot_date"], y=hist["days_past_due"], name="Days past due",
                        marker_color=[BUCKET_COLORS.get(b, INK)
                                      for b in hist["delinquency_bucket"]],
                        customdata=hist[["delinquency_bucket", "payment_received"]].to_numpy(),
                        hovertemplate="%{x}<br>%{y} DPD (%{customdata[0]})<br>Paid "
                                      "₹%{customdata[1]:,.0f}<extra></extra>")
            fig.add_scatter(x=hist["snapshot_date"], y=hist["outstanding_principal"],
                            name="Outstanding (₹, right axis)", yaxis="y2", mode="lines",
                            line={"color": ACCENT})
            if not ev.empty:
                fig.add_scatter(x=ev["event_date"], y=[0] * len(ev), mode="markers",
                                name="Collection action", marker={"symbol": "triangle-up",
                                                                  "size": 10, "color": INK},
                                text=ev["action_type"] + " — " + ev["contact_outcome"],
                                hovertemplate="%{x}<br>%{text}<extra></extra>")
            style(fig, "Repayment and delinquency history (observed)", height=340,
                  y_title="Days past due")
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "showgrid": False,
                                      "title": "₹ outstanding"})
            st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Borrower and loan features (origination)**")
        feats = pd.DataFrame({
            "Feature": ["Purpose", "Annual income", "Employment (years)", "Debt-to-income",
                        "Payment-to-income", "Credit utilisation", "Credit history (months)",
                        "Prior delinquencies", "Open credit lines"],
            "Value": [loan["loan_purpose"], inr(loan["annual_income"]),
                      "missing" if pd.isna(loan["employment_length_years"])
                      else f"{loan['employment_length_years']:.1f}",
                      pct(loan["debt_to_income_ratio"]), pct(loan["payment_to_income_ratio"]),
                      pct(loan["credit_utilization_ratio"]),
                      num(loan["credit_history_length_months"]),
                      num(loan["prior_delinquency_count"]), num(loan["open_credit_accounts"])]})
        st.dataframe(feats, hide_index=True, width="stretch")
        if isinstance(loan["contact_restriction"], str):
            note(f"Contact restriction on file: <b>{loan['contact_restriction']}</b>.",
                 adverse=True)

    row = q[q["loan_id"] == loan_id]
    st.subheader("Model estimates and recommendation")
    if row.empty:
        note(f"This account is not in the collections population at the as-of date (status "
             f"<b>{loan['loan_status']}</b>). Collection models apply to accounts 1+ days past "
             "due or charged off within the recovery window, so no estimate is shown.")
        _events_table(ev)
        return
    r = row.iloc[0]
    c3, c4 = st.columns([2, 3])
    with c3:
        st.markdown("**Roll-forward risk (Model A)**")
        if pd.notna(r["risk_probability"]):
            st.markdown(f"<div style='font-size:1.6rem;font-weight:600;color:"
                        f"{ADVERSE if r['risk_probability'] > 0.5 else INK}'>"
                        f"{pct(r['risk_probability'])}</div>"
                        "<div style='font-size:0.82rem;color:#51607A'>probability of reaching "
                        "90+ DPD or charge-off within 3 months</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"{tag(r['risk_note'], 'neutral')}", unsafe_allow_html=True)
        st.markdown("**Eligibility**")
        kind = "ok" if r["eligible"] else "warn"
        st.markdown(tag(r["eligibility_status"], kind), unsafe_allow_html=True)
        st.markdown("**Recommended next action**")
        st.markdown(f"<div style='font-size:1.1rem;font-weight:600'>{r['recommended_action']}"
                    f"</div>", unsafe_allow_html=True)
        st.caption(r["recommendation_explanation"])
    with c4:
        cand = data.parquet("action_candidates.parquet")
        cand = cand[cand["loan_id"] == loan_id]
        if cand.empty:
            empty_state("No candidate actions for this stage.")
        else:
            fig = go.Figure()
            fig.add_bar(x=cand["action_type"], y=cand["expected_cash_with_action"],
                        name="Expected cash with action", marker_color=ACCENT)
            fig.add_bar(x=cand["action_type"], y=cand["expected_cash_no_action"],
                        name="Expected cash with no action", marker_color="#C3CBD6")
            fig.add_scatter(x=cand["action_type"], y=cand["expected_net_recovery"],
                            mode="markers", name="Net value (incremental − cost)",
                            marker={"color": INK, "size": 11, "symbol": "diamond"})
            style(fig, "Recovery estimate by permitted action (Models B and C)", height=320,
                  y_title="₹ over next month")
            fig.update_layout(barmode="group")
            st.plotly_chart(fig, width="stretch")
            if not r["eligible"]:
                st.caption("Shown for information; the account is not eligible for contact under "
                            "the current policy.")
    if pd.notna(r["risk_probability"]):
        _risk_explanation(loan_id)
    uncertainty_warning("Roll-forward risk and recovery value answer different questions: a "
                        "high-risk account can still have low recoverable value.")
    _events_table(ev)


def _risk_explanation(loan_id: str) -> None:
    pop = data.parquet("collections_population.parquet")
    x = pop[pop["loan_id"] == loan_id]
    if x.empty:
        return
    try:
        pipe = data.model("risk_model")
    except Exception as exc:
        note(f"Risk model could not be loaded ({exc}); explanation unavailable.", adverse=True)
        return
    contrib, base = explain(pipe, x, RISK_NUMERIC, RISK_CATEGORICAL)
    s = contrib.iloc[0].sort_values(key=abs, ascending=False).head(10)[::-1]
    fig = go.Figure(go.Bar(
        x=s.values, y=[FEATURE_LABELS.get(f, f) for f in s.index], orientation="h",
        marker_color=[ADVERSE if v > 0 else ACCENT for v in s.values],
        hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
    style(fig, "Why the risk model scored this account (top 10 contributions)", height=360,
          x_title="Contribution to log-odds of roll-forward (right = higher risk)", legend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Contributions are relative to the model's average account (base log-odds "
               f"{base:+.2f}). They explain the model's arithmetic, not causes of delinquency.")


def _events_table(ev: pd.DataFrame) -> None:
    st.subheader("Collection history (simulated events)")
    if ev.empty:
        empty_state("No collection actions recorded for this account.")
        return
    st.dataframe(ev[["event_date", "delinquency_stage_at_event", "action_type", "contact_channel",
                     "attempt_number", "contact_outcome", "promise_to_pay_indicator",
                     "subsequent_payment_amount", "estimated_action_cost"]],
                 hide_index=True, width="stretch", column_config={
                     "subsequent_payment_amount": st.column_config.NumberColumn(
                         "Paid in 30 days (₹)", format="localized"),
                     "estimated_action_cost": st.column_config.NumberColumn("Cost (₹)")})
