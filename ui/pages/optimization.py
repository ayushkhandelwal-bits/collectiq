"""Recovery Optimization: allocate budget and agent capacity across eligible accounts."""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_settings
from src.optimization.optimizer import (
    STATUS_EMPTY,
    STATUS_HEURISTIC,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    Constraints,
    solve_greedy,
)
from src.optimization.service import optimize_and_compare, persist_run
from src.prioritization.engine import Policy, eligible_candidates
from ui import data, filters
from ui.charts.plots import ACCENT, INK, SLATE, style
from ui.components.core import (
    download_csv,
    empty_state,
    inr,
    kpi_strip,
    note,
    num,
    page_header,
    pct,
    provenance_band,
    tag,
)


def render() -> None:
    s = load_settings()
    ocfg, ccfg = s["optimization"], s["collections"]
    page_header("Recovery Optimization",
                "Choose which accounts to work this period, and with which action, to maximise "
                "estimated net recovery within budget and agent capacity.")
    planning_default = pd.Timestamp(s["generation"]["as_of_date"]) + timedelta(days=1)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        with st.form("opt_form"):
            st.markdown("**Planning period**")
            start = st.date_input("Start", planning_default.date())
            days = st.number_input("Length (days)", 1, 31, ccfg["planning_period_days"])
            st.markdown("**Resources**")
            budget = st.number_input("Budget (₹)", 0, 50_000_000, int(ocfg["default_budget"]),
                                     step=25_000)
            hours = st.number_input("Agent capacity (hours)", 0, 20_000,
                                    int(ocfg["default_capacity_hours"]), step=25)
            max_per = st.selectbox("Maximum actions per account", [1, 2], index=0)
            st.markdown("**Action-specific limits**")
            caps = {}
            for a, v in ocfg["default_action_caps"].items():
                caps[a] = st.number_input(a, 0, 100_000, int(v), step=10, key=f"cap_{a}")
            st.markdown("**Eligibility rules**")
            cool = st.number_input("Cooling-off (days)", 0, 30, ccfg["cooling_off_days"])
            max30 = st.number_input("Max contacts in 30 days", 1, 10, ccfg["max_contacts_30d"])
            inc_co = st.toggle("Include charged-off accounts", True)
            min_cov = st.number_input("Minimum 90+ DPD accounts to work", 0, 5_000, 0,
                                      help="Optional policy constraint. Can make the problem "
                                           "infeasible if resources are too tight.")
            method = st.radio("Method", ["Optimal (HiGHS MILP)", "Greedy heuristic (comparison)"])
            submitted = st.form_submit_button("Run optimization", type="primary",
                                              width="stretch")
    with right:
        pop = filters.apply(data.parquet("collections_population.parquet"))
        cand = data.parquet("action_candidates.parquet")
        cand = cand[cand["loan_id"].isin(pop["loan_id"])]
        base = Policy.from_settings()
        pol = Policy(cooling_off_days=int(cool), max_contacts_30d=int(max30),
                     include_charged_off=bool(inc_co), restricted_states=base.restricted_states,
                     digital_only_states=base.digital_only_states)
        if submitted:
            cons = Constraints(budget=float(budget), capacity_hours=float(hours),
                               action_caps={k: float(v) for k, v in caps.items()},
                               max_actions_per_account=int(max_per),
                               min_coverage={"90+ DPD": int(min_cov)} if min_cov else {},
                               time_limit=float(ocfg["time_limit_seconds"]),
                               mip_rel_gap=float(ocfg["mip_rel_gap"]))
            with st.spinner("Solving…"):
                ec = eligible_candidates(pop, cand, pol)
                queue = filters.apply(data.queue_for_policy(int(cool), int(max30), bool(inc_co),
                                                            1, ()))
                res, baselines = optimize_and_compare(ec, queue, cons)
                if method.startswith("Greedy"):
                    res = solve_greedy(ec, cons)
                    tot = res.totals(cons)["net"]
                    for b in baselines.values():
                        b["incremental_net_vs_baseline"] = tot - b["net"]
                end = (pd.Timestamp(start) + timedelta(days=int(days) - 1)).strftime("%Y-%m-%d")
                run_id = None
                try:
                    run_id = persist_run(res, baselines, cons, queue, str(start), end,
                                         {"policy": pol.__dict__, "method": method,
                                          "filters": filters.describe()})
                except Exception as exc:  # persistence failure must not hide the result
                    st.warning(f"Result not saved to the database: {exc}")
            st.session_state["opt_result"] = {"res": res, "baselines": baselines, "cons": cons,
                                              "run_id": run_id, "eligible": ec["loan_id"].nunique(),
                                              "candidates": len(ec)}
        out = st.session_state.get("opt_result")
        if not out:
            provenance_band([("Accounts in view", num(len(pop))),
                             ("Global filters", filters.describe())])
            empty_state("Set the constraints on the left and select 'Run optimization'. The "
                        "pipeline's default run is listed under Previous runs below.")
        else:
            _results(out)
        _history()


def _status_tag(status: str) -> str:
    kind = {"OPTIMAL": "ok", "FEASIBLE_TIME_LIMIT": "warn", "HEURISTIC": "warn",
            "INFEASIBLE": "syn", "FAILED": "syn"}.get(status, "neutral")
    return tag(status.replace("_", " ").title(), kind)


def _results(out: dict) -> None:
    res, baselines, cons = out["res"], out["baselines"], out["cons"]
    tot = res.totals(cons)
    st.markdown(f"Solver status {_status_tag(res.status)} &nbsp; <span style='color:#51607A;"
                f"font-size:0.85rem'>{res.solver_name} · {res.candidate_count:,} candidate "
                f"account-actions from {out['eligible']:,} eligible accounts · "
                f"{res.runtime_seconds:.2f}s{' · run ' + out['run_id'] if out['run_id'] else ''}"
                "</span>", unsafe_allow_html=True)
    if res.status == STATUS_INFEASIBLE:
        note(f"<b>No feasible plan.</b> {res.message}", adverse=True)
        return
    if res.status == STATUS_EMPTY:
        note(res.message, adverse=True)
        return
    if res.status == STATUS_HEURISTIC:
        note("Heuristic result: approximate and <b>not guaranteed optimal</b>. "
             + res.message, adverse=True)
    elif res.status != STATUS_OPTIMAL:
        note(res.message, adverse=True)
    kpi_strip([
        {"label": "Selected accounts", "value": num(tot["selected_accounts"]),
         "detail": f"{num(tot['selected_actions'])} actions"},
        {"label": "Est. gross recovery", "value": inr(tot["gross"], True),
         "detail": "incremental vs no action", "estimate": True},
        {"label": "Operating cost", "value": inr(tot["cost"], True),
         "detail": f"{pct(tot['budget_utilization'])} of budget"},
        {"label": "Est. net recovery", "value": inr(tot["net"], True),
         "detail": "gross − cost", "estimate": True},
        {"label": "Capacity used", "value": f"{tot['hours']:,.1f} h",
         "detail": f"{pct(tot['capacity_utilization'])} of {cons.capacity_hours:,.0f} h"},
    ])
    note("Outputs are model-based estimates on synthetic data. They do not establish that the "
         "optimised plan would actually improve recoveries.")

    st.subheader("Comparison with baseline allocation rules")
    rows = [{"Strategy": "Optimised plan", "Accounts": tot["selected_accounts"],
             "Gross (est.)": tot["gross"], "Cost": tot["cost"], "Net (est.)": tot["net"],
             "Hours": tot["hours"], "Optimised advantage (₹)": 0.0}]
    for name, b in baselines.items():
        rows.append({"Strategy": name, "Accounts": b["selected_accounts"],
                     "Gross (est.)": b["gross"], "Cost": b["cost"], "Net (est.)": b["net"],
                     "Hours": b["hours"],
                     "Optimised advantage (₹)": b.get("incremental_net_vs_baseline")})
    comp = pd.DataFrame(rows).round({"Gross (est.)": 0, "Cost": 0, "Net (est.)": 0,
                                     "Hours": 1, "Optimised advantage (₹)": 0})
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = go.Figure(go.Bar(
            y=comp["Strategy"][::-1], x=comp["Net (est.)"][::-1] / 1e5, orientation="h",
            marker_color=[ACCENT if s == "Optimised plan" else SLATE
                          for s in comp["Strategy"][::-1]],
            hovertemplate="%{y}<br>₹%{x:,.1f} L estimated net<extra></extra>"))
        style(fig, "Estimated net recovery by strategy", height=300, x_title="₹ lakh",
              legend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(comp, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(c, format="localized")
            for c in ["Gross (est.)", "Cost", "Net (est.)", "Optimised advantage (₹)"]})
        with st.expander("Baseline definitions"):
            for name, b in baselines.items():
                st.markdown(f"* **{name}:** {b['description']}")
            st.markdown("All strategies use the same eligible accounts, budget, capacity, "
                        "action limits and per-account limit.")

    sel = res.selected
    if sel.empty:
        empty_state("The solver selected no actions: no candidate adds positive estimated net "
                    "value within the constraints.")
        return
    st.subheader("Selected collection queue")
    by = sel.groupby("action_type").agg(accounts=("loan_id", "size"),
                                        cost=("action_cost", "sum"),
                                        hours=("agent_hours", "sum"),
                                        net=("expected_net_recovery", "sum")).reset_index()
    by["cap"] = by["action_type"].map(cons.action_caps)
    fig = go.Figure()
    fig.add_bar(y=by["action_type"], x=by["accounts"], name="Selected", marker_color=ACCENT,
                orientation="h")
    fig.add_scatter(y=by["action_type"], x=by["cap"], mode="markers", name="Action limit",
                    marker={"symbol": "line-ns-open", "size": 22, "color": INK,
                            "line": {"width": 2}})
    style(fig, "Actions selected vs action-specific limits", height=260, x_title="Accounts")
    st.plotly_chart(fig, width="stretch")
    q = data.default_queue().set_index("loan_id")
    sel = sel.copy()
    for c in ["expected_gross_recovery", "action_cost", "expected_net_recovery"]:
        sel[c] = sel[c].round(0)
    sel["eligibility"] = sel["loan_id"].map(q["eligibility_status"])
    sel["rationale"] = [
        f"{a}: P(payment) {p:.0%} vs {p0:.0%} without action; est. incremental "
        f"{inr(g)} − cost {inr(c)}"
        for a, p, p0, g, c in zip(sel["action_type"], sel["p_payment"],
                                  sel["p_payment_no_action"], sel["expected_gross_recovery"],
                                  sel["action_cost"])]
    show = sel[["assigned_priority", "loan_id", "stage", "action_type",
                "expected_gross_recovery", "action_cost", "expected_net_recovery",
                "agent_hours", "eligibility", "rationale"]]
    st.dataframe(show, hide_index=True, width="stretch", height=380, column_config={
        "assigned_priority": "Priority", "loan_id": "Loan", "stage": "Stage",
        "action_type": "Selected action",
        "expected_gross_recovery": st.column_config.NumberColumn("Est. recovery value (₹)",
                                                                 format="localized"),
        "action_cost": st.column_config.NumberColumn("Est. cost (₹)", format="localized"),
        "expected_net_recovery": st.column_config.NumberColumn("Est. net (₹)",
                                                               format="localized"),
        "agent_hours": st.column_config.NumberColumn("Capacity (h)", format="%.2f"),
        "eligibility": "Eligibility", "rationale": st.column_config.TextColumn("Rationale",
                                                                              width="large")})
    download_csv(show, "Download selected queue (CSV)", "collectiq_optimised_queue.csv",
                 "dl_opt")
    with st.expander("Assumptions used"):
        st.json({**cons.as_dict(), "objective": "maximise Σ estimated net recovery",
                 "value_definition": "incremental expected cash vs no action − action cost",
                 "solver_message": res.message})


def _history() -> None:
    st.subheader("Previous runs")
    runs = data.optimization_runs(15)
    if runs.empty:
        empty_state("No optimisation runs recorded yet.")
        return
    runs = runs.copy()
    runs["source"] = runs["assumptions_json"].map(
        lambda j: json.loads(j).get("source") or json.loads(j).get("method", ""))
    st.dataframe(runs[["run_timestamp", "solver_status", "solver_name", "budget_limit",
                       "available_capacity", "selected_account_count",
                       "estimated_net_recovery", "source"]], hide_index=True,
                 width="stretch", column_config={
                     "run_timestamp": "Run at (UTC)", "solver_status": "Status",
                     "solver_name": "Solver", "available_capacity": "Agent hours",
                     "selected_account_count": "Accounts", "source": "Source",
                     "budget_limit": st.column_config.NumberColumn("Budget (₹)",
                                                                   format="localized"),
                     "estimated_net_recovery": st.column_config.NumberColumn(
                         "Est. net (₹)", format="localized")})
