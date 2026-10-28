"""Scenario Lab: hypothetical changes to resources and recovery assumptions."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.scenario import ScenarioAssumptions, run_scenario, sensitivity
from src.optimization.service import default_constraints
from src.prioritization.engine import Policy, eligible_candidates
from ui import data, filters
from ui.charts.plots import ACCENT, INK, style
from ui.components.core import (
    empty_state,
    inr,
    kpi_strip,
    note,
    num,
    page_header,
    pct,
    provenance_band,
)
from ui.filters import PRODUCTS


def render() -> None:
    page_header("Scenario Lab",
                "Test how budget, capacity, delinquency volume and recovery assumptions would "
                "change the optimised plan. Results are hypothetical.")
    note("<b>Hypothetical scenarios.</b> Inputs adjust a copy of the model estimates. Historical "
         "data and trained models are never changed or retrained.")
    base_cons = default_constraints()
    pop = filters.apply(data.parquet("collections_population.parquet"))
    cand = data.parquet("action_candidates.parquet")
    cand = cand[cand["loan_id"].isin(pop["loan_id"])]
    ec = eligible_candidates(pop, cand, Policy.from_settings())
    provenance_band([("Eligible accounts", num(ec["loan_id"].nunique())),
                     ("Global filters", filters.describe())])
    if ec.empty:
        empty_state("No eligible accounts under the current filters.")
        return

    with st.form("scenario"):
        c1, c2, c3 = st.columns(3)
        budget = c1.number_input("Budget (₹)", 0, 50_000_000, int(base_cons.budget), 25_000)
        hours = c1.number_input("Agent capacity (hours)", 0, 20_000,
                                int(base_cons.capacity_hours), 25)
        dlq = c2.slider("Delinquent population (× today)", 0.5, 2.0, 1.0, 0.05,
                        help="Scales the number of eligible accounts by resampling.")
        p_mult = c2.slider("Payment probability (× model estimate)", 0.5, 1.5, 1.0, 0.05)
        a_mult = c3.slider("Amount if paid (× model estimate)", 0.5, 1.5, 1.0, 0.05)
        prod = c3.selectbox("Composition change: product", ["None"] + PRODUCTS)
        weight = c3.slider("…its share of delinquent accounts (× today)", 0.0, 3.0, 1.0, 0.1)
        sens = st.checkbox("Include sensitivity analysis (re-optimises 12 times)", value=True)
        go_ = st.form_submit_button("Run scenario", type="primary")
    if go_:
        base_a = ScenarioAssumptions(budget=base_cons.budget,
                                     capacity_hours=base_cons.capacity_hours)
        scen_a = ScenarioAssumptions(budget=float(budget), capacity_hours=float(hours),
                                     delinquency_multiplier=dlq, recovery_prob_multiplier=p_mult,
                                     recovery_amount_multiplier=a_mult,
                                     product_weights={prod: weight} if prod != "None" else {})
        with st.spinner("Optimising baseline and scenario…"):
            accounts = pop[["loan_id", "loan_product"]]
            out = {"base": run_scenario(ec, accounts, base_a, base_cons),
                   "scen": run_scenario(ec, accounts, scen_a, base_cons)}
            if sens:
                mults = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
                out["sens_budget"] = sensitivity(ec, accounts, scen_a, base_cons, "budget",
                                                 mults)
                out["sens_capacity"] = sensitivity(ec, accounts, scen_a, base_cons, "capacity",
                                                   mults)
        st.session_state["scenario_out"] = out
    out = st.session_state.get("scenario_out")
    if not out:
        empty_state("Adjust the assumptions and select 'Run scenario'.")
        return
    b, s = out["base"], out["scen"]
    if s["status"] not in ("OPTIMAL", "FEASIBLE_TIME_LIMIT", "HEURISTIC"):
        note(f"Scenario status {s['status']}: {s['message']}", adverse=True)
    kpi_strip([
        {"label": "Scenario net recovery (est.)", "value": inr(s["net"], True),
         "detail": f"baseline {inr(b['net'], True)}", "estimate": True,
         "tone": "down-good" if s["net"] >= b["net"] else "up-bad"},
        {"label": "Change vs baseline", "value": inr(s["net"] - b["net"], True),
         "detail": pct((s["net"] - b["net"]) / b["net"]) if b["net"] else "—", "estimate": True},
        {"label": "Eligible accounts (scenario)", "value": num(s["eligible_accounts"]),
         "detail": f"baseline {num(b['eligible_accounts'])}"},
        {"label": "Accounts worked", "value": num(s["selected_accounts"]),
         "detail": f"baseline {num(b['selected_accounts'])}"},
        {"label": "Budget / capacity used",
         "value": f"{pct(s['budget_utilization'], 0)} / {pct(s['capacity_utilization'], 0)}",
         "detail": "of scenario limits"},
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Assumptions**")
        rows = []
        labels = {"budget": "Budget (₹)", "capacity_hours": "Agent hours",
                  "delinquency_multiplier": "Delinquent population ×",
                  "recovery_prob_multiplier": "Payment probability ×",
                  "recovery_amount_multiplier": "Amount if paid ×",
                  "product_weights": "Product composition"}
        def fmt(k, v):
            if k == "budget":
                return inr(v)
            if k == "capacity_hours":
                return f"{v:,.0f} h"
            if k == "product_weights":
                return ", ".join(f"{p} × {w:.1f}" for p, w in v.items()) or "unchanged"
            return f"× {v:.2f}"

        for k, lab in labels.items():
            rows.append({"Assumption": lab, "Baseline": fmt(k, b["assumptions"][k]),
                         "Scenario": fmt(k, s["assumptions"][k])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with c2:
        acts = sorted(set(b["by_action"]) | set(s["by_action"]))
        fig = go.Figure()
        fig.add_bar(y=acts, x=[b["by_action"].get(a, 0) for a in acts], name="Baseline",
                    marker_color="#C3CBD6", orientation="h")
        fig.add_bar(y=acts, x=[s["by_action"].get(a, 0) for a in acts], name="Scenario",
                    marker_color=ACCENT, orientation="h")
        style(fig, "Actions in the optimised plan", 300, x_title="Accounts")
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, width="stretch")
    if "sens_budget" in out:
        st.subheader("Sensitivity of estimated net recovery")
        c3, c4 = st.columns(2)
        for col, key, lab, unit in [(c3, "sens_budget", "budget", "budget"),
                                    (c4, "sens_capacity", "capacity_hours", "agent hours")]:
            d = out[key]
            with col:
                fig = go.Figure(go.Scatter(
                    x=d[lab], y=d["net"] / 1e5, mode="lines+markers", line={"color": INK},
                    customdata=d[["selected_accounts"]].to_numpy(),
                    hovertemplate="%{x:,.0f}<br>₹%{y:,.1f} L<br>%{customdata[0]:,} accounts"
                                  "<extra></extra>"))
                style(fig, f"Net recovery vs {unit} (other scenario inputs fixed)", 300,
                      x_title="₹ budget" if lab == "budget" else "Agent hours",
                      y_title="₹ lakh (est.)", legend=False)
                st.plotly_chart(fig, width="stretch")
        st.caption("Flattening curves show where the other resource becomes the binding "
                   "constraint, which indicates where extra budget or staff adds little.")
    note("Limitations: scenarios scale model estimates mechanically. Resampled accounts are "
         "hypothetical replicas of today's eligible accounts, and multipliers assume effects are "
         "uniform across segments.")
