"""Executive Overview: what is happening, where the problem is, what deserves attention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import kpis as K
from src.config import load_settings
from ui import data, filters
from ui.charts.plots import ACCENT, ADVERSE, BUCKET_COLORS, INK, style
from ui.components.core import (
    empty_state,
    fmt_date,
    inr,
    insights,
    kpi_strip,
    num,
    page_header,
    pct,
    provenance_band,
)


def render() -> None:
    as_of = load_settings()["generation"]["as_of_date"]
    page_header("Executive Overview",
                "Portfolio health, collections performance and recovery opportunities.")
    provenance_band([("As of", fmt_date(as_of)), ("Filters", filters.describe()),
                     ("Comparison", "previous month-end")])

    loans = filters.apply(data.loans())
    port = filters.apply(data.portfolio_snapshots())
    queue = filters.apply(data.default_queue())
    if loans.empty:
        empty_state("No loans match the selected filters. Clear the sidebar filters to see "
                    "the full portfolio.")
        return
    k = K.portfolio_kpis(loans)
    trends = K.monthly_trends(port)
    prev = trends.iloc[-2] if len(trends) > 1 else None
    last = trends.iloc[-1]
    d_rate = last["delinquency_rate_30plus"]
    delta = ""
    tone = ""
    if prev is not None and pd.notna(prev["delinquency_rate_30plus"]):
        diff = d_rate - prev["delinquency_rate_30plus"]
        delta = f"{diff * 100:+.2f} pp vs {fmt_date(prev['snapshot_date'])}"
        tone = "up-bad" if diff > 0 else "down-good"
    enrv = queue["priority_score"].sum(skipna=True)
    kpi_strip([
        {"label": "Total accounts", "value": num(k["total_accounts"]),
         "detail": f"{num(k['open_accounts'])} open at as-of",
         "help": "All loans originated to the as-of date; open = not repaid or charged off."},
        {"label": "Outstanding exposure", "value": inr(k["outstanding_exposure"], True),
         "detail": "open accounts, principal",
         "help": "Sum of outstanding principal on open accounts at the as-of date. Excludes "
                 "charged-off balances."},
        {"label": "Delinquency rate (30+ DPD)", "value": pct(d_rate, 2), "detail": delta,
         "tone": tone,
         "help": "Open accounts 30+ days past due / open accounts, at month-end."},
        {"label": "Observed recovery rate", "value": pct(k["observed_recovery_rate"]),
         "detail": f"{inr(k['observed_recovery_amount'], True)} recovered to date",
         "help": "Cumulative post-charge-off recoveries / charged-off principal, all "
                 "charged-off loans to the as-of date."},
        {"label": "Estimated net recovery opportunity", "value": inr(enrv, True),
         "detail": "model estimate, before capacity limits", "estimate": True,
         "help": "Sum of positive expected net recovery value (incremental cash vs no "
                 "action, minus cost) over eligible accounts' best action. Unconstrained; "
                 "see Recovery Optimization for a budget-feasible plan."},
    ])

    c1, c2 = st.columns(2)
    with c1:
        open_l = loans[loans["loan_status"].isin(K.OPEN_STATUSES)]
        dist = (open_l.groupby("loan_status")
                .agg(accounts=("loan_id", "size"),
                     exposure=("current_outstanding_principal", "sum"))
                .reindex(K.OPEN_STATUSES).fillna(0).reset_index())
        labels = ["Current", "1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD"]
        dist["bucket"] = labels
        dlq = dist[dist["bucket"] != "Current"]
        fig = go.Figure(go.Bar(
            x=dlq["bucket"], y=dlq["accounts"],
            marker_color=[BUCKET_COLORS[b] for b in dlq["bucket"]],
            customdata=np.stack([dlq["exposure"] / 1e5, dlq["accounts"] / max(len(open_l), 1)],
                                axis=1),
            hovertemplate="%{x}<br>%{y:,} accounts (%{customdata[1]:.2%} of open)"
                          "<br>Exposure ₹%{customdata[0]:,.1f} L<extra></extra>"))
        style(fig, "Delinquent open accounts by bucket", x_title="Delinquency bucket",
              y_title="Accounts", legend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{num(dist.loc[0, 'accounts'])} open accounts are current and not shown.")
    with c2:
        t = trends.tail(24)
        fig = go.Figure()
        fig.add_bar(x=t["snapshot_date"], y=t["recoveries"] / 1e5, name="Post-charge-off "
                    "recoveries", marker_color=ACCENT,
                    hovertemplate="%{x}<br>₹%{y:,.1f} L<extra>Recoveries</extra>")
        fig.add_scatter(x=t["snapshot_date"], y=t["collections_cash"] / 1e5,
                        name="Cash from delinquent accounts", mode="lines", line_color=INK,
                        hovertemplate="%{x}<br>₹%{y:,.1f} L<extra>Delinquent cash</extra>")
        style(fig, "Recovery performance, last 24 months", y_title="₹ lakh per month")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        open_l = loans[loans["loan_status"].isin(K.OPEN_STATUSES)].copy()
        comp = (open_l.groupby(["risk_grade", "loan_status"])["current_outstanding_principal"]
                .sum().reset_index())
        fig = go.Figure()
        for st_, lab in zip(K.OPEN_STATUSES, ["Current", "1-29 DPD", "30-59 DPD", "60-89 DPD",
                                               "90+ DPD"]):
            s = comp[comp["loan_status"] == st_]
            fig.add_bar(x=s["risk_grade"], y=s["current_outstanding_principal"] / 1e7, name=lab,
                        marker_color=BUCKET_COLORS[lab],
                        hovertemplate="Grade %{x}<br>₹%{y:,.2f} Cr<extra>" + lab + "</extra>")
        fig.update_layout(barmode="stack")
        style(fig, "Exposure by risk grade and delinquency", x_title="Origination risk grade",
              y_title="₹ crore outstanding")
        st.plotly_chart(fig, width="stretch")
    with c4:
        seg = K.segment_table(loans, "loan_product")
        seg = seg.sort_values("outstanding_exposure")
        fig = go.Figure(go.Bar(
            y=seg["segment"], x=seg["exposure_share"], orientation="h", marker_color=ACCENT,
            customdata=np.stack([seg["delinquency_rate"], seg["outstanding_exposure"] / 1e7,
                                 seg["open_accounts"]], axis=1),
            hovertemplate="%{y}<br>%{x:.1%} of exposure (₹%{customdata[1]:,.2f} Cr)"
                          "<br>30+ DPD rate %{customdata[0]:.2%}<br>%{customdata[2]:,} open "
                          "accounts<extra></extra>"))
        style(fig, "Exposure concentration by product", x_title="Share of outstanding exposure",
              legend=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Priority insights")
    st.caption("Rule-based observations are computed from observed data with fixed thresholds; "
               "model estimates are labelled as such.")
    insights(_insights(loans, trends, queue, k))

    c5, c6 = st.columns([1, 1])
    with c5:
        st.subheader("Segments requiring investigation")
        _investigate(loans)
    with c6:
        st.subheader("Highest-priority eligible accounts")
        top = queue[queue["priority_score"].notna()].head(10)
        if top.empty:
            empty_state("No eligible accounts with positive expected net value for these "
                        "filters.")
        else:
            st.dataframe(
                top[["loan_id", "delinquency_bucket", "outstanding_principal",
                     "recommended_action", "expected_net_recovery"]],
                hide_index=True, width="stretch",
                column_config={
                    "loan_id": "Loan", "delinquency_bucket": "Stage",
                    "outstanding_principal": st.column_config.NumberColumn(
                        "Outstanding (₹)", format="localized"),
                    "recommended_action": "Recommended action",
                    "expected_net_recovery": st.column_config.NumberColumn(
                        "Est. net value (₹)", format="localized",
                        help="Model estimate: incremental cash vs no action, minus cost")})
            st.caption("Full queue in Collections Workbench.")


def _insights(loans: pd.DataFrame, trends: pd.DataFrame, queue: pd.DataFrame,
              k: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if len(trends) >= 4:
        r = trends["delinquency_rate_30plus"]
        chg = r.iloc[-1] - r.iloc[-4]
        direction = "risen" if chg > 0 else "fallen"
        out.append(("Rule-based observation: trend",
                    f"The 30+ DPD rate has {direction} by {abs(chg) * 100:.2f} percentage points "
                    f"over the last three months, to {pct(r.iloc[-1], 2)}."))
    seg = K.segment_table(loans, "loan_product")
    seg = seg[~seg["small_sample"] & seg["delinquency_rate"].notna()]
    if not seg.empty and pd.notna(k["delinquency_rate_30plus"]):
        w = seg.sort_values("delinquency_rate", ascending=False).iloc[0]
        ratio = w["delinquency_rate"] / k["delinquency_rate_30plus"]
        if ratio > 1.1:
            out.append(("Rule-based observation: concentration",
                        f"{w['segment']} has the highest 30+ DPD rate ({pct(w['delinquency_rate'])}"
                        f", {ratio:.1f}× the portfolio) and holds "
                        f"{pct(w['exposure_share'])} of open exposure."))
    bands = K.add_bands(loans)
    dti = K.segment_table(bands, "dti_band")
    dti = dti[~dti["small_sample"]]
    if len(dti) > 1:
        hi, lo = dti.iloc[-1], dti.iloc[0]
        out.append(("Rule-based observation: borrower segment",
                    f"Borrowers with DTI {hi['segment']} default at {pct(hi['default_rate'])} "
                    f"(95% CI {pct(hi['default_rate_ci_low'])}–{pct(hi['default_rate_ci_high'])}) "
                    f"versus {pct(lo['default_rate'])} for DTI {lo['segment']}."))
    el = queue[queue["eligible"]]
    if not el.empty:
        hp = queue[queue["priority_category"] == "High"]
        low = el[el["expected_net_recovery"] <= 0]
        out.append(("Model estimate: prioritisation",
                    f"{num(len(hp))} eligible accounts are High priority, carrying "
                    f"{inr(hp['priority_score'].sum(), True)} of estimated net value. "
                    f"{num(len(low))} eligible accounts show no positive value from any permitted "
                    "action this period."))
        both = el.dropna(subset=["risk_probability", "expected_net_recovery"])
        if len(both) >= 50:
            rho = both["risk_probability"].rank().corr(both["expected_net_recovery"].rank())
            top_risk = set(both.nlargest(100, "risk_probability")["loan_id"])
            top_val = set(both.nlargest(100, "expected_net_recovery")["loan_id"])
            overlap = len(top_risk & top_val)
            out.append(("Model estimate: risk vs value",
                        f"Among {num(len(both))} eligible 1-89 DPD accounts, roll-forward risk and "
                        f"estimated net value have a rank correlation of {rho:+.2f}; only "
                        f"{overlap} of the 100 highest-risk accounts are also among the 100 "
                        "highest-value ones. Ranking by risk alone would work a different set "
                        "of accounts."))
    fc = data.report("delinquency_forecast.json")
    if fc.get("available") and not filters.selected()[0] and not filters.selected()[1]:
        out.append(("Model estimate: forecast (synthetic series)",
                    f"{fc['selected_method']} projects a 30+ DPD rate of "
                    f"{pct(fc['forecast'][-1], 2)} by {fmt_date(fc['forecast_dates'][-1])} "
                    f"(95% interval {pct(fc['lower'][-1], 2)}–{pct(fc['upper'][-1], 2)})."))
    return out


def _investigate(loans: pd.DataFrame) -> None:
    d = loans.copy()
    d["segment"] = d["loan_product"] + " · grade " + d["risk_grade"]
    seg = K.segment_table(d, "segment")
    base = K.portfolio_kpis(loans)["delinquency_rate_30plus"]
    flag = seg[(~seg["small_sample"]) & (seg["delinquency_rate"] > 1.5 * base)]
    flag = flag.sort_values("delinquency_rate", ascending=False)
    if flag.empty:
        empty_state("No segment (≥100 accounts) has a 30+ DPD rate above 1.5× the portfolio "
                    "rate.")
        return
    st.dataframe(flag[["segment", "open_accounts", "delinquency_rate", "default_rate",
                       "outstanding_exposure"]].head(10), hide_index=True,
                 width="stretch",
                 column_config={
                     "segment": "Segment", "open_accounts": st.column_config.NumberColumn(
                         "Open accounts", format="localized"),
                     "delinquency_rate": st.column_config.NumberColumn("30+ DPD rate",
                                                                       format="percent"),
                     "default_rate": st.column_config.NumberColumn("Default rate",
                                                                   format="percent"),
                     "outstanding_exposure": st.column_config.NumberColumn(
                         "Exposure (₹)", format="localized")})
    st.caption(f"Rule: segments with ≥100 accounts and a 30+ DPD rate above 1.5× the "
               f"portfolio rate ({pct(base, 2)}).")


__all__ = ["render", "ADVERSE"]
