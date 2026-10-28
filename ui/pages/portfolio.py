"""Portfolio Explorer: segments, vintages, roll rates, trends and drill-down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import kpis as K
from ui import data, filters
from ui.charts.plots import ACCENT, ADVERSE_SEQ, BUCKET_COLORS, INK, SLATE, style
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
)

STATUS_TO_STAGE = {"Current": "Current", "Delinquent 1-29": "1-29 DPD",
                   "Delinquent 30-59": "30-59 DPD", "Delinquent 60-89": "60-89 DPD",
                   "Delinquent 90+": "90+ DPD", "Charged Off": "Charged Off",
                   "Fully Paid": "Fully Paid"}
DIMENSIONS = {"Loan product": "loan_product", "Risk grade": "risk_grade",
              "Loan purpose": "loan_purpose", "Loan term": "term_band",
              "Income band": "income_band", "Debt-to-income band": "dti_band",
              "Utilisation band": "utilization_band", "Origination cohort": "origination_cohort",
              "Region (reporting only)": "region", "Delinquency stage": "stage"}


def render() -> None:
    page_header("Portfolio Explorer",
                "Segment performance, vintage behaviour and delinquency roll rates for the loan "
                "book.")
    all_loans = K.add_bands(filters.apply(data.loans()))
    all_loans["stage"] = all_loans["loan_status"].map(STATUS_TO_STAGE)
    cohorts = sorted(all_loans["origination_cohort"].unique())
    c1, c2, c3 = st.columns([2, 2, 1.4])
    sel_cohorts = c1.multiselect("Origination cohort", cohorts, placeholder="All cohorts")
    sel_stage = c2.multiselect("Delinquency stage (as of)", list(STATUS_TO_STAGE.values()),
                               placeholder="All stages")
    od = pd.to_datetime(all_loans["origination_date"])
    rng = c3.date_input("Origination dates", (od.min().date(), od.max().date()),
                        min_value=od.min().date(), max_value=od.max().date())
    loans = all_loans
    if sel_cohorts:
        loans = loans[loans["origination_cohort"].isin(sel_cohorts)]
    if sel_stage:
        loans = loans[loans["stage"].isin(sel_stage)]
    if isinstance(rng, tuple) and len(rng) == 2:
        od = pd.to_datetime(loans["origination_date"])
        loans = loans[(od >= pd.Timestamp(rng[0])) & (od <= pd.Timestamp(rng[1]))]
    provenance_band([("Accounts in view", num(len(loans))), ("Global filters", filters.describe())])
    if loans.empty:
        empty_state("No accounts match these filters. Widen the cohort, stage or date range.")
        return

    k = K.portfolio_kpis(loans)
    kpi_strip([
        {"label": "Accounts", "value": num(k["total_accounts"]),
         "detail": f"{num(k['open_accounts'])} open"},
        {"label": "Original principal", "value": inr(k["original_principal_total"], True),
         "detail": "at origination"},
        {"label": "Outstanding exposure", "value": inr(k["outstanding_exposure"], True),
         "detail": "open accounts"},
        {"label": "30+ DPD rate", "value": pct(k["delinquency_rate_30plus"], 2),
         "detail": f"{num(k['delinquent_30plus_accounts'])} of open accounts"},
        {"label": "Default rate", "value": pct(k["default_rate"], 2),
         "detail": f"{num(k['charged_off_accounts'])} charged off (cumulative)"},
    ])
    methodology("Definitions and denominators", (
        "* **30+ DPD rate** = open accounts 30 or more days past due ÷ open accounts at the "
        "as-of date.\n* **Default rate** = loans charged off ÷ loans originated in the segment "
        "(cumulative; younger cohorts have had less time to default).\n* **Observed recovery "
        "rate** = cumulative post-charge-off recoveries ÷ charged-off principal.\n"
        "* **Buckets:** Current 0, 1-29, 30-59, 60-89, 90+ days past due; charge-off after six "
        "consecutive missed instalments.\n* Segments under 100 accounts are flagged as small "
        "samples."))

    t1, t2, t3, t4 = st.tabs(["Segments", "Vintages", "Roll rates", "Trend & forecast"])
    with t1:
        _segments(loans)
    with t2:
        _vintage(loans)
    with t3:
        _roll_rates(loans)
    with t4:
        _trend()

    st.subheader("Drill-down: collections-stage accounts in view")
    q = data.default_queue()
    q = q[q["loan_id"].isin(loans["loan_id"])]
    if q.empty:
        empty_state("No accounts in a collections stage within the current view.")
    else:
        show = q[["loan_id", "loan_product", "risk_grade", "delinquency_bucket", "days_past_due",
                  "outstanding_principal", "eligibility_status", "recommended_action",
                  "expected_net_recovery"]]
        st.dataframe(show, hide_index=True, width="stretch", height=300,
                     column_config={
                         "outstanding_principal": st.column_config.NumberColumn(
                             "Outstanding (₹)", format="localized"),
                         "expected_net_recovery": st.column_config.NumberColumn(
                             "Est. net value (₹)", format="localized")})
        download_csv(show, "Download drill-down (CSV)", "collectiq_portfolio_drilldown.csv",
                     "dl_portfolio")


def _segments(loans: pd.DataFrame) -> None:
    dim_label = st.selectbox("Segment by", list(DIMENSIONS), index=0)
    seg = K.segment_table(loans, DIMENSIONS[dim_label])
    if dim_label.startswith("Region"):
        note("Region is shown for reporting and fairness monitoring only; it is not used by any "
             "model or collection rule.")
    c1, c2 = st.columns(2)
    with c1:
        labels = seg["segment"].astype(str).str.replace(" ", "<br>", regex=False)
        fig = go.Figure()
        fig.add_bar(x=labels, y=seg["delinquency_rate"], name="30+ DPD rate",
                    marker_color=ACCENT,
                    customdata=seg[["open_accounts"]].to_numpy(),
                    hovertemplate="%{x}<br>%{y:.2%}<br>n open = %{customdata[0]:,}<extra></extra>")
        fig.add_scatter(x=labels, y=seg["default_rate"], name="Default rate",
                        mode="markers", marker={"color": INK, "size": 9, "symbol": "diamond"},
                        error_y={"type": "data", "symmetric": False,
                                 "array": seg["default_rate_ci_high"] - seg["default_rate"],
                                 "arrayminus": seg["default_rate"] - seg["default_rate_ci_low"],
                                 "color": SLATE},
                        hovertemplate="%{x}<br>Default %{y:.2%}<extra>95% Wilson CI shown"
                                      "</extra>")
        style(fig, f"Delinquency and default by {dim_label.lower()}", height=360, y_title="Rate")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(tickangle=0, automargin=True)
        fig.update_layout(legend={"y": -0.32})
        st.plotly_chart(fig, width="stretch")
    with c2:
        s2 = seg.sort_values("outstanding_exposure", ascending=True)
        fig = go.Figure(go.Bar(y=s2["segment"].astype(str), x=s2["outstanding_exposure"] / 1e7,
                               orientation="h", marker_color=INK,
                               customdata=s2[["exposure_share"]].to_numpy(),
                               hovertemplate="%{y}<br>₹%{x:,.2f} Cr (%{customdata[0]:.1%})"
                                             "<extra></extra>"))
        style(fig, "Exposure concentration", x_title="₹ crore outstanding", legend=False)
        st.plotly_chart(fig, width="stretch")
    cols = ["segment", "account_count", "original_principal", "outstanding_exposure",
            "delinquent_accounts", "delinquency_rate", "default_rate",
            "observed_recovery_amount", "observed_recovery_rate", "small_sample"]
    st.dataframe(seg[cols], hide_index=True, width="stretch", column_config={
        "segment": "Segment", "account_count": st.column_config.NumberColumn("Accounts"),
        "original_principal": st.column_config.NumberColumn("Original principal (₹)",
                                                            format="localized"),
        "outstanding_exposure": st.column_config.NumberColumn("Outstanding (₹)",
                                                              format="localized"),
        "delinquent_accounts": st.column_config.NumberColumn("30+ DPD accounts"),
        "delinquency_rate": st.column_config.NumberColumn("30+ DPD rate", format="percent"),
        "default_rate": st.column_config.NumberColumn("Default rate", format="percent"),
        "observed_recovery_amount": st.column_config.NumberColumn("Recovered (₹)",
                                                                  format="localized"),
        "observed_recovery_rate": st.column_config.NumberColumn("Recovery rate",
                                                                format="percent"),
        "small_sample": st.column_config.CheckboxColumn("Small sample (<100)")})
    download_csv(seg, "Download segment table (CSV)", "collectiq_segments.csv", "dl_seg")


def _vintage(loans: pd.DataFrame) -> None:
    base = data.parquet("vintage_base.parquet")
    base = base[base["loan_id"].isin(loans["loan_id"])]
    if base.empty:
        empty_state("No vintage history for the selected accounts.")
        return
    metric = st.radio("Metric", ["Ever 30+ DPD", "Charged off"], horizontal=True)
    col = "first_30plus_mob" if metric == "Ever 30+ DPD" else "first_co_mob"
    cur = K.vintage_curves(base, col, max_mob=30)
    piv = cur.pivot(index="cohort", columns="mob", values="rate")
    sizes = cur.groupby("cohort")["accounts"].first()
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=piv.columns, y=[f"{c} (n={sizes[c]:,})" for c in piv.index],
        colorscale=ADVERSE_SEQ, colorbar={"title": "Cumulative", "tickformat": ".0%"},
        hovertemplate="%{y}<br>Month on book %{x}<br>%{z:.2%}<extra></extra>",
        zmin=0))
    style(fig, f"Cumulative share {metric.lower()} by origination quarter and month on book",
          height=420, x_title="Months on book", legend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Blank cells: the cohort has not yet been observed for that many months. Cohorts "
               "are compared only at account ages every loan in the cohort has reached.")


def _roll_rates(loans: pd.DataFrame) -> None:
    rr = filters.apply(data.parquet("roll_rate_counts.parquet"))
    if rr.empty:
        empty_state("Roll rates need consecutive monthly snapshots; none are available.")
        return
    dates = sorted(rr["snapshot_date"].unique())
    start, end = st.select_slider("Transition months (to month-end)", options=dates,
                                  value=(dates[-12] if len(dates) > 12 else dates[0], dates[-1]))
    sub = rr[(rr["snapshot_date"] >= start) & (rr["snapshot_date"] <= end)]
    m = K.roll_rate_matrix(sub)
    counts = sub.groupby("from_bucket")["transitions"].sum().reindex(m.index).fillna(0)
    fig = go.Figure(go.Heatmap(
        z=m.to_numpy(), x=m.columns, y=[f"{i} (n={int(counts[i]):,})" for i in m.index],
        colorscale=[[0, "#FFFFFF"], [0.25, "#C9E3E1"], [1, "#0B4F50"]], zmin=0, zmax=1,
        text=np.vectorize(lambda v: "" if np.isnan(v) or v < 0.005 else f"{v:.0%}")(m.to_numpy()),
        texttemplate="%{text}", hovertemplate="From %{y}<br>to %{x}: %{z:.2%}<extra></extra>",
        colorbar={"tickformat": ".0%"}))
    style(fig, "Month-over-month roll-rate matrix (row = start bucket)", height=360,
          x_title="Bucket at next month-end", legend=False)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.caption("Computed only from consecutive month-end snapshots of the same account. "
               "Applies the sidebar product and grade filters; cohort and stage filters do not "
               "apply to transitions.")


def _trend() -> None:
    port = filters.apply(data.portfolio_snapshots())
    t = K.monthly_trends(port)
    fc = data.report("delinquency_forecast.json")
    fig = go.Figure()
    fig.add_scatter(x=t["snapshot_date"], y=t["delinquency_rate_30plus"], mode="lines",
                    name="Observed 30+ DPD rate (filtered view)", line={"color": INK})
    unfiltered = not filters.selected()[0] and not filters.selected()[1]
    if fc.get("available") and unfiltered:
        fig.add_scatter(x=fc["forecast_dates"], y=fc["upper"], mode="lines",
                        line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fc["forecast_dates"], y=fc["lower"], mode="lines", fill="tonexty",
                        fillcolor="rgba(11,110,110,0.15)", line={"width": 0},
                        name="95% interval")
        fig.add_scatter(x=fc["forecast_dates"], y=fc["forecast"], mode="lines+markers",
                        name=f"Forecast: {fc['selected_method']}",
                        line={"color": ACCENT, "dash": "dash"})
    style(fig, "30+ DPD rate: history and 6-month forecast", y_title="Share of open accounts")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    if not fc.get("available"):
        note(f"Forecast unavailable: {fc.get('reason', 'not computed')}")
        return
    if not unfiltered:
        note("The forecast is fitted on the whole portfolio; clear the sidebar filters to "
             "overlay it.")
    v = pd.DataFrame(fc["validation"])
    st.markdown(f"**Time-ordered validation** (last {len(fc['holdout_dates'])} months held out; "
                f"first {fc['burn_in_months']} ramp-up months excluded). Selected: "
                f"{fc['selected_method']}.")
    st.dataframe(v, hide_index=True, column_config={
        "method": "Method", "mae": st.column_config.NumberColumn("MAE (rate)", format="%.4f"),
        "mape": st.column_config.NumberColumn("MAPE", format="percent")})
    note("Forecast of a synthetic series. The simple naive baseline is competitive, which is "
         "expected with ~3 years of monthly history; treat the projection as indicative.")


__all__ = ["render", "BUCKET_COLORS"]
