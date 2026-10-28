"""Portfolio, collections and segment KPIs. Every metric documents its numerator,
denominator and observation period (see docs/KPI_DICTIONARY.md)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

OPEN_STATUSES = ["Current", "Delinquent 1-29", "Delinquent 30-59", "Delinquent 60-89",
                 "Delinquent 90+"]
DLQ30_STATUSES = ["Delinquent 30-59", "Delinquent 60-89", "Delinquent 90+"]
DLQ_BUCKETS = ["1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD"]
DLQ30_BUCKETS = ["30-59 DPD", "60-89 DPD", "90+ DPD"]
OPEN_BUCKETS = ["Current"] + DLQ_BUCKETS


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def portfolio_kpis(loans: pd.DataFrame) -> dict[str, Any]:
    """As-of portfolio KPIs from loan-level as-of fields."""
    is_open = loans["loan_status"].isin(OPEN_STATUSES)
    is_co = loans["loan_status"] == "Charged Off"
    n_open = int(is_open.sum())
    dlq30 = int(loans["loan_status"].isin(DLQ30_STATUSES).sum())
    dlq1 = int((is_open & (loans["loan_status"] != "Current")).sum())
    co_principal = float(loans.loc[is_co, "charged_off_principal"].sum())
    recovered = float(loans.loc[is_co, "final_recovery_amount"].sum())
    return {
        "total_accounts": int(len(loans)),
        "open_accounts": n_open,
        "original_principal_total": float(loans["original_principal"].sum()),
        "outstanding_exposure": float(loans.loc[is_open, "current_outstanding_principal"].sum()),
        "delinquent_30plus_accounts": dlq30,
        "delinquency_rate_30plus": dlq30 / n_open if n_open else np.nan,
        "delinquent_1plus_accounts": dlq1,
        "delinquency_rate_1plus": dlq1 / n_open if n_open else np.nan,
        "delinquent_exposure_30plus": float(loans.loc[loans["loan_status"].isin(DLQ30_STATUSES),
                                                      "current_outstanding_principal"].sum()),
        "charged_off_accounts": int(is_co.sum()),
        "default_rate": float(is_co.mean()) if len(loans) else np.nan,
        "charged_off_principal": co_principal,
        "observed_recovery_amount": recovered,
        "observed_recovery_rate": recovered / co_principal if co_principal else np.nan,
        "average_loan_amount": float(loans["loan_amount"].mean()),
        "average_dti": float(loans["debt_to_income_ratio"].mean()),
    }


def monthly_trends(port: pd.DataFrame) -> pd.DataFrame:
    """Month-level portfolio indicators from ``portfolio_snapshots``."""
    p = port.copy()
    p["open"] = p["delinquency_bucket"].isin(OPEN_BUCKETS)
    p["dlq30"] = p["delinquency_bucket"].isin(DLQ30_BUCKETS)
    p["dlq1"] = p["delinquency_bucket"].isin(DLQ_BUCKETS)
    p["co"] = p["delinquency_bucket"] == "Charged Off"
    g = p.groupby("snapshot_date")

    def s(mask: str, col: str) -> pd.Series:
        return p[p[mask]].groupby("snapshot_date")[col].sum()

    out = pd.DataFrame({
        "open_accounts": s("open", "account_count"),
        "outstanding_exposure": s("open", "outstanding_principal_total"),
        "dlq30_accounts": s("dlq30", "account_count"),
        "dlq1_accounts": s("dlq1", "account_count"),
        "delinquent_exposure": g["delinquent_exposure"].sum(),
        "charge_offs": g["defaulted_account_count"].sum(),
        "recoveries": s("co", "observed_recovery_total"),
        "collections_cash": s("dlq1", "observed_recovery_total"),
        "scheduled_cash": s("open", "observed_recovery_total"),
    }).fillna(0.0)
    out["delinquency_rate_30plus"] = out["dlq30_accounts"] / out["open_accounts"].replace(0, np.nan)
    out["delinquency_rate_1plus"] = out["dlq1_accounts"] / out["open_accounts"].replace(0, np.nan)
    return out.reset_index()


def segment_table(loans: pd.DataFrame, by: str, min_n: int = 100) -> pd.DataFrame:
    """Segment performance with consistent denominators.

    * Delinquency rate = open accounts 30+ DPD / open accounts (as-of)
    * Default rate = charged-off loans / loans originated in segment (cumulative)
    * Observed recovery rate = cumulative recoveries / charged-off principal
    """
    d = loans.copy()
    d["is_open"] = d["loan_status"].isin(OPEN_STATUSES)
    d["is_dlq30"] = d["loan_status"].isin(DLQ30_STATUSES)
    d["is_co"] = d["loan_status"] == "Charged Off"
    d["open_exposure"] = np.where(d["is_open"], d["current_outstanding_principal"], 0.0)
    g = d.groupby(by, observed=True).agg(
        account_count=("loan_id", "size"), original_principal=("original_principal", "sum"),
        open_accounts=("is_open", "sum"), outstanding_exposure=("open_exposure", "sum"),
        delinquent_accounts=("is_dlq30", "sum"), defaulted_accounts=("is_co", "sum"),
        charged_off_principal=("charged_off_principal", "sum"),
        observed_recovery_amount=("final_recovery_amount", "sum")).reset_index()
    g = g.rename(columns={by: "segment"})
    g["delinquency_rate"] = g["delinquent_accounts"] / g["open_accounts"].replace(0, np.nan)
    g["default_rate"] = g["defaulted_accounts"] / g["account_count"]
    g["observed_recovery_rate"] = (g["observed_recovery_amount"]
                                   / g["charged_off_principal"].replace(0, np.nan))
    ci = [wilson_ci(k, n) for k, n in zip(g["defaulted_accounts"], g["account_count"])]
    g["default_rate_ci_low"] = [c[0] for c in ci]
    g["default_rate_ci_high"] = [c[1] for c in ci]
    g["exposure_share"] = g["outstanding_exposure"] / max(g["outstanding_exposure"].sum(), 1e-9)
    g["small_sample"] = g["account_count"] < min_n
    for c in ["original_principal", "outstanding_exposure", "charged_off_principal",
              "observed_recovery_amount"]:
        g[c] = g[c].round(0)
    g.insert(0, "segment_type", by)
    return g


def add_bands(loans: pd.DataFrame) -> pd.DataFrame:
    """Add reporting bands used for segmentation."""
    d = loans.copy()
    d["income_band"] = pd.cut(d["annual_income"], [0, 3e5, 5e5, 8e5, 1.2e6, np.inf],
                              labels=["< ₹3L", "₹3-5L", "₹5-8L", "₹8-12L", "≥ ₹12L"])
    d["dti_band"] = pd.cut(d["debt_to_income_ratio"], [-0.01, 0.2, 0.35, 0.5, 0.65, 1.5],
                           labels=["< 20%", "20-35%", "35-50%", "50-65%", "≥ 65%"])
    d["utilization_band"] = pd.cut(d["credit_utilization_ratio"], [-0.01, 0.3, 0.5, 0.7, 1.0],
                                   labels=["< 30%", "30-50%", "50-70%", "≥ 70%"])
    od = pd.to_datetime(d["origination_date"])
    d["origination_cohort"] = od.dt.year.astype(str) + "-Q" + od.dt.quarter.astype(str)
    d["term_band"] = d["loan_term_months"].astype(str) + " months"
    return d


def vintage_base(snaps: pd.DataFrame, loans: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Loan-level first month-on-book reaching 30+ DPD and charge-off."""
    s = snaps[["loan_id", "snapshot_date", "delinquency_bucket"]].merge(
        loans[["loan_id", "origination_date"]], on="loan_id")
    sd = pd.to_datetime(s["snapshot_date"])
    od = pd.to_datetime(s["origination_date"])
    s["mob"] = (sd.dt.year - od.dt.year) * 12 + (sd.dt.month - od.dt.month)
    bad30 = s[s["delinquency_bucket"].isin(DLQ30_BUCKETS + ["Charged Off"])]
    first30 = bad30.groupby("loan_id")["mob"].min().rename("first_30plus_mob")
    firstco = (s[s["delinquency_bucket"] == "Charged Off"].groupby("loan_id")["mob"].min()
               .rename("first_co_mob"))
    base = add_bands(loans)[["loan_id", "origination_cohort", "loan_product", "risk_grade",
                             "origination_date"]].copy()
    od = pd.to_datetime(base["origination_date"])
    a = pd.Timestamp(as_of)
    base["observable_mob"] = (a.year - od.dt.year) * 12 + (a.month - od.dt.month)
    return base.merge(first30, on="loan_id", how="left").merge(firstco, on="loan_id", how="left")


def vintage_curves(base: pd.DataFrame, metric: str = "first_30plus_mob",
                   max_mob: int = 36) -> pd.DataFrame:
    """Cumulative share of a cohort reaching the event by month-on-book.

    A cell is shown only if every loan in the cohort has been observable for that
    many months, so cohorts are compared only over comparable account ages.
    """
    rows = []
    for cohort, g in base.groupby("origination_cohort"):
        n = len(g)
        max_obs = int(g["observable_mob"].min())
        for m in range(1, max_mob + 1):
            val = (g[metric] <= m).sum() / n if m <= max_obs else np.nan
            rows.append({"cohort": cohort, "mob": m, "rate": val, "accounts": n})
    return pd.DataFrame(rows)


def roll_rate_counts(snaps: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month bucket transitions (counts) by product and grade."""
    s = snaps[["loan_id", "snapshot_date", "delinquency_bucket"]].sort_values(
        ["loan_id", "snapshot_date"])
    s["from_bucket"] = s.groupby("loan_id")["delinquency_bucket"].shift(1)
    s["prev_date"] = s.groupby("loan_id")["snapshot_date"].shift(1)
    s = s.dropna(subset=["from_bucket"])
    # Only consecutive month-ends are true transitions.
    gap = (pd.to_datetime(s["snapshot_date"]) - pd.to_datetime(s["prev_date"])).dt.days
    s = s[gap <= 31]
    s = s.merge(loans[["loan_id", "loan_product", "risk_grade"]], on="loan_id")
    s = s[s["from_bucket"].isin(OPEN_BUCKETS)]
    return (s.groupby(["snapshot_date", "loan_product", "risk_grade", "from_bucket",
                       "delinquency_bucket"]).size().rename("transitions").reset_index()
            .rename(columns={"delinquency_bucket": "to_bucket"}))


def roll_rate_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised transition matrix from aggregated counts."""
    order_from = OPEN_BUCKETS
    order_to = OPEN_BUCKETS + ["Charged Off", "Closed"]
    m = counts.pivot_table(index="from_bucket", columns="to_bucket", values="transitions",
                           aggfunc="sum", fill_value=0)
    m = m.reindex(index=order_from, columns=order_to, fill_value=0)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0)


def collections_kpis(events: pd.DataFrame, since: str | None = None) -> dict[str, Any]:
    """Collections operations KPIs from collection events (synthetic)."""
    e = events if since is None else events[events["event_date"] >= since]
    if e.empty:
        return {}
    phone = e[e["contact_channel"] == "Phone"]
    rpc = phone["contact_outcome"].isin(["Right-party contact", "Promise to pay"])
    ptp = e["promise_to_pay_indicator"] == 1
    closed_ptp = e[ptp & e["promise_fulfilled_indicator"].notna()]
    return {
        "events": int(len(e)),
        "accounts_worked": int(e["loan_id"].nunique()),
        "contact_success_rate": float(rpc.mean()) if len(phone) else np.nan,
        "promise_to_pay_rate": float(ptp[phone.index].sum() / max(rpc.sum(), 1)),
        "promise_fulfillment_rate": float(closed_ptp["promise_fulfilled_indicator"].mean())
        if len(closed_ptp) else np.nan,
        "recovery_per_action": float(e["subsequent_payment_amount"].mean()),
        "average_action_cost": float(e["estimated_action_cost"].mean()),
        "net_recovery_value": float((e["subsequent_payment_amount"]
                                     - e["estimated_action_cost"]).sum()),
        "actions_per_account": float(len(e) / e["loan_id"].nunique()),
        "payment_rate_30d": float((e["subsequent_payment_amount"] > 0).mean()),
    }


def action_performance(events: pd.DataFrame) -> pd.DataFrame:
    """Observed outcomes by stage and action (associations, not causal effects)."""
    e = events.copy()
    e["paid"] = e["subsequent_payment_amount"] > 0
    g = e.groupby(["delinquency_stage_at_event", "action_type"]).agg(
        events=("event_id", "size"), payment_rate=("paid", "mean"),
        avg_payment=("subsequent_payment_amount", "mean"),
        avg_cost=("estimated_action_cost", "mean")).reset_index()
    g["avg_net"] = g["avg_payment"] - g["avg_cost"]
    return g
