"""Leakage-safe feature engineering.

Prediction timestamps
---------------------
* Risk model (Model A): month-end snapshot date ``t``. Features use only snapshot
  rows dated <= t plus origination attributes. Target looks at t+1 .. t+3.
* Recovery models (B, C): unit = a *collection opportunity*: an account in a
  collections stage (1-89 DPD, 90+ DPD or charged off within the recovery window)
  at month-end ``t-1``. The decision is the action taken in month ``t`` - one of the
  five actions or "No action" (historical months without an event). Features use
  the snapshot at ``t-1`` and collection history whose 30-day outcome windows had
  closed before the start of month ``t``. Outcome = cash received in month ``t``.

Outcome / post-event fields listed in ``PROHIBITED_FEATURES`` must never appear in
any feature list; this is enforced by ``assert_no_leakage`` and by tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKET_ORDER = {"Current": 0, "1-29 DPD": 1, "30-59 DPD": 2, "60-89 DPD": 3, "90+ DPD": 4,
                "Charged Off": 5, "Closed": -1}
OPEN_PRE_SERIOUS = ["1-29 DPD", "30-59 DPD", "60-89 DPD"]  # Model A population
SERIOUS = ["90+ DPD", "Charged Off"]

ORIGINATION_NUMERIC = [
    "bureau_score", "interest_rate", "loan_term_months", "debt_to_income_ratio",
    "payment_to_income_ratio", "log_annual_income", "employment_length_years",
    "credit_history_length_months", "credit_utilization_ratio", "prior_delinquency_count",
    "open_credit_accounts", "log_loan_amount",
]
BEHAVIOUR_NUMERIC = [
    "months_on_book", "days_past_due", "bucket_ordinal", "missed_last_3m", "missed_last_6m",
    "max_dpd_6m", "times_30plus_12m", "months_since_last_missed", "payment_ratio_3m",
    "outstanding_ratio", "remaining_term_ratio",
]
RISK_NUMERIC = ORIGINATION_NUMERIC + BEHAVIOUR_NUMERIC
RISK_CATEGORICAL = ["loan_product", "risk_grade"]

RECOVERY_NUMERIC = ORIGINATION_NUMERIC + [
    "months_on_book", "days_past_due", "missed_last_6m", "max_dpd_6m",
    "months_since_last_payment", "log_collectible_balance", "log_amount_due", "log_emi",
    "months_since_charge_off", "prior_events_90d", "prior_ptp_count",
    "prior_ptp_fulfilled_rate",
]
NO_ACTION = "No action"
COLLECTION_STAGES = ["1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD", "Charged Off"]
RECOVERY_CATEGORICAL = ["loan_product", "risk_grade", "stage", "action_type"]

# Fields that are outcomes or only observable after the decision point.
PROHIBITED_FEATURES = {
    "loan_status", "final_recovery_amount", "final_recovery_date", "charge_off_date",
    "charged_off_principal", "current_outstanding_principal", "contact_outcome",
    "promise_to_pay_indicator", "promise_to_pay_amount", "promise_fulfilled_indicator",
    "subsequent_payment_amount", "observed_recovery_amount", "net_observed_recovery",
    "recovery_success_indicator", "target", "target_amount", "payment_received_next",
    "account_status", "region",  # region excluded by fairness policy, not leakage
}


def assert_no_leakage(features: list[str]) -> None:
    """Raise if any prohibited field is used as a feature."""
    bad = sorted(set(features) & PROHIBITED_FEATURES)
    if bad:
        raise ValueError(f"Leakage / policy violation - prohibited features: {bad}")


def _months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return ((later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month))


def origination_features(loans: pd.DataFrame) -> pd.DataFrame:
    """Origination-time attributes (available at every later prediction date)."""
    out = loans[["loan_id", "loan_product", "risk_grade", "bureau_score", "interest_rate",
                 "loan_term_months", "debt_to_income_ratio", "payment_to_income_ratio",
                 "employment_length_years", "credit_history_length_months",
                 "credit_utilization_ratio", "prior_delinquency_count", "open_credit_accounts",
                 "original_principal", "emi_amount", "origination_date", "region"]].copy()
    out["log_annual_income"] = np.log(loans["annual_income"].astype(float))
    out["log_loan_amount"] = np.log(loans["loan_amount"].astype(float))
    return out


def snapshot_features(snaps: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features for every loan-month, using history up to that month only."""
    s = snaps.sort_values(["loan_id", "snapshot_date"]).reset_index(drop=True).copy()
    g = s.groupby("loan_id", sort=False)
    missed = s["missed_payment_indicator"].astype(float)
    cs = missed.groupby(s["loan_id"]).cumsum()

    def window_sum(cumsum: pd.Series, k: int) -> pd.Series:
        prev = cumsum.groupby(s["loan_id"]).shift(k).fillna(0.0)
        return cumsum - prev

    s["missed_last_3m"] = window_sum(cs, 3)
    s["missed_last_6m"] = window_sum(cs, 6)
    dpd = s["days_past_due"].astype(float)
    s["max_dpd_6m"] = pd.concat([dpd] + [dpd.groupby(s["loan_id"]).shift(i) for i in range(1, 6)],
                                axis=1).max(axis=1)
    is30 = s["delinquency_bucket"].isin(["30-59 DPD", "60-89 DPD", "90+ DPD"]).astype(float)
    cs30 = is30.groupby(s["loan_id"]).cumsum()
    s["times_30plus_12m"] = window_sum(cs30, 12)
    pay = s["payment_received"].astype(float).groupby(s["loan_id"]).cumsum()
    sch = s["scheduled_payment"].astype(float).groupby(s["loan_id"]).cumsum()
    pay3 = window_sum(pay, 3)
    sch3 = window_sum(sch, 3)
    s["payment_ratio_3m"] = np.clip(np.where(sch3 > 0, pay3 / sch3.replace(0, np.nan), 1.0), 0, 3)
    row = g.cumcount()
    last_miss = row.where(missed > 0).groupby(s["loan_id"]).ffill()
    s["months_since_last_missed"] = (row - last_miss).fillna(99).clip(upper=99)
    last_pay = row.where(s["payment_received"] > 0).groupby(s["loan_id"]).ffill()
    s["months_since_last_payment"] = (row - last_pay).fillna(row + 1).clip(upper=99)
    s["bucket_ordinal"] = s["delinquency_bucket"].map(BUCKET_ORDER).astype(float)

    o = origination_features(loans)
    s = s.merge(o, on="loan_id", how="left")
    sd = pd.to_datetime(s["snapshot_date"])
    od = pd.to_datetime(s["origination_date"])
    s["months_on_book"] = _months_between(sd, od).astype(float)
    s["outstanding_ratio"] = s["outstanding_principal"] / s["original_principal"]
    s["remaining_term_ratio"] = np.clip((s["loan_term_months"] - s["months_on_book"])
                                        / s["loan_term_months"], 0, 1)
    first_co = (s.loc[s["delinquency_bucket"] == "Charged Off"]
                .groupby("loan_id")["snapshot_date"].min().rename("first_co_date"))
    s = s.merge(first_co, on="loan_id", how="left")
    co_dt = pd.to_datetime(s["first_co_date"])
    s["months_since_charge_off"] = np.where(co_dt.notna() & (sd >= co_dt),
                                            _months_between(sd, co_dt), -1).astype(float)
    missed_count = np.where(s["days_past_due"] > 0, s["days_past_due"] // 30 + 1, 0)
    amount_due = np.where(s["delinquency_bucket"] == "Charged Off", s["outstanding_principal"],
                          missed_count * s["emi_amount"])
    s["amount_due"] = amount_due
    s["log_amount_due"] = np.log1p(amount_due)
    s["log_collectible_balance"] = np.log1p(s["outstanding_principal"].astype(float))
    s["log_emi"] = np.log(s["emi_amount"].astype(float))
    return s


def risk_panel(sf: pd.DataFrame, horizon: int, last_label_date: str,
               sample_fraction: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Labelled panel for Model A.

    Population: month-end rows of open accounts that are 1-89 DPD (delinquent but
    not yet seriously delinquent). Current accounts are excluded because they
    cannot structurally reach 90+ DPD within 3 months (one bucket per month).
    Target: 1 if the account is 90+ DPD or charged off at any of the next
    ``horizon`` month-ends; accounts that close (repay) in the window are 0.
    """
    s = sf.sort_values(["loan_id", "snapshot_date"]).reset_index(drop=True)
    serious = s["delinquency_bucket"].isin(SERIOUS).astype(int)
    sd = pd.to_datetime(s["snapshot_date"])
    target = np.zeros(len(s), dtype=int)
    for k in range(1, horizon + 1):
        nxt = serious.groupby(s["loan_id"]).shift(-k)
        nxt_date = sd.groupby(s["loan_id"]).shift(-k)
        within = (nxt_date - sd).dt.days <= 31 * k + 3
        target |= ((nxt == 1) & within).to_numpy()
    s["target"] = target
    pop = s["delinquency_bucket"].isin(OPEN_PRE_SERIOUS) & (s["snapshot_date"] <= last_label_date)
    panel = s.loc[pop].copy()
    if sample_fraction < 1.0:
        panel = panel.sample(frac=sample_fraction, random_state=seed)
    return panel.sort_values("snapshot_date").reset_index(drop=True)


def _prior_event_stats(events: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """For each reference row (loan_id, ref_date) compute collection-history features
    from events whose 30-day outcome window closed on or before ref_date."""
    ev = events[["loan_id", "event_date", "promise_to_pay_indicator",
                 "promise_fulfilled_indicator"]].copy()
    ev["ev_dt"] = pd.to_datetime(ev["event_date"])
    r = ref[["key", "loan_id", "ref_date"]].copy()
    r["ref_dt"] = pd.to_datetime(r["ref_date"])
    m = r.merge(ev, on="loan_id", how="inner")
    m = m[m["ev_dt"] < m["ref_dt"]]
    closed = m["ev_dt"] + pd.Timedelta(days=30) <= m["ref_dt"]
    m["recent90"] = (m["ref_dt"] - m["ev_dt"]).dt.days <= 90
    m["ptp_closed"] = np.where(closed, m["promise_to_pay_indicator"].fillna(0), 0)
    m["ful_closed"] = np.where(closed, m["promise_fulfilled_indicator"].fillna(0), 0)
    agg = m.groupby("key").agg(prior_events_90d=("recent90", "sum"),
                               prior_ptp_count=("ptp_closed", "sum"),
                               prior_ptp_fulfilled=("ful_closed", "sum"),
                               last_event_date=("ev_dt", "max"))
    out = r[["key"]].merge(agg, on="key", how="left")
    out["prior_events_90d"] = out["prior_events_90d"].fillna(0)
    out["prior_ptp_count"] = out["prior_ptp_count"].fillna(0)
    out["prior_ptp_fulfilled_rate"] = np.where(
        out["prior_ptp_count"] > 0, out["prior_ptp_fulfilled"] / out["prior_ptp_count"].clip(1),
        np.nan)
    return out.drop(columns=["prior_ptp_fulfilled"])


def opportunity_features(events: pd.DataFrame, sf: pd.DataFrame) -> pd.DataFrame:
    """Collection opportunities with the action taken (or "No action") and outcome.

    Including untreated account-months lets the models estimate cash expected
    *without* an action, so the prioritisation can value an action by its
    estimated incremental effect rather than by cash that would arrive anyway.
    """
    s = sf.sort_values(["loan_id", "snapshot_date"]).reset_index(drop=True)
    nxt_date = s.groupby("loan_id")["snapshot_date"].shift(-1)
    nxt_pay = s.groupby("loan_id")["payment_received"].shift(-1)
    gap = (pd.to_datetime(nxt_date) - pd.to_datetime(s["snapshot_date"])).dt.days
    opp = s[s["delinquency_bucket"].isin(COLLECTION_STAGES) & (gap <= 31)].copy()
    opp["target_amount"] = nxt_pay[opp.index].astype(float)
    opp["outcome_month_end"] = nxt_date[opp.index]
    opp["decision_date"] = (pd.to_datetime(opp["snapshot_date"]) + pd.Timedelta(days=1)
                            ).dt.strftime("%Y-%m-%d")
    ev = events[["loan_id", "prediction_timestamp", "event_id", "action_type",
                 "subsequent_payment_amount"]].copy()
    ev["snapshot_date"] = ev["prediction_timestamp"].str[:10]
    opp = opp.merge(ev.drop(columns="prediction_timestamp"), on=["loan_id", "snapshot_date"],
                    how="left")
    opp["action_type"] = opp["action_type"].fillna(NO_ACTION)
    opp["stage"] = opp["delinquency_bucket"]
    opp["key"] = opp["loan_id"] + "|" + opp["snapshot_date"]
    opp["ref_date"] = opp["decision_date"]
    prior = _prior_event_stats(events, opp[["key", "loan_id", "ref_date"]])
    opp = opp.merge(prior, on="key", how="left")
    opp["target"] = (opp["target_amount"] > 0).astype(int)
    return opp.sort_values("decision_date").reset_index(drop=True)


def latest_account_state(sf: pd.DataFrame, events: pd.DataFrame, as_of: str,
                         planning_start: str) -> pd.DataFrame:
    """Feature row per loan at the as-of month-end, with collection-history features
    computed relative to the planning-period start."""
    cur = sf[sf["snapshot_date"] == as_of].copy()
    cur["stage"] = cur["delinquency_bucket"]
    cur["key"] = cur["loan_id"]
    cur["ref_date"] = planning_start
    prior = _prior_event_stats(events, cur[["key", "loan_id", "ref_date"]])
    cur = cur.merge(prior, on="key", how="left")
    # Attempt number for a new action = actions in the current episode + 1.
    ev = events[["loan_id", "event_date", "attempt_number"]].sort_values("event_date")
    last = ev.groupby("loan_id").tail(1).set_index("loan_id")
    cur = cur.merge(last.rename(columns={"event_date": "last_contact_date",
                                         "attempt_number": "last_attempt"}),
                    left_on="loan_id", right_index=True, how="left")
    # Attempt counters reset when an account cures or is charged off (episode start).
    hist = sf[sf["snapshot_date"] <= as_of]
    last_current = (hist.loc[hist["delinquency_bucket"] == "Current"]
                    .groupby("loan_id")["snapshot_date"].max().rename("last_current_date"))
    cur = cur.merge(last_current, left_on="loan_id", right_index=True, how="left")
    episode_start = pd.concat([pd.to_datetime(cur["last_current_date"]),
                               pd.to_datetime(cur["first_co_date"])], axis=1).max(axis=1)
    last_contact = pd.to_datetime(cur["last_contact_date"])
    new_episode = last_contact.isna() | (episode_start.notna() & (last_contact <= episode_start))
    cur["attempt_number"] = np.where(new_episode, 1, cur["last_attempt"].fillna(0) + 1)
    pd_start = pd.Timestamp(planning_start)
    last_dt = pd.to_datetime(cur["last_contact_date"])
    cur["days_since_last_contact"] = (pd_start - last_dt).dt.days
    ev30 = events[pd.to_datetime(events["event_date"]) >= pd_start - pd.Timedelta(days=30)]
    cnt = ev30.groupby("loan_id").size().rename("contacts_last_30d")
    cur = cur.merge(cnt, left_on="loan_id", right_index=True, how="left")
    cur["contacts_last_30d"] = cur["contacts_last_30d"].fillna(0).astype(int)
    return cur.reset_index(drop=True)
