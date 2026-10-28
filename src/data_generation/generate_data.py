"""Deterministic synthetic lending-portfolio generator for CollectIQ.

The generator simulates a consumer-lending book month by month so that every
table is internally consistent:

* ``loans``       - one row per loan with origination attributes and as-of outcomes
* ``snapshots``   - one row per loan per month-end (repayment / delinquency history)
* ``events``      - one row per simulated collection action, with its 30-day outcome

Mechanism (all parameters are documented in ``GENERATION_PARAMETERS``):

1. Borrower and loan attributes are drawn from skewed, correlated distributions.
2. A latent credit-risk index ``z`` combines observable attributes with a hidden
   component, so outcomes are stochastic and not a copy of any single feature.
3. Each month an account that is current may miss its instalment (logistic hazard
   driven by ``z``, loan age, seasonality and a macro stress factor) or prepay.
4. A delinquent account is worked by a *historical* collection policy that picks an
   action at random conditional on the delinquency stage only. The month's
   outcome (cure / partial payment / further miss) depends on ``z``, stage, the
   action and whether a promise to pay was obtained.
5. After 6 consecutive missed instalments the account is charged off; recoveries
   on the charged-off balance are then simulated, capped at the balance.

All values are synthetic. Currency convention: Indian Rupees (INR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_settings

# --------------------------------------------------------------------------- #
# Static generation parameters (persisted to metadata for reproducibility)
# --------------------------------------------------------------------------- #
PRODUCTS: dict[str, dict[str, Any]] = {
    "Personal Loan": {
        "share": 0.38, "median_amount": 300_000, "amount_sigma": 0.55,
        "min_amount": 50_000, "max_amount": 2_500_000,
        "terms": [12, 24, 36, 48, 60], "term_p": [0.10, 0.25, 0.35, 0.18, 0.12],
        "base_rate": 12.5, "risk_effect": 0.10,
        "purposes": ["Debt consolidation", "Medical", "Home improvement", "Wedding",
                     "Education", "Travel", "Other"],
        "purpose_p": [0.32, 0.12, 0.16, 0.12, 0.10, 0.08, 0.10],
    },
    "Two-Wheeler Loan": {
        "share": 0.22, "median_amount": 95_000, "amount_sigma": 0.30,
        "min_amount": 30_000, "max_amount": 300_000,
        "terms": [12, 18, 24, 36], "term_p": [0.20, 0.25, 0.35, 0.20],
        "base_rate": 11.0, "risk_effect": 0.25,
        "purposes": ["Vehicle purchase"], "purpose_p": [1.0],
    },
    "Consumer Durable Loan": {
        "share": 0.15, "median_amount": 45_000, "amount_sigma": 0.45,
        "min_amount": 8_000, "max_amount": 250_000,
        "terms": [6, 9, 12], "term_p": [0.35, 0.30, 0.35],
        "base_rate": 14.0, "risk_effect": 0.20,
        "purposes": ["Electronics & appliances"], "purpose_p": [1.0],
    },
    "Business Loan": {
        "share": 0.12, "median_amount": 900_000, "amount_sigma": 0.60,
        "min_amount": 150_000, "max_amount": 5_000_000,
        "terms": [24, 36, 48, 60], "term_p": [0.20, 0.35, 0.25, 0.20],
        "base_rate": 15.0, "risk_effect": 0.30,
        "purposes": ["Working capital", "Equipment purchase", "Business expansion"],
        "purpose_p": [0.50, 0.25, 0.25],
    },
    "Used Car Loan": {
        "share": 0.13, "median_amount": 450_000, "amount_sigma": 0.40,
        "min_amount": 100_000, "max_amount": 1_800_000,
        "terms": [24, 36, 48, 60], "term_p": [0.15, 0.35, 0.30, 0.20],
        "base_rate": 12.0, "risk_effect": 0.0,
        "purposes": ["Vehicle purchase"], "purpose_p": [1.0],
    },
}
REGIONS = {"North": 0.27, "West": 0.26, "South": 0.27, "East": 0.12, "Central": 0.08}
GRADE_SHARES = {"A": 0.20, "B": 0.30, "C": 0.25, "D": 0.15, "E": 0.10}
GRADE_SPREAD = {"A": 0.0, "B": 2.0, "C": 4.0, "D": 6.5, "E": 9.5}

# Historical (simulated) collection policy: action probabilities by stage.
HISTORICAL_POLICY: dict[str, dict[str, float]] = {
    "1-29 DPD": {"Reminder notification": 0.60, "Outbound call": 0.40},
    "30-59 DPD": {"Reminder notification": 0.15, "Outbound call": 0.45,
                  "Follow-up call": 0.20, "Payment-plan review": 0.20},
    "60-89 DPD": {"Follow-up call": 0.35, "Payment-plan review": 0.35,
                  "Specialist review": 0.30},
    "90+ DPD": {"Follow-up call": 0.30, "Payment-plan review": 0.35,
                "Specialist review": 0.35},
    "Charged Off": {"Payment-plan review": 0.50, "Specialist review": 0.50},
}
EVENT_PROBABILITY = {"1-29 DPD": 0.85, "30-59 DPD": 0.90, "60-89 DPD": 0.90,
                     "90+ DPD": 0.90, "Charged Off": 0.55}
# Simulated (true) logit uplifts on cure / partial-payment / recovery when contact is made.
ACTION_UPLIFT = {
    "Reminder notification": {"cure": 0.25, "partial": 0.05, "recovery": 0.0},
    "Outbound call": {"cure": 0.40, "partial": 0.15, "recovery": 0.0},
    "Follow-up call": {"cure": 0.30, "partial": 0.20, "recovery": 0.0},
    "Payment-plan review": {"cure": 0.10, "partial": 0.65, "recovery": 0.45},
    "Specialist review": {"cure": 0.55, "partial": 0.35, "recovery": 0.75},
}
ACTIONS = list(ACTION_UPLIFT)

BUCKETS = ["Current", "1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD", "Charged Off", "Closed"]
LOAN_STATUSES = ["Current", "Delinquent 1-29", "Delinquent 30-59", "Delinquent 60-89",
                 "Delinquent 90+", "Charged Off", "Fully Paid"]
CONTACT_OUTCOMES = ["Delivered", "Not delivered", "Right-party contact", "Promise to pay",
                    "No contact"]
CONTACT_RESTRICTIONS = {"Dispute under review": 0.015, "Hardship assessment hold": 0.025,
                        "Do-not-call request": 0.015}


@dataclass
class GenerationResult:
    """Container for generated tables plus the parameters used."""

    loans: pd.DataFrame
    snapshots: pd.DataFrame
    events: pd.DataFrame
    parameters: dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _std(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def stage_from_missed(d: np.ndarray) -> np.ndarray:
    """Map missed-instalment counts to delinquency bucket labels."""
    out = np.full(d.shape, "Current", dtype=object)
    out[d == 1] = "1-29 DPD"
    out[d == 2] = "30-59 DPD"
    out[d == 3] = "60-89 DPD"
    out[d >= 4] = "90+ DPD"
    return out


def status_from_missed(d: np.ndarray) -> np.ndarray:
    """Map missed-instalment counts to loan-status labels for open accounts."""
    out = np.full(d.shape, "Current", dtype=object)
    out[d == 1] = "Delinquent 1-29"
    out[d == 2] = "Delinquent 30-59"
    out[d == 3] = "Delinquent 60-89"
    out[d >= 4] = "Delinquent 90+"
    return out


def annuity_payment(principal: np.ndarray, monthly_rate: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Equated monthly instalment for an amortising loan."""
    growth = (1 + monthly_rate) ** n
    return principal * monthly_rate * growth / (growth - 1)


def remaining_principal(principal: np.ndarray, monthly_rate: np.ndarray, n: np.ndarray,
                        k: np.ndarray) -> np.ndarray:
    """Outstanding principal after ``k`` scheduled instalments."""
    g_n = (1 + monthly_rate) ** n
    g_k = (1 + monthly_rate) ** np.minimum(k, n)
    return np.maximum(principal * (g_n - g_k) / (g_n - 1), 0.0)


def _month_starts(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=f"{start}-01", end=end, freq="MS")


def _draw_borrowers(rng: np.random.Generator, n: int, months: pd.DatetimeIndex,
                    n_orig_months: int) -> pd.DataFrame:
    """Draw origination-time borrower and loan attributes."""
    names = list(PRODUCTS)
    product = rng.choice(names, size=n, p=[PRODUCTS[p]["share"] for p in names])
    region = rng.choice(list(REGIONS), size=n, p=list(REGIONS.values()))

    # Origination volume grows over time (linear ramp of monthly weights).
    w = np.linspace(1.0, 2.2, n_orig_months)
    orig_idx = rng.choice(n_orig_months, size=n, p=w / w.sum())
    orig_day = rng.integers(0, 28, size=n)
    origination_date = months[orig_idx] + pd.to_timedelta(orig_day, unit="D")

    income = np.exp(rng.normal(np.log(600_000), 0.55, n))
    income = np.where(product == "Business Loan", income * 1.6, income)
    income = np.round(np.clip(income, 180_000, 12_000_000), -3)

    amount = np.empty(n)
    term = np.empty(n, dtype=int)
    purpose = np.empty(n, dtype=object)
    base_rate = np.empty(n)
    prod_eff = np.empty(n)
    for name, spec in PRODUCTS.items():
        m = product == name
        k = int(m.sum())
        log_amt = (np.log(spec["median_amount"]) + 0.45 * (np.log(income[m]) - np.log(600_000))
                   + rng.normal(0, spec["amount_sigma"], k))
        amount[m] = np.clip(np.exp(log_amt), spec["min_amount"], spec["max_amount"])
        term[m] = rng.choice(spec["terms"], size=k, p=spec["term_p"])
        purpose[m] = rng.choice(spec["purposes"], size=k, p=spec["purpose_p"])
        base_rate[m] = spec["base_rate"]
        prod_eff[m] = spec["risk_effect"]

    util = rng.beta(2.2, 3.0, n)
    credit_hist = np.clip(rng.normal(90, 55, n), 6, 360).astype(int)
    emp = np.clip(np.round(rng.gamma(2.0, 2.8, n) * 2) / 2, 0, 35)
    hidden_behaviour = rng.normal(0, 1, n)
    prior_delinq = rng.poisson(np.exp(-1.4 + 0.7 * hidden_behaviour + 0.4 * _std(util)))
    open_accounts = 1 + rng.poisson(3.5, n)
    revolving = np.round(util * income * rng.uniform(0.05, 0.35, n), -2)
    other_debt_ratio = rng.beta(2.0, 7.0, n) * 0.7

    # Observable risk index and hidden component.
    observable = (0.50 * _std(other_debt_ratio) + 0.45 * _std(util)
                  + 0.45 * _std(np.log1p(prior_delinq)) - 0.35 * _std(np.log(income))
                  - 0.30 * _std(credit_hist) - 0.15 * _std(emp) + prod_eff)
    hidden = 0.45 * hidden_behaviour + rng.normal(0, 0.55, n)

    # Bureau-style score observed at origination -> risk grade (A best ... E worst).
    bureau_score = np.round(np.clip(760 - 55 * (observable + rng.normal(0, 0.45, n)), 300, 900))
    cuts = np.quantile(bureau_score, np.cumsum([0.10, 0.15, 0.25, 0.30])[:4])
    grade = np.select(
        [bureau_score <= cuts[0], bureau_score <= cuts[1], bureau_score <= cuts[2],
         bureau_score <= cuts[3]], ["E", "D", "C", "B"], default="A")
    rate = np.clip(base_rate + np.vectorize(GRADE_SPREAD.get)(grade) + rng.normal(0, 0.6, n),
                   9.5, 34.0).round(2)
    r = rate / 1200.0

    # Affordability rule: cap payment-to-income at 45%.
    emi = annuity_payment(amount, r, term)
    pti = emi * 12 / income
    scale = np.where(pti > 0.45, 0.45 / pti, 1.0)
    amount = np.floor(amount * scale / 100.0) * 100.0  # floor keeps PTI within the cap
    emi = np.round(annuity_payment(amount, r, term), 2)
    pti = np.floor(emi * 12 / income * 1e4) / 1e4
    dti = np.round(np.minimum(other_debt_ratio + pti, 0.85), 4)

    # Latent risk index (drives simulated behaviour; never stored as a feature).
    z = observable + 0.30 * _std(dti) + hidden

    emp_out = emp.astype(float)
    emp_out[rng.random(n) < 0.04] = np.nan  # documented 4% missingness

    return pd.DataFrame({
        "loan_product": product, "loan_purpose": purpose, "region": region,
        "origination_date": origination_date, "orig_month_idx": orig_idx,
        "loan_amount": amount, "annual_income": income, "employment_length_years": emp_out,
        "debt_to_income_ratio": dti, "payment_to_income_ratio": np.round(pti, 4),
        "interest_rate": rate, "loan_term_months": term,
        "credit_history_length_months": credit_hist, "credit_utilization_ratio": np.round(util, 4),
        "prior_delinquency_count": prior_delinq, "open_credit_accounts": open_accounts,
        "revolving_balance": revolving, "bureau_score": bureau_score, "risk_grade": grade,
        "emi_amount": emi, "_z": z,
    })


def generate(n_loans: int | None = None, seed: int | None = None) -> GenerationResult:
    """Generate the synthetic portfolio.

    Args:
        n_loans: number of loans; defaults to the active profile size.
        seed: random seed; defaults to ``generation.seed`` (42).

    Returns:
        GenerationResult with loans, snapshots and events DataFrames.
    """
    settings = load_settings()
    gcfg = settings["generation"]
    if n_loans is None:
        from src.config import get_profile

        n_loans = settings["profiles"][get_profile()]["n_loans"]
    seed = gcfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    as_of = pd.Timestamp(gcfg["as_of_date"])
    months = _month_starts(gcfg["start_month"], as_of.strftime("%Y-%m-%d"))
    n_months = len(months)
    n_orig = len(_month_starts(gcfg["start_month"], f"{gcfg['last_origination_month']}-01"))
    month_ends = months + pd.offsets.MonthEnd(0)
    co_threshold = int(gcfg["charge_off_missed_installments"])
    rec_window = int(gcfg["recovery_window_months"])
    action_cfg = settings["collections"]["actions"]

    b = _draw_borrowers(rng, n_loans, months, n_orig)
    n = n_loans
    z = b["_z"].to_numpy()
    principal = b["loan_amount"].to_numpy()
    r = b["interest_rate"].to_numpy() / 1200.0
    term = b["loan_term_months"].to_numpy()
    emi = b["emi_amount"].to_numpy()
    o = b["orig_month_idx"].to_numpy()
    dti = b["debt_to_income_ratio"].to_numpy()
    dpd_offset = rng.integers(5, 30, size=n)  # days between due date and month end

    k = np.zeros(n, dtype=int)            # instalments paid
    status = np.zeros(n, dtype=int)       # 0 open, 1 fully paid, 2 charged off
    co_m = np.full(n, -1)
    co_bal = np.zeros(n)
    recovered = np.zeros(n)
    last_rec_m = np.full(n, -1)
    last_pay_m = o.copy()
    attempts = np.zeros(n, dtype=int)
    closed_m = np.full(n, -1)

    snap_parts: list[dict[str, np.ndarray]] = []
    event_parts: list[dict[str, np.ndarray]] = []
    idx_all = np.arange(n)

    for m in range(n_months):
        stress = 0.18 * np.sin(2 * np.pi * (m - 2) / 12) + 0.40 * max(0.0, (m - 27) / 14)
        age = m - o
        is_open = (status == 0) & (age >= 1)
        due_now = np.minimum(age, term)
        d_start = np.minimum(age - 1, term) - k

        cur = is_open & (d_start <= 0)
        dlq = is_open & (d_start > 0)
        k_before = k.copy()
        pay = np.zeros(n)
        prepaid = np.zeros(n, dtype=bool)

        # ---- current accounts: pay, miss or prepay -------------------------- #
        ci = idx_all[cur]
        if ci.size:
            seasoning = np.where((age[ci] >= 3) & (age[ci] <= 15), 0.35,
                                 np.where(age[ci] <= 2, -0.2, 0.0))
            p_miss = _sigmoid(-4.55 + 0.95 * z[ci] + seasoning + stress)
            p_pre = np.where(age[ci] >= 4, 0.012, 0.0)
            u = rng.random(ci.size)
            pre = u < p_pre
            miss = (~pre) & (u < p_pre + p_miss)
            payk = ~(pre | miss)
            k[ci[payk]] += 1
            pay[ci[payk]] = emi[ci[payk]]
            pre_i = ci[pre]
            pay[pre_i] = remaining_principal(principal[pre_i], r[pre_i], term[pre_i], k[pre_i])
            k[pre_i] = term[pre_i]
            prepaid[pre_i] = True

        # ---- delinquent accounts: collection event + outcome ---------------- #
        di = idx_all[dlq]
        if di.size:
            stage = stage_from_missed(d_start[di])
            act, contact, ptp, outcome, out_label = _simulate_actions(
                rng, di, stage, z, attempts, dti, recovery=False)
            cu = np.zeros(di.size)
            pu = np.zeros(di.size)
            for a in ACTIONS:
                sel = (act == a) & contact
                cu[sel] = ACTION_UPLIFT[a]["cure"]
                pu[sel] = ACTION_UPLIFT[a]["partial"]
            cu += np.where((act == "Outbound call") & contact & (z[di] < 0), 0.25, 0.0)
            pu += np.where((act == "Payment-plan review") & contact & (dti[di] > 0.45), 0.35, 0.0)
            dd = d_start[di]
            l_cure = 0.55 - 0.40 * (dd - 1) - 0.65 * z[di] + cu + 0.9 * ptp - 0.6 * stress
            l_part = 0.05 - 0.10 * (dd - 1) - 0.30 * z[di] + pu + 0.5 * ptp
            denom = np.exp(l_cure) + np.exp(l_part) + 1.0
            p_c = np.exp(l_cure) / denom
            p_p = np.exp(l_part) / denom
            u = rng.random(di.size)
            cure = u < p_c
            part = (~cure) & (u < p_c + p_p)
            k[di[cure]] = due_now[di[cure]]
            pi_ = di[part]
            k[pi_] = np.minimum(k[pi_] + 1, due_now[pi_])
            pay[di] = (k[di] - k_before[di]) * emi[di]
            has_event = act != ""
            attempts[di[has_event]] += 1
            ptp_amt = np.where(ptp, np.round(dd * emi[di], 2), np.nan)
            event_parts.append(_event_rows(
                rng, m, months, month_ends, di[has_event], stage[has_event], act[has_event],
                out_label[has_event], ptp[has_event], ptp_amt[has_event],
                pay[di][has_event], attempts[di[has_event]], action_cfg))
            attempts[di[cure]] = 0

        # ---- transitions to fully paid / charged off ----------------------- #
        open_idx = idx_all[is_open]
        d_end = due_now - k
        newly_paid = is_open & (k >= term)
        status[newly_paid] = 1
        closed_m[newly_paid] = m
        newly_co = is_open & (~newly_paid) & (d_end >= co_threshold)
        status[newly_co] = 2
        co_m[newly_co] = m
        co_bal[newly_co] = remaining_principal(principal[newly_co], r[newly_co],
                                                term[newly_co], k[newly_co])
        attempts[newly_co] = 0
        last_pay_m[pay > 0] = m

        # Snapshot rows for accounts that were open during the month.
        oi = open_idx
        bal = remaining_principal(principal[oi], r[oi], term[oi], k[oi])
        de = np.maximum(d_end[oi], 0)
        st = status_from_missed(de)
        bk = stage_from_missed(de)
        st = np.where(newly_paid[oi], "Fully Paid", st)
        bk = np.where(newly_paid[oi], "Closed", bk)
        st = np.where(newly_co[oi], "Charged Off", st)
        bk = np.where(newly_co[oi], "Charged Off", bk)
        bal = np.where(newly_paid[oi], 0.0, bal)
        sched = np.where(age[oi] <= term[oi], emi[oi], 0.0)
        snap_parts.append({
            "loan_idx": oi, "month": np.full(oi.size, m),
            "scheduled_payment": np.round(sched, 2), "payment_received": np.round(pay[oi], 2),
            "outstanding_principal": np.round(bal, 2),
            "days_past_due": np.where(de > 0, 30 * (de - 1) + dpd_offset[oi], 0),
            "delinquency_bucket": bk,
            "missed_payment_indicator": ((sched > 0) & (pay[oi] < sched - 0.01)).astype(int),
            "cumulative_principal_paid": np.round(principal[oi] - bal, 2),
            "account_status": st,
        })

        # ---- charged-off accounts: recovery activity ------------------------ #
        mco = m - co_m
        rec = (status == 2) & (co_m >= 0) & (mco >= 1) & (mco <= rec_window)
        rec &= (co_bal - recovered) > 1.0
        ri = idx_all[rec]
        if ri.size:
            stage = np.full(ri.size, "Charged Off", dtype=object)
            act, contact, ptp, outcome, out_label = _simulate_actions(
                rng, ri, stage, z, attempts, dti, recovery=True)
            up = np.zeros(ri.size)
            for a in ACTIONS:
                up[(act == a) & contact] = ACTION_UPLIFT[a]["recovery"]
            p_rec = _sigmoid(-1.55 - 0.45 * z[ri] - 0.04 * mco[ri] + up + 1.1 * ptp - 0.3 * stress)
            remaining = co_bal[ri] - recovered[ri]
            settle_frac = rng.uniform(0.20, 0.55, ri.size)
            ptp_amt = np.where(ptp, np.round(remaining * settle_frac, 2), np.nan)
            got = rng.random(ri.size) < p_rec
            partial_amt = remaining * rng.beta(1.6, 9.0, ri.size)
            amt = np.where(ptp & (rng.random(ri.size) < 0.8), ptp_amt, partial_amt)
            amt = np.where(got, np.minimum(np.nan_to_num(amt), remaining), 0.0)
            amt = np.round(amt, 2)
            recovered[ri] += amt
            last_rec_m[ri[amt > 0]] = m
            has_event = act != ""
            attempts[ri[has_event]] += 1
            event_parts.append(_event_rows(
                rng, m, months, month_ends, ri[has_event], stage[has_event], act[has_event],
                out_label[has_event], ptp[has_event], ptp_amt[has_event], amt[has_event],
                attempts[ri[has_event]], action_cfg))
            snap_parts.append({
                "loan_idx": ri, "month": np.full(ri.size, m),
                "scheduled_payment": np.zeros(ri.size), "payment_received": amt,
                "outstanding_principal": np.round(co_bal[ri] - recovered[ri], 2),
                "days_past_due": 30 * (co_threshold - 1 + mco[ri]) + dpd_offset[ri],
                "delinquency_bucket": np.full(ri.size, "Charged Off", dtype=object),
                "missed_payment_indicator": np.zeros(ri.size, dtype=int),
                "cumulative_principal_paid": np.round(principal[ri] - co_bal[ri], 2),
                "account_status": np.full(ri.size, "Charged Off", dtype=object),
            })

    loans = _finalise_loans(rng, b, k, status, co_m, co_bal, recovered, last_rec_m, closed_m,
                            principal, r, term, o, month_ends, as_of, gcfg)
    snapshots = _finalise_snapshots(snap_parts, loans, month_ends)
    events = _finalise_events(event_parts, loans, gcfg)

    params = {
        "seed": seed, "n_loans": n_loans, "generation_version": gcfg["version"],
        "dataset_name": gcfg["dataset_name"], "as_of_date": gcfg["as_of_date"],
        "start_month": gcfg["start_month"],
        "last_origination_month": gcfg["last_origination_month"],
        "charge_off_missed_installments": co_threshold,
        "recovery_window_months": rec_window, "currency": "INR",
        "products": PRODUCTS, "regions": REGIONS, "grade_shares": GRADE_SHARES,
        "grade_rate_spread": GRADE_SPREAD, "historical_policy": HISTORICAL_POLICY,
        "event_probability": EVENT_PROBABILITY, "action_uplift_logit": ACTION_UPLIFT,
        "contact_restrictions": CONTACT_RESTRICTIONS,
        "mechanism": __doc__.strip(),
    }
    return GenerationResult(loans=loans, snapshots=snapshots, events=events, parameters=params)


def _simulate_actions(rng: np.random.Generator, idx: np.ndarray, stage: np.ndarray,
                      z: np.ndarray, attempts: np.ndarray, dti: np.ndarray,
                      recovery: bool) -> tuple[np.ndarray, ...]:
    """Pick historical actions and simulate contact / promise-to-pay."""
    size = idx.size
    act = np.full(size, "", dtype=object)
    for st, probs in HISTORICAL_POLICY.items():
        sel = np.where(stage == st)[0]
        if not sel.size:
            continue
        worked = rng.random(sel.size) < EVENT_PROBABILITY[st]
        chosen = rng.choice(list(probs), size=sel.size, p=list(probs.values()))
        act[sel[worked]] = chosen[worked]
    zz = z[idx]
    att = attempts[idx]
    is_rem = act == "Reminder notification"
    is_phone = (act != "") & ~is_rem
    delivered = is_rem & (rng.random(size) < 0.95)
    rpc = is_phone & (rng.random(size) < _sigmoid(0.75 - 0.45 * zz - 0.12 * att))
    p_ptp = _sigmoid(-0.1 - 0.45 * zz + 0.35 * (act == "Payment-plan review")
                     + 0.25 * (act == "Specialist review") - (0.3 if recovery else 0.0))
    ptp = rpc & (rng.random(size) < p_ptp)
    contact = delivered | rpc
    label = np.full(size, "", dtype=object)
    label[is_rem] = np.where(delivered[is_rem], "Delivered", "Not delivered")
    label[is_phone] = np.where(ptp[is_phone], "Promise to pay",
                               np.where(rpc[is_phone], "Right-party contact", "No contact"))
    return act, contact, ptp, None, label


def _event_rows(rng: np.random.Generator, m: int, months: pd.DatetimeIndex,
                month_ends: pd.DatetimeIndex, idx: np.ndarray, stage: np.ndarray,
                act: np.ndarray, out_label: np.ndarray, ptp: np.ndarray, ptp_amt: np.ndarray,
                paid: np.ndarray, attempt_no: np.ndarray,
                action_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    day = rng.integers(0, 28, size=idx.size)
    cost = np.array([action_cfg[a]["cost"] for a in act], dtype=float)
    fulfilled = np.where(ptp, (paid >= 0.9 * np.nan_to_num(ptp_amt) - 0.01) & (paid > 0), False)
    return {
        "loan_idx": idx, "month": np.full(idx.size, m),
        "event_date": (months[m] + pd.to_timedelta(day, unit="D")).to_numpy(),
        "prediction_timestamp": np.full(idx.size, month_ends[m - 1] if m > 0 else months[m]),
        "delinquency_stage_at_event": stage, "action_type": act,
        "contact_channel": np.array([action_cfg[a]["channel"] for a in act], dtype=object),
        "attempt_number": attempt_no, "estimated_action_cost": cost,
        "contact_outcome": out_label, "promise_to_pay_indicator": ptp.astype(int),
        "promise_to_pay_amount": ptp_amt,
        "promise_fulfilled_indicator": np.where(ptp, fulfilled.astype(float), np.nan),
        "subsequent_payment_amount": np.round(paid, 2),
        "agent_capacity_category": np.array(
            [action_cfg[a]["capacity_category"] for a in act], dtype=object),
    }


def _finalise_loans(rng: np.random.Generator, b: pd.DataFrame, k: np.ndarray, status: np.ndarray,
                    co_m: np.ndarray, co_bal: np.ndarray, recovered: np.ndarray,
                    last_rec_m: np.ndarray, closed_m: np.ndarray, principal: np.ndarray,
                    r: np.ndarray, term: np.ndarray, o: np.ndarray,
                    month_ends: pd.DatetimeIndex, as_of: pd.Timestamp,
                    gcfg: dict[str, Any]) -> pd.DataFrame:
    n = len(b)
    last_m = len(month_ends) - 1
    age = last_m - o
    d_final = np.maximum(np.minimum(age, term) - k, 0)
    loan_status = status_from_missed(d_final)
    loan_status = np.where(status == 1, "Fully Paid", loan_status)
    loan_status = np.where(status == 2, "Charged Off", loan_status)
    outstanding = remaining_principal(principal, r, term, k)
    outstanding = np.where(status == 1, 0.0, outstanding)
    outstanding = np.where(status == 2, co_bal - recovered, outstanding)

    restriction = np.full(n, None, dtype=object)
    in_coll = np.isin(loan_status, ["Delinquent 1-29", "Delinquent 30-59", "Delinquent 60-89",
                                    "Delinquent 90+", "Charged Off"])
    u = rng.random(n)
    cum = 0.0
    for label, p in CONTACT_RESTRICTIONS.items():
        sel = in_coll & (u >= cum) & (u < cum + p)
        restriction[sel] = label
        cum += p

    ids = np.array([f"LN{i + 1:07d}" for i in range(n)])
    orig = pd.to_datetime(b["origination_date"])
    maturity = pd.Series([d + pd.DateOffset(months=int(t)) for d, t in zip(orig, term)])
    co_dates = pd.Series([month_ends[i] if i >= 0 else pd.NaT for i in co_m])
    rec_dates = pd.Series([month_ends[i] if i >= 0 else pd.NaT for i in last_rec_m])
    created = pd.Timestamp(gcfg["as_of_date"]).strftime("%Y-%m-%d") + "T00:00:00"

    loans = pd.DataFrame({
        "loan_id": ids,
        "customer_id": np.array([f"CU{i + 1:07d}" for i in rng.permutation(n)]),
        "source_dataset": gcfg["dataset_name"],
        "data_origin": "SYNTHETIC",
        "loan_product": b["loan_product"], "loan_purpose": b["loan_purpose"],
        "region": b["region"], "risk_grade": b["risk_grade"], "bureau_score": b["bureau_score"],
        "loan_amount": b["loan_amount"], "annual_income": b["annual_income"],
        "employment_length_years": b["employment_length_years"],
        "debt_to_income_ratio": b["debt_to_income_ratio"],
        "payment_to_income_ratio": b["payment_to_income_ratio"],
        "interest_rate": b["interest_rate"], "loan_term_months": b["loan_term_months"],
        "emi_amount": b["emi_amount"],
        "credit_history_length_months": b["credit_history_length_months"],
        "credit_utilization_ratio": b["credit_utilization_ratio"],
        "prior_delinquency_count": b["prior_delinquency_count"],
        "open_credit_accounts": b["open_credit_accounts"],
        "revolving_balance": b["revolving_balance"],
        "origination_date": orig.dt.strftime("%Y-%m-%d"),
        "maturity_date": maturity.dt.strftime("%Y-%m-%d"),
        "loan_status": loan_status,
        "original_principal": b["loan_amount"],
        "current_outstanding_principal": np.round(outstanding, 2),
        "charge_off_date": co_dates.dt.strftime("%Y-%m-%d").where(co_dates.notna(), None),
        "charged_off_principal": np.where(status == 2, np.round(co_bal, 2), np.nan),
        "final_recovery_amount": np.where(status == 2, np.round(recovered, 2), np.nan),
        "final_recovery_date": rec_dates.dt.strftime("%Y-%m-%d").where(rec_dates.notna(), None),
        "contact_restriction": restriction,
        "source_record_id": None,
        "created_at": created,
    })
    return loans


def _finalise_snapshots(parts: list[dict[str, np.ndarray]], loans: pd.DataFrame,
                        month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({key: np.concatenate([p[key] for p in parts]) for key in parts[0]})
    ids = loans["loan_id"].to_numpy()
    df["loan_id"] = ids[df["loan_idx"].to_numpy()]
    df["snapshot_date"] = month_ends[df["month"].to_numpy()].strftime("%Y-%m-%d")
    df["snapshot_id"] = df["loan_id"] + "-" + month_ends[df["month"].to_numpy()].strftime("%Y%m")
    df["data_origin"] = "SYNTHETIC"
    df = df.sort_values(["loan_id", "snapshot_date"]).reset_index(drop=True)
    cols = ["snapshot_id", "loan_id", "snapshot_date", "scheduled_payment", "payment_received",
            "outstanding_principal", "days_past_due", "delinquency_bucket",
            "missed_payment_indicator", "cumulative_principal_paid", "account_status",
            "data_origin"]
    return df[cols]


def _finalise_events(parts: list[dict[str, np.ndarray]], loans: pd.DataFrame,
                     gcfg: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame({key: np.concatenate([p[key] for p in parts]) for key in parts[0]})
    ids = loans["loan_id"].to_numpy()
    df["loan_id"] = ids[df["loan_idx"].to_numpy()]
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
    df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"]).dt.strftime(
        "%Y-%m-%dT23:59:59")
    df = df.sort_values(["event_date", "loan_id"]).reset_index(drop=True)
    df["event_id"] = [f"EV{i + 1:08d}" for i in range(len(df))]
    df["outcome_window_days"] = 30
    df["data_origin"] = "SYNTHETIC"
    df["generation_version"] = gcfg["version"]
    cols = ["event_id", "loan_id", "event_date", "prediction_timestamp",
            "delinquency_stage_at_event", "action_type", "contact_channel", "attempt_number",
            "estimated_action_cost", "contact_outcome", "promise_to_pay_indicator",
            "promise_to_pay_amount", "promise_fulfilled_indicator", "subsequent_payment_amount",
            "outcome_window_days", "agent_capacity_category", "data_origin",
            "generation_version"]
    return df[cols]


def summarise(result: GenerationResult) -> dict[str, Any]:
    """Compact summary statistics of a generated dataset."""
    loans, snaps, ev = result.loans, result.snapshots, result.events
    return {
        "loan_count": len(loans),
        "snapshot_count": len(snaps),
        "event_count": len(ev),
        "loan_status_counts": loans["loan_status"].value_counts().to_dict(),
        "charge_off_rate": float((loans["loan_status"] == "Charged Off").mean()),
        "products": loans["loan_product"].value_counts().to_dict(),
        "risk_grades": loans["risk_grade"].value_counts().sort_index().to_dict(),
        "event_success_rate": float((ev["subsequent_payment_amount"] > 0).mean()),
        "action_counts": ev["action_type"].value_counts().to_dict(),
        "origination_range": [loans["origination_date"].min(), loans["origination_date"].max()],
        "snapshot_range": [snaps["snapshot_date"].min(), snaps["snapshot_date"].max()],
    }
