"""Data-quality checks. Each check returns a result row; CRITICAL failures block
model training. Results are stored in ``data_quality_results`` verbatim - nothing
is pre-filled with success messages and no records are silently dropped."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data_generation.generate_data import ACTIONS, BUCKETS, LOAN_STATUSES

CRITICAL, WARNING, INFO = "CRITICAL", "WARNING", "INFO"


@dataclass
class QualityReport:
    """Collection of check results."""

    results: list[dict] = field(default_factory=list)

    def add(self, table: str, check: str, failed: int, severity: str, details: str,
            warn_only: bool = False) -> None:
        if failed == 0:
            status = "PASS"
        elif severity == CRITICAL and not warn_only:
            status = "FAIL"
        else:
            status = "WARN"
        self.results.append({"table_name": table, "check_name": check, "check_status": status,
                             "affected_record_count": int(failed), "severity": severity,
                             "details": details})

    def info(self, table: str, check: str, details: str, count: int = 0) -> None:
        self.results.append({"table_name": table, "check_name": check, "check_status": "INFO",
                             "affected_record_count": int(count), "severity": INFO,
                             "details": details})

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    @property
    def critical_failures(self) -> pd.DataFrame:
        df = self.frame
        if df.empty:
            return df
        return df[(df["check_status"] == "FAIL") & (df["severity"] == CRITICAL)]

    @property
    def passed(self) -> bool:
        return self.critical_failures.empty


def _dup(df: pd.DataFrame, cols: list[str]) -> int:
    return int(df.duplicated(cols).sum())


def run_quality_checks(frames: dict[str, pd.DataFrame], as_of: str) -> QualityReport:
    """Run all column-level and cross-table validations."""
    rep = QualityReport()
    loans = frames["loan_accounts"]
    snaps = frames["repayment_snapshots"]
    ev = frames["collection_events"]
    rec = frames.get("recovery_outcomes")
    port = frames.get("portfolio_snapshots")
    loan_ids = set(loans["loan_id"])

    # ---------------- loan_accounts ---------------- #
    t = "loan_accounts"
    rep.add(t, "primary_key_unique", _dup(loans, ["loan_id"]), CRITICAL, "loan_id duplicates")
    req = ["loan_id", "customer_id", "loan_product", "loan_amount", "interest_rate",
           "loan_term_months", "origination_date", "loan_status", "original_principal"]
    rep.add(t, "required_fields_not_null", int(loans[req].isna().any(axis=1).sum()), CRITICAL,
            f"Nulls in required fields {req}")
    money = ["loan_amount", "annual_income", "revolving_balance", "original_principal",
             "current_outstanding_principal", "final_recovery_amount", "emi_amount"]
    neg = int((loans[money].fillna(0) < 0).any(axis=1).sum())
    rep.add(t, "non_negative_amounts", neg, CRITICAL, f"Negative values in {money}")
    bad_ratio = int(((loans["debt_to_income_ratio"] < 0) | (loans["debt_to_income_ratio"] > 1.5)
                     | (loans["credit_utilization_ratio"] < 0)
                     | (loans["credit_utilization_ratio"] > 1)
                     | (loans["payment_to_income_ratio"] < 0)
                     | (loans["payment_to_income_ratio"] > 1)).sum())
    rep.add(t, "ratio_ranges", bad_ratio, CRITICAL,
            "DTI in [0,1.5], utilisation and PTI in [0,1] (stored as fractions)")
    bad_rate = int(((loans["interest_rate"] <= 0) | (loans["interest_rate"] > 60)).sum())
    rep.add(t, "interest_rate_range", bad_rate, WARNING, "interest_rate (% p.a.) in (0, 60]")
    orig = pd.to_datetime(loans["origination_date"], errors="coerce")
    mat = pd.to_datetime(loans["maturity_date"], errors="coerce")
    co = pd.to_datetime(loans["charge_off_date"], errors="coerce")
    asof = pd.Timestamp(as_of)
    rep.add(t, "valid_dates", int(orig.isna().sum() + (orig > asof).sum()), CRITICAL,
            "origination_date parseable and not after as-of date")
    rep.add(t, "date_ordering", int((mat <= orig).sum() + (co <= orig).sum()), CRITICAL,
            "maturity and charge-off dates after origination")
    rep.add(t, "valid_loan_status", int((~loans["loan_status"].isin(LOAN_STATUSES)).sum()),
            CRITICAL, f"loan_status in {LOAN_STATUSES}")
    is_co = loans["loan_status"] == "Charged Off"
    incons = int((is_co & loans["charged_off_principal"].isna()).sum()
                 + (~is_co & loans["charged_off_principal"].notna()).sum()
                 + ((loans["loan_status"] == "Fully Paid")
                    & (loans["current_outstanding_principal"] > 0.01)).sum())
    rep.add(t, "status_consistency", incons, CRITICAL,
            "Charged Off <=> charged_off_principal present; Fully Paid => zero outstanding")
    over = int((loans["final_recovery_amount"].fillna(0)
                > loans["charged_off_principal"].fillna(0) + 0.01).sum())
    rep.add(t, "recovery_within_collectible", over, CRITICAL,
            "Cumulative recovery <= charged-off principal")
    for c in loans.columns:
        rate = loans[c].isna().mean()
        if rate > 0.30 and c not in ("charge_off_date", "charged_off_principal",
                                     "final_recovery_amount", "final_recovery_date",
                                     "contact_restriction", "source_record_id"):
            rep.add(t, f"high_missingness:{c}", int(loans[c].isna().sum()), WARNING,
                    f"{rate:.1%} missing", warn_only=True)
    emp_missing = loans["employment_length_years"].isna().mean()
    rep.info(t, "missingness:employment_length_years",
             f"{emp_missing:.2%} missing - median-imputed inside model pipelines",
             int(loans["employment_length_years"].isna().sum()))
    for c in ["annual_income", "loan_amount", "revolving_balance"]:
        q1, q3 = loans[c].quantile([0.25, 0.75])
        n_out = int(((loans[c] < q1 - 3 * (q3 - q1)) | (loans[c] > q3 + 3 * (q3 - q1))).sum())
        rep.info(t, f"outliers_3iqr:{c}", f"{n_out} values beyond 3xIQR fences (retained; "
                 "plausible skewed tail)", n_out)
    rep.info(t, "data_origin_counts",
             "; ".join(f"{k}={v}" for k, v in loans["data_origin"].value_counts().items()),
             len(loans))
    rep.info(t, "target_prevalence:charge_off",
             f"Charged-off share {is_co.mean():.2%} of all loans", int(is_co.sum()))

    # ---------------- repayment_snapshots ---------------- #
    t = "repayment_snapshots"
    rep.add(t, "primary_key_unique", _dup(snaps, ["snapshot_id"]), CRITICAL, "snapshot_id")
    rep.add(t, "no_duplicate_monthly_snapshots", _dup(snaps, ["loan_id", "snapshot_date"]),
            CRITICAL, "(loan_id, snapshot_date) unique")
    rep.add(t, "foreign_key_loan", int((~snaps["loan_id"].isin(loan_ids)).sum()), CRITICAL,
            "loan_id exists in loan_accounts")
    neg = int((snaps[["scheduled_payment", "payment_received", "outstanding_principal",
                      "days_past_due"]] < 0).any(axis=1).sum())
    rep.add(t, "non_negative_values", neg, CRITICAL, "amounts and DPD >= 0")
    rep.add(t, "valid_bucket", int((~snaps["delinquency_bucket"].isin(BUCKETS)).sum()),
            CRITICAL, f"bucket in {BUCKETS}")
    so = snaps.merge(loans[["loan_id", "origination_date"]], on="loan_id", how="left")
    rep.add(t, "snapshot_after_origination",
            int((so["snapshot_date"] < so["origination_date"]).sum()), CRITICAL,
            "snapshot_date >= origination_date")
    dpd = snaps["days_past_due"]
    bk = snaps["delinquency_bucket"]
    mism = (((bk == "Current") & (dpd != 0)) | ((bk == "1-29 DPD") & ~dpd.between(1, 29))
            | ((bk == "30-59 DPD") & ~dpd.between(30, 59))
            | ((bk == "60-89 DPD") & ~dpd.between(60, 89)) | ((bk == "90+ DPD") & (dpd < 90)))
    rep.add(t, "dpd_bucket_consistency", int(mism.sum()), CRITICAL,
            "days_past_due agrees with bucket definition")
    last = snaps.sort_values("snapshot_date").groupby("loan_id").tail(1)
    chk = last.merge(loans[["loan_id", "loan_status"]], on="loan_id")
    rep.add(t, "final_status_matches_loan", int((chk["account_status"]
                                                 != chk["loan_status"]).sum()), CRITICAL,
            "Last snapshot status equals loan_status")

    # ---------------- collection_events ---------------- #
    t = "collection_events"
    rep.add(t, "primary_key_unique", _dup(ev, ["event_id"]), CRITICAL, "event_id")
    rep.add(t, "foreign_key_loan", int((~ev["loan_id"].isin(loan_ids)).sum()), CRITICAL,
            "loan_id exists")
    rep.add(t, "no_duplicate_events", _dup(ev, ["loan_id", "event_date", "action_type"]),
            CRITICAL, "(loan_id, event_date, action_type) unique")
    rep.add(t, "valid_action_type", int((~ev["action_type"].isin(ACTIONS)).sum()), CRITICAL,
            f"action in {ACTIONS}")
    rep.add(t, "non_negative_amounts",
            int(((ev["estimated_action_cost"] < 0) | (ev["subsequent_payment_amount"] < 0)).sum()),
            CRITICAL, "cost and payment >= 0")
    eo = ev.merge(loans[["loan_id", "origination_date"]], on="loan_id", how="left")
    rep.add(t, "event_after_origination", int((eo["event_date"] <= eo["origination_date"]).sum()),
            CRITICAL, "event_date after origination")
    rep.add(t, "prediction_timestamp_before_event",
            int((ev["prediction_timestamp"].str[:10] >= ev["event_date"]).sum()), CRITICAL,
            "Feature cut-off strictly precedes the action (leakage guard)")
    succ = (ev["subsequent_payment_amount"] > 0).mean()
    rep.info(t, "target_prevalence:payment_within_30d", f"{succ:.2%} of events", len(ev))

    # ---------------- recovery_outcomes ---------------- #
    if rec is not None:
        t = "recovery_outcomes"
        rep.add(t, "primary_key_unique", _dup(rec, ["recovery_outcome_id"]), CRITICAL, "id")
        rep.add(t, "foreign_key_event", int((~rec["event_id"].isin(set(ev["event_id"]))).sum()),
                CRITICAL, "event_id exists")
        rep.add(t, "recovery_within_opening_balance",
                int((rec["observed_recovery_amount"]
                     > rec["opening_collectible_balance"] + 0.01).sum()), CRITICAL,
                "Window recovery <= opening collectible balance")
        rep.add(t, "opening_balance_present", int(rec["opening_collectible_balance"].isna().sum()),
                CRITICAL, "Every window has an opening balance (prior snapshot found)")
        co_rec = rec[rec["outcome_definition"].str.startswith("Recovery")]
        per_loan = co_rec.groupby("loan_id")["observed_recovery_amount"].sum()
        cmp_ = (loans.drop_duplicates("loan_id").set_index("loan_id")["final_recovery_amount"]
                .reindex(per_loan.index))
        rep.add(t, "event_recoveries_not_exceed_account_total",
                int((per_loan > cmp_.fillna(0) + 0.01).sum()), CRITICAL,
                "Sum of event-attributed recoveries <= account-level cumulative recovery "
                "(no double counting)")

    # ---------------- portfolio_snapshots ---------------- #
    if port is not None:
        t = "portfolio_snapshots"
        diff_n = abs(int(port["account_count"].sum()) - len(snaps))
        rep.add(t, "reconcile_account_count", diff_n, CRITICAL,
                "Sum of account_count equals repayment_snapshots rows")
        diff_amt = abs(port["outstanding_principal_total"].sum()
                       - snaps["outstanding_principal"].sum())
        rep.add(t, "reconcile_outstanding", int(diff_amt > 1.0 * max(1, len(port)) * 0.01),
                CRITICAL, f"Outstanding totals reconcile (abs diff {diff_amt:,.2f})")
        diff_pay = abs(port["observed_recovery_total"].sum() - snaps["payment_received"].sum())
        rep.add(t, "reconcile_cash", int(diff_pay > len(port) * 0.01), CRITICAL,
                f"Cash totals reconcile (abs diff {diff_pay:,.2f})")
    return rep


def leakage_check(feature_lists: dict[str, list[str]], prohibited: set[str]) -> QualityReport:
    """Verify that no model feature list contains outcome / post-event fields."""
    rep = QualityReport()
    for model, feats in feature_lists.items():
        bad = sorted(set(feats) & prohibited)
        rep.add("model_features", f"target_leakage:{model}", len(bad), CRITICAL,
                "Prohibited post-outcome fields in features: " + (", ".join(bad) or "none"))
    return rep


def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value counts and rates per column."""
    return (pd.DataFrame({"missing_count": df.isna().sum(), "missing_rate": df.isna().mean()})
            .reset_index(names="column_name").sort_values("missing_rate", ascending=False))


__all__ = ["run_quality_checks", "leakage_check", "describe_missing", "QualityReport"]
