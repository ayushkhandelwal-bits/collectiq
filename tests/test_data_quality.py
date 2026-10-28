"""Data-quality checks detect real failures and block the pipeline."""

from __future__ import annotations

import pytest

from src.features.build_features import PROHIBITED_FEATURES
from src.preprocessing.quality import leakage_check, run_quality_checks
from src.preprocessing.standardize import (
    build_portfolio_snapshots,
    build_recovery_outcomes,
    standardize,
)

AS_OF = "2026-06-30"


@pytest.fixture(scope="module")
def frames(small_generation):
    g = small_generation
    f = standardize(g.loans, g.snapshots, g.events)
    f["recovery_outcomes"] = build_recovery_outcomes(f["collection_events"],
                                                     f["repayment_snapshots"],
                                                     f["loan_accounts"])
    f["portfolio_snapshots"] = build_portfolio_snapshots(f["repayment_snapshots"],
                                                         f["loan_accounts"])
    return f


def _status(rep, table, check):
    df = rep.frame
    return df[(df.table_name == table) & (df.check_name == check)]["check_status"].iloc[0]


def test_clean_data_passes(frames):
    rep = run_quality_checks(frames, AS_OF)
    assert rep.passed, rep.critical_failures


def test_duplicate_primary_key_fails(frames):
    f = dict(frames)
    f["loan_accounts"] = frames["loan_accounts"].copy()
    f["loan_accounts"].loc[1, "loan_id"] = f["loan_accounts"].loc[0, "loan_id"]
    rep = run_quality_checks(f, AS_OF)
    assert _status(rep, "loan_accounts", "primary_key_unique") == "FAIL"
    assert not rep.passed


def test_negative_amount_fails(frames):
    f = dict(frames)
    f["repayment_snapshots"] = frames["repayment_snapshots"].copy()
    f["repayment_snapshots"].loc[0, "payment_received"] = -10.0
    rep = run_quality_checks(f, AS_OF)
    assert _status(rep, "repayment_snapshots", "non_negative_values") == "FAIL"


def test_recovery_exceeding_collectible_fails(frames):
    f = dict(frames)
    L = frames["loan_accounts"].copy()
    i = L.index[L["loan_status"] == "Charged Off"][0]
    L.loc[i, "final_recovery_amount"] = L.loc[i, "charged_off_principal"] * 2
    f["loan_accounts"] = L
    rep = run_quality_checks(f, AS_OF)
    assert _status(rep, "loan_accounts", "recovery_within_collectible") == "FAIL"


def test_orphan_foreign_key_fails(frames):
    f = dict(frames)
    ev = frames["collection_events"].copy()
    ev.loc[0, "loan_id"] = "LN9999999"
    f["collection_events"] = ev
    rep = run_quality_checks(f, AS_OF)
    assert _status(rep, "collection_events", "foreign_key_loan") == "FAIL"


def test_impossible_dates_fail(frames):
    f = dict(frames)
    L = frames["loan_accounts"].copy()
    L.loc[0, "maturity_date"] = "2000-01-01"
    f["loan_accounts"] = L
    rep = run_quality_checks(f, AS_OF)
    assert _status(rep, "loan_accounts", "date_ordering") == "FAIL"


def test_leakage_check_flags_outcome_fields():
    rep = leakage_check({"m": ["bureau_score", "final_recovery_amount"]}, PROHIBITED_FEATURES)
    assert not rep.passed
    rep_ok = leakage_check({"m": ["bureau_score"]}, PROHIBITED_FEATURES)
    assert rep_ok.passed


def test_portfolio_reconciles(frames):
    p, s = frames["portfolio_snapshots"], frames["repayment_snapshots"]
    assert p["account_count"].sum() == len(s)
    assert abs(p["observed_recovery_total"].sum() - s["payment_received"].sum()) < 1.0


def test_blocked_pipeline_records_status(built, monkeypatch):
    """A critical DQ failure stops loading/training and is recorded as BLOCKED."""
    from src.database import db
    from src.pipeline import run_pipeline as rp
    from src.preprocessing.quality import QualityReport

    def failing(frames, as_of):
        r = QualityReport()
        r.add("loan_accounts", "forced_failure", 1, "CRITICAL", "injected by test")
        return r

    before = db.scalar("SELECT COUNT(*) FROM loan_accounts")
    monkeypatch.setattr(rp, "run_quality_checks", failing)
    out = rp.run(profile="test", skip_train=True)
    assert out["status"] == "BLOCKED"
    last = db.query("SELECT pipeline_status FROM pipeline_runs ORDER BY started_at DESC LIMIT 1")
    assert last["pipeline_status"].iloc[0] == "BLOCKED"
    assert db.scalar("SELECT COUNT(*) FROM loan_accounts") == before
