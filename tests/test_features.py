"""Feature engineering: leakage controls and correct time windows."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import (
    PROHIBITED_FEATURES,
    RECOVERY_CATEGORICAL,
    RECOVERY_NUMERIC,
    RISK_CATEGORICAL,
    RISK_NUMERIC,
    _prior_event_stats,
    assert_no_leakage,
    opportunity_features,
    risk_panel,
    snapshot_features,
)


@pytest.fixture(scope="module")
def sf(small_generation):
    return snapshot_features(small_generation.snapshots, small_generation.loans)


def test_feature_lists_have_no_prohibited_fields():
    assert_no_leakage(RISK_NUMERIC + RISK_CATEGORICAL)
    assert_no_leakage(RECOVERY_NUMERIC + RECOVERY_CATEGORICAL)
    with pytest.raises(ValueError):
        assert_no_leakage(["bureau_score", "loan_status"])
    assert "region" in PROHIBITED_FEATURES  # fairness policy


def test_rolling_features_use_only_past(sf):
    one = sf[sf["loan_id"] == sf["loan_id"].iloc[0]].sort_values("snapshot_date")
    expected = one["missed_payment_indicator"].rolling(3, min_periods=1).sum().to_numpy()
    assert (one["missed_last_3m"].to_numpy() == expected).all()


def test_risk_panel_target_definition(sf):
    panel = risk_panel(sf, 3, "2026-03-31")
    assert panel["delinquency_bucket"].isin(["1-29 DPD", "30-59 DPD", "60-89 DPD"]).all()
    assert panel["snapshot_date"].max() <= "2026-03-31"
    s = sf.sort_values(["loan_id", "snapshot_date"])
    row = panel[panel["target"] == 1].iloc[0]
    fut = s[(s["loan_id"] == row["loan_id"]) & (s["snapshot_date"] > row["snapshot_date"])].head(3)
    assert fut["delinquency_bucket"].isin(["90+ DPD", "Charged Off"]).any()
    row0 = panel[panel["target"] == 0].iloc[0]
    fut0 = s[(s["loan_id"] == row0["loan_id"]) & (s["snapshot_date"] > row0["snapshot_date"])
             ].head(3)
    assert not fut0["delinquency_bucket"].isin(["90+ DPD", "Charged Off"]).any()


def test_prior_event_stats_exclude_open_windows():
    events = pd.DataFrame({
        "loan_id": ["L1", "L1", "L1"],
        "event_date": ["2025-01-05", "2025-02-20", "2025-03-01"],
        "promise_to_pay_indicator": [1, 1, 1],
        "promise_fulfilled_indicator": [1, 0, 1],
    })
    ref = pd.DataFrame({"key": ["k"], "loan_id": ["L1"], "ref_date": ["2025-03-01"]})
    out = _prior_event_stats(events, ref).iloc[0]
    # Only the 2025-01-05 event's 30-day window has closed by 2025-03-01; the event
    # on the reference date itself is excluded.
    assert out["prior_ptp_count"] == 1
    assert out["prior_ptp_fulfilled_rate"] == 1.0
    assert out["prior_events_90d"] == 2


def test_opportunity_outcomes_match_events(small_generation, sf):
    opp = opportunity_features(small_generation.events, sf)
    ev_rows = opp[opp["event_id"].notna()]
    assert (ev_rows["target_amount"] - ev_rows["subsequent_payment_amount"]).abs().max() < 0.01
    assert (opp["action_type"] == "No action").any()
    assert (opp["decision_date"] > opp["snapshot_date"]).all()
