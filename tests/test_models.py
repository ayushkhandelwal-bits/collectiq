"""Trained models: artifacts, predictions, splits, security of loading."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import path
from src.features.build_features import (
    RECOVERY_CATEGORICAL,
    RECOVERY_NUMERIC,
    RISK_CATEGORICAL,
    RISK_NUMERIC,
)
from src.models.registry import UntrustedArtifactError, load_artifact, load_registry


def test_artifacts_registered_and_reload(built):
    reg = load_registry()
    for name in ("risk_model", "propensity_model", "amount_model"):
        assert name in reg and len(reg[name]["sha256"]) == 64
        assert load_artifact(name) is not None


def test_predictions_finite_and_bounded(built):
    panel = pd.read_parquet(path("processed_dir") / "risk_model_panel.parquet").head(500)
    p = load_artifact("risk_model").predict_proba(panel[RISK_NUMERIC + RISK_CATEGORICAL])[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    opp = pd.read_parquet(path("processed_dir") / "opportunity_features.parquet").head(500)
    q = load_artifact("propensity_model").predict_proba(
        opp[RECOVERY_NUMERIC + RECOVERY_CATEGORICAL])[:, 1]
    assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()


def test_feature_order_preserved(built):
    pipe = load_artifact("risk_model")
    assert list(pipe.feature_names_in_) == RISK_NUMERIC + RISK_CATEGORICAL
    meta = load_registry()["risk_model"]
    assert meta["numeric"] == RISK_NUMERIC


def test_handles_missing_values(built):
    panel = pd.read_parquet(path("processed_dir") / "risk_model_panel.parquet").head(20).copy()
    panel["employment_length_years"] = np.nan
    panel.loc[panel.index[0], "bureau_score"] = np.nan
    p = load_artifact("risk_model").predict_proba(panel[RISK_NUMERIC + RISK_CATEGORICAL])[:, 1]
    assert np.isfinite(p).all()


def test_metrics_calculated_and_splits_disjoint(built):
    for name in ("risk_model", "propensity_model"):
        rep = json.loads((path("reports_dir") / f"{name}_report.json").read_text())
        sel = rep["results"][rep["selected"]]
        for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier"):
            assert sel["test"][k] is not None
        d = rep["split_dates"]
        assert d["train"][1] < d["valid"][0] <= d["valid"][1] < d["test"][0]
    amt = json.loads((path("reports_dir") / "amount_model_report.json").read_text())
    assert amt["results"][amt["selected"]]["test_positive"]["mae"] > 0


def test_model_runs_recorded(built):
    from src.database import db

    runs = db.query("SELECT model_name, model_status, training_data_origin FROM model_runs")
    assert (runs["model_status"] == "SELECTED").sum() >= 3
    assert (runs["training_data_origin"] == "SYNTHETIC").all()


def test_tampered_artifact_refused(built, tmp_path):
    reg = load_registry()
    f = path("artifacts_dir") / reg["amount_model"]["file"]
    original = f.read_bytes()
    try:
        f.write_bytes(original + b"tamper")
        with pytest.raises(UntrustedArtifactError):
            load_artifact("amount_model")
    finally:
        f.write_bytes(original)
    with pytest.raises(UntrustedArtifactError):
        load_artifact("not_registered")
