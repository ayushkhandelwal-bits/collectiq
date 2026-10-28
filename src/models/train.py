"""Training pipelines for CollectIQ's three models.

Model A - Delinquency roll-forward risk (monthly, account level)
    Target: account reaches 90+ DPD or is charged off within the next 3 month-ends,
    given it is open and 1-89 DPD at the month-end prediction date.
Model B - Payment propensity per action (collection-opportunity level)
    Unit: an account in a collections stage at month-end t-1; the decision is the
    action in month t (one of five actions, or "No action").
    Target: any payment (recovery cash if charged off) received in month t.
Model C - Payment amount given payment (hurdle second stage)
    Target: log(1 + amount) for opportunities with a payment. Expected cash is
    ``P(payment | action) x E[amount | payment, action]`` (zero-inflation handled).

Because "No action" is a modelled decision, the prioritisation can estimate the
*incremental* cash associated with an action. In this synthetic data the historical
action assignment was randomised within stage, so these differences are
identifiable by construction; on real (observational) data they are associations
and would need experimental evidence before being read as causal effects.

All models are trained and evaluated on SYNTHETIC data. Splits are chronological:
train -> validation (model and threshold selection) -> held-out test (reporting only).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import load_settings, path
from src.features.build_features import (
    RECOVERY_CATEGORICAL,
    RECOVERY_NUMERIC,
    RISK_CATEGORICAL,
    RISK_NUMERIC,
    assert_no_leakage,
)
from src.models import evaluation as ev
from src.models.explain import explain, global_importance
from src.models.registry import save_artifact
from src.utils.helpers import get_logger, to_json, utc_now, write_json

log = get_logger(__name__)

RISK_TARGET = ("Account reaches 90+ DPD or is charged off at any of the next 3 month-ends; "
               "population = open accounts at 1-89 DPD at the month-end prediction date.")
PROPENSITY_TARGET = ("Any payment (recovery cash for charged-off accounts) received in the "
                     "month following a month-end in a collections stage, given the action "
                     "taken (incl. 'No action'); features as of that month-end.")
AMOUNT_TARGET = ("Cash received in the month following the decision, given a payment occurred "
                 "(log1p-transformed, Duan smearing on back-transform).")


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Imputation (+ missing indicators), optional scaling, one-hot encoding."""
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median",
                                                                 add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", Pipeline(num_steps), numeric), ("cat", cat, categorical)],
                             verbose_feature_names_out=True)


def _xgb_params() -> dict[str, Any]:
    s = load_settings()["modeling"]
    return {**s["xgboost"], "random_state": s["random_seed"], "n_jobs": 2,
            "tree_method": "hist", "eval_metric": "logloss"}


def classifier_candidates(numeric: list[str], categorical: list[str],
                          include_rf: bool) -> dict[str, Pipeline]:
    """Candidate classification pipelines (preprocessing fitted on training data only)."""
    seed = load_settings()["modeling"]["random_seed"]
    cands = {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(numeric, categorical, scale=True)),
            ("model", LogisticRegression(C=0.5, max_iter=3000, random_state=seed))]),
        "XGBoost": Pipeline([
            ("prep", make_preprocessor(numeric, categorical, scale=False)),
            ("model", XGBClassifier(**_xgb_params()))]),
    }
    if include_rf:
        cands["Random Forest"] = Pipeline([
            ("prep", make_preprocessor(numeric, categorical, scale=False)),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=10,
                                             min_samples_leaf=30, n_jobs=2,
                                             random_state=seed))])
    return cands


def chrono_split(df: pd.DataFrame, date_col: str, train_end: str, valid_end: str,
                 test_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split on a date column."""
    d = df[date_col]
    train = df[d <= train_end]
    valid = df[(d > train_end) & (d <= valid_end)]
    test = df[(d > valid_end) & (d <= test_end)]
    return train, valid, test


def _set_train_means(pipe: Pipeline, X: pd.DataFrame, cols: list[str]) -> None:
    xt = np.asarray(pipe.named_steps["prep"].transform(X[cols]), dtype=float)
    pipe._collectiq_train_means = xt.mean(axis=0)


def _model_run_row(name: str, version: str, target: str, train: pd.DataFrame,
                   valid: pd.DataFrame, test: pd.DataFrame, date_col: str, params: dict,
                   metrics: dict, dataset_version: str, artifact: str | None,
                   status: str) -> dict[str, Any]:
    return {
        "model_run_id": f"MR-{uuid.uuid4().hex[:12]}", "model_name": name,
        "model_version": version, "target_definition": target,
        "training_data_origin": "SYNTHETIC",
        "training_start_date": str(train[date_col].min())[:10],
        "training_end_date": str(train[date_col].max())[:10],
        "training_record_count": len(train), "validation_record_count": len(valid),
        "test_record_count": len(test), "model_parameters_json": to_json(params),
        "metrics_json": to_json(metrics), "dataset_version": dataset_version,
        "artifact_path": artifact, "trained_at": utc_now(), "model_status": status,
    }


def _params_of(pipe: Pipeline) -> dict[str, Any]:
    p = pipe.named_steps["model"].get_params()
    return {k: v for k, v in p.items() if isinstance(v, (int, float, str, bool, type(None)))}


def _select(results: dict[str, dict], key: str, tolerance: float) -> str:
    """Pick the best validation score; prefer Logistic Regression within tolerance."""
    best = max(results, key=lambda m: results[m]["valid"][key])
    lr = "Logistic Regression"
    if lr in results and best != lr:
        if results[best]["valid"][key] - results[lr]["valid"][key] <= tolerance:
            return lr
    return best


def _classification_suite(name_prefix: str, df: pd.DataFrame, numeric: list[str],
                          categorical: list[str], date_col: str, splits: tuple[str, str, str],
                          threshold_rule: str, include_rf: bool, dataset_version: str,
                          target: str, artifact_name: str) -> dict[str, Any]:
    cols = numeric + categorical
    assert_no_leakage(cols)
    train, valid, test = chrono_split(df, date_col, *splits)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"{name_prefix}: empty split - check date configuration")
    mcfg = load_settings()["modeling"]["risk_model"]
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for mname, pipe in classifier_candidates(numeric, categorical, include_rf).items():
        t0 = time.time()
        pipe.fit(train[cols], train["target"])
        fit_s = time.time() - t0
        pv = pipe.predict_proba(valid[cols])[:, 1]
        pt = pipe.predict_proba(test[cols])[:, 1]
        if threshold_rule == "cost":
            thr = ev.choose_threshold_cost(valid["target"].to_numpy(), pv,
                                           mcfg["cost_false_negative"],
                                           mcfg["cost_false_positive"])
        else:
            thr = ev.choose_threshold_f1(valid["target"].to_numpy(), pv)
        results[mname] = {
            "valid": ev.classification_metrics(valid["target"], pv, thr),
            "test": ev.classification_metrics(test["target"], pt, thr),
            "fit_seconds": round(fit_s, 2), "threshold": thr,
            "curves_test": ev.curves(test["target"].to_numpy(), pt),
        }
        fitted[mname] = pipe
        log.info("%s | %s | valid AUC %.4f | test AUC %.4f | %.1fs", name_prefix, mname,
                 results[mname]["valid"]["roc_auc"], results[mname]["test"]["roc_auc"], fit_s)

    selected = _select(results, "roc_auc", tolerance=0.003)
    sel_pipe = fitted[selected]
    _set_train_means(sel_pipe, train, cols)
    test = test.copy()
    test["score"] = sel_pipe.predict_proba(test[cols])[:, 1]
    train_scores = sel_pipe.predict_proba(train[cols])[:, 1]

    # Segment-level and stability evaluation on held-out test data.
    seg = pd.concat([ev.segment_performance(test, "target", "score", s)
                     for s in ["loan_product", "risk_grade", "region", "stage", "action_type"]
                if s in test.columns], ignore_index=True)
    test["month"] = test[date_col].astype(str).str[:7]
    stab = []
    for mth, g in test.groupby("month"):
        if g["target"].nunique() > 1:
            stab.append({"month": mth, "n": len(g), "roc_auc": ev.roc_auc_score(g["target"],
                                                                               g["score"]),
                         "observed_rate": g["target"].mean(), "mean_predicted": g["score"].mean()})
    # Drift: train vs test distributions of the score and numeric features.
    drift = [{"variable": "model_score", "psi": ev.psi(train_scores, test["score"])}]
    for c in numeric:
        drift.append({"variable": c, "psi": ev.psi(train[c].to_numpy(float),
                                                   test[c].to_numpy(float))})
    for d in drift:
        d["status"] = ev.psi_label(d["psi"])

    # Explanations on a test sample.
    n_shap = min(load_settings()["modeling"]["shap_sample_size"], len(test))
    sample = test.sample(n=n_shap, random_state=42)
    contribs, base = explain(sel_pipe, sample, numeric, categorical)
    importance = global_importance(contribs)
    expl_frame = contribs.add_prefix("contrib__")
    for c in cols:
        expl_frame[f"value__{c}"] = sample[c].to_numpy()

    version = f"{artifact_name}-{dataset_version[:8]}"
    artifact = save_artifact(artifact_name, sel_pipe, {
        "model_name": f"{name_prefix} - {selected}", "model_version": version,
        "numeric": numeric, "categorical": categorical, "target_definition": target,
        "threshold": results[selected]["threshold"], "trained_at": utc_now(),
        "dataset_version": dataset_version, "training_data_origin": "SYNTHETIC",
        "base_value": base})
    runs = []
    for mname in results:
        runs.append(_model_run_row(
            f"{name_prefix} - {mname}", version, target, train, valid, test, date_col,
            _params_of(fitted[mname]),
            {k: v for k, v in results[mname].items() if k != "curves_test"}, dataset_version,
            artifact if mname == selected else None,
            "SELECTED" if mname == selected else "CHALLENGER"))
    return {
        "selected": selected, "results": results, "runs": runs, "pipeline": sel_pipe,
        "segments": seg, "stability": pd.DataFrame(stab), "drift": pd.DataFrame(drift),
        "importance": importance, "explanations": expl_frame, "base_value": base,
        "test_predictions": test, "split_sizes": {"train": len(train), "valid": len(valid),
                                                  "test": len(test)},
        "split_dates": {"train": [str(train[date_col].min())[:10], str(train[date_col].max())[:10]],
                        "valid": [str(valid[date_col].min())[:10], str(valid[date_col].max())[:10]],
                        "test": [str(test[date_col].min())[:10], str(test[date_col].max())[:10]]},
    }


def train_risk_model(panel: pd.DataFrame, dataset_version: str) -> dict[str, Any]:
    """Train and evaluate Model A (Logistic Regression, Random Forest, XGBoost)."""
    c = load_settings()["modeling"]["risk_model"]
    return _classification_suite(
        "Model A: Roll-forward risk", panel, RISK_NUMERIC, RISK_CATEGORICAL,
        "snapshot_date", (c["train_end"], c["valid_end"], c["test_end"]), "cost",
        include_rf=True, dataset_version=dataset_version, target=RISK_TARGET,
        artifact_name="risk_model")


def train_propensity_model(opportunities: pd.DataFrame, dataset_version: str) -> dict[str, Any]:
    """Train and evaluate Model B (Logistic Regression vs XGBoost)."""
    c = load_settings()["modeling"]["recovery_model"]
    return _classification_suite(
        "Model B: Payment propensity", opportunities, RECOVERY_NUMERIC, RECOVERY_CATEGORICAL,
        "decision_date", (c["train_end"], c["valid_end"], c["test_end"]), "f1",
        include_rf=False, dataset_version=dataset_version, target=PROPENSITY_TARGET,
        artifact_name="propensity_model")


def train_amount_model(events: pd.DataFrame, propensity: Pipeline,
                       dataset_version: str) -> dict[str, Any]:
    """Train Model C (hurdle second stage) and evaluate the combined expected amount."""
    c = load_settings()["modeling"]["recovery_model"]
    numeric, categorical = RECOVERY_NUMERIC, RECOVERY_CATEGORICAL
    cols = numeric + categorical
    assert_no_leakage(cols)
    train, valid, test = chrono_split(events, "decision_date", c["train_end"], c["valid_end"],
                                      c["test_end"])
    pos = {k: d[d["target_amount"] > 0] for k, d in
           {"train": train, "valid": valid, "test": test}.items()}
    seed = load_settings()["modeling"]["random_seed"]
    xgbp = {k: v for k, v in _xgb_params().items() if k != "eval_metric"}
    cands = {
        "Ridge Regression": Pipeline([("prep", make_preprocessor(numeric, categorical, True)),
                                      ("model", Ridge(alpha=3.0, random_state=seed))]),
        "XGBoost Regressor": Pipeline([("prep", make_preprocessor(numeric, categorical, False)),
                                       ("model", XGBRegressor(**xgbp))]),
    }
    results: dict[str, dict] = {}
    fitted = {}
    p_valid = propensity.predict_proba(valid[cols])[:, 1]
    p_test = propensity.predict_proba(test[cols])[:, 1]
    for mname, pipe in cands.items():
        t0 = time.time()
        ytr = np.log1p(pos["train"]["target_amount"])
        pipe.fit(pos["train"][cols], ytr)
        resid = ytr - pipe.predict(pos["train"][cols])
        smear = float(np.mean(np.exp(resid)))
        pipe._collectiq_smearing = smear

        def amount(df: pd.DataFrame, _pipe: Pipeline = pipe, _s: float = smear) -> np.ndarray:
            return np.maximum(np.expm1(_pipe.predict(df[cols])) * _s, 0.0)

        res = {"smearing_factor": smear, "fit_seconds": round(time.time() - t0, 2)}
        for split, d, p in [("valid", valid, p_valid), ("test", test, p_test)]:
            pdd = pos[split]
            res[f"{split}_positive"] = ev.regression_metrics(pdd["target_amount"], amount(pdd))
            res[f"{split}_positive_log"] = ev.regression_metrics(
                np.log1p(pdd["target_amount"]), pipe.predict(pdd[cols]))
            expected = p * amount(d)
            res[f"{split}_expected_value"] = ev.regression_metrics(d["target_amount"], expected)
            res[f"{split}_expected_value"]["sum_actual"] = float(d["target_amount"].sum())
            res[f"{split}_expected_value"]["sum_predicted"] = float(expected.sum())
        results[mname] = res
        fitted[mname] = pipe
        log.info("Model C | %s | valid MAE(pos) %.0f | test MAE(pos) %.0f", mname,
                 res["valid_positive"]["mae"], res["test_positive"]["mae"])

    selected = min(results, key=lambda m: results[m]["valid_expected_value"]["mae"])
    sel = fitted[selected]
    _set_train_means(sel, pos["train"], cols)
    # Decile calibration of the combined expected amount on test.
    t = test.copy()
    t["expected"] = p_test * np.maximum(np.expm1(sel.predict(t[cols])) * sel._collectiq_smearing,
                                        0)
    t["decile"] = pd.qcut(t["expected"].rank(method="first"), 10, labels=False) + 1
    dec = (t.groupby("decile").agg(n=("expected", "size"), mean_expected=("expected", "mean"),
                                   mean_actual=("target_amount", "mean")).reset_index())
    seg = (t.groupby("stage").agg(n=("expected", "size"), mean_expected=("expected", "mean"),
                                  mean_actual=("target_amount", "mean")).reset_index())
    seg["abs_error"] = (seg["mean_expected"] - seg["mean_actual"]).abs()
    version = f"amount_model-{dataset_version[:8]}"
    artifact = save_artifact("amount_model", sel, {
        "model_name": f"Model C: Payment amount - {selected}", "model_version": version,
        "numeric": numeric, "categorical": categorical, "target_definition": AMOUNT_TARGET,
        "smearing_factor": sel._collectiq_smearing, "trained_at": utc_now(),
        "dataset_version": dataset_version, "training_data_origin": "SYNTHETIC"})
    runs = [_model_run_row(f"Model C: Payment amount - {m}", version, AMOUNT_TARGET,
                           pos["train"], pos["valid"], pos["test"], "decision_date",
                           _params_of(fitted[m]), results[m], dataset_version,
                           artifact if m == selected else None,
                           "SELECTED" if m == selected else "CHALLENGER") for m in results]
    return {"selected": selected, "results": results, "runs": runs, "pipeline": sel,
            "deciles": dec, "stage_errors": seg,
            "split_sizes": {k: len(v) for k, v in pos.items()}}


def save_reports(name: str, out: dict[str, Any]) -> None:
    """Persist JSON summaries and parquet detail tables for the UI."""
    rep_dir = path("reports_dir")
    proc = path("processed_dir")
    summary = {k: v for k, v in out.items()
               if k in ("selected", "results", "split_sizes", "split_dates", "base_value")}
    write_json(summary, rep_dir / f"{name}_report.json")
    for key in ("segments", "stability", "drift", "importance", "explanations", "deciles",
                "stage_errors"):
        if key in out and isinstance(out[key], pd.DataFrame):
            out[key].to_parquet(proc / f"{name}_{key}.parquet", index=False)
    if "test_predictions" in out:
        keep = ["loan_id", "snapshot_date", "decision_date", "target", "score",
                "loan_product", "risk_grade", "region", "stage", "action_type"]
        tp = out["test_predictions"]
        tp[[c for c in keep if c in tp.columns]].to_parquet(
            proc / f"{name}_test_predictions.parquet", index=False)
