"""Model & Monitoring: comparison, evaluation, explanations, segments, drift."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.evaluation import classification_metrics
from src.prioritization.engine import FEATURE_LABELS
from ui import data
from ui.charts.plots import ACCENT, ADVERSE, CATEGORICAL, INK, RULE, SLATE, style
from ui.components.core import empty_state, fmt_date, note, page_header, provenance_band

CLS_KEYS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision", "Precision"),
            ("recall", "Recall"), ("f1", "F1"), ("brier", "Brier")]
INTERP = {"Logistic Regression": "High (coefficients)", "XGBoost": "Medium (SHAP)",
          "Random Forest": "Medium (feature importance)", "Ridge Regression": "High",
          "XGBoost Regressor": "Medium (SHAP)"}


def render() -> None:
    page_header("Model & Monitoring",
                "How the three models were trained, how they perform on held-out data, what "
                "drives them and whether inputs have shifted.")
    reg = data.registry()
    if not reg:
        empty_state("No registered models. Run the pipeline from Data & Governance.")
        return
    provenance_band([("Training data", "synthetic only"),
                     ("Validation", "chronological train → validation → test"),
                     ("Last trained", fmt_date(reg.get("risk_model", {}).get("trained_at")))])
    t1, t2, t3, t4 = st.tabs(["Roll-forward risk (A)", "Payment propensity (B)",
                              "Payment amount (C)", "Training metadata"])
    with t1:
        _classifier("risk_model", "Model A: probability that an account at 1-89 DPD reaches 90+ "
                    "DPD or charge-off within the next 3 month-ends.")
    with t2:
        _classifier("propensity_model", "Model B: probability of any payment in the month after "
                    "a decision, given the action taken (including 'No action').")
    with t3:
        _amount()
    with t4:
        _metadata(reg)


def _classifier(name: str, description: str) -> None:
    rep = data.report(f"{name}_report.json")
    if not rep:
        empty_state("Evaluation report not found for this model.")
        return
    meta = data.registry().get(name, {})
    st.markdown(description)
    st.markdown(f"**Selected model:** {rep['selected']} — chosen on validation ROC-AUC (Logistic "
                "Regression preferred when within 0.003 for interpretability). Test data was "
                "not used for selection.")
    rows = []
    for m, r in rep["results"].items():
        row = {"Model": m + (" (selected)" if m == rep["selected"] else ""),
               "Valid ROC-AUC": r["valid"]["roc_auc"]}
        row.update({f"Test {lab}": r["test"][k] for k, lab in CLS_KEYS})
        row["Threshold"] = r["threshold"]
        row["Fit (s)"] = r["fit_seconds"]
        row["Interpretability"] = INTERP.get(m, "")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                for c in ["Valid ROC-AUC"] + [f"Test {lab}" for _, lab in CLS_KEYS]
                                + ["Threshold"]})
    sel = rep["results"][rep["selected"]]
    ss = rep.get("split_sizes", {})
    sd = rep.get("split_dates", {})
    st.caption(f"Rows — train {ss.get('train', 0):,} ({' to '.join(sd.get('train', []))}), "
               f"validation {ss.get('valid', 0):,} ({' to '.join(sd.get('valid', []))}), test "
               f"{ss.get('test', 0):,} ({' to '.join(sd.get('test', []))}). Test prevalence "
               f"{sel['test']['prevalence']:.1%}.")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure()
        for i, (m, r) in enumerate(rep["results"].items()):
            fig.add_scatter(x=r["curves_test"]["roc"]["fpr"], y=r["curves_test"]["roc"]["tpr"],
                            mode="lines", name=m, line={"color": CATEGORICAL[i]})
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": RULE},
                        showlegend=False)
        style(fig, "ROC curve (test)", 320, "False positive rate", "True positive rate")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        for i, (m, r) in enumerate(rep["results"].items()):
            fig.add_scatter(x=r["curves_test"]["pr"]["recall"],
                            y=r["curves_test"]["pr"]["precision"], mode="lines", name=m,
                            line={"color": CATEGORICAL[i]})
        style(fig, "Precision–recall (test)", 320, "Recall", "Precision")
        st.plotly_chart(fig, width="stretch")
    with c3:
        cal = sel["curves_test"]["calibration"]
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": RULE},
                        name="Perfect calibration")
        fig.add_scatter(x=cal["mean_predicted"], y=cal["observed_rate"], mode="lines+markers",
                        name=rep["selected"], line={"color": ACCENT})
        style(fig, f"Calibration (test, Brier {sel['test']['brier']:.3f})", 320,
              "Mean predicted probability", "Observed rate")
        st.plotly_chart(fig, width="stretch")

    preds = data.parquet(f"{name}_test_predictions.parquet")
    if not preds.empty:
        c4, c5 = st.columns([1, 1.4])
        with c4:
            thr = st.slider("Decision threshold", 0.05, 0.95, float(round(sel["threshold"], 2)),
                            0.01, key=f"thr_{name}",
                            help="Validation-selected threshold is the default. Moving it "
                                 "recomputes the confusion matrix on test data.")
            m = classification_metrics(preds["target"], preds["score"], thr)
            cm = m["confusion_matrix"]
            z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            fig = go.Figure(go.Heatmap(z=z, x=["Predicted 0", "Predicted 1"],
                                       y=["Actual 0", "Actual 1"], text=z,
                                       texttemplate="%{text:,}", showscale=False,
                                       colorscale=[[0, "#F5F7F9"], [1, "#5FA3A0"]]))
            style(fig, f"Confusion matrix at {thr:.2f}", 260, legend=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Precision {m['precision']:.3f} · recall {m['recall']:.3f} · "
                       f"F1 {m['f1']:.3f}")
            if name == "risk_model":
                st.caption("Default threshold minimises expected cost on validation data, "
                           "assuming a missed roll-forward costs 5× an unnecessary contact.")
            else:
                st.caption("Default threshold maximises F1 on validation data. Prioritisation "
                           "uses the probability itself, not the threshold.")
        with c5:
            _importance(name, meta)
    _segments_and_drift(name)


def _importance(name: str, meta: dict) -> None:
    imp = data.parquet(f"{name}_importance.parquet")
    expl = data.parquet(f"{name}_explanations.parquet")
    if imp.empty:
        empty_state("Explanations not available.")
        return
    top = imp.head(12)[::-1]
    kind = "SHAP (TreeExplainer)" if "XGB" in meta.get("model_name", "") else \
        "coefficient × standardised value"
    view = st.radio("Explanation view", ["Global importance", "SHAP summary",
                                         "Individual prediction"], horizontal=True,
                    key=f"ev_{name}")
    if view == "Global importance":
        fig = go.Figure(go.Bar(x=top["mean_abs_contribution"],
                               y=[FEATURE_LABELS.get(f, f) for f in top["feature"]],
                               orientation="h", marker_color=INK))
        style(fig, f"Mean |contribution| — {kind}", 380, "Mean absolute log-odds contribution",
              legend=False)
        st.plotly_chart(fig, width="stretch")
    elif view == "SHAP summary":
        feats = imp.head(10)["feature"].tolist()[::-1]
        fig = go.Figure()
        for i, f in enumerate(feats):
            cvals = expl[f"contrib__{f}"].to_numpy()
            vcol = f"value__{f}"
            raw = expl[vcol] if vcol in expl.columns else pd.Series(np.zeros(len(expl)))
            if raw.dtype == object:
                color = pd.factorize(raw)[0].astype(float)
            else:
                color = raw.astype(float).rank(pct=True).to_numpy()
            jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(cvals))
            fig.add_scatter(x=cvals, y=i + jitter, mode="markers", showlegend=False,
                            marker={"size": 4, "color": color, "colorscale":
                                    [[0, "#5B8DB8"], [1, ADVERSE]], "opacity": 0.6},
                            hovertemplate=FEATURE_LABELS.get(f, f) + ": %{x:+.3f}<extra></extra>")
        fig.update_yaxes(tickvals=list(range(len(feats))),
                         ticktext=[FEATURE_LABELS.get(f, f) for f in feats])
        style(fig, "Contribution distribution (colour: low → high feature value)", 400,
              "Contribution to log-odds", legend=False)
        st.plotly_chart(fig, width="stretch")
    else:
        idx = st.number_input("Test-sample row", 0, len(expl) - 1, 0, key=f"row_{name}")
        row = expl.iloc[int(idx)]
        contrib = row[[c for c in expl.columns if c.startswith("contrib__")]]
        contrib.index = [c.replace("contrib__", "") for c in contrib.index]
        s = contrib.astype(float).sort_values(key=abs, ascending=False).head(10)[::-1]
        fig = go.Figure(go.Bar(x=s.values, y=[FEATURE_LABELS.get(f, f) for f in s.index],
                               orientation="h",
                               marker_color=[ADVERSE if v > 0 else ACCENT for v in s.values]))
        style(fig, f"Row {int(idx)}: top contributions (base {meta.get('base_value', 0):+.2f})",
              380, "Log-odds contribution", legend=False)
        st.plotly_chart(fig, width="stretch")
    st.caption("Contributions describe the model's arithmetic; they are not causal effects.")


def _segments_and_drift(name: str) -> None:
    c1, c2 = st.columns(2)
    seg = data.parquet(f"{name}_segments.parquet")
    with c1:
        st.markdown("**Segment-level performance (test)**")
        if seg.empty:
            empty_state("No segment evaluation.")
        else:
            st.dataframe(seg, hide_index=True, width="stretch", height=330,
                         column_config={
                             "observed_rate": st.column_config.NumberColumn("Observed",
                                                                            format="percent"),
                             "mean_predicted": st.column_config.NumberColumn("Predicted",
                                                                             format="percent"),
                             "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                             "sufficient_sample": st.column_config.CheckboxColumn("n ≥ 200")})
            reg = seg[seg["segment_type"] == "region"]
            if len(reg) > 1:
                spread = reg["roc_auc"].max() - reg["roc_auc"].min()
                st.caption(f"Fairness check: ROC-AUC spread across regions = {spread:.3f}. "
                           "Region is not a model input; it is monitored only.")
    with c2:
        stab = data.parquet(f"{name}_stability.parquet")
        drift = data.parquet(f"{name}_drift.parquet")
        if not stab.empty:
            fig = go.Figure()
            fig.add_scatter(x=stab["month"], y=stab["roc_auc"], mode="lines+markers",
                            name="ROC-AUC", line={"color": INK})
            style(fig, "Stability: monthly ROC-AUC on test months", 200, y_title="ROC-AUC",
                  legend=False)
            st.plotly_chart(fig, width="stretch")
        if not drift.empty:
            d = drift.sort_values("psi", ascending=False).head(8)
            st.markdown("**Drift: population stability index, training vs test period**")
            st.dataframe(d, hide_index=True, width="stretch", column_config={
                "variable": "Variable", "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                "status": "Status"})
            st.caption("PSI < 0.10 stable; 0.10–0.25 moderate; > 0.25 significant. Shifts in "
                       "months on book reflect a seasoning book, not necessarily a problem.")


def _amount() -> None:
    rep = data.report("amount_model_report.json")
    if not rep:
        empty_state("Evaluation report not found.")
        return
    st.markdown("Model C estimates the amount paid **given** a payment (hurdle second stage). "
                "Expected cash = P(payment | action) × E[amount | payment, action]. The target "
                "is zero-inflated and right-skewed, so the regression is fitted to log(1 + "
                "amount) on paying cases only, with Duan smearing on back-transformation.")
    rows = []
    for m, r in rep["results"].items():
        rows.append({"Model": m + (" (selected)" if m == rep["selected"] else ""),
                     "Test MAE | paid (₹)": r["test_positive"]["mae"],
                     "Test RMSE | paid (₹)": r["test_positive"]["rmse"],
                     "Test R² log scale": r["test_positive_log"]["r2"],
                     "Test MAE expected cash (₹)": r["test_expected_value"]["mae"],
                     "Σ predicted / Σ actual": r["test_expected_value"]["sum_predicted"]
                     / r["test_expected_value"]["sum_actual"],
                     "Valid MAE expected cash (₹)": r["valid_expected_value"]["mae"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                                for c in ["Test R² log scale", "Σ predicted / Σ actual"]})
    st.caption("Selected on validation MAE of expected cash (all opportunities, including "
               "zeros).")
    c1, c2 = st.columns(2)
    dec = data.parquet("amount_model_deciles.parquet")
    with c1:
        if not dec.empty:
            fig = go.Figure()
            fig.add_bar(x=dec["decile"], y=dec["mean_actual"], name="Observed mean cash",
                        marker_color="#C3CBD6")
            fig.add_scatter(x=dec["decile"], y=dec["mean_expected"], name="Expected (B × C)",
                            mode="lines+markers", line={"color": ACCENT})
            style(fig, "Expected vs observed cash by predicted decile (test)", 320,
                  "Decile of expected cash", "₹ per opportunity")
            st.plotly_chart(fig, width="stretch")
    with c2:
        se = data.parquet("amount_model_stage_errors.parquet")
        if not se.empty:
            st.markdown("**Error by delinquency stage (test)**")
            st.dataframe(se, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(c, format="localized")
                for c in ["mean_expected", "mean_actual", "abs_error"]})
    note("Recovery models are trained on simulated collection outcomes and have not been "
         "validated on real collection-event data.", adverse=True)


def _metadata(reg: dict) -> None:
    runs = data.table("model_runs", "trained_at", 50)
    if runs.empty:
        empty_state("No model runs recorded.")
        return
    st.dataframe(runs[["model_name", "model_status", "model_version", "training_data_origin",
                       "training_start_date", "training_end_date", "training_record_count",
                       "validation_record_count", "test_record_count", "dataset_version",
                       "artifact_path", "trained_at"]], hide_index=True,
                 width="stretch")
    pick = st.selectbox("Inspect run", runs["model_run_id"] + " — " + runs["model_name"])
    r = runs[runs["model_run_id"] == pick.split(" — ")[0]].iloc[0]
    c1, c2 = st.columns(2)
    c1.markdown("**Target definition**")
    c1.write(r["target_definition"])
    c1.markdown("**Parameters**")
    c1.json(json.loads(r["model_parameters_json"]), expanded=False)
    c2.markdown("**Metrics (validation and test)**")
    c2.json(json.loads(r["metrics_json"]), expanded=False)
    st.markdown("**Registered artifacts** (SHA-256 verified before every load)")
    st.dataframe(pd.DataFrame([{"artifact": k, "model": v.get("model_name"),
                                "version": v.get("model_version"), "sha256": v.get("sha256"),
                                "trained_at": v.get("trained_at")} for k, v in reg.items()]),
                 hide_index=True, width="stretch")
    note("Limitations: all models are trained on synthetic data generated by a known mechanism, "
         "so performance here says nothing about real portfolios. Model A applies only to "
         "accounts 1-89 DPD. Action effects in Models B/C are identifiable only because the "
         "simulated historical policy was randomised within stage.")


__all__ = ["render", "SLATE"]
