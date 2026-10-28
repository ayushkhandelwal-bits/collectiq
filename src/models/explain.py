"""Prediction explanations.

* Tree models (XGBoost): SHAP TreeExplainer values in log-odds space. If the
  ``shap`` package is unavailable, XGBoost's native TreeSHAP (``pred_contribs``)
  is used - the same algorithm.
* Logistic regression: exact additive contributions ``coef * standardised value``
  relative to the training mean, also in log-odds space.

One-hot and missing-indicator columns are aggregated back to their source
feature so explanations read in business terms. Contributions describe how the
*model* reached a score; they are not causal effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline


def base_feature_map(pipeline: Pipeline, numeric: list[str], categorical: list[str]) -> list[str]:
    """Map transformed column names back to source features."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    out = []
    for n in names:
        raw = n.split("__", 1)[1]
        if raw.startswith("missingindicator_"):
            out.append(raw.replace("missingindicator_", ""))
            continue
        if n.startswith("cat__"):
            match = [c for c in categorical if raw.startswith(c + "_")]
            out.append(max(match, key=len) if match else raw)
        else:
            out.append(raw)
    return out


def _tree_contribs(model, xt: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(xt)
        if isinstance(vals, list):
            vals = vals[-1]
        base = explainer.expected_value
        base = float(np.ravel(base)[-1])
        return np.asarray(vals), base
    except Exception:  # pragma: no cover - fallback path
        import xgboost as xgb

        contrib = model.get_booster().predict(xgb.DMatrix(xt), pred_contribs=True)
        return contrib[:, :-1], float(contrib[0, -1])


def explain(pipeline: Pipeline, X: pd.DataFrame, numeric: list[str],
            categorical: list[str]) -> tuple[pd.DataFrame, float]:
    """Per-row contributions aggregated to source features, plus the base value."""
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    xt = prep.transform(X[numeric + categorical])
    xt = np.asarray(xt, dtype=float)
    feats = base_feature_map(pipeline, numeric, categorical)
    if isinstance(model, (LogisticRegression, Ridge)):
        coef = np.ravel(model.coef_)
        # Contribution relative to an average account (scaled numeric mean = 0;
        # for one-hot columns relative to the training category frequencies).
        means = getattr(pipeline, "_collectiq_train_means", np.zeros(xt.shape[1]))
        contrib = (xt - means) * coef
        base = float(np.ravel(model.intercept_)[0] + np.dot(means, coef))
    else:
        contrib, base = _tree_contribs(model, xt)
    df = pd.DataFrame(contrib, columns=range(len(feats)))
    df.columns = feats
    agg = df.T.groupby(level=0).sum().T
    return agg, base


def global_importance(contribs: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute contribution per feature, descending."""
    imp = contribs.abs().mean().sort_values(ascending=False)
    return imp.rename("mean_abs_contribution").rename_axis("feature").reset_index()


def top_reasons(contrib_row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    """Largest absolute contributions for one prediction."""
    s = contrib_row.reindex(contrib_row.abs().sort_values(ascending=False).index)
    return [(f, float(v)) for f, v in s.head(k).items()]
