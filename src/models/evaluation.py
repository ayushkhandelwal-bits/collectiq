"""Model evaluation: classification / regression metrics, curves, calibration,
segment-level performance, stability and drift (PSI)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Threshold-free and thresholded metrics for a binary classifier."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "mean_predicted": float(p.mean()),
    }


def choose_threshold_cost(y: np.ndarray, p: np.ndarray, c_fn: float, c_fp: float) -> float:
    """Threshold minimising expected misclassification cost on validation data."""
    grid = np.unique(np.quantile(p, np.linspace(0.0, 0.995, 400)))
    y = np.asarray(y).astype(int)
    costs = [c_fn * ((p < t) & (y == 1)).sum() + c_fp * ((p >= t) & (y == 0)).sum() for t in grid]
    return float(grid[int(np.argmin(costs))])


def choose_threshold_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on validation data."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.nanargmax(f1))])


def _downsample(x: np.ndarray, y: np.ndarray, k: int = 200) -> tuple[list, list]:
    if len(x) <= k:
        return x.tolist(), y.tolist()
    idx = np.unique(np.linspace(0, len(x) - 1, k).astype(int))
    return x[idx].tolist(), y[idx].tolist()


def curves(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """ROC, precision-recall and calibration curves (down-sampled for storage)."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fx, fy = _downsample(fpr, tpr)
    rx, ry = _downsample(rec[::-1], prec[::-1])
    return {"roc": {"fpr": fx, "tpr": fy}, "pr": {"recall": rx, "precision": ry},
            "calibration": {"mean_predicted": mean_pred.tolist(),
                            "observed_rate": frac_pos.tolist()}}


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R-squared."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
            "mean_actual": float(y.mean()), "mean_predicted": float(pred.mean())}


def segment_performance(df: pd.DataFrame, y_col: str, p_col: str, segment: str,
                        min_n: int = 200) -> pd.DataFrame:
    """AUC and calibration-in-the-large by segment; flags small samples."""
    rows = []
    for seg, g in df.groupby(segment, observed=True):
        y, p = g[y_col].to_numpy(), g[p_col].to_numpy()
        auc = roc_auc_score(y, p) if (len(np.unique(y)) > 1) else np.nan
        rows.append({"segment_type": segment, "segment": seg, "n": len(g),
                     "positives": int(y.sum()), "observed_rate": y.mean(),
                     "mean_predicted": p.mean(), "roc_auc": auc,
                     "sufficient_sample": bool(len(g) >= min_n and y.sum() >= 20)})
    return pd.DataFrame(rows)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the expected distribution."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    """Conventional PSI interpretation bands."""
    if value < 0.10:
        return "Stable"
    if value < 0.25:
        return "Moderate shift"
    return "Significant shift"
