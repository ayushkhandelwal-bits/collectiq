"""Baseline forecasting of the portfolio 30+ DPD delinquency rate.

Methods: naive (last value), damped-trend exponential smoothing (ETS) and
ARIMA(1,1,0). Validation is time-ordered: the final ``holdout`` months are held
out, models are fitted on earlier months only, and MAE / MAPE are reported. The
best validated method is refitted on the full history to forecast ahead.

The series is built from SYNTHETIC monthly snapshots; the first ``burn_in``
months (book ramp-up, few seasoned accounts) are excluded.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def _naive(train: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sd = np.std(np.diff(train.to_numpy())) if len(train) > 2 else 0.0
    f = np.repeat(train.iloc[-1], h)
    width = 1.96 * sd * np.sqrt(np.arange(1, h + 1))
    return f, f - width, f + width


def _ets(train: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    m = ETSModel(train.reset_index(drop=True), error="add", trend="add",
                 damped_trend=True).fit(disp=False)
    pr = m.get_prediction(start=len(train), end=len(train) + h - 1)
    sf = pr.summary_frame(alpha=0.05)
    return sf["mean"].to_numpy(), sf["pi_lower"].to_numpy(), sf["pi_upper"].to_numpy()


def _arima(train: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from statsmodels.tsa.arima.model import ARIMA

    m = ARIMA(train.reset_index(drop=True), order=(1, 1, 0)).fit()
    fc = m.get_forecast(h)
    ci = fc.conf_int(alpha=0.05).to_numpy()
    return fc.predicted_mean.to_numpy(), ci[:, 0], ci[:, 1]


METHODS = {"Naive (last value)": _naive, "Exponential smoothing (damped trend)": _ets,
           "ARIMA(1,1,0)": _arima}


def forecast_series(series: pd.Series, holdout: int = 6, horizon: int = 6,
                    burn_in: int = 9, min_train: int = 18) -> dict[str, Any]:
    """Validate methods on a time-ordered holdout and forecast ahead."""
    s = series.dropna().iloc[burn_in:]
    if len(s) < min_train + holdout:
        return {"available": False,
                "reason": f"Only {len(s)} usable months; need at least {min_train + holdout}."}
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    evals, preds = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in METHODS.items():
            try:
                f, _, _ = fn(train, holdout)
                err = test.to_numpy() - f
                evals.append({"method": name, "mae": float(np.mean(np.abs(err))),
                              "mape": float(np.mean(np.abs(err / test.to_numpy())))})
                preds[name] = f.tolist()
            except Exception as exc:  # pragma: no cover - numerical failure
                evals.append({"method": name, "mae": np.nan, "mape": np.nan,
                              "error": str(exc)[:120]})
        ev = pd.DataFrame(evals)
        best = ev.dropna(subset=["mae"]).sort_values("mae").iloc[0]["method"]
        f, lo, hi = METHODS[best](s, horizon)
    last = pd.to_datetime(s.index[-1])
    future = pd.date_range(last + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")
    return {
        "available": True, "selected_method": best, "validation": ev.to_dict("records"),
        "holdout_dates": [str(d)[:10] for d in test.index],
        "holdout_actual": test.tolist(), "holdout_predictions": preds,
        "history_dates": [str(d)[:10] for d in s.index], "history": s.tolist(),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future],
        "forecast": np.asarray(f).tolist(), "lower": np.clip(lo, 0, None).tolist(),
        "upper": np.asarray(hi).tolist(), "burn_in_months": burn_in,
    }
