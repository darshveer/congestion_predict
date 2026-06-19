"""Slide-grounded evaluation: rolling-origin CV, seasonal-naive skill, residual Moran's I.

- Rolling-origin (expanding-window) CV  -> AID843 2026-02-17 (time-series CV, ex-ante).
- Seasonal-naive baseline (same hour last week) -> AID843 2026-03-10.
- Global Moran's I on forecast residuals -> AID843 2026-01-13/02-03 (is spatial
  structure left unmodelled?). Computed on residuals because raw counts carry a strong
  calendar mean that would dominate (2026-01-20 constant-mean caveat).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def rolling_origin_cv(df, fit_predict, n_folds=4, min_frac=0.5):
    """Expanding-window CV over the timeline. `fit_predict(train_df, test_df)->(y,yhat)`.

    Returns per-fold MAE list. Each fold trains on [start..t_k], tests on the next slice.
    """
    t0, t1 = df["start"].min(), df["start"].max()
    edges = pd.to_datetime(np.linspace(
        (t0 + (t1 - t0) * min_frac).value, t1.value, n_folds + 1)).tz_localize("UTC")
    maes = []
    for k in range(n_folds):
        tr = df[df["start"] <= edges[k]]
        te = df[(df["start"] > edges[k]) & (df["start"] <= edges[k + 1])]
        if len(te) < 50:
            continue
        y, yhat = fit_predict(tr.copy(), te.copy())
        maes.append(mean_absolute_error(y, yhat))
    return maes


def seasonal_naive_panel(panel, season=168):
    """Forecast each corridor-hour as its value `season` hours earlier (same hr last week)."""
    p = panel.sort_values(["corridor", "bin"]).copy()
    p["snaive"] = p.groupby("corridor")["n_events"].shift(season)
    return p


def morans_i(values, W):
    """Global Moran's I for a vector aligned to the rows/cols of weights matrix W."""
    x = np.asarray(values, float)
    x = x - x.mean()
    n = len(x)
    S0 = W.sum()
    num = x @ (W @ x)
    den = (x * x).sum()
    if den == 0 or S0 == 0:
        return np.nan
    return (n / S0) * (num / den)
