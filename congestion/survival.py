"""Survival analysis for clearance time (research recommendation #2).

The plain duration regressor trains only on the ~34% of events with a logged end time
and silently ignores the rest. Survival analysis instead treats events that are still
*active* (no end logged, status='active') as RIGHT-CENSORED: we know they lasted at
least their elapsed time. A Weibull Accelerated-Failure-Time (AFT) model uses both the
observed and censored events to estimate clearance time, and is evaluated with the
concordance index (a ranking metric that is valid under censoring).

This module is a comparison/augmentation: train.py reports whether it beats the GBM
regressor's concordance. Requires `lifelines`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Compact, low-cardinality covariates (AFT is a parametric linear model -> keep it lean).
_CAUSES = ["vehicle_breakdown", "accident", "construction", "water_logging",
           "tree_fall", "public_event", "procession", "vip_movement", "pot_holes"]


def build_survival_frame(df: pd.DataFrame, snapshot=None) -> pd.DataFrame:
    """Return per-event (duration T in minutes, event-observed E, covariates).

    Observed (E=1): events with a valid clearance time.
    Censored (E=0): status=='active' events -> elapsed time since start (>= true duration),
                    capped at 24 h (our operational horizon).
    Data-gap rows (closed/resolved but no logged time) are dropped: not genuinely censored.
    """
    snapshot = snapshot or df["start_datetime"].max()
    out = pd.DataFrame(index=df.index)
    observed = df["duration_min"].notna()
    elapsed = (snapshot - df["start_datetime"]).dt.total_seconds() / 60.0

    T = pd.Series(np.nan, index=df.index)
    E = pd.Series(0, index=df.index)
    T[observed] = df["duration_min"][observed]; E[observed] = 1
    active = (~observed) & df["status"].eq("active") & (elapsed > 0)
    T[active] = elapsed[active].clip(upper=24 * 60); E[active] = 0

    keep = observed | active
    out["T"] = T; out["E"] = E
    out["is_corridor"] = (df["corridor"] != "Non-corridor").astype(int)
    out["is_planned"] = df["event_type"].eq("planned").astype(int)
    out["hour"] = df["start"].dt.hour
    out["is_weekend"] = (df["start"].dt.dayofweek >= 5).astype(int)
    for c in _CAUSES:
        out[f"cause_{c}"] = (df["event_cause"] == c).astype(int)
    return out[keep].copy()


def fit_eval(tr: pd.DataFrame, te: pd.DataFrame):
    """Fit Weibull AFT on train survival frame; return (model, train_ci, test_ci, median_pred_fn)."""
    from lifelines import WeibullAFTFitter
    from lifelines.utils import concordance_index

    aft = WeibullAFTFitter(penalizer=0.05)
    aft.fit(tr, duration_col="T", event_col="E")
    # median predicted clearance time for ranking/eval
    med_tr = aft.predict_median(tr).clip(1, 24 * 60)
    med_te = aft.predict_median(te).clip(1, 24 * 60)
    ci_tr = concordance_index(tr["T"], med_tr, tr["E"])
    ci_te = concordance_index(te["T"], med_te, te["E"])
    return aft, ci_tr, ci_te, med_te
