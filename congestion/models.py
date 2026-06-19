"""Predictive models.

1. ImpactModel  - gradient-boosted classifiers for road-closure & high-priority,
                  plus a clearance-duration regressor. Drives the recommendation engine.
2. HotspotModel - spatio-temporal forecaster: expected event load per corridor per
                  hour-of-week (Poisson gradient boosting on lag/rolling/calendar feats).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import features as FE


class ImpactModel:
    """Predicts, at report time: P(road closure), P(high priority), expected clearance minutes.

    Config validated by experiments.py (rolling-origin CV): classifiers use the lean
    forward-selected feature set FE.CLF_FEATURES with shallow, regularised trees
    (lr=0.03, max_leaf_nodes=15) which generalised best across time folds; the duration
    regressor keeps the richer set (density/KDE features help duration, not closure).
    """

    def __init__(self):
        self.corridor_rate = None
        self.cat_levels = None
        self.clf_closure = None
        self.clf_priority = None
        self.reg_duration = None

    def _hgb_clf(self):
        return HistGradientBoostingClassifier(
            categorical_features="from_dtype", learning_rate=0.03, max_iter=600,
            max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=30,
            early_stopping=True, validation_fraction=0.15, random_state=0)

    def fit(self, df: pd.DataFrame):
        X, self.corridor_rate = FE.build(df)
        X, self.cat_levels = FE.cap_categoricals(X)
        Xc = X[[c for c in FE.CLF_FEATURES if c in X.columns]]

        # Keep probabilities well-calibrated (the recommender consumes them directly);
        # rare-class recall is handled downstream by threshold tuning, not class weights.
        self.clf_closure = self._hgb_clf().fit(Xc, df["road_closure"].values)
        self.clf_priority = self._hgb_clf().fit(Xc, df["high_priority"].values)

        # duration regressor on the subset with a valid clearance time (log-minutes).
        # AID843 (2026-02-03): weight noisy/partial observations -> recency weighting,
        # so the model trusts recent operations more than 5-month-old clearances.
        m = df["duration_min"].notna().values
        if m.sum() > 200:
            age_days = (df["start"].max() - df["start"]).dt.total_seconds().values / 86400.0
            w = np.exp(-age_days[m] / 60.0)  # ~2-month half-life
            self.reg_duration = HistGradientBoostingRegressor(
                categorical_features="from_dtype", loss="absolute_error",
                learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                min_samples_leaf=30, early_stopping=True, random_state=0)
            self.reg_duration.fit(X[m], np.log1p(df["duration_min"].values[m]), sample_weight=w)
        return self

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _ = FE.build(df, corridor_rate=self.corridor_rate)
        X, _ = FE.cap_categoricals(X, levels=self.cat_levels)
        return X

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._features(df)
        Xc = X[[c for c in FE.CLF_FEATURES if c in X.columns]]
        out = pd.DataFrame(index=df.index)
        out["p_road_closure"] = self.clf_closure.predict_proba(Xc)[:, 1]
        out["p_high_priority"] = self.clf_priority.predict_proba(Xc)[:, 1]
        if self.reg_duration is not None:
            out["exp_duration_min"] = np.expm1(self.reg_duration.predict(X)).clip(1, 24 * 60)
        else:
            out["exp_duration_min"] = np.nan
        return out


# --------------------------------------------------------------------------- #
#  Spatio-temporal hotspot / event-load forecaster
# --------------------------------------------------------------------------- #
class HotspotModel:
    """Forecast number of events (and high-impact events) per corridor per hourly bin.

    Aggregates the log to a corridor x hour panel, then learns counts from calendar
    features + autoregressive lags (last hour, same-hour-yesterday, 7-day) and rolling
    means. Uses Poisson-loss gradient boosting (proper for non-negative counts).
    """
    LAGS = [1, 2, 3, 24, 168]
    ROLLS = [24, 168]
    SLAGS = [1, 24]          # spatial-lag horizons (neighbour spillover)
    KNN = 5                  # neighbours per corridor in the weights matrix W

    def __init__(self):
        self.model = None
        self.model_impact = None
        self.corridors = None
        self.W = None          # row-normalised inverse-distance weights (n_corr x n_corr)
        self.corr_order = None  # corridor order matching W rows/cols

    def _build_weights(self, df: pd.DataFrame):
        """Spatial weights W from corridor centroids (AID843 2026-01-13 spatial lag).

        Geometry only (no road graph): k-nearest corridors by centroid haversine,
        inverse-distance weighted and row-normalised. A spatial-lag feature built from
        W injects neighbouring-corridor load that pure own-history AR lags cannot see.
        """
        cen = df.groupby("corridor")[["latitude", "longitude"]].mean()
        self.corr_order = list(cen.index)
        from congestion.features import _haversine_km
        lat = cen["latitude"].to_numpy(); lon = cen["longitude"].to_numpy()
        n = len(cen)
        W = np.zeros((n, n))
        for i in range(n):
            d = _haversine_km(lat[i], lon[i], lat, lon)
            d[i] = np.inf
            nb = np.argsort(d)[:self.KNN]
            W[i, nb] = 1.0 / np.maximum(d[nb], 0.1)
        W /= W.sum(axis=1, keepdims=True)  # row-normalise
        self.W = W

    @staticmethod
    def _panel(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["bin"] = d["start"].dt.floor("h")
        g = d.groupby(["corridor", "bin"]).agg(
            n_events=("id", "size"),
            n_impact=("road_closure", "sum")).reset_index()
        # dense corridor x hour grid so lags are well-defined
        full_idx = pd.MultiIndex.from_product(
            [g["corridor"].unique(),
             pd.date_range(g["bin"].min(), g["bin"].max(), freq="h", tz=g["bin"].dt.tz)],
            names=["corridor", "bin"])
        g = g.set_index(["corridor", "bin"]).reindex(full_idx, fill_value=0).reset_index()
        return g

    def _featurize(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["corridor", "bin"]).copy()
        b = g["bin"]
        g["hour"] = b.dt.hour
        g["dow"] = b.dt.dayofweek
        g["is_weekend"] = (g["dow"] >= 5).astype(int)
        g["hour_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
        g["hour_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
        gb = g.groupby("corridor")["n_events"]
        for L in self.LAGS:
            g[f"lag_{L}"] = gb.shift(L)
        for R in self.ROLLS:
            g[f"roll_{R}"] = gb.shift(1).rolling(R, min_periods=1).mean()
        g["corridor_cat"] = g["corridor"].astype("category")

        # spatial-lag features: W-weighted neighbouring-corridor counts at each horizon.
        if self.W is not None:
            M = (g.pivot(index="bin", columns="corridor", values="n_events")
                 .reindex(columns=self.corr_order).fillna(0.0))
            for L in self.SLAGS:
                slag = M.shift(L).to_numpy() @ self.W.T  # (n_bins, n_corr)
                sdf = pd.DataFrame(slag, index=M.index, columns=self.corr_order)
                long = sdf.reset_index().melt("bin", var_name="corridor",
                                              value_name=f"slag_{L}")
                g = g.merge(long, on=["corridor", "bin"], how="left")
        return g

    FEATS = (["hour", "dow", "is_weekend", "hour_sin", "hour_cos", "corridor_cat"]
             + [f"lag_{L}" for L in LAGS] + [f"roll_{R}" for R in ROLLS]
             + [f"slag_{L}" for L in SLAGS])

    def fit(self, df: pd.DataFrame):
        self._build_weights(df)
        g = self._featurize(self._panel(df)).dropna(subset=[f"lag_{max(self.LAGS)}"])
        X = g[self.FEATS]
        common = dict(categorical_features=["corridor_cat"], loss="poisson",
                      learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                      min_samples_leaf=50, early_stopping=True, random_state=0)
        self.model = HistGradientBoostingRegressor(**common).fit(X, g["n_events"])
        self.model_impact = HistGradientBoostingRegressor(**common).fit(X, g["n_impact"])
        self.corridors = sorted(df["corridor"].unique())
        return self

    def forecast_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """In-sample/rolling forecast over the observed panel (for evaluation & hotspot maps)."""
        g = self._featurize(self._panel(df)).dropna(subset=[f"lag_{max(self.LAGS)}"]).copy()
        g["pred_events"] = self.model.predict(g[self.FEATS])
        g["pred_impact"] = self.model_impact.predict(g[self.FEATS])
        return g[["corridor", "bin", "n_events", "pred_events", "n_impact", "pred_impact",
                  "hour", "dow"]]
