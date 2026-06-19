"""Feature engineering for the per-event impact models.

All features are computable at the moment an event is *reported* (causal / no leakage):
event attributes + location + time + historical spatial density. We never use the
event's own outcome, end time, or any post-report column.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from . import CITY_CENTER

CAT_FEATURES = ["event_type", "event_cause", "corridor", "veh_type", "zone",
                "authenticated", "is_peak", "daypart",
                "junction", "police_station", "pin", "cause_corridor"]
# Spatial-density / KDE features. Ablation (see ablation.py) showed they HELP the
# clearance-duration regressor but HURT the road-closure classifier (which is already
# saturated by cause+corridor+coords). So the classifiers drop this group; the
# regressor keeps it. Data-driven, not faith-based.
SPATIAL_DENSITY = ["hist_density_1km", "hist_density_3km",
                   "kde_intensity", "kde_gathering", "kde_accident", "kde_closure"]

# Feature groups for systematic ablation (experiments.py decides which to keep per target).
GROUPS = {
    "temporal": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "dow", "is_weekend",
                 "month", "is_peak", "daypart"],
    "spatial_coord": ["lat", "lon", "dist_center_km"],
    "density_kde": SPATIAL_DENSITY,
    "recency": ["hrs_since_nearby"],
    "cat_core": ["event_type", "event_cause", "corridor", "veh_type", "zone",
                 "authenticated", "is_gathering", "is_corridor"],
    "cat_extra": ["junction", "police_station", "pin", "cause_corridor"],
    "text": ["has_desc", "desc_len", "desc_nonascii"],
    "target_enc": ["corridor_event_rate"],
}

# Classifier feature set chosen by forward selection in experiments.py (rolling-origin CV):
# cat_core + spatial_coord + text reached AUC 0.793, beating the full 35-feature set (0.756).
# Adding density/recency/cat_extra/temporal/target_enc did NOT help -> excluded.
CLF_FEATURES = GROUPS["cat_core"] + GROUPS["spatial_coord"] + GROUPS["text"]


def cap_categoricals(X: pd.DataFrame, levels: dict | None = None, max_card: int = 250):
    """Cast CAT_FEATURES to category, capping high-cardinality columns (HistGBM limit 255).

    Rare categories collapse to '__other__'. `levels` is learned on TRAIN and reused at
    predict time so train/test share the same category set.
    """
    X = X.copy()
    if levels is None:
        levels = {}
        for c in CAT_FEATURES:
            if c in X.columns:
                levels[c] = set(X[c].astype(str).value_counts().head(max_card).index)
    for c, lv in levels.items():
        if c in X.columns:
            s = X[c].astype(str)
            X[c] = s.where(s.isin(lv), "__other__").astype("category")
    return X, levels
NUM_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "dow", "is_weekend",
                "month", "dist_center_km", "lat", "lon", "is_gathering", "is_corridor",
                "hist_density_1km", "hist_density_3km", "corridor_event_rate",
                "kde_intensity", "kde_gathering", "kde_accident"]

_EARTH_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * _EARTH_KM * np.arcsin(np.sqrt(d))


def _historical_density(df: pd.DataFrame, radius_km: float) -> np.ndarray:
    """For each event, count prior events within `radius_km` (causal: strictly earlier).

    Uses a BallTree (haversine) over all points, then subtracts non-causal neighbours
    by checking timestamps. Cheap because neighbour sets within a small radius are small.
    """
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    r = radius_km / _EARTH_KM
    idx = tree.query_radius(coords, r=r)
    t = df["start"].to_numpy()
    out = np.empty(len(df), dtype=float)
    for i, neigh in enumerate(idx):
        out[i] = (t[neigh] < t[i]).sum()  # only events strictly before this one
    return out


def _kde_intensity(df: pd.DataFrame, bw_km: float, mask: np.ndarray | None = None,
                   cutoff_km: float = 4.0) -> np.ndarray:
    """Causal Gaussian-kernel intensity (a KDE surface sampled at each event).

    AID843 (2026-02-12): point *patterns* -> KDE, not kriging. We weight each prior
    event by exp(-d^2 / 2*bw^2) (distance-decay, vs the arbitrary hard radius), so a
    breakdown 50 m away counts far more than one 3 km away. Strictly causal: only
    events earlier in time contribute, so there is no leakage. `mask` selects a
    sub-pattern (e.g. one event_cause) for cause-specific intensity surfaces.
    """
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    idx = tree.query_radius(coords, r=cutoff_km / _EARTH_KM, return_distance=True)
    neigh_idx, neigh_dist = idx
    t = df["start"].to_numpy()
    sel = np.ones(len(df), bool) if mask is None else mask
    bw = bw_km / _EARTH_KM
    out = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        nb, dd = neigh_idx[i], neigh_dist[i]
        keep = (t[nb] < t[i]) & sel[nb]          # causal + in the chosen sub-pattern
        out[i] = np.exp(-(dd[keep] ** 2) / (2 * bw * bw)).sum()
    return out


def _time_since_nearby(df: pd.DataFrame, radius_km: float, cap_hr: float = 72.0) -> np.ndarray:
    """Hours since the most recent prior event within `radius_km` (causal recency)."""
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    idx = tree.query_radius(coords, r=radius_km / _EARTH_KM)
    t = df["start"].astype("int64").to_numpy() / 1e9 / 3600.0  # hours
    out = np.full(len(df), cap_hr, dtype=float)
    for i, neigh in enumerate(idx):
        prev = t[neigh][t[neigh] < t[i]]
        if prev.size:
            out[i] = min(t[i] - prev.max(), cap_hr)
    return out


def build(df: pd.DataFrame, corridor_rate: dict | None = None):
    """Return (X feature frame, corridor_rate map). Pass the train-derived map at predict time."""
    f = pd.DataFrame(index=df.index)
    s = df["start"]

    # temporal
    hour = s.dt.hour + s.dt.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = s.dt.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * f["dow"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * f["dow"] / 7)
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["month"] = s.dt.month
    f["is_peak"] = hour.between(8, 11).astype(int) | hour.between(17, 20).astype(int)
    f["is_peak"] = f["is_peak"].astype(int)
    f["daypart"] = pd.cut(hour, [-1, 6, 11, 16, 21, 24],
                          labels=["night", "morning", "afternoon", "evening", "latenight"]).astype(str)

    # spatial
    f["lat"] = df["latitude"]
    f["lon"] = df["longitude"]
    f["dist_center_km"] = _haversine_km(df["latitude"], df["longitude"], *CITY_CENTER)
    f["hist_density_1km"] = _historical_density(df, 1.0)
    f["hist_density_3km"] = _historical_density(df, 3.0)
    # Gaussian-kernel causal KDE intensity (overall + cause-specific sub-patterns).
    f["kde_intensity"] = _kde_intensity(df, bw_km=1.0)
    f["kde_gathering"] = _kde_intensity(df, bw_km=1.5,
                                        mask=df["is_gathering"].to_numpy().astype(bool))
    f["kde_accident"] = _kde_intensity(df, bw_km=1.0,
                                       mask=(df["event_cause"] == "accident").to_numpy())

    # causal recency: hours since the previous event within ~1 km (nearby activity).
    f["hrs_since_nearby"] = _time_since_nearby(df, 1.0)
    # causal KDE of PAST road closures nearby (past outcomes are known at report time).
    f["kde_closure"] = _kde_intensity(df, bw_km=1.5,
                                      mask=df["road_closure"].to_numpy().astype(bool))

    # categorical passthrough (core)
    for c in ["event_type", "event_cause", "corridor", "veh_type", "zone"]:
        f[c] = df[c].astype(str)
    f["authenticated"] = df["authenticated"].fillna("unknown").astype(str)
    f["is_gathering"] = df["is_gathering"].astype(int)
    f["is_corridor"] = (df["corridor"] != "Non-corridor").astype(int)

    # categorical extra (higher cardinality) + cause x corridor interaction
    f["junction"] = df["junction"].fillna("none").astype(str)
    f["police_station"] = df["police_station"].fillna("none").astype(str)
    f["pin"] = df["address"].astype(str).str.extract(r"Pin-?\s?(\d{6})")[0].fillna("none")
    f["cause_corridor"] = (df["event_cause"].astype(str) + "|" + df["corridor"].astype(str))

    # text signals from the free-text description
    desc = df["description"].fillna("").astype(str)
    f["has_desc"] = (desc.str.len() > 0).astype(int)
    f["desc_len"] = desc.str.len().clip(0, 200)
    # non-ASCII (mostly Kannada) descriptions correlate with local/citizen reports
    f["desc_nonascii"] = desc.apply(lambda x: int(any(ord(c) > 127 for c in x)))

    # corridor historical road-closure rate (target encoding learned on TRAIN only)
    if corridor_rate is None:
        corridor_rate = df.assign(c=f["corridor"]).groupby("c")["road_closure"].mean().to_dict()
    global_rate = np.mean(list(corridor_rate.values())) if corridor_rate else 0.0
    f["corridor_event_rate"] = f["corridor"].map(corridor_rate).fillna(global_rate)

    return f, corridor_rate
