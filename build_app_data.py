"""Train the production models on ALL data and bundle everything the UI needs.

Writes models/app_bundle.pkl: the fitted ImpactModel, a corridor-level hotspot forecast
(centroid + predicted events/hr), dropdown option lists, and a sample of historical
events for the map. Run this once before launching the dashboard (the app also auto-runs
it if the bundle is missing).
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd

from congestion import data as D
from congestion.models import ImpactModel, HotspotModel
from congestion.logging_utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("build")
OUT = "models/app_bundle.pkl"


def main():
    os.makedirs("models", exist_ok=True)
    log.info("loading data ...")
    df = D.load_clean()

    log.info(f"training ImpactModel on all {len(df)} events ...")
    impact = ImpactModel().fit(df)

    log.info("training HotspotModel + building corridor forecast ...")
    hot = HotspotModel().fit(df)
    fc = hot.forecast_panel(df)

    # corridor-level forecast: mean predicted events/hr + centroid for the map
    cen = df.groupby("corridor")[["latitude", "longitude"]].mean()
    corr_fc = (fc.groupby("corridor")
               .agg(pred_events=("pred_events", "mean"),
                    pred_impact=("pred_impact", "mean"),
                    actual_events=("n_events", "mean")).join(cen).reset_index())
    corr_fc = corr_fc[corr_fc["corridor"] != "Non-corridor"].sort_values(
        "pred_events", ascending=False)

    # hour-of-week profile per corridor for the time slider on the map
    fc2 = fc.assign(how=fc["dow"] * 24 + fc["hour"])
    how_profile = (fc2.pivot_table(index="corridor", columns="how",
                                   values="pred_events", aggfunc="mean", fill_value=0))

    # sample of historical events for the scatter/heatmap layers
    sample = df[["latitude", "longitude", "event_cause", "corridor", "road_closure",
                 "start"]].copy()
    sample["start"] = sample["start"].astype(str)

    bundle = {
        "impact": impact,
        "corridor_forecast": corr_fc,
        "how_profile": how_profile,
        "events_sample": sample,
        "options": {
            "event_cause": sorted(df["event_cause"].dropna().unique().tolist()),
            "event_type": sorted(df["event_type"].dropna().unique().tolist()),
            "corridor": sorted(df["corridor"].dropna().unique().tolist()),
            "veh_type": sorted(df["veh_type"].dropna().unique().tolist()),
        },
        "centroids": cen.reset_index().to_dict("records"),
        "meta": {"n_events": int(len(df)),
                 "date_min": str(df["start"].min().date()),
                 "date_max": str(df["start"].max().date()),
                 "built_at": pd.Timestamp.now().isoformat(timespec="seconds")},
    }
    with open(OUT, "wb") as fh:
        pickle.dump(bundle, fh)
    log.info(f"wrote {OUT}  ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
