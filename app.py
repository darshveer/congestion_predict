"""ASTraM Event-Congestion dashboard (Streamlit + pydeck map).

Run:  streamlit run app.py
The app loads models/app_bundle.pkl (auto-builds it if missing).
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from congestion import data as D
from congestion import recommend as R
from congestion.logging_utils import get_logger

LOG = get_logger("ui")
BUNDLE = "models/app_bundle.pkl"
DATA_FILE = D.DATA_FILE
CENTER = [12.9716, 77.5946]

st.set_page_config(page_title="ASTraM Congestion Intelligence", page_icon="🚦",
                   layout="wide")


# ----------------------------- data loading -----------------------------
@st.cache_resource(show_spinner="Loading trained models …")
def load_bundle():
    if not os.path.exists(BUNDLE):
        LOG.info("bundle missing -> building it")
        subprocess.run([sys.executable, "build_app_data.py"], check=True)
    with open(BUNDLE, "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def load_metrics():
    import json
    if os.path.exists("results.json"):
        return json.load(open("results.json"))
    return {}


@st.cache_data
def raw_columns():
    return list(pd.read_csv(DATA_FILE, nrows=1).columns)


def score_event(_impact, inp: dict):
    """Build a raw 1-row frame, clean it, and run the impact model + recommender."""
    row = {c: np.nan for c in raw_columns()}
    row.update(inp)
    raw = pd.DataFrame([row])
    clean = D.clean(raw)
    preds = _impact.predict(clean)
    recs = R.recommend(clean, preds)
    return clean, preds.iloc[0], recs.iloc[0]


bundle = load_bundle()
metrics = load_metrics()
impact = bundle["impact"]
opt = bundle["options"]

# ----------------------------- sidebar nav ------------------------------
st.sidebar.title("🚦 ASTraM Congestion Intelligence")
st.sidebar.caption(f"Bengaluru Traffic Police event data · "
                   f"{bundle['meta']['date_min']} → {bundle['meta']['date_max']} · "
                   f"{bundle['meta']['n_events']:,} events")
page = st.sidebar.radio("View", ["📊 Overview", "🗺️ Hotspot Map",
                                 "🎯 Score an Event", "📈 Model & Analysis"])
st.sidebar.markdown("---")
st.sidebar.caption("Models trained on the full history. Re-run `python build_app_data.py` "
                   "after retraining to refresh.")


def metric_card(col, label, value, help_=None):
    col.metric(label, value, help=help_)


# ============================== OVERVIEW ================================
if page == "📊 Overview":
    st.title("Event-Driven Congestion — Operational Intelligence")
    st.markdown("Forecast the traffic impact of planned & unplanned events and turn it "
                "into **manpower / barricading / diversion** plans.")

    rc = metrics.get("road_closure", {})
    du = metrics.get("duration", {})
    ho = metrics.get("hotspot", {})
    sv = metrics.get("survival", {})
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Road-closure AUC", f"{rc.get('auc', float('nan')):.3f}",
                "Ranking quality (0.5=chance, 1.0=perfect). Drives barricading/diversion.")
    metric_card(c2, "Clearance error (median)", f"±{du.get('median_ae_min', float('nan')):.0f} min",
                f"Mean ±{du.get('mae_min', 0):.0f} min · naive baseline ±{du.get('naive_mae_min', 0):.0f}")
    metric_card(c3, "Hotspot skill", f"{ho.get('skill_vs_seasonal_naive', 0)*100:+.1f}%",
                "Improvement over a 'same hour last week' forecast.")
    metric_card(c4, "Survival concordance", f"{sv.get('aft_concordance', float('nan')):.3f}",
                f"Clearance-time ranking (beats plain model's {sv.get('gbm_concordance', 0):.3f}).")

    st.markdown("---")
    a, b = st.columns([3, 2])
    with a:
        st.subheader("How the system answers the problem")
        st.markdown(
            "- **Quantify impact in advance** → road-closure probability + expected "
            "clearance time for every reported event.\n"
            "- **Forecast where/when** → expected event load per corridor per hour "
            "(see the Hotspot Map).\n"
            "- **Recommend resources** → officers, barricades, diversion plan from the "
            "predictions (see Score an Event).\n"
            "- **Post-event learning** → models retrain on new events as they accumulate.")
    with b:
        st.subheader("Top forecast corridors")
        cf = bundle["corridor_forecast"].head(8)[["corridor", "pred_events"]].copy()
        cf.columns = ["Corridor", "Pred. events/hr"]
        st.dataframe(cf.style.format({"Pred. events/hr": "{:.3f}"}),
                     hide_index=True, width="stretch")

    st.caption("All metrics measured on data the model never saw during training "
               "(future time periods). Source: results.json / logs/metrics.jsonl.")


# ============================== HOTSPOT MAP ============================
elif page == "🗺️ Hotspot Map":
    st.title("🗺️ Predicted Event Hotspots — Bengaluru")
    ev = bundle["events_sample"].copy()
    cf = bundle["corridor_forecast"].copy()

    left, right = st.columns([1, 3])
    with left:
        layer_choice = st.multiselect(
            "Map layers",
            ["Historical event density (heatmap)", "Road closures", "Corridor forecast"],
            default=["Historical event density (heatmap)", "Corridor forecast"])
        cause_filter = st.multiselect("Filter events by cause", opt["event_cause"],
                                      default=[])
        st.caption("Heatmap = where incidents historically cluster. Columns = corridors "
                   "ranked by **predicted** event load (taller/redder = busier).")

    if cause_filter:
        ev = ev[ev["event_cause"].isin(cause_filter)]

    layers = []
    if "Historical event density (heatmap)" in layer_choice:
        layers.append(pdk.Layer(
            "HeatmapLayer", data=ev, get_position="[longitude, latitude]",
            aggregation="SUM", opacity=0.6, get_weight=1))
    if "Road closures" in layer_choice:
        cl = ev[ev["road_closure"] == 1]
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=cl, get_position="[longitude, latitude]",
            get_fill_color="[216, 83, 59, 160]", get_radius=120, pickable=True))
    if "Corridor forecast" in layer_choice:
        m = cf["pred_events"].max() or 1
        cf["norm"] = cf["pred_events"] / m
        cf["r"] = (40 + 200 * cf["norm"]).astype(int)
        cf["g"] = (180 * (1 - cf["norm"])).astype(int)
        layers.append(pdk.Layer(
            "ColumnLayer", data=cf, get_position="[longitude, latitude]",
            get_elevation="pred_events", elevation_scale=3000, radius=350,
            get_fill_color="[r, g, 60, 200]", pickable=True, auto_highlight=True))

    tooltip = {"html": "<b>{corridor}</b><br/>pred events/hr: {pred_events}",
               "style": {"color": "white"}}
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=CENTER[0], longitude=CENTER[1],
                                         zoom=10.4, pitch=45),
        map_provider="carto", map_style="light", tooltip=tooltip),
        width="stretch", height=560)

    st.subheader("Corridor forecast table")
    show = cf[["corridor", "pred_events", "pred_impact", "actual_events"]].copy()
    show.columns = ["Corridor", "Pred events/hr", "Pred closures/hr", "Actual events/hr"]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[1:]}),
                 hide_index=True, width="stretch")


# ============================== SCORE EVENT ===========================
elif page == "🎯 Score an Event":
    st.title("🎯 Score an Event → Deployment Plan")
    st.caption("Enter a (planned or reported) event and get its predicted impact and a "
               "recommended manpower / barricading / diversion plan.")

    c1, c2, c3 = st.columns(3)
    with c1:
        etype = st.selectbox("Event type", opt["event_type"],
                             index=opt["event_type"].index("unplanned")
                             if "unplanned" in opt["event_type"] else 0)
        cause = st.selectbox("Cause", opt["event_cause"],
                             index=opt["event_cause"].index("accident")
                             if "accident" in opt["event_cause"] else 0)
    with c2:
        corridor = st.selectbox("Corridor", opt["corridor"])
        veh = st.selectbox("Vehicle type", opt["veh_type"])
    with c3:
        date = st.date_input("Date", value=pd.Timestamp("2024-04-10"))
        hour = st.slider("Hour of day (IST)", 0, 23, 18)

    st.markdown("**Location** — click a corridor preset or set coordinates")
    cc = {r["corridor"]: (r["latitude"], r["longitude"]) for r in bundle["centroids"]}
    use_centroid = st.checkbox(f"Use centroid of '{corridor}'", value=True)
    if use_centroid and corridor in cc and not (np.isnan(cc[corridor][0])):
        lat, lon = cc[corridor]
    else:
        lat, lon = CENTER
    cla, clo = st.columns(2)
    lat = cla.number_input("Latitude", value=float(lat), format="%.5f")
    lon = clo.number_input("Longitude", value=float(lon), format="%.5f")

    if st.button("🚀 Predict & recommend", type="primary"):
        # IST hour -> UTC for the raw timestamp the cleaner expects
        ts = pd.Timestamp(date) + pd.Timedelta(hours=hour) - pd.Timedelta(hours=5, minutes=30)
        inp = {"event_type": etype, "event_cause": cause, "corridor": corridor,
               "veh_type": veh, "latitude": lat, "longitude": lon,
               "requires_road_closure": "FALSE", "authenticated": "yes",
               "priority": "Low", "zone": "unknown",
               "start_datetime": ts.strftime("%Y-%m-%d %H:%M:%S+00")}
        _, p, rec = score_event(impact, inp)
        LOG.info(f"scored event cause={cause} corridor={corridor} "
                 f"-> p_close={p['p_road_closure']:.3f} tier={rec['severity_tier']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Road-closure risk", f"{p['p_road_closure']*100:.0f}%")
        m2.metric("Expected clearance", f"{p['exp_duration_min']:.0f} min")
        m3.metric("Severity", rec["severity_tier"])
        m4.metric("Major-road event", "Yes" if p["p_high_priority"] > 0.5 else "No")

        tier_color = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}
        st.markdown(f"### {tier_color.get(rec['severity_tier'],'⚪')} Deployment plan — "
                    f"**{rec['severity_tier']}** (score {rec['severity_score']})")
        d1, d2, d3 = st.columns(3)
        d1.metric("👮 Officers", int(rec["rec_officers"]))
        d2.metric("🚧 Barricades", int(rec["rec_barricades"]))
        d3.metric("🚛 Tow crane", "Yes" if rec["rec_tow_crane"] else "No")
        st.info(f"**Traffic management:** {rec['rec_diversion']}")

        # map: the event + nearby corridor hotspots
        pt = pd.DataFrame([{"latitude": lat, "longitude": lon}])
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer("ColumnLayer", data=bundle["corridor_forecast"],
                          get_position="[longitude, latitude]",
                          get_elevation="pred_events", elevation_scale=3000, radius=300,
                          get_fill_color="[120,120,120,120]"),
                pdk.Layer("ScatterplotLayer", data=pt,
                          get_position="[longitude, latitude]",
                          get_fill_color="[216, 30, 30, 220]", get_radius=300)],
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12,
                                             pitch=45),
            map_provider="carto", map_style="light"),
            width="stretch", height=420)


# ============================== ANALYSIS ==============================
elif page == "📈 Model & Analysis":
    st.title("📈 Model Performance & Data Analysis")

    st.subheader("Metrics (held-out future test set)")
    rows = []
    for sec, d in metrics.items():
        if sec == "_meta":
            continue
        for k, v in d.items():
            rows.append({"section": sec, "metric": k, "value": v})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     height=320)
    st.caption("Full history in logs/metrics.jsonl (one JSON line per run).")

    st.subheader("Analysis figures")
    figs = [
        ("06_kde_hotspots.png", "Where events & closures cluster (KDE intensity)"),
        ("12_hotspot_forecast.png", "Hotspot forecast: predicted vs actual + corridor×hour"),
        ("04_closure_rate_by_cause.png", "Road-closure rate by cause"),
        ("02_hour_dow_heatmap.png", "When events happen (hour × weekday)"),
        ("09_roc_pr_closure.png", "Road-closure ROC & precision-recall"),
        ("10_calibration.png", "Probability calibration"),
        ("11_feature_importance.png", "What drives road closures"),
        ("01_volume_over_time.png", "Daily event volume"),
        ("13_monthly_cause_trend.png", "Monthly trend by cause"),
        ("08_duration_distribution.png", "Clearance-time distribution"),
    ]
    cols = st.columns(2)
    for i, (f, cap) in enumerate(figs):
        p = os.path.join("figures", f)
        if os.path.exists(p):
            cols[i % 2].image(p, caption=cap, width="stretch")
