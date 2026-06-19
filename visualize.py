"""Generate analysis figures into figures/. Each is referenced from the README.

Run: python visualize.py   (writes PNGs; safe to re-run)
"""
import warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             confusion_matrix, roc_auc_score)
from sklearn.calibration import calibration_curve

from congestion import data as D
from congestion.models import ImpactModel, HotspotModel
from congestion import features as FE

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
OUT = "figures"
import os; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}", bbox_inches="tight"); plt.close(fig)
    print("wrote", f"{OUT}/{name}", flush=True)

df = D.load_clean()
cut = df["start"].quantile(0.8)
tr, te = df[df["start"] <= cut].copy(), df[df["start"] > cut].copy()

# ---- 1. event volume over time (daily, planned vs unplanned) ----
daily = (df.set_index("start").groupby([pd.Grouper(freq="D"), "event_type"]).size()
         .unstack(fill_value=0))
fig, ax = plt.subplots(figsize=(11, 3.4))
daily.rolling(3).mean().plot(ax=ax, lw=1.6)
ax.set_title("Daily traffic-event volume (3-day rolling)"); ax.set_ylabel("events/day")
ax.set_xlabel(""); ax.legend(title="")
save(fig, "01_volume_over_time.png")

# ---- 2. hour x day-of-week heatmap ----
piv = (df.assign(hour=df["start"].dt.hour, dow=df["start"].dt.dayofweek)
       .pivot_table(index="dow", columns="hour", values="id", aggfunc="count", fill_value=0))
fig, ax = plt.subplots(figsize=(11, 3.2))
im = ax.imshow(piv, aspect="auto", cmap="magma")
ax.set_yticks(range(7)); ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(0, 24, 2))
ax.set_title("When events happen (count by hour × weekday, IST)"); ax.set_xlabel("hour of day")
ax.grid(False); fig.colorbar(im, ax=ax, label="events")
save(fig, "02_hour_dow_heatmap.png")

# ---- 3. event cause distribution ----
vc = df["event_cause"].value_counts().head(12)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(vc.index, vc.values, color="#3b7dd8")
ax.set_title("Event causes (top 12)"); ax.set_xlabel("count")
save(fig, "03_cause_distribution.png")

# ---- 4. road-closure rate by cause ----
rc = (df.groupby("event_cause")["road_closure"].agg(["mean", "count"])
      .query("count >= 30").sort_values("mean").tail(12))
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(rc.index, rc["mean"], color="#d8533b")
ax.set_title("Road-closure rate by cause (n≥30)"); ax.set_xlabel("P(road closure)")
for i, (m, n) in enumerate(zip(rc["mean"], rc["count"])):
    ax.text(m + 0.005, i, f"n={n}", va="center", fontsize=7)
save(fig, "04_closure_rate_by_cause.png")

# ---- 5. spatial scatter: closures vs normal ----
fig, ax = plt.subplots(figsize=(6.4, 6))
ok = df["road_closure"] == 0
ax.scatter(df.loc[ok, "longitude"], df.loc[ok, "latitude"], s=3, alpha=0.18,
           c="#7a7a7a", label="event")
cl = df["road_closure"] == 1
ax.scatter(df.loc[cl, "longitude"], df.loc[cl, "latitude"], s=14, alpha=0.7,
           c="#d8533b", label="road closure", edgecolors="none")
ax.set_title("Event locations — road closures highlighted"); ax.legend()
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
save(fig, "05_spatial_closures.png")

# ---- 6. KDE hotspot surface (all events) ----
def kde_grid(d, bins=160, sigma=3):
    H, xe, ye = np.histogram2d(d["longitude"], d["latitude"], bins=bins)
    return gaussian_filter(H.T, sigma), xe, ye
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, (title, sub) in zip(axes, [("All events", df),
                                   ("Road closures", df[df["road_closure"] == 1])]):
    Hs, xe, ye = kde_grid(sub)
    im = ax.imshow(Hs, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   aspect="auto", cmap="inferno", norm=LogNorm(vmin=max(Hs.max()/1e3, 1e-2)))
    ax.set_title(f"Hotspot intensity — {title}"); ax.set_xlabel("longitude"); ax.grid(False)
axes[0].set_ylabel("latitude")
save(fig, "06_kde_hotspots.png")

# ---- 7. corridor load ----
cc = df["corridor"].value_counts().head(14)[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(cc.index, cc.values, color="#2a9d8f")
ax.set_title("Events by corridor (top 14)"); ax.set_xlabel("count")
save(fig, "07_corridor_load.png")

# ---- 8. clearance duration distribution ----
dur = df["duration_min"].dropna()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(dur, bins=60, color="#577590")
ax.axvline(dur.median(), color="#d8533b", ls="--", label=f"median {dur.median():.0f} min")
ax.set_title("Clearance duration (events with logged end, ≤24h)")
ax.set_xlabel("minutes"); ax.legend()
save(fig, "08_duration_distribution.png")

# ================= MODEL FIGURES =================
print("training models for evaluation figures ...", flush=True)
im = ImpactModel().fit(tr)
pte = im.predict(te)
y = te["road_closure"].values; p = pte["p_road_closure"].values

# ---- 9. ROC + PR curves ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
fpr, tpr, _ = roc_curve(y, p)
a1.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y,p):.3f}")
a1.plot([0,1],[0,1],"k--",lw=0.8); a1.set_title("ROC — road closure")
a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate"); a1.legend()
prec, rec, _ = precision_recall_curve(y, p)
a2.plot(rec, prec, lw=2, color="#d8533b", label=f"PR-AUC={auc(rec,prec):.3f}")
a2.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"base rate={y.mean():.3f}")
a2.set_title("Precision-Recall — road closure"); a2.set_xlabel("recall"); a2.set_ylabel("precision"); a2.legend()
save(fig, "09_roc_pr_closure.png")

# ---- 10. calibration ----
fig, ax = plt.subplots(figsize=(5, 4.5))
frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
ax.plot(mean_pred, frac, "o-", label="model")
ax.plot([0,1],[0,1],"k--",lw=0.8,label="perfect")
ax.set_title("Calibration — road-closure probabilities")
ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.legend()
save(fig, "10_calibration.png")

# ---- 11. permutation importance ----
from sklearn.inspection import permutation_importance
_Xfull = im._features(te)
Xte = _Xfull[[c for c in FE.CLF_FEATURES if c in _Xfull.columns]]
pi = permutation_importance(im.clf_closure, Xte, y, scoring="roc_auc", n_repeats=5, random_state=0)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp.index, imp.values, color="#9b5de5")
ax.set_title("Road-closure drivers (permutation importance, AUC drop)")
ax.set_xlabel("mean AUC decrease when shuffled")
save(fig, "11_feature_importance.png")

# ---- 12. hotspot forecast: predicted vs actual + corridor×hour-of-week ----
hm = HotspotModel().fit(tr)
fc = hm.forecast_panel(df); fc_te = fc[fc["bin"] > cut]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.hexbin(fc_te["pred_events"], fc_te["n_events"], gridsize=30, cmap="viridis", mincnt=1, bins="log")
lim = max(fc_te["pred_events"].max(), fc_te["n_events"].max())
a1.plot([0, lim], [0, lim], "r--", lw=1)
a1.set_title("Hotspot forecast: predicted vs actual (per corridor-hour)")
a1.set_xlabel("predicted events"); a1.set_ylabel("actual events"); a1.grid(False)
how = (fc.assign(how=fc["dow"] * 24 + fc["hour"])
       .pivot_table(index="corridor", columns="how", values="pred_events", aggfunc="mean", fill_value=0))
how = how.loc[how.mean(1).sort_values().tail(12).index]
im2 = a2.imshow(how, aspect="auto", cmap="magma")
a2.set_yticks(range(len(how.index))); a2.set_yticklabels(how.index, fontsize=7)
a2.set_xticks(range(0, 168, 24)); a2.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
a2.set_title("Predicted event load (corridor × hour-of-week)"); a2.grid(False)
fig.colorbar(im2, ax=a2, label="pred events/hr")
save(fig, "12_hotspot_forecast.png")

# ---- 13. monthly cause trend ----
mc = (df.assign(month=df["start"].dt.to_period("M").astype(str))
      .groupby(["month", "event_cause"]).size().unstack(fill_value=0))
top_causes = df["event_cause"].value_counts().head(6).index
fig, ax = plt.subplots(figsize=(9, 4))
mc[top_causes].plot(ax=ax, marker="o", lw=1.4)
ax.set_title("Monthly trend by cause (top 6)"); ax.set_ylabel("events"); ax.set_xlabel("")
ax.legend(fontsize=7, ncol=2)
save(fig, "13_monthly_cause_trend.png")

print("\nAll figures written to figures/")
