"""Find sensible severity weights by anchoring them to RECORDED impact.

Severity has no ground-truth label, so we make one from what actually happened:
  - did the event require a road closure?  (acute disruption — recorded for every event)
  - how long did it take to clear?         (resource-hours — recorded for ~34%)

We then choose the weights (w_closure, w_duration, w_cause) so the severity score best
*ranks* events by that recorded impact, measured by Spearman rank correlation on a
held-out (future) test set. Objective gives equal say to the closure facet and the
duration facet so both matter. Thresholds are set from score quantiles so the four tiers
spread sensibly and the actual closure rate rises monotonically across them.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from congestion import data as D
from congestion.models import ImpactModel
from congestion.recommend import _BARRICADE_CAUSES

df = D.load_clean()
cut = df["start"].quantile(0.8)
tr, te = df[df["start"] <= cut].copy(), df[df["start"] > cut].copy()
im = ImpactModel().fit(tr)
p = im.predict(te)

pc = p["p_road_closure"].values
dn = (p["exp_duration_min"] / 240).clip(0, 1).values          # predicted duration, normalised
cause_flag = te["event_cause"].isin(_BARRICADE_CAUSES).astype(float).values

# ----- recorded ground truth -----
closure_actual = te["road_closure"].values.astype(float)       # known for all
known = te["duration_min"].notna().values
dur_actual = (te["duration_min"] / 240).clip(0, 1).fillna(0).values

def score(wc, wd, wg):
    return wc * pc + wd * dn + wg * cause_flag

def objective(s):
    # equal weight to ranking real closures and ranking real clearance times
    rc = spearmanr(s, closure_actual).correlation
    rd = spearmanr(s[known], dur_actual[known]).correlation
    return 0.5 * rc + 0.5 * rd, rc, rd

best = None
for wc in np.arange(0, 1.001, 0.05):
    for wd in np.arange(0, 1.001 - wc, 0.05):
        wg = round(1 - wc - wd, 2)
        if wg < -1e-9:
            continue
        obj, rc, rd = objective(score(round(wc, 2), round(wd, 2), wg))
        if best is None or obj > best[0]:
            best = (obj, round(wc, 2), round(wd, 2), wg, rc, rd)

obj, wc, wd, wg, rc, rd = best
print(f"BEST weights: closure={wc}  duration={wd}  cause={wg}")
print(f"  objective={obj:.3f}  (closure-rank corr={rc:.3f}, duration-rank corr={rd:.3f})")

# compare with the OLD formula (0.55 closure, 0.30 corridor-flag, 0.15 duration)
old = 0.55 * pc + 0.30 * p["p_high_priority"].values + 0.15 * dn
o_obj, o_rc, o_rd = objective(old)
print(f"OLD formula objective={o_obj:.3f} (closure corr={o_rc:.3f}, duration corr={o_rd:.3f})")

# ----- thresholds from score quantiles on the full dataset (stable, sensible spread) -----
p_all = im.predict(df)  # production-style scoring on everything for threshold calibration
s_all = (wc * p_all["p_road_closure"].values
         + wd * (p_all["exp_duration_min"] / 240).clip(0, 1).values
         + wg * df["event_cause"].isin(_BARRICADE_CAUSES).astype(float).values)
qs = np.quantile(s_all, [0.50, 0.82, 0.97])
print(f"\nthresholds (Low<{qs[0]:.3f}  Moderate<{qs[1]:.3f}  High<{qs[2]:.3f}  Critical>=)")

# show resulting distribution + actual closure rate per tier (sanity: should rise)
def tier(s):
    return np.where(s >= qs[2], "Critical",
           np.where(s >= qs[1], "High",
           np.where(s >= qs[0], "Moderate", "Low")))
t_all = tier(s_all)
out = pd.DataFrame({"tier": t_all, "closure": df["road_closure"].values})
order = ["Low", "Moderate", "High", "Critical"]
print("\ntier            count   actual_closure_rate")
for tname in order:
    m = t_all == tname
    print(f"  {tname:<10} {m.sum():>6}   {df['road_closure'].values[m].mean():.3f}")
