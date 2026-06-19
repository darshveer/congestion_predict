"""Ablation: does each slide-derived feature group actually help? Keep only what does."""
import warnings, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from congestion import data as D, features as FE
warnings.filterwarnings("ignore", category=UserWarning)

df = D.load_clean()
cut = df["start"].quantile(0.8)
tr, te = df[df["start"] <= cut].copy(), df[df["start"] > cut].copy()
Xtr, rate = FE.build(tr); Xte, _ = FE.build(te, corridor_rate=rate)
for c in FE.CAT_FEATURES:
    Xtr[c] = Xtr[c].astype("category"); Xte[c] = Xte[c].astype("category")

KDE = ["kde_intensity", "kde_gathering", "kde_accident"]
DENS = ["hist_density_1km", "hist_density_3km"]

def auc(cols_drop):
    a, b = Xtr.drop(columns=cols_drop), Xte.drop(columns=cols_drop)
    m = HistGradientBoostingClassifier(categorical_features="from_dtype", learning_rate=0.05,
        max_iter=600, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=30,
        early_stopping=True, validation_fraction=0.15, random_state=0).fit(a, tr["road_closure"].values)
    p = m.predict_proba(b)[:, 1]
    return roc_auc_score(te["road_closure"], p), average_precision_score(te["road_closure"], p)

for name, drop in [("full (KDE + density)", []),
                   ("no KDE (density only)", KDE),
                   ("no density (KDE only)", DENS),
                   ("neither", KDE + DENS)]:
    a, pr = auc(drop)
    print(f"{name:<26} AUC={a:.3f}  PR-AUC={pr:.3f}")
