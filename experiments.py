"""Systematic model search for the road-closure target (the key predictive task).

Tests, with leakage-safe rolling-origin CV and LIVE progress:
  STUDY 1  model families        (HistGBM / LightGBM / XGBoost / RandomForest / LogReg)
  STUDY 2  leave-one-group-out   (which feature groups actually help)
  STUDY 3  forward selection      (best minimal group subset)
  STUDY 4  hyperparameter search  (tune the winner)

Selection metric: mean ROC-AUC across folds (PR-AUC reported alongside). Nothing is
adopted because it's fashionable — only if it beats the current config out-of-time.
"""
import time, json, warnings, itertools
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb, xgboost as xgb
from congestion import data as D, features as FE

warnings.filterwarnings("ignore")
T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

TARGET = "road_closure"
df = D.load_clean()
log(f"loaded {len(df)} events; building features once ...")
X_ALL, _ = FE.build(df)            # causal feats are label-free except target_enc
Y = df[TARGET].values
START = df["start"].values
log(f"features ready: {X_ALL.shape[1]} cols")

CV_EDGES = np.quantile(df["start"].astype("int64"), [0.5, 0.625, 0.75, 0.875, 1.0])

def folds():
    for k in range(len(CV_EDGES) - 1):
        tr = START.astype("int64") <= CV_EDGES[k]
        te = (START.astype("int64") > CV_EDGES[k]) & (START.astype("int64") <= CV_EDGES[k+1])
        if te.sum() >= 40 and Y[tr].sum() >= 10:
            yield tr, te

def _target_enc(cols, tr, te):
    """Leakage-safe corridor_event_rate: learn the rate on this fold's TRAIN only."""
    X = X_ALL.copy()
    rate = (pd.Series(Y[tr]).groupby(X_ALL["corridor"].values[tr]).mean())
    gmean = rate.mean()
    X["corridor_event_rate"] = X_ALL["corridor"].map(rate).fillna(gmean)
    return X[cols]

CAT = [c for c in FE.CAT_FEATURES]
def cats_in(cols): return [c for c in CAT if c in cols]

# Pre-compute top-200 categories per high-card column (HistGBM caps cardinality at 255)
# and a FIXED CategoricalDtype so train/test share identical category codes (XGBoost needs this).
_TOPCAT = {c: set(X_ALL[c].astype(str).value_counts().head(200).index) for c in CAT}
_CATDTYPE = {c: pd.CategoricalDtype(categories=sorted(_TOPCAT[c] | {"__other__"})) for c in CAT}
def _cap(s, col):
    return s.astype(str).where(s.astype(str).isin(_TOPCAT[col]), "__other__").astype(_CATDTYPE[col])

def fit_predict(name, cols, tr, te):
    X = _target_enc(cols, tr, te)
    Xtr, Xte, ytr = X[tr], X[te], Y[tr]
    cc = cats_in(cols)
    if name in ("hgb", "lgbm", "xgb"):
        Xtr = Xtr.copy(); Xte = Xte.copy()
        for c in cc:
            Xtr[c] = _cap(Xtr[c], c); Xte[c] = _cap(Xte[c], c)
    if name == "hgb":
        m = HistGradientBoostingClassifier(categorical_features="from_dtype",
            learning_rate=0.05, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0,
            min_samples_leaf=30, early_stopping=True, validation_fraction=0.15, random_state=0)
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
    if name == "lgbm":
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
            random_state=0, verbose=-1)
        m.fit(Xtr, ytr, categorical_feature=cc); return m.predict_proba(Xte)[:, 1]
    if name == "xgb":
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
            min_child_weight=5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, eval_metric="logloss", random_state=0)
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
    # encoded baselines (drop very-high-card cats for one-hot sanity)
    enc_cols = [c for c in cols if c not in ("junction", "police_station", "pin", "cause_corridor")]
    if not enc_cols:  # nothing the baseline can use -> predict base rate
        return np.full(int(te.sum()), ytr.mean())
    Xtr2, Xte2 = X[tr][enc_cols].copy(), X[te][enc_cols].copy()
    cc2 = cats_in(enc_cols)
    if name == "rf":
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr2[cc2] = oe.fit_transform(Xtr2[cc2].astype(str)); Xte2[cc2] = oe.transform(Xte2[cc2].astype(str))
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=0)
        m.fit(Xtr2, ytr); return m.predict_proba(Xte2)[:, 1]
    if name == "logreg":
        allc = pd.concat([Xtr2, Xte2])
        d = pd.get_dummies(allc, columns=cc2, dummy_na=True)
        a, b = d.iloc[:len(Xtr2)], d.iloc[len(Xtr2):]
        m = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(a, ytr); return m.predict_proba(b)[:, 1]

def cv(name, cols):
    aucs, prs = [], []
    for tr, te in folds():
        p = fit_predict(name, cols, tr, te)
        aucs.append(roc_auc_score(Y[te], p)); prs.append(average_precision_score(Y[te], p))
    return np.mean(aucs), np.mean(prs), np.std(aucs)

ALL_COLS = list(X_ALL.columns)
RESULTS = {}

# ---------------- STUDY 1: model families ----------------
log("STUDY 1: model families (all features)")
fam = {}
for name in ["hgb", "lgbm", "xgb", "rf", "logreg"]:
    a, pr, sd = cv(name, ALL_COLS)
    fam[name] = a; log(f"  {name:<7} AUC={a:.4f}±{sd:.3f}  PR-AUC={pr:.4f}")
best_fam = max(fam, key=fam.get); RESULTS["families"] = {k: round(v, 4) for k, v in fam.items()}
log(f"  -> best family: {best_fam} (AUC {fam[best_fam]:.4f})")

# ---------------- STUDY 2: leave-one-group-out ----------------
log(f"STUDY 2: leave-one-group-out with {best_fam}")
base_a, base_pr, _ = cv(best_fam, ALL_COLS)
log(f"  ALL groups: AUC={base_a:.4f} PR-AUC={base_pr:.4f}")
loo = {}
for gname, gcols in FE.GROUPS.items():
    cols = [c for c in ALL_COLS if c not in gcols]
    a, pr, _ = cv(best_fam, cols)
    loo[gname] = a - base_a  # negative => dropping it hurt => group is useful
    log(f"  drop {gname:<14} AUC={a:.4f} (delta {a-base_a:+.4f})  PR={pr:.4f}")
RESULTS["leave_one_out_delta"] = {k: round(v, 4) for k, v in loo.items()}

# ---------------- STUDY 3: forward selection over groups ----------------
log(f"STUDY 3: forward group selection with {best_fam}")
chosen, chosen_cols, cur = [], [], 0.5
remaining = list(FE.GROUPS)
while remaining:
    scored = []
    for g in remaining:
        cols = list(dict.fromkeys(chosen_cols + FE.GROUPS[g]))
        a, _, _ = cv(best_fam, cols); scored.append((a, g))
    a, g = max(scored)
    if a <= cur + 0.0005:
        break
    chosen.append(g); chosen_cols = list(dict.fromkeys(chosen_cols + FE.GROUPS[g]))
    cur = a; remaining.remove(g)
    log(f"  + {g:<14} -> AUC={a:.4f}  (set={chosen})")
RESULTS["forward_selected_groups"] = chosen
RESULTS["forward_auc"] = round(cur, 4)

# ---------------- STUDY 4: hyperparameter search on winner ----------------
# Family AUCs are within ~1 CV std of each other, so family is not the lever. We tune
# the production model (HistGBM: native categoricals, calibrated probs, no libomp dep).
tune_fam = best_fam if best_fam in ("lgbm", "hgb", "xgb") else "hgb"
log(f"STUDY 4: hyperparameter search (tuning {tune_fam}) on selected groups")
sel_cols = chosen_cols or ALL_COLS
grid = list(itertools.product([0.03, 0.05, 0.1], [400, 800], [15, 31, 63]))
best = (base_a, None)
for lr, n, leaves in grid:
    def mk(name=best_fam, lr=lr, n=n, leaves=leaves):
        return (lr, n, leaves)
    # only tune the native-cat GBMs cleanly
    aucs = []
    for tr, te in folds():
        X = _target_enc(sel_cols, tr, te); cc = cats_in(sel_cols)
        Xtr, Xte = X[tr].copy(), X[te].copy()
        for c in cc:
            Xtr[c] = _cap(Xtr[c], c); Xte[c] = _cap(Xte[c], c)
        if tune_fam == "lgbm":
            m = lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=leaves,
                min_child_samples=30, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                random_state=0, verbose=-1); m.fit(Xtr, Y[tr], categorical_feature=cc)
        elif tune_fam == "hgb":
            m = HistGradientBoostingClassifier(categorical_features="from_dtype",
                learning_rate=lr, max_iter=n, max_leaf_nodes=leaves, l2_regularization=1.0,
                min_samples_leaf=30, early_stopping=True, random_state=0); m.fit(Xtr, Y[tr])
        else:
            md = max(3, int(np.log2(leaves)) + 1)
            m = xgb.XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=md,
                min_child_weight=5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", enable_categorical=True, eval_metric="logloss",
                random_state=0); m.fit(Xtr, Y[tr])
        aucs.append(roc_auc_score(Y[te], m.predict_proba(Xte)[:, 1]))
    a = float(np.mean(aucs))
    if a > best[0]: best = (a, dict(learning_rate=lr, n=n, leaves=leaves))
    log(f"  lr={lr} n={n} leaves={leaves} -> AUC={a:.4f}")
RESULTS["best_hyperparams"] = best[1]; RESULTS["best_cv_auc"] = round(best[0], 4)
log(f"DONE. best family={best_fam}, AUC={best[0]:.4f}, params={best[1]}")

with open("experiment_results.json", "w") as fh:
    json.dump({"best_family": best_fam, **RESULTS}, fh, indent=2)
log("saved -> experiment_results.json")
