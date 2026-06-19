"""Train & evaluate the event-driven congestion prediction system.

Evaluation uses a strict TIME-BASED split (train on the earlier 80% of the timeline,
test on the most recent 20%) because the real task is forecasting the future — a random
split would leak future information and overstate accuracy.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, mean_absolute_error,
                             confusion_matrix)

from congestion import data as D
from congestion import features as FE
from congestion.models import ImpactModel, HotspotModel
from congestion import recommend as R
from congestion.logging_utils import get_logger, log_metrics, write_results

warnings.filterwarnings("ignore", category=UserWarning)
RESULTS = {}

_LOG = get_logger("train")
_T0 = time.time()
def log(msg):
    """Timestamped progress line so training is visible live in the terminal + log file."""
    _LOG.info(f"[{time.time()-_T0:5.1f}s] {msg}")


def time_split(df, frac=0.8):
    cut = df["start"].quantile(frac)
    tr, te = df[df["start"] <= cut].copy(), df[df["start"] > cut].copy()
    return tr.reset_index(drop=True), te.reset_index(drop=True), cut


def best_threshold(y, p):
    """Pick the probability threshold maximising F1 on the given set."""
    ts = np.quantile(p, np.linspace(0.5, 0.99, 50))
    return max(ts, key=lambda t: f1_score(y, p >= t, zero_division=0))


def eval_classifier(name, y_tr, p_tr, y_te, p_te):
    thr = best_threshold(y_tr, p_tr)
    yp = (p_te >= thr).astype(int)
    m = dict(
        auc=roc_auc_score(y_te, p_te),
        pr_auc=average_precision_score(y_te, p_te),
        brier=brier_score_loss(y_te, p_te),
        precision=precision_score(y_te, yp, zero_division=0),
        recall=recall_score(y_te, yp, zero_division=0),
        f1=f1_score(y_te, yp, zero_division=0),
        threshold=float(thr), positive_rate=float(y_te.mean()))
    cm = confusion_matrix(y_te, yp)
    print(f"\n### {name}")
    print(f"  AUC={m['auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  Brier={m['brier']:.3f}  "
          f"(base rate {m['positive_rate']:.3f})")
    print(f"  @thr={thr:.3f}: precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"  confusion [tn fp / fn tp]: {cm.ravel().tolist()}")
    RESULTS[name] = {k: round(float(v), 4) for k, v in m.items()}
    log_metrics(name, RESULTS[name], _LOG)  # -> logs/metrics.jsonl
    return m


def main():
    log("loading + cleaning data ...")
    df = D.load_clean()
    print(f"Loaded {len(df)} cleaned events | "
          f"{df['start'].min().date()} -> {df['start'].max().date()}")
    tr, te, cut = time_split(df)
    print(f"Train {len(tr)}  Test {len(te)}  (cut at {cut})")

    # ---------------- Impact models ----------------
    log("building features + fitting impact models (closure, priority, duration) ...")
    im = ImpactModel().fit(tr)
    log("impact models fitted; scoring train/test ...")
    p_tr = im.predict(tr)
    p_te = im.predict(te)
    log("impact scoring done -> evaluating")

    eval_classifier("road_closure", tr["road_closure"].values, p_tr["p_road_closure"].values,
                    te["road_closure"].values, p_te["p_road_closure"].values)
    eval_classifier("high_priority", tr["high_priority"].values, p_tr["p_high_priority"].values,
                    te["high_priority"].values, p_te["p_high_priority"].values)
    print("  NOTE: 'priority' is an administrative rule (High == event on a named corridor,"
          " 99.9% match), so AUC~1.0 reflects recovering that rule, not genuine forecasting."
          " It is used only as a 'major-road importance' signal in recommendations.")

    # duration regressor
    if im.reg_duration is not None:
        m = te["duration_min"].notna().values
        if m.sum() > 20:
            yt = te["duration_min"].values[m]
            yp = p_te["exp_duration_min"].values[m]
            mae = mean_absolute_error(yt, yp)
            med_ae = np.median(np.abs(yt - yp))
            base = mean_absolute_error(yt, np.full_like(yt, np.median(tr["duration_min"].dropna())))
            print(f"\n### clearance_duration  (n_test={m.sum()})")
            print(f"  MAE={mae:.1f} min  MedianAE={med_ae:.1f} min  (naive-median MAE={base:.1f})")
            RESULTS["duration"] = {"mae_min": round(float(mae), 1),
                                   "median_ae_min": round(float(med_ae), 1),
                                   "naive_mae_min": round(float(base), 1),
                                   "n_test": int(m.sum())}
            log_metrics("duration", RESULTS["duration"], _LOG)

    # ---- survival analysis for clearance time (uses censored / still-active events) ----
    log("fitting Weibull AFT survival model (censored-aware clearance time) ...")
    try:
        from congestion import survival as SV
        from lifelines.utils import concordance_index
        sf_tr, sf_te = SV.build_survival_frame(tr), SV.build_survival_frame(te)
        aft, ci_tr, ci_te, _ = SV.fit_eval(sf_tr, sf_te)
        # Fair head-to-head: both predictors scored on the SAME survival test frame (T,E).
        gpred = p_te["exp_duration_min"].reindex(sf_te.index)
        ci_gbm = concordance_index(sf_te["T"], gpred, sf_te["E"])
        print(f"\n### clearance survival (Weibull AFT, censored-aware)")
        print(f"  train rows={len(sf_tr)} (obs={int(sf_tr['E'].sum())}, "
              f"censored={int((1-sf_tr['E']).sum())}) | test rows={len(sf_te)}")
        print(f"  concordance:  AFT={ci_te:.3f}   GBM-regressor={ci_gbm:.3f}   "
              f"(higher=better ranking of clearance time; AFT also exploits censored events)")
        RESULTS["survival"] = {"aft_concordance": round(float(ci_te), 4),
                               "gbm_concordance": round(float(ci_gbm), 4),
                               "censored_train_events": int((1 - sf_tr['E']).sum())}
        log_metrics("survival", RESULTS["survival"], _LOG)
    except Exception as e:
        print(f"\n[survival skipped: {e}]")

    # ---- permutation importance for the road-closure model (drivers of impact) ----
    log("computing permutation importance for road-closure drivers ...")
    from sklearn.inspection import permutation_importance
    from congestion import features as _FE
    _Xfull = im._features(te)
    Xte = _Xfull[[c for c in _FE.CLF_FEATURES if c in _Xfull.columns]]
    pi = permutation_importance(im.clf_closure, Xte, te["road_closure"].values,
                                scoring="roc_auc", n_repeats=5, random_state=0)
    imp = (pd.Series(pi.importances_mean, index=Xte.columns)
           .sort_values(ascending=False).head(8))
    print("\n### road_closure drivers (permutation importance, AUC drop)")
    for k, v in imp.items():
        print(f"    {k:<22} {v:+.4f}")
    RESULTS["closure_drivers"] = {k: round(float(v), 4) for k, v in imp.items()}

    # ---------------- Spatio-temporal hotspot forecaster ----------------
    log("fitting spatio-temporal hotspot forecaster (Poisson GBM + spatial lags) ...")
    from congestion import evaluate as EV
    hm = HotspotModel().fit(tr)
    fc = hm.forecast_panel(df)
    fc_te = fc[fc["bin"] > cut]
    mae_e = mean_absolute_error(fc_te["n_events"], fc_te["pred_events"])
    # Baseline 1: each corridor's own historical mean hourly load (no time signal).
    corridor_mean = fc_te.groupby("corridor")["n_events"].transform("mean")
    base_e = mean_absolute_error(fc_te["n_events"], corridor_mean)
    # Baseline 2 (slides): seasonal-naive = same hour last week.
    sn = EV.seasonal_naive_panel(fc).dropna(subset=["snaive"])
    sn_te = sn[sn["bin"] > cut]
    base_sn = mean_absolute_error(sn_te["n_events"], sn_te["snaive"])
    corr = np.corrcoef(fc_te["pred_events"], fc_te["n_events"])[0, 1]
    skill = 1 - mae_e / base_sn
    print(f"\n### hotspot_forecast (corridor x hour, n_test_bins={len(fc_te)})")
    print(f"  model MAE={mae_e:.3f}/bin | corridor-mean baseline={base_e:.3f} | "
          f"seasonal-naive baseline={base_sn:.3f}")
    print(f"  skill vs seasonal-naive={skill:+.1%}   pred-vs-actual corr={corr:.3f}")

    # Rolling-origin CV (expanding window) for an honest, multi-fold error estimate.
    log("rolling-origin CV for hotspot model (4 folds, refits per fold) ...")
    _fold = [0]
    def _fp(tr_df, te_df):
        _fold[0] += 1; log(f"  CV fold {_fold[0]}: train={len(tr_df)} test={len(te_df)}")
        m = HotspotModel().fit(tr_df)
        f = m.forecast_panel(pd.concat([tr_df, te_df]))
        f = f[f["bin"] > te_df["start"].min().floor("h")]
        return f["n_events"].values, f["pred_events"].values
    cv = EV.rolling_origin_cv(df, _fp, n_folds=4)
    print(f"  rolling-origin CV MAE: {[round(x,3) for x in cv]}  "
          f"mean={np.mean(cv):.3f} +/- {np.std(cv):.3f}")

    # Residual Moran's I: is spatial structure left unmodelled across corridors?
    res = (fc_te.assign(resid=fc_te["n_events"] - fc_te["pred_events"])
           .groupby("corridor")["resid"].mean().reindex(hm.corr_order).fillna(0).values)
    mi = EV.morans_i(res, hm.W)
    print(f"  residual Moran's I (corridor)={mi:+.3f}  "
          f"(~0 => little spatial signal left; note: 22 corridors < 30, so indicative only)")
    RESULTS["hotspot"] = {"event_mae_per_bin": round(float(mae_e), 4),
                          "corridor_mean_mae": round(float(base_e), 4),
                          "seasonal_naive_mae": round(float(base_sn), 4),
                          "skill_vs_seasonal_naive": round(float(skill), 4),
                          "pred_actual_corr": round(float(corr), 4),
                          "cv_mae_mean": round(float(np.mean(cv)), 4),
                          "cv_mae_std": round(float(np.std(cv)), 4),
                          "residual_morans_i": round(float(mi), 4),
                          "n_test_bins": int(len(fc_te))}
    log_metrics("hotspot", RESULTS["hotspot"], _LOG)

    # top forecast hotspots in the test horizon (corridor x hour-of-week)
    fc_te = fc_te.assign(how=fc_te["dow"] * 24 + fc_te["hour"])
    top = (fc_te.groupby("corridor")["pred_events"].mean()
           .sort_values(ascending=False).head(8).round(3))
    print("  top forecast corridors (mean predicted events/hr):")
    for c, v in top.items():
        print(f"    {c:<20} {v}")

    # ---------------- Recommendation demo ----------------
    print("\n### sample recommendations (most recent 6 events)")
    sample = te.tail(6)
    recs = R.recommend(sample, im.predict(sample))
    sp = im.predict(sample)
    for idx in sample.index:
        print(f"  [{sample.at[idx,'event_cause']:<16} {sample.at[idx,'corridor']:<16}] "
              f"P(close)={sp.at[idx,'p_road_closure']:.2f} P(high)={sp.at[idx,'p_high_priority']:.2f} "
              f"~{sp.at[idx,'exp_duration_min']:.0f}min -> {recs.at[idx,'severity_tier']:<8} "
              f"{recs.at[idx,'rec_officers']}off/{recs.at[idx,'rec_barricades']}barr | {recs.at[idx,'rec_diversion'][:42]}")

    RESULTS["_meta"] = {"n_events": int(len(df)), "n_train": int(len(tr)),
                        "n_test": int(len(te)), "split_at": str(cut),
                        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    write_results(RESULTS)
    log(f"saved metrics snapshot -> results.json; appended rows -> logs/metrics.jsonl")


if __name__ == "__main__":
    main()
