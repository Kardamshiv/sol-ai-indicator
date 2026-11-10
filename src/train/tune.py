"""Step 7-9: tuning, baselines, model comparison and full metrics.  TRAINING + VALIDATION DATA ONLY.

Run from repo root:   python -m src.train.tune            (optional: --n-iter 24)

Protocol
  * Only rows up to the end of the validation period are used; the test period is dropped immediately.
  * Rolling-origin (expanding window) CV, 6 folds, gap of HORIZON candles between train and evaluation block.
  * Hyper-parameters come from a PREDEFINED search space (random search, fixed seed); chosen by mean fold ROC-AUC.
  * 'Confident signals' = the top MIN_COVERAGE share of candles by confidence in each fold (uses no labels).
  * Confidence intervals: cluster bootstrap over calendar days (signals on one day are not independent).
Outputs -> results/: experiment_log.csv, model_comparison.csv, cv_folds.csv, confusion_matrices.json,
                     best_params.json, environment.json
This script never saves or replaces the deployed model.
"""
import argparse, json, os, pickle, platform, sys, time
import numpy as np, pandas as pd
import lightgbm, sklearn
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from src.config import *
from src.features.build import FEATURES
from src.data.split import build_dataset, split_bounds
from src.models import DirectionModel

OUT = ROOT / "results"
N_FOLDS = 6
SPACE = dict(num_leaves=[7, 15, 31, 63], min_child_samples=[50, 100, 200, 400], learning_rate=[0.01, 0.03, 0.05],
             n_estimators=[150, 300, 600], colsample_bytree=[0.5, 0.7, 0.9], subsample=[0.6, 0.8, 1.0], reg_lambda=[0, 1, 10, 50])
WHY = {"Always Up": "Trivial baseline: majority direction",
       "Persistence": "Forecasting baseline: next hour continues the last hour's direction",
       "RSI rule (30/70)": "Classic technical-indicator baseline named in the brief",
       "RSI extremes (top 3%)": "Same RSI idea at the same coverage as the ML models (fair comparison)",
       "Reversal (top 3%)": "Contrarian persistence: the next hour reverses the last hour, at the same coverage",
       "Logistic regression": "Simple linear ML baseline (quantile-normalised inputs)",
       "Random forest": "Conventional non-linear ML model, bagged trees",
       "LightGBM (default)": "Gradient boosting with the settings used for the deployed model",
       "LightGBM (tuned)": "Gradient boosting with the best settings from the predefined search"}


def cluster_boot(vals, groups, n=2000):
    vals = np.asarray(vals, float); g, _ = pd.factorize(groups); G = g.max() + 1
    sums = np.bincount(g, weights=vals, minlength=G); cnt = np.bincount(g, minlength=G).astype(float)
    idx = np.random.default_rng(SEED).integers(0, G, (n, G))
    est = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(vals.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def run_cv(dev, folds, kind, params=None):
    out = []
    for a, b in folds:
        tr, te = dev.iloc[a], dev.iloc[b]
        t0 = time.perf_counter(); m = DirectionModel(kind, params).fit(tr[FEATURES], tr["y"].astype(int)); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(te[FEATURES])[:, 2]; pr = time.perf_counter() - t0
        out.append(dict(p=p, fit_s=fit_s, pred_ms_1k=pr * 1e6 / len(te), size_kb=len(pickle.dumps(m)) / 1024))
    return out


def auc_clear(te, score):
    k = (te["y"] != 1).values; yt = (te["y"].values[k] == 2).astype(int)
    return roc_auc_score(yt, score[k]) if 0 < yt.sum() < len(yt) else np.nan


def evaluate(name, dev, folds, scores, conf=None, sig=None, probs=None, extra=None):
    """scores: per-fold score (>0 means Up). conf: per-fold confidence for top-q signals. sig: per-fold natural signals."""
    extra = extra or {}
    fold_rows, S, F, D, YT, PRED, PRB, AUCS = [], [], [], [], [], [], [], []
    for i, (a, b) in enumerate(folds):
        te = dev.iloc[b]; sc = scores[i]
        k = (te["y"] != 1).values; yt = (te["y"].values[k] == 2).astype(int)
        if sig is not None: sg = sig[i]
        else:
            thr = np.quantile(conf[i], 1 - MIN_COVERAGE); sg = np.where(conf[i] >= thr, np.sign(sc), 0)
        f = te["fwd"].values; m = sg != 0
        hit = (np.sign(f[m]) == sg[m]).mean() if m.any() else np.nan
        net = (sg[m] * f[m] - COST).mean() if m.any() else np.nan
        auc = auc_clear(te, sc); AUCS.append(auc)
        fold_rows.append(dict(method=name, fold=i + 1, test_start=str(te["time"].iloc[0])[:10], test_end=str(te["time"].iloc[-1])[:10],
                              rows=len(te), auc=auc, signals=int(m.sum()), hit=hit, net_per_trade=net))
        S.append(sg); F.append(f); D.append(te["time"].dt.strftime("%Y-%m-%d").values)
        YT.append(yt); PRED.append((sc[k] > 0).astype(int))
        if probs is not None: PRB.append(np.clip(probs[i][k], 1e-6, 1 - 1e-6))
    S, F, D, YT, PRED = map(np.concatenate, (S, F, D, YT, PRED)); m = S != 0
    pr, rc, f1, _ = precision_recall_fscore_support(YT, PRED, average="binary", zero_division=0)
    row = dict(method=name, why_included=WHY.get(name, ""), cv_auc_mean=np.nanmean(AUCS), cv_auc_std=np.nanstd(AUCS),
               accuracy_clear=(YT == PRED).mean(), precision_up=pr, recall_up=rc, f1_up=f1)
    if probs is not None:
        P, Y = np.concatenate(PRB), YT
        row.update(pr_auc=average_precision_score(Y, P), logloss=log_loss(Y, P), logloss_constant=log_loss(Y, np.full(len(Y), Y.mean())), brier=brier_score_loss(Y, P))
    else:
        row["pr_auc"] = np.nan
    row.update(signals=int(m.sum()), coverage=m.mean())
    if m.sum() >= 20:
        h, hl, hh = cluster_boot((np.sign(F[m]) == S[m]).astype(float), D[m]); r, rl, rh = cluster_boot(S[m] * F[m] - COST, D[m])
        row.update(hit=h, hit_ci_low=hl, hit_ci_high=hh, net_per_trade=r, net_ci_low=rl, net_ci_high=rh)
    row["always_up_hit_all_candles"] = float((F > 0).mean())
    row.update(extra)
    return row, fold_rows, confusion_matrix(YT, PRED, labels=[0, 1]).tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-iter", type=int, default=24); args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", parse_dates=["time"])
    _, d = build_dataset(df); b = split_bounds(len(d))
    test_start = d["time"].iloc[b["test"][0]]
    dev = d.iloc[:b["val"][1]].copy()                      # training + validation ONLY
    del df, d
    assert dev["time"].max() < test_start
    print(f"Using {len(dev):,} rows up to {dev['time'].iloc[-1]}; the test period (from {test_start}) is not used.")
    json.dump(dict(python=sys.version, platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                   pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, seed=SEED),
              open(OUT / "environment.json", "w"), indent=2)
    folds = list(TimeSeriesSplit(n_splits=N_FOLDS, gap=HORIZON).split(dev))

    # ---- predefined random search (+ the deployed defaults as E000)
    rng = np.random.default_rng(SEED); configs, seen = [{}], {"{}"}
    while len(configs) < args.n_iter + 1:
        c = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}; key = json.dumps(c, sort_keys=True)
        if key not in seen: seen.add(key); configs.append(c)
    log, runs = [], []
    for i, c in enumerate(configs):
        t0 = time.perf_counter(); res = run_cv(dev, folds, "lgbm", c)
        aucs = [auc_clear(dev.iloc[bb], r["p"] - 0.5) for (_, bb), r in zip(folds, res)]
        ll = [log_loss((dev.iloc[bb]["y"].values[dev.iloc[bb]["y"].values != 1] == 2).astype(int),
                       np.clip(r["p"][dev.iloc[bb]["y"].values != 1], 1e-6, 1 - 1e-6)) for (_, bb), r in zip(folds, res)]
        runs.append(res)
        log.append(dict(run_id=f"E{i:03d}", split=f"rolling-origin {N_FOLDS} folds, gap={HORIZON}", seed=SEED, model="LightGBM" + (" (default)" if i == 0 else ""),
                        hyperparameters=json.dumps(c), cv_auc_mean=np.mean(aucs), cv_auc_std=np.std(aucs), cv_logloss_mean=np.mean(ll),
                        fit_time_s_mean=np.mean([r["fit_s"] for r in res]), runtime_s=time.perf_counter() - t0))
        print(f"{log[-1]['run_id']}  AUC {log[-1]['cv_auc_mean']:.4f} +/- {log[-1]['cv_auc_std']:.4f}  logloss {log[-1]['cv_logloss_mean']:.4f}  {c}")
    best = int(np.argmax([r["cv_auc_mean"] for r in log]))
    for i, r in enumerate(log): r["is_best"] = i == best
    pd.DataFrame(log).to_csv(OUT / "experiment_log.csv", index=False)
    json.dump(dict(run_id=log[best]["run_id"], params=configs[best], cv_auc_mean=log[best]["cv_auc_mean"]), open(OUT / "best_params.json", "w"), indent=2)

    # ---- model comparison under identical folds and metrics
    def ml(name, res, timing=True):
        sc = [r["p"] - 0.5 for r in res]
        ex = dict(fit_time_s=np.mean([r["fit_s"] for r in res]), model_size_kb=res[-1]["size_kb"], predict_ms_per_1k=np.mean([r["pred_ms_1k"] for r in res]))
        return evaluate(name, dev, folds, sc, conf=[np.abs(s) for s in sc], probs=[r["p"] for r in res], extra=ex)
    te_of = [dev.iloc[bb] for _, bb in folds]
    rows = [evaluate("Always Up", dev, folds, [np.ones(len(t)) for t in te_of], sig=[np.ones(len(t), int) for t in te_of]),
            evaluate("Persistence", dev, folds, [t["ret_4"].values for t in te_of], conf=[np.abs(t["ret_4"].values) for t in te_of]),
            evaluate("RSI rule (30/70)", dev, folds, [50 - t["rsi_14"].values for t in te_of],
                     sig=[np.where(t["rsi_14"] < 30, 1, np.where(t["rsi_14"] > 70, -1, 0)) for t in te_of]),
            evaluate("RSI extremes (top 3%)", dev, folds, [50 - t["rsi_14"].values for t in te_of], conf=[np.abs(t["rsi_14"].values - 50) for t in te_of]),
            evaluate("Reversal (top 3%)", dev, folds, [-t["ret_4"].values for t in te_of], conf=[np.abs(t["ret_4"].values) for t in te_of]),
            ml("Logistic regression", run_cv(dev, folds, "logreg_q")), ml("Random forest", run_cv(dev, folds, "rf")),
            ml("LightGBM (default)", runs[0]), ml("LightGBM (tuned)", runs[best])]
    comp = pd.DataFrame([r[0] for r in rows]); comp["test_score"] = "not evaluated (reserved for the final one-shot test)"
    comp.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame([x for r in rows for x in r[1]]).to_csv(OUT / "cv_folds.csv", index=False)
    pd.concat([dev.iloc[bb][["time", "y", "fwd"]].assign(fold=i + 1, p_default=runs[0][i]["p"], p_tuned=runs[best][i]["p"])
               for i, (_, bb) in enumerate(folds)]).to_csv(OUT / "oof_predictions.csv", index=False)   # out-of-fold predictions (train+val only)
    json.dump({r[0]["method"]: r[2] for r in rows}, open(OUT / "confusion_matrices.json", "w"), indent=2)
    show = ["method", "cv_auc_mean", "cv_auc_std", "accuracy_clear", "f1_up", "logloss", "logloss_constant", "signals", "hit", "hit_ci_low", "hit_ci_high", "net_per_trade", "net_ci_low", "net_ci_high", "fit_time_s", "model_size_kb"]
    print("\nValidation-protocol comparison (rolling-origin CV; NO test data):")
    print(comp[[c for c in show if c in comp]].round(4).to_string(index=False))
    print(f"\nBest search config: {log[best]['run_id']} {configs[best]}")
    print("Saved results/experiment_log.csv, model_comparison.csv, cv_folds.csv, confusion_matrices.json, best_params.json, environment.json, oof_predictions.csv")


if __name__ == "__main__":
    main()
