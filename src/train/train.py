"""Train, validate chronologically, compare with baselines, save model.

Run from repo root:  python -m src.train.train
"""
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from src.config import *
from src.features.build import make_features, make_labels, FEATURES


def signals(proba, cutoff):
    """+1 Up, -1 Down, 0 no signal. Only confident Up/Down count."""
    pred, conf = proba.argmax(1), proba.max(1)
    return np.where((conf >= cutoff) & (pred == 2), 1, np.where((conf >= cutoff) & (pred == 0), -1, 0))


def report(name, sig, fwd):
    m = sig != 0
    if m.sum() == 0:
        print(f"{name:28s} no signals"); return
    hit = (np.sign(fwd[m]) == sig[m]).mean()
    pnl = (sig[m] * fwd[m] - COST)
    print(f"{name:28s} signals={m.sum():5d} coverage={m.mean():.1%} hit={hit:.3f} "
          f"net/trade={pnl.mean():+.4%}")


def main():
    df = pd.read_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", parse_dates=["time"])
    X = make_features(df)
    fwd, y = make_labels(df)
    d = pd.concat([X, fwd.rename("fwd"), y.rename("y")], axis=1).dropna()
    n = len(d)
    # chronological split with an embargo of HORIZON candles between sets
    i1, i2 = int(n * .70), int(n * .85)
    tr, va, te = d.iloc[:i1], d.iloc[i1 + HORIZON:i2], d.iloc[i2 + HORIZON:]
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    params = dict(objective="multiclass", num_class=3, n_estimators=300, learning_rate=0.03,
                  num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=SEED, verbose=-1)

    # 1) walk-forward CV on train+val (with gap) -> honest stability check
    tv = pd.concat([tr, va])
    accs = []
    for a, b in TimeSeriesSplit(n_splits=5, gap=HORIZON).split(tv):
        m = lgb.LGBMClassifier(**params).fit(tv.iloc[a][FEATURES], tv.iloc[a]["y"].astype(int))
        accs.append((m.predict(tv.iloc[b][FEATURES]) == tv.iloc[b]["y"]).mean())
    print("walk-forward 3-class accuracy per fold:", np.round(accs, 3))

    # 2) fit on train, pick confidence cutoff on validation
    model = lgb.LGBMClassifier(**params).fit(tr[FEATURES], tr["y"].astype(int))
    pv = model.predict_proba(va[FEATURES])
    best, best_score = 0.5, -1
    for cutoff in np.arange(0.40, 0.76, 0.02):
        s = signals(pv, cutoff); mk = s != 0
        if mk.sum() < 30: continue
        score = (s[mk] * va["fwd"].values[mk] - COST).mean()
        if score > best_score: best, best_score = cutoff, score
    print(f"chosen confidence cutoff: {best:.2f}")

    # 3) untouched test set: model vs baselines
    print("\n--- TEST SET (never used for tuning) ---")
    pt = model.predict_proba(te[FEATURES])
    f = te["fwd"].values
    report("LightGBM (confident)", signals(pt, best), f)
    report("Always Up", np.ones(len(te), int), f)
    rsi_sig = np.where(te["rsi_14"] < 30, 1, np.where(te["rsi_14"] > 70, -1, 0))
    report("RSI rule (30/70)", rsi_sig, f)

    imp = pd.Series(model.feature_importances_, FEATURES).sort_values(ascending=False)
    print("\ntop features:\n", imp.head(8).to_string())

    # 4) refit on train+val for deployment, save everything the server needs
    final = lgb.LGBMClassifier(**params).fit(tv[FEATURES], tv["y"].astype(int))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "model.joblib")
    json.dump(dict(symbol=SYMBOL, interval=INTERVAL, horizon=HORIZON, thresh=THRESH,
                   cutoff=float(best), features=FEATURES, classes={0: "Down", 1: "Neutral", 2: "Up"}),
              open(MODEL_DIR / "config.json", "w"), indent=2)
    print("saved models/model.joblib and models/config.json")


if __name__ == "__main__":
    main()
