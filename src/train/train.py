"""Train, validate chronologically, compare with baselines, save model.
Run from repo root:  python -m src.train.train
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.config import *
from src.features.build import make_features, make_labels, FEATURES
from src.models import make_model, signals


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
    fwd, y = make_labels(df)
    d = pd.concat([make_features(df), fwd.rename("fwd"), y.rename("y")], axis=1).dropna()
    n = len(d)
    i1, i2 = int(n * .70), int(n * .85)          # chronological split, embargo = HORIZON candles
    tr, va, te = d.iloc[:i1], d.iloc[i1 + HORIZON:i2], d.iloc[i2 + HORIZON:]
    tv = pd.concat([tr, va])
    print(f"train={len(tr)} val={len(va)} test={len(te)} | model={make_model().__class__.__name__}")
    clear = tv[tv["y"] != 1]
    print(f"clear moves (|ret|>{THRESH:.2%}): {len(clear) / len(tv):.1%} of candles; "
          f"Up share among them: {(clear['y'] == 2).mean():.3f}  <- 'always Up' direction baseline")

    # 1) walk-forward: direction accuracy on clear moves, per fold
    accs = []
    for a, b in TimeSeriesSplit(n_splits=5, gap=HORIZON).split(tv):
        m = make_model().fit(tv.iloc[a][FEATURES], tv.iloc[a]["y"].astype(int))
        p, yb = m.predict_proba(tv.iloc[b][FEATURES]), tv.iloc[b]["y"].astype(int).values
        k = yb != 1
        accs.append(((p[:, 2] > p[:, 0])[k] == (yb[k] == 2)).mean())
    print("walk-forward direction accuracy (clear moves) per fold:", np.round(accs, 3))

    # 2) fit on train, pick confidence cutoff on validation (needs at least MIN_COVERAGE signals)
    model = make_model().fit(tr[FEATURES], tr["y"].astype(int))
    pv = model.predict_proba(va[FEATURES])
    best, best_score = 0.40, -9
    for cutoff in np.arange(0.40, 0.80, 0.01):
        s = signals(pv, cutoff); mk = s != 0
        if mk.mean() < MIN_COVERAGE: continue
        score = (s[mk] * va["fwd"].values[mk] - COST).mean()
        if score > best_score: best, best_score = cutoff, score
    print(f"chosen confidence cutoff: {best:.2f}")
    cpd = 86400 / {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[INTERVAL]
    cov = (signals(pv, best) != 0).mean()
    print(f"validation coverage {cov:.2%} -> expect about {cov * cpd * 7:.0f} live signals per 7 days")

    # 3) untouched test set: model vs baselines
    print("\n--- TEST SET (never used for tuning) ---")
    f = te["fwd"].values
    report("Model (confident)", signals(model.predict_proba(te[FEATURES]), best), f)
    report("Always Up", np.ones(len(te), int), f)
    rsi_sig = np.where(te["rsi_14"] < 30, 1, np.where(te["rsi_14"] > 70, -1, 0))
    report("RSI rule (30/70)", rsi_sig, f)

    # 4) refit on train+val for deployment, save everything the server needs
    final = make_model().fit(tv[FEATURES], tv["y"].astype(int))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "model.joblib")
    json.dump(dict(symbol=SYMBOL, interval=INTERVAL, horizon=HORIZON, thresh=THRESH,
                   model=getattr(__import__("src.config", fromlist=["x"]), "MODEL_NAME", "lgbm_dir"),
                   cutoff=float(best), features=FEATURES, classes={0: "Down", 1: "Neutral", 2: "Up"}),
              open(MODEL_DIR / "config.json", "w"), indent=2)
    print("saved models/model.joblib and models/config.json")


if __name__ == "__main__":
    main()
