"""Runs the experiments your report needs as evidence. Run from repo root:
    python -m src.train.experiments
Results -> reports/experiments.csv

IMPORTANT: choose settings by the VALIDATION column (val_net_per_trade), not by test.
Test numbers are for honest reporting only. Picking the best test row = leakage.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.config import *
from src.features.build import make_features, make_labels, FEATURES
from src.train.train import signals

IV_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[INTERVAL]


def make_model(name="lgbm"):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=300, learning_rate=0.03,
                              num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=SEED, verbose=-1)


def evaluate(df, horizon=HORIZON, thresh=THRESH, model="lgbm", months=None, feats=FEATURES, tag=""):
    if months:
        df = df.iloc[-int(months * 30 * 86400 / IV_SEC):]
    X = make_features(df)
    fwd, y = make_labels(df, horizon, thresh)
    d = pd.concat([X, fwd.rename("fwd"), y.rename("y")], axis=1).dropna()
    n = len(d); i1, i2 = int(n * .70), int(n * .85)
    tr, va, te = d.iloc[:i1], d.iloc[i1 + horizon:i2], d.iloc[i2 + horizon:]
    m = make_model(model).fit(tr[feats], tr["y"].astype(int))
    pv, pt = m.predict_proba(va[feats]), m.predict_proba(te[feats])
    best, best_val = 0.5, np.nan
    for cut in np.arange(0.40, 0.76, 0.02):
        s = signals(pv, cut); mk = s != 0
        if mk.sum() < 30: continue
        sc = (s[mk] * va["fwd"].values[mk] - COST).mean()
        if np.isnan(best_val) or sc > best_val: best, best_val = cut, sc
    s, f = signals(pt, best), te["fwd"].values
    mk = s != 0
    return dict(group=tag, model=model, horizon=horizon, thresh=thresh, months=months or "all",
                n_feats=len(feats), cutoff=round(best, 2), val_net_per_trade=round(best_val, 5),
                test_signals=int(mk.sum()), test_hit=round((np.sign(f[mk]) == s[mk]).mean(), 4) if mk.any() else np.nan,
                test_net_per_trade=round((s[mk] * f[mk] - COST).mean(), 5) if mk.any() else np.nan,
                always_up_hit=round((f > 0).mean(), 4))


def main():
    df = pd.read_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", parse_dates=["time"])
    jobs = []
    for h in (1, 2, 4, 8):                                   # horizon x dead-zone
        for t in (0.0015, 0.002, 0.003, 0.004):   # SOL moves more than BTC
            jobs.append(dict(horizon=h, thresh=t, tag="horizon_thresh"))
    for mo in (3, 6, 12, 24):                                # history length
        jobs.append(dict(months=mo, tag="history"))
    jobs.append(dict(model="logreg", tag="model"))           # model comparison
    jobs.append(dict(model="lgbm", tag="model"))
    groups = {"no_time": ["hour_sin", "hour_cos", "dow"],    # feature ablation
              "no_flow": ["taker_ratio", "trades_z", "vol_z"],
              "no_classic": ["rsi_14", "macd_hist", "bb_width", "bb_pos", "dist_vwap"]}
    jobs.append(dict(tag="ablation_none_removed"))
    for name, cols in groups.items():
        jobs.append(dict(feats=[c for c in FEATURES if c not in cols], tag=f"ablation_{name}"))

    rows = []
    for j in jobs:
        try:
            r = evaluate(df, **j); rows.append(r); print(r)
        except Exception as e:
            print("skipped", {k: v for k, v in j.items() if k != "feats"}, repr(e))
    out = pd.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    out.to_csv(ROOT / "reports" / "experiments.csv", index=False)
    print("\nSaved reports/experiments.csv")


if __name__ == "__main__":
    main()
