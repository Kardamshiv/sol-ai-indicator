"""Reliability plot: when the model says 'confidence 60%', is it right ~60% of the time?
Run from repo root:  python -m src.train.calibration   -> reports/calibration.png
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import *
from src.features.build import make_features, make_labels, FEATURES
from src.train.experiments import make_model


def main():
    df = pd.read_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", parse_dates=["time"])
    fwd, y = make_labels(df)
    d = pd.concat([make_features(df), y.rename("y")], axis=1).dropna()
    n = len(d); i1, i2 = int(n * .70), int(n * .85)
    tr, te = d.iloc[:i1], d.iloc[i2 + HORIZON:]
    p = make_model().fit(tr[FEATURES], tr["y"].astype(int)).predict_proba(te[FEATURES])
    conf, ok = p.max(1), (p.argmax(1) == te["y"].astype(int).values)
    bins = np.linspace(conf.min(), conf.max(), 8)
    idx = np.digitize(conf, bins)
    xs, ys, cnt = [], [], []
    for b in np.unique(idx):
        m = idx == b
        if m.sum() >= 30:
            xs.append(conf[m].mean()); ys.append(ok[m].mean()); cnt.append(int(m.sum()))
    print(pd.DataFrame(dict(mean_confidence=xs, actual_accuracy=ys, n=cnt)).round(3).to_string(index=False))
    plt.figure(figsize=(5, 5)); lo = min(xs + ys) - .02; hi = max(xs + ys) + .02
    plt.plot([lo, hi], [lo, hi], "--", label="perfect"); plt.plot(xs, ys, "o-", label="model")
    plt.xlabel("Predicted confidence"); plt.ylabel("Actual accuracy"); plt.title("Calibration (test set)")
    plt.legend(); plt.tight_layout()
    (ROOT / "reports").mkdir(exist_ok=True)
    plt.savefig(ROOT / "reports" / "calibration.png", dpi=150)
    print("saved reports/calibration.png")


if __name__ == "__main__":
    main()
