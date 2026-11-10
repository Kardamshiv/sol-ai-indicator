"""Does probability calibration help?  Uses ONLY out-of-fold predictions saved by tune.py (train + validation data).

Run from repo root, after  python -m src.train.tune :
    python -m src.train.calibration_cv         -> results/calibration_cv.csv, results/calibration_cv.png

Rolling scheme: a calibrator is fitted on the out-of-fold predictions of folds 1..k-1 and evaluated on fold k,
so it is never evaluated on data it was fitted on. Calibrators: Platt (logistic on logit(p)) and isotonic.
Metrics are on clear moves (P(Up) vs actual Up). The constant-prior log loss (about 0.693) is the bar to beat.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from src.config import *

OUT = ROOT / "results"
KINDS = ("none", "platt", "isotonic")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def ece(y, p, bins=10):
    d = pd.DataFrame(dict(y=y, p=p, q=pd.qcut(p, bins, duplicates="drop"))); g = d.groupby("q", observed=True)
    return float((g.size() / len(d) * (g["y"].mean() - g["p"].mean()).abs()).sum())


def fit_cal(kind, p, y):
    if kind == "platt":
        m = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
        return lambda q: m.predict_proba(logit(q).reshape(-1, 1))[:, 1]
    if kind == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p, y)
        return m.predict
    return lambda q: q


def main():
    oof = pd.read_csv(OUT / "oof_predictions.csv", parse_dates=["time"])
    oof = oof[oof.y != 1].copy(); oof["t"] = (oof.y == 2).astype(int)
    folds = int(oof.fold.max()); rows, pooled = [], {}
    for col in ("p_default", "p_tuned"):
        for k in range(2, folds + 1):
            trn, te = oof[oof.fold < k], oof[oof.fold == k]
            for kind in KINDS:
                q = np.clip(fit_cal(kind, trn[col].values, trn.t.values)(te[col].values), 1e-4, 1 - 1e-4)
                pooled.setdefault((col, kind), []).append(pd.DataFrame(dict(q=q, t=te.t.values, fold=k)))
                rows.append(dict(model=col, calibrator=kind, fold=k, logloss=log_loss(te.t, q), brier=brier_score_loss(te.t, q),
                                 ece=ece(te.t.values, q), logloss_constant=log_loss(te.t, np.full(len(te), trn.t.mean()))))
    r = pd.DataFrame(rows); r.to_csv(OUT / "calibration_cv.csv", index=False)
    base = r[r.calibrator == "none"].set_index(["model", "fold"])["logloss"]
    r["better_than_none"] = [row.logloss < base[(row.model, row.fold)] for row in r.itertuples()]
    s = r.groupby(["model", "calibrator"]).agg(logloss=("logloss", "mean"), constant=("logloss_constant", "mean"), brier=("brier", "mean"),
                                                ece=("ece", "mean"), folds_better_than_none=("better_than_none", "sum")).round(4)
    print(f"Calibration, rolling scheme (folds 2-{folds}); constant-prior log loss is the bar to beat:\n{s.to_string()}\n")

    print("Top 3% most confident directions: does the displayed confidence match the hit rate? (clear moves)")
    for kind in ("none", "platt"):
        d = pd.concat(pooled[("p_default", kind)]); hits, confs = [], []
        for _, g in d.groupby("fold"):
            c = np.maximum(g.q, 1 - g.q); m = c >= np.quantile(c, 0.97)
            hits += list(((g.q[m] > 0.5).astype(int) == g.t[m]).astype(float)); confs += list(c[m])
        print(f"  {kind:6s} mean displayed confidence {np.mean(confs):.3f} | actual direction accuracy {np.mean(hits):.3f} | n={len(hits)}")

    fig, ax = plt.subplots(figsize=(5.5, 5.5)); ax.plot([0.4, 0.6], [0.4, 0.6], "--", color="gray", label="perfect")
    for kind, colr in (("none", "#c44e52"), ("platt", "#4c72b0"), ("isotonic", "#55a868")):
        d = pd.concat(pooled[("p_default", kind)]); d["b"] = pd.qcut(d.q, 10, duplicates="drop")
        g = d.groupby("b", observed=True).agg(p=("q", "mean"), a=("t", "mean")); ax.plot(g.p, g.a, "o-", color=colr, label=kind)
    ax.set_xlabel("Predicted P(Up)"); ax.set_ylabel("Actual share Up"); ax.set_title("Reliability, default LightGBM (folds 2-6)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "calibration_cv.png", dpi=130)
    print("Saved results/calibration_cv.csv and results/calibration_cv.png")


if __name__ == "__main__":
    main()
