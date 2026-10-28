"""Models and the signal rule, shared by training, experiments, backtest and the live server.

Every model exposes predict_proba() -> columns [Down, Neutral, Up], so one signal rule fits all.
  lgbm / logreg         : 3-class (Down / Neutral / Up)
  lgbm_dir / logreg_dir : binary Up-vs-Down, trained ONLY on clear moves (|return| > THRESH).
                          Separates "which direction" from "is a big move coming".
Set the default with MODEL_NAME in src/config.py (default: lgbm_dir).
"""
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import src.config as C

LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED, verbose=-1)


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))


class DirectionModel:
    def __init__(self, kind="lgbm"):
        self.kind = kind

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        keep = y != 1
        self.m_ = lgb.LGBMClassifier(objective="binary", **LGB) if self.kind == "lgbm" else _logreg()
        self.m_.fit(X[keep], (y[keep] == 2).astype(int))
        return self

    def predict_proba(self, X):
        p = self.m_.predict_proba(X)[:, 1]
        return np.column_stack([1 - p, np.zeros(len(p)), p])


def make_model(name=None):
    name = name or getattr(C, "MODEL_NAME", "lgbm_dir")
    if name == "lgbm_dir":
        return DirectionModel("lgbm")
    if name == "logreg_dir":
        return DirectionModel("logreg")
    if name == "logreg":
        return _logreg()
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, **LGB)


def signals(proba, cutoff):
    """+1 Up, -1 Down, 0 no signal. Only a confident Up/Down becomes a signal."""
    pred, conf = proba.argmax(1), proba.max(1)
    return np.where((conf >= cutoff) & (pred == 2), 1, np.where((conf >= cutoff) & (pred == 0), -1, 0))
