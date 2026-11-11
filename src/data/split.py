"""Single source of truth for the chronological split (same numbers as src/train/train.py).

    train      first 70% of usable rows
    validation next 15%, after an embargo of HORIZON candles
    test       last 15%, after an embargo of HORIZON candles  -> reserved for the final one-shot evaluation

save_boundaries() writes results/split_boundaries.json (row numbers and timestamps only, no prices),
which is the 'saved split indices' the project guidelines ask for.
"""
import json
import pandas as pd
from src.config import *
from src.features.build import make_features, make_labels

TRAIN_FRAC, TRAINVAL_FRAC = 0.70, 0.85


def build_dataset(df, horizon=HORIZON, thresh=THRESH):
    """Return (features with warm-up NaNs, usable rows with features + forward return + class)."""
    X = make_features(df)
    fwd, y = make_labels(df, horizon, thresh)
    d = pd.concat([X, fwd.rename("fwd"), y.rename("y"), df[["time", "close"]]], axis=1).dropna()
    return X, d


def split_bounds(n, horizon=HORIZON):
    i1, i2 = int(n * TRAIN_FRAC), int(n * TRAINVAL_FRAC)
    return dict(train=(0, i1), val=(i1 + horizon, i2), test=(i2 + horizon, n))


def save_boundaries(d, horizon=HORIZON, thresh=THRESH):
    out = {}
    for name, (a, e) in split_bounds(len(d), horizon).items():
        out[name] = dict(start_row=a, end_row_exclusive=e, rows=e - a,
                         start_time=str(d["time"].iloc[a]), end_time=str(d["time"].iloc[e - 1]))
    out.update(symbol=SYMBOL, interval=INTERVAL, horizon=horizon, thresh=thresh, embargo_candles=horizon, seed=SEED)
    p = ROOT / "results" / "split_boundaries.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    return out
