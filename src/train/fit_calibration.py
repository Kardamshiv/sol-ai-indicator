"""Fit the probability calibrator used by the live indicator (Platt scaling).

Run after tune.py:   python -m src.train.fit_calibration      -> models/calibration.json

Uses ONLY the out-of-fold predictions of the default LightGBM on training + validation rows
(results/oof_predictions.csv). No test data. Fold 1 is skipped: its model saw only about 8,500 rows, far fewer than the
final model (about 59,000), so its probabilities are noisier and more extreme than the deployed model's will be. The confidence cutoff is chosen so that the top MIN_COVERAGE share
of candles signal; this rule uses no labels.
"""
import argparse, datetime, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from src.config import *


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coverage', type=float, default=MIN_COVERAGE, help='target share of candles that signal')
    ap.add_argument('--no-save', action='store_true', help='print the report only; do not write models/calibration.json')
    args = ap.parse_args()
    target = args.coverage
    oof = pd.read_csv(ROOT / "results" / "oof_predictions.csv", parse_dates=["time"])
    oof = oof[oof.fold >= 2].copy()                    # skip fold 1 (model trained on too little data)
    clear = oof[oof.y != 1]
    m = LogisticRegression(C=1e6).fit(logit(clear.p_default.values).reshape(-1, 1), (clear.y == 2).astype(int).values)
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    if a <= 0.05:        # a non-positive slope would flip or flatten the model's direction: refuse to save
        raise SystemExit(f"Calibration slope {a:.3f} is not clearly positive: the model has no usable ranking skill on "
                         "out-of-fold data. Not saving calibration.json; do not deploy this model without review.")
    q = 1 / (1 + np.exp(-(a * logit(oof.p_default.values) + b)))
    conf = np.maximum(q, 1 - q)
    # Models trained on more data give less extreme probabilities, so the cutoff is set on the LAST TWO folds
    # (training sets closest in size to the final model) to keep the live signal rate near the target.
    late = (oof.fold >= oof.fold.max() - 1).values
    cut = float(np.quantile(conf[late], 1 - target))
    cpd = 86400 / {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[INTERVAL]
    cov = float((conf[late] >= cut).mean())
    cal = dict(method="platt", coef=a, intercept=b, cutoff_calibrated=cut, coverage_target=target, rows_fitted=int(len(clear)),
               fitted_on="out-of-fold predictions, training + validation rows only", created=datetime.date.today().isoformat())
    if not args.no_save:
        MODEL_DIR.mkdir(exist_ok=True)
        json.dump(cal, open(MODEL_DIR / "calibration.json", "w"), indent=2)
    print(f"Platt: P(Up) = sigmoid({a:.3f} * logit(raw) + {b:.3f}); fitted on {len(clear):,} clear-move predictions")
    print("Raw P(Up) -> calibrated P(Up):", {r: round(float(1 / (1 + np.exp(-(a * logit(np.array(r)) + b)))), 3) for r in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80)})
    print(f"Calibrated cutoff {cut:.3f} -> coverage {cov:.2%} on the last two folds, about {cov * cpd * 7:.0f} live signals per 7 days")
    clear = clear.assign(q=1 / (1 + np.exp(-(a * logit(clear.p_default.values) + b))))
    clear["c"] = np.maximum(clear.q, 1 - clear.q)
    oof["c"] = np.maximum(1 / (1 + np.exp(-(a * logit(oof.p_default.values) + b))), 1 - 1 / (1 + np.exp(-(a * logit(oof.p_default.values) + b))))
    print("\nStability by fold at this cutoff (signals per week varies with the market regime):")
    allhit, alln = [], 0
    for f in sorted(oof.fold.unique()):
        cv, cc = oof[oof.fold == f], clear[(clear.fold == f) & (clear.c >= cut)]
        hit = ((cc.q > 0.5).astype(int) == (cc.y == 2).astype(int))
        allhit += list(hit); alln += len(cc)
        print(f"  fold {f}: {cv.time.iloc[0].date()} to {cv.time.iloc[-1].date()}  coverage {(cv.c >= cut).mean():.1%} "
              f"(about {(cv.c >= cut).mean() * cpd * 7:.0f}/week)  clear-move signals {len(cc):4d}  accuracy {hit.mean():.3f}")
    h = float(np.mean(allhit)); se = (h * (1 - h) / max(alln, 1)) ** 0.5
    top = clear[clear.c >= cut]
    print(f"\nIn-sample check (folds 2-6, clear moves above the cutoff): mean displayed confidence {top.c.mean():.3f} | "
          f"direction accuracy {h:.3f} +/- {1.96 * se:.3f} (naive 95%) | n={alln}")
    print("Report only, nothing saved." if args.no_save else "Saved models/calibration.json (commit it with the model)")


if __name__ == "__main__":
    main()
