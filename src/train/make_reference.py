"""Reference numbers shown in the app: validation-protocol results at the LIVE cutoff.
These are NOT live results and NOT test-period results (no test data is used).

Run after fit_calibration:   python -m src.train.make_reference    -> models/reference.json
Source: out-of-fold predictions of rolling-origin folds 2-6 (training + validation rows only), calibrated with
models/calibration.json, signalling where calibrated confidence >= the saved cutoff.
"""
import json
import numpy as np, pandas as pd
from src.config import *
from src.train.fit_calibration import logit
from src.train.tune import cluster_boot


def main():
    cal = json.load(open(MODEL_DIR / "calibration.json"))
    oof = pd.read_csv(ROOT / "results" / "oof_predictions.csv", parse_dates=["time"])
    oof = oof[oof.fold >= 2]
    q = 1 / (1 + np.exp(-(cal["coef"] * logit(oof.p_default.values) + cal["intercept"])))
    sig = np.where(np.maximum(q, 1 - q) >= cal["cutoff_calibrated"], np.where(q > 0.5, 1, -1), 0)
    m, f, days = sig != 0, oof.fwd.values, oof.time.dt.strftime("%Y-%m-%d").values
    h, hl, hh = cluster_boot((np.sign(f[m]) == sig[m]).astype(float), days[m])
    r, rl, rh = cluster_boot(sig[m] * f[m] - COST, days[m])
    ref = dict(source="rolling-origin validation folds 2-6 (training + validation data only)", period=f"{oof.time.min().date()} to {oof.time.max().date()}",
               trades=int(m.sum()), coverage=float(m.mean()), hit_rate=h, hit_ci_low=hl, hit_ci_high=hh, always_up_hit=float((f > 0).mean()),
               net_per_trade=r, net_ci_low=rl, net_ci_high=rh)
    json.dump(ref, open(MODEL_DIR / "reference.json", "w"), indent=2)
    print(f"{ref['trades']} signals ({ref['coverage']:.1%} of candles) | hit {h:.3f} (95% {hl:.3f} to {hh:.3f}) vs always-up {ref['always_up_hit']:.3f} | "
          f"net per trade {r:+.4%} (95% {rl:+.4%} to {rh:+.4%})")
    print("Saved models/reference.json")


if __name__ == "__main__":
    main()
