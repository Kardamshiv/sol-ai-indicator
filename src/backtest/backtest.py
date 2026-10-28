"""Realistic simulated trading on the untouched TEST period.
Rules: one trade at a time, enter at the signal candle's close, exit HORIZON candles later,
pay COST (fees + slippage) on every trade, compound the equity.

Run from repo root:  python -m src.backtest.backtest   -> reports/backtest.png
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import *
from src.features.build import make_features, make_labels, FEATURES
from src.models import make_model, signals


def main():
    df = pd.read_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", parse_dates=["time"])
    fwd, y = make_labels(df)
    d = pd.concat([make_features(df), fwd.rename("fwd"), y.rename("y"), df["close"]], axis=1).dropna()
    n = len(d); i1, i2 = int(n * .70), int(n * .85)
    tr, va, te = d.iloc[:i1], d.iloc[i1 + HORIZON:i2], d.iloc[i2 + HORIZON:]
    model = make_model().fit(tr[FEATURES], tr["y"].astype(int))

    pv, best, best_val = model.predict_proba(va[FEATURES]), 0.40, None      # cutoff from validation only
    for cut in np.arange(0.40, 0.80, 0.01):
        s = signals(pv, cut); mk = s != 0
        if mk.mean() < MIN_COVERAGE: continue
        sc = (s[mk] * va["fwd"].values[mk] - COST).mean()
        if best_val is None or sc > best_val: best, best_val = cut, sc

    sig, f = signals(model.predict_proba(te[FEATURES]), best), te["fwd"].values
    eq, rets, i = [1.0], [], 0
    while i < len(te):
        if sig[i] != 0:
            r = sig[i] * f[i] - COST
            rets.append(r); eq.append(eq[-1] * (1 + r)); i += HORIZON
        else:
            i += 1
    eq = np.array(eq)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    bh = te["close"].iloc[-1] / te["close"].iloc[0] - 1
    print(f"cutoff {best:.2f} | test candles {len(te)} | trades {len(rets)}")
    if rets:
        print(f"win rate {np.mean(np.array(rets) + COST > COST):.3f} (hit before costs: "
              f"{np.mean(np.array(rets) + COST > 0):.3f})")
        print(f"avg net/trade {np.mean(rets):+.4%} | total return {eq[-1] - 1:+.2%} | max drawdown {dd:.2%}")
    print(f"buy & hold over same period: {bh:+.2%}")
    plt.figure(figsize=(7, 4)); plt.plot(eq, label="Model strategy (after costs)")
    plt.axhline(1, color="gray", ls="--"); plt.xlabel("Trade number"); plt.ylabel("Equity (start = 1)")
    plt.title(f"{SYMBOL} backtest on test period (buy & hold {bh:+.1%})"); plt.legend(); plt.tight_layout()
    (ROOT / "reports").mkdir(exist_ok=True)
    plt.savefig(ROOT / "reports" / "backtest.png", dpi=150)
    print("saved reports/backtest.png")


if __name__ == "__main__":
    main()
