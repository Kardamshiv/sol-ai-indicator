# User guide: SOL AI indicator

## What it does
Every 15 minutes, when a candle **closes**, a trained model (LightGBM) looks at the last few days of SOLUSDT price,
volume and buying-pressure data and predicts whether the price will be higher or lower **1 hour later**.

## How to read the chart
| You see | Meaning |
|---|---|
| Green arrow up under a candle, "Up 68%" | The model expects a rise over the next 4 candles, with 68% confidence |
| Red arrow down above a candle, "Down 64%" | The model expects a fall over the next 4 candles |
| No arrow / "No clear signal" | Confidence is below the cutoff (see `models/config.json`). **Most candles look like this on purpose.** |

- Arrows are made once, when the candle closes, and **never move or disappear** (stored in a log).
- The side panel shows the latest signal, a countdown to the next candle close, and a live scorecard
  (signals scored, hit rate, always-up baseline, net profit per trade after costs).
- Confidence is the model's own probability. Higher is not a guarantee: check the calibration plot in `reports/`.

## How to use it
1. Treat an arrow as one input among several, not as an order to trade.
2. Prefer higher-confidence signals. Ignore anything the scorecard shows is not beating "always up".
3. A signal is about the **next 1 hour**. Do not hold the idea much longer.
4. Costs matter: a move smaller than about 0.1% loses money after fees.

## When NOT to trust it
- **Major news or events** (exchange problems, regulation, large token unlocks, macro announcements): the model only sees prices.
- **Sudden volatility spikes or illiquid hours**: patterns learned from normal conditions break.
- **Right after a restart or data gap**: the first signals use a freshly rebuilt history.
- **A market regime unlike the training period** (for example a crash after a long calm uptrend).
- **When the scorecard hit rate falls to or below the always-up baseline** over many signals.
- Fewer than about 30 scored signals prove nothing. Judge it over weeks, not hours.

## Limits
Trained on SOLUSDT 15m data from Binance only. It predicts direction, not size. Paper trading only. Not financial advice.
