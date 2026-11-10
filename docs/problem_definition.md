# Problem Definition (Project Step 1)

**Working title:** Hourly Direction Prediction for SOLUSDT Using Gradient Boosting With a Live Non-Repainting Confidence Indicator

## 1. Application and target users

Retail and student traders who watch SOLUSDT (Solana / Tether) on a live 15-minute chart and want a quick, honest hint about the likely direction of the next hour, with a confidence score they can interpret.

## 2. Task type

Binary classification (Up vs Down) of the future close, with **abstention**: when the model's confidence is below a cutoff the indicator shows "No clear signal". Because the data are a time series, all evaluation is chronological (forecasting protocol).

## 3. Problem statement

> Given the 15-minute candle history of SOLUSDT (OHLC, volume, trade count, taker-buy volume) up to the close of a candle, predict whether the close four candles (1 hour) later will be above or below the current close, and report a confidence, for a trader using a live chart, under the constraints below, so that confident signals are more accurate than the Always-Up and RSI baselines with statistical support.

## 4. Inputs, outputs, constraints, assumptions

| Item | Definition |
|---|---|
| Inputs | 18 features computed only from candles that have already closed (returns, volatility, ATR, RSI, MACD, Bollinger, VWAP distance, volume and flow measures, time of day) |
| Output | Up, Down or No signal, plus a confidence between 50% and 100% |
| Source | Binance public market-data API only (no API key) |
| Constraint: no repainting | A signal is created only when a candle closes, stored once, and never edited |
| Constraint: one code path | The same feature code is used in training and in the live server |
| Constraint: paper trading | No real money; not financial advice |
| Assumption: costs | Round-trip cost of 0.1% (fee plus slippage), a simplification; real spot fees can be higher |
| Assumption: noise | Moves smaller than +/-0.2% in an hour are treated as noise when training the direction model |

## 5. Hypotheses

- **H0:** Hourly SOLUSDT direction cannot be predicted from these features better than the baselines (Always-Up, RSI rule).
- **H1:** A gradient-boosting classifier predicts hourly direction with accuracy above the baselines.
- **H2:** Accuracy increases with model confidence (the confidence score is informative and reasonably calibrated).

## 6. Success criteria (declared before further experiments)

| # | Criterion | Target | Evaluated on |
|---|---|---|---|
| 1 | Direction accuracy of confident signals vs Always-Up | Lower bound of the 95% bootstrap confidence interval is above the baseline's accuracy | One-shot final test on fresh candles (live window) |
| 2 | Same metric vs the RSI rule (30/70) | Point estimate higher | Same |
| 3 | Signal coverage | At least 3% of candles (roughly 20 or more signals per week) | Validation and live |
| 4 | Calibration | Per-bin gap between confidence and accuracy at most 5 points (bins with at least 30 signals) | Validation and live |
| 5 | Net return per trade after costs | Greater than 0 (**stretch goal**, not required; failure will be reported honestly) | Backtest and live |
| 6 | Repainting | Zero events | Live |
| 7 | Live sample size | At least 15 signals in 7 days; otherwise the live result is reported as inconclusive | Live |

Also reported, not used as targets: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, training time and model size.

## 7. Honest status of the evidence

During development we inspected validation and test-period results for several settings (horizon, dead zone, model type). **The current test period is therefore a development set, not an untouched one.** Criteria 1 and 2 will be judged one-shot on candles that arrive **after the model-freeze date** (the live window), not on the earlier test period.

## 8. Deliverable of this step

This one-page definition, plus the literature table in `docs/literature_review.md`.
