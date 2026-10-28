# SOL AI Trading Indicator: Report

Team: ______  Asset: SOLUSDT (declared on ____)

## 1. Choices and evidence  (source: reports/experiments.csv)
| Choice | Options tried | Chosen | Why (cite validation numbers) |
|---|---|---|---|
| Asset | | SOLUSDT | unique, 24/7, free live data, same source for train and live |
| Timeframe | | 15m | |
| Horizon | 1, 2, 4, 8 candles | | |
| Dead zone (THRESH) | 0.15%, 0.2%, 0.3%, 0.4% | | |
| History length | 3, 6, 12, 24 months | | |
| Model | logistic regression, LightGBM (, LSTM) | | |

## 2. Data
Source, endpoint, date range, number of candles, gaps found and how handled, UTC alignment.

## 3. Features
List of 18 features and why. Ablation results (no_time / no_flow / no_classic) and what they show.

## 4. Validation and leakage prevention
Chronological 70/15/15 split, embargo of HORIZON candles, walk-forward folds (paste per-fold accuracy),
cutoff chosen on validation only, test set used once. Closed-candle-only features. Same feature code in training and live.

## 5. Results (test set)
Paste the TEST SET table from `src.train.train`: model vs Always Up vs RSI rule. Calibration plot (reports/calibration.png).

## 6. Backtest after costs  (reports/backtest.png)
Trades, win rate, net per trade, total return, max drawdown, buy & hold. State the cost assumption (COST) and its limits
(no order-book impact, perfect fills).

## 7. Live results
Signals scored, hit rate vs always-up, net per trade (from the app scorecard / data/signals.db), uptime, incidents.

## 8. What did not work
(Marks are awarded for this.) Failed features, models, thresholds, bugs, and what you learned.

## 9. Limitations and future work
Single asset, single exchange, regime dependence, no news, execution costs, more data sources.

## 10. Citations
Binance API docs, LightGBM, scikit-learn, pandas, FastAPI, TradingView Lightweight Charts, any paper or code you reused.
