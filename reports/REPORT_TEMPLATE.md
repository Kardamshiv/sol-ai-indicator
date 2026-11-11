# SOL AI Trading Indicator: Report

**Team:** SOL AI Indicator Research Team | **Asset:** SOLUSDT (Declared: 27 October 2026)  
**Detailed Report Reference:** See [`reports/PROJECT_REPORT.md`](file:///c:/Users/karda/Downloads/sol-ai-indicator/sol-ai-indicator/reports/PROJECT_REPORT.md) for full academic text, diagrams, and proofs.

---

## 1. Choices and Evidence (Source: `results/model_comparison.csv` and `results/experiment_log.csv`)
| Choice | Options tried | Chosen | Why (cite validation numbers) |
|---|---|---|---|
| **Asset** | BTCUSDT, ETHUSDT, SOLUSDT | **SOLUSDT** | High volatility/liquidity, 24/7 trading, free WebSocket/REST data via Binance Vision. |
| **Timeframe** | 1m, 5m, 15m, 1h | **15m** | Balanced signal frequency and fee drag; 96 candles/day provides adequate sample size. |
| **Horizon** | 1, 2, 4, 8 candles | **4 candles (1 hour)** | Intraday predictability peaks at 1 hour; shorter horizons lose edge to transaction fees. |
| **Dead zone (`THRESH`)** | 0.10%, 0.15%, 0.20%, 0.30% | **0.20% (0.002)** | Separates noise from clear directional moves; prevents fee churn on flat candles. |
| **History length** | 6, 12, 24 months | **24 months (2 years)** | 70,079 candles cover multiple bull/bear/sideways regimes without seasonal overfitting. |
| **Model** | Logistic Reg, Random Forest, LightGBM | **LightGBM (Direction)** | Best tradeoff: CV AUC $0.521$, $54.7\%$ hit rate, $529\text{ KB}$ size, and $4.4\text{ ms}$ inference. |

---

## 2. Data
- **Source & Endpoint:** Binance Vision Public REST (`https://data-api.binance.vision/api/v3/klines`) and WebSocket (`wss://data-stream.binance.vision`).
- **Date Range:** 2024-10-28 11:00 UTC to 2026-10-28 10:30 UTC (70,079 15m candles).
- **Gaps & Continuity:** Verified exactly 0 gaps, 0 duplicates, and 0 missing values across all OHLCV, trade count, and taker volume fields.
- **UTC Alignment:** All candles open/close strictly on 15m UTC multiples (00, 15, 30, 45).

---

## 3. Features
18 strictly causal features computed on closed candles:
- Returns: `ret_1`, `ret_4`, `ret_16`, `ret_96`
- Volatility: `vol_16`, `vol_96`, `atr_14`, `bb_width`, `bb_pos`
- Indicators: `rsi_14`, `macd_hist`, `dist_vwap`
- Order Flow: `vol_z`, `trades_z`, `taker_ratio`
- Cyclical Time: `hour_sin`, `hour_cos`, `dow`
- **Ablation Findings:** Removing cyclical time causes the largest AUC loss ($\Delta \text{AUC} = -0.0069$), followed by classic indicators ($-0.0060$) and returns ($-0.0060$).

---

## 4. Validation and Leakage Prevention
- **Chronological Split:** 70% Train (48,981 candles), 15% Validation (10,492 candles), 15% Held-out Test (10,493 candles).
- **Embargo:** $H=4$ candles dropped between every split and CV fold to prevent overlapping forward label leakage.
- **Walk-Forward Validation:** 6-fold expanding-window cross-validation with an embargo.
- **Probability Calibration:** Out-of-fold Platt scaling ($a = 0.253, b = 0.020$) fitted on training+validation only.
- **Single Pipeline:** Identical feature code in [`src/features/build.py`](file:///c:/Users/karda/Downloads/sol-ai-indicator/sol-ai-indicator/src/features/build.py) shared across training, backtesting, and live server.

---

## 5. Results (Held-Out Test Set)
Evaluated on 10,493 untouched test candles (2026-07-11 to 2026-10-28):
| Strategy / Rule | Direction Hit Rate | Signals Fired | Net Return / Trade (After Fees) |
|---|---|---|---|
| **Always Up** | $49.9\%$ | 10,493 | $-0.106\%$ |
| **RSI Rule (30/70)** | $51.9\%$ | 898 | $-0.111\%$ |
| **Model (Confident Signals)** | $\mathbf{58.3\%}$ | 168 | $\mathbf{+0.043\%}$ gross / $\mathbf{-0.057\%}$ net |

---

## 6. Backtest After Costs (`reports/backtest.png`)
- **Trades Executed:** 168 non-overlapping trades over test period.
- **Direction Win Rate:** $58.3\%$ on confident signals vs $49.9\%$ Always-Up baseline.
- **Cost Assumption:** Round-trip fee + slippage of $0.10\%$ ($0.05\%$ per side).
- **Strategy Realities:** Transaction fees consume gross alpha; trading net profitability requires fee discounts (VIP/Maker tiers $\le 0.04\%$).

---

## 7. Live Results & System Operation
- **Repainting Events:** 0 (guaranteed by SQLite `INSERT OR IGNORE` on closed candles).
- **Server Architecture:** FastAPI backend with automated 4-candle forward scoring, WebSocket streaming, and REST backfill.
- **Interactive UI:** Lightweight Charts candlestick rendering, confidence histogram strip, what-if cutoff simulator, feature explainability drivers, and one-click CSV export.

---

## 8. What Did Not Work (Lessons Learned)
1. **Uncalibrated Raw Tree Probabilities:** Raw LightGBM leaves yielded clustered, overconfident extremes. Platt calibration was essential to map log-odds to genuine frequencies.
2. **Predicting Every Candle:** Forcing predictions on all candles degraded hit rates to $\approx 50.3\%$. An explicit dead zone ($\pm 0.20\%$) and confidence cutoff raised hit rate to $54.7\%-58.3\%$.
3. **Short Time Horizons (< 15m):** 1-minute and 5-minute models exhibited lower signal-to-noise ratios and were completely unviable after transaction costs.
4. **Complex Neural Networks (LSTM):** Tested deep architectures required excessive training time and yielded no statistical benefit over gradient boosted decision trees.

---

## 9. Limitations and Future Work
- **Limitations:** Single exchange (Binance), single pair (SOLUSDT), sensitivity to sudden macro flash liquidation events.
- **Future Work:** Multi-asset cross-attention (BTC + ETH + SOL), order book depth features, and dynamic funding rate integration.

---

## 10. Citations
1. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*.
2. Jaquart, P. et al. (2021). Short-term bitcoin market prediction via machine learning. *J. Finance Data Sci.*
3. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
4. TradingView Lightweight Charts & Binance Public API Documentation.
