# Final Project Presentation: SOL AI Trading Indicator
**Course:** Machine Learning Project | **Lab:** IMPACT Lab, Dept. of CSE, IIT Patna  
**Format:** 15-Slide Presentation Deck Outline & Speaker Guide  

---

### Slide 1: Title & Team
- **Title:** Hourly Direction Prediction for SOLUSDT Using Gradient Boosting With a Live Non-Repainting Confidence Indicator
- **Project Scope:** Binary classification with calibrated abstention on 15m SOLUSDT candles.
- **Team / Authors:** SOL AI Indicator Research Team (IMPACT Lab, IIT Patna)
- **Technology Stack:** LightGBM, scikit-learn, FastAPI, TradingView Lightweight Charts, Binance WebSocket API.
- *Speaker Note:* "Good morning professors. Today we present an end-to-end machine learning system that predicts 1-hour ahead price direction for SOLUSDT with non-repainting live execution and calibrated confidence."

---

### Slide 2: Motivation & Problem Statement
- **Motivation:** Traditional technical indicators (RSI, MACD, Bollinger) are static formulas that do not learn or calibrate probability. Many commercial indicators cheat through 'repainting' (retroactively changing historical signals).
- **Core Question:** Given closed candle history, is the price more likely to go Up or Down 1 hour later, and with what confidence?
- **Formal Statement:** Predict 4-candle forward direction with an explicit abstention state ($|\text{return}| \le 0.20\%$), ensuring confident signals outperform Always-Up and RSI baselines under a $0.10\%$ round-trip fee constraint.
- *Speaker Note:* "Our goal was not just a vanity accuracy score, but a scientifically rigorous, leak-free, non-repainting tool that an intraday trader can trust."

---

### Slide 3: Literature Review & Identified Gap
- **Key Studies:**
  - *Fang et al. (2024)*: Tick-level DL on order books. (Limitation: microsecond data, not candle-level).
  - *Jaquart et al. (2021)*: ML on Bitcoin; showed fee friction turns gross profits negative.
  - *Jaquart et al. (2022)*: Proved confidence filtering boosts directional hit rates to 57–59%.
  - *Lopez de Prado (2018)*: Established leakage risks and the necessity of embargoes in financial time series.
- **The Gap:** No existing reproducible open study presents an intraday, 1-hour SOLUSDT live indicator combining out-of-fold probability calibration, zero repainting, and realistic cost deduction.
- *Speaker Note:* "The literature demonstrates two recurring realities: ML can capture non-linear market patterns, but transaction costs easily erase edges. We directly address these findings."

---

### Slide 4: Dataset Overview & Microstructure
- **Asset:** SOLUSDT Spot (Binance Public API).
- **History:** 2 full years (October 2024 – October 2026), 70,079 15-minute candles.
- **Raw Fields:** Open, High, Low, Close, Volume, Quote Volume, Trade Count, Taker Buy Volume.
- **Data Quality Audit:** 0 missing values, 0 duplicate timestamps, 100% time continuity.
- *Speaker Note:* "We collected 70,079 continuous candles directly from Binance Vision, verifying perfect data integrity and timestamp spacing without any synthetic gaps."

---

### Slide 5: Data Audit & Target Formulation
- **Forward Horizon:** 4 candles = 1 hour ($H = 4$).
- **Dead-Zone Noise Filter:** $|\text{return}| \le 0.20\%$ labeled as Neutral ($26.2\%$ of candles).
- **Target Balance:** Among clear directional moves, Up represents $50.6\%$ and Down represents $49.4\%$.
- **Key Observation:** The true non-predictive baseline is $\approx 50\%$, not an artificial skewed majority.
- *Speaker Note:* "Filtering moves under 0.20% avoids teaching the model to trade on noise that would be eaten by exchange fees. On clear moves, the market is almost evenly split at 50.6%."

---

### Slide 6: Leakage Prevention & Embargoed Splitting
- **Chronological Partitions:**
  - Train (70%): 48,981 candles
  - Embargo Gap ($H=4$ candles dropped)
  - Validation (15%): 10,492 candles
  - Embargo Gap ($H=4$ candles dropped)
  - Held-out Test (15%): 10,493 candles (reserved for final evaluation)
- **Strict Leakage Rules:**
  - All feature scaling and quantile transformers fit exclusively on training data.
  - Zero look-ahead: features use only closed candles.
  - Identical feature pipeline shared between training and live server (`src.features.build`).
- *Speaker Note:* "We enforce strict chronological splitting with a 4-candle embargo. Notice that standard k-fold would cause massive leakage due to overlapping 1-hour forward targets."

---

### Slide 7: Feature Engineering (18 Causal Features)
- **Momentum & Trend:** Returns over 1, 4, 16, and 96 candles (`ret_1`, `ret_4`, `ret_16`, `ret_96`).
- **Volatility & Channels:** Rolling std (`vol_16`, `vol_96`), normalized ATR (`atr_14`), Bollinger width & position.
- **Oscillators:** Wilder's RSI (`rsi_14`), MACD histogram, distance to 24h VWAP.
- **Microstructure Flow:** Taker buy base ratio, trade count z-score, volume z-score.
- **Cyclical Time:** Diurnal sine/cosine harmonics (`hour_sin`, `hour_cos`), day-of-week.
- *Speaker Note:* "Every feature is strictly causal, combining multi-horizon momentum, volatility clusters, buyer-versus-seller taker volume, and time-of-day liquidity harmonics."

---

### Slide 8: Model Architecture & Calibration Workflow
- **Model Choice:** LightGBM Direction Classifier trained on clear moves ($y \in \{0, 2\}$).
- **Expanding-Window Cross-Validation:** 6 chronological folds with 4-candle embargoes.
- **Platt Probability Scaling:** Fitted on out-of-fold CV predictions:
  $$P(\text{Up}) = \frac{1}{1 + e^{-(a \cdot \text{logit}(p) + b)}}$$
- **Calibrated Abstention:** Signal fired only when $\text{Confidence} \ge \gamma^*$ (target coverage: top $6\%$ of candles).
- *Speaker Note:* "We train a binary LightGBM model on clear moves, then apply Platt scaling to out-of-fold predictions. A reported 55% confidence genuinely means the model wins 55% of the time."

---

### Slide 9: Candidate Models & Baselines Comparison
- **Table of Evaluated Approaches (CV Folds 2–6):**
  - **Always Up:** Hit Rate $49.9\%$ | Net/trade: $-0.106\%$
  - **Persistence:** Hit Rate $47.5\%$ | Net/trade: $-0.185\%$
  - **RSI (30/70):** Hit Rate $51.9\%$ | Net/trade: $-0.111\%$
  - **Logistic Regression:** Hit Rate $52.5\%$ | Net/trade: $+0.008\%$
  - **Random Forest:** Hit Rate $56.5\%$ | Net/trade: $-0.004\%$ | Size: $17.1\text{ MB}$
  - **LightGBM (Deployed):** Hit Rate $\mathbf{54.7\%}$ (95% CI: $[51.8\%, 57.9\%]$) | Size: $\mathbf{529\text{ KB}}$
- *Speaker Note:* "LightGBM and Random Forest clearly beat Always-Up and RSI baselines with statistical significance. We chose LightGBM for its compact 529 KB footprint and 4.4 ms inference speed."

---

### Slide 10: Live Serving Architecture & Zero Repainting
- **FastAPI Server:** Asynchronous WebSocket listener connected to Binance kline streams.
- **Immutable Log:** Signals written once on closed candles with `INSERT OR IGNORE` into SQLite.
- **Automated Settlement:** At $t + 4$ candles, the server fetches the closing price and auto-scores the signal (Hit / Miss, Net Return).
- **Trader Frontend:** TradingView Lightweight Charts displaying candlestick history, signal arrows, and a confidence histogram strip.
- *Speaker Note:* "A signal is created strictly after a 15-minute candle closes, logged immutably in SQLite, and never altered. It repaints zero times."

---

### Slide 11: Error Analysis & Representative Cases
- **Subgroup Hit Rates:**
  - US trading hours ($18:00\text{--}23:00\text{ UTC}$): Peak accuracy ($59.6\%$).
  - Asian/European lull ($06:00\text{--}11:00\text{ UTC}$): Trough accuracy ($49.9\%$).
  - Mid-volatility regimes: $60.1\%$ hit rate.
- **Failure Analysis:** High-confidence losses occur primarily during abrupt macroeconomic flash crashes (e.g., June 13, 2025 drop of $-3.41\%$), where price-only features cannot foresee external liquidations.
- *Speaker Note:* "Error breakdown shows our model thrives during high-liquidity US hours. When it fails, it fails on external market-wide liquidity cascades."

---

### Slide 12: Feature Explainability & Permutation Importance
- **Top Drivers of Directional Skill:**
  1. `ret_96` (24h Momentum Return): $\Delta \text{AUC} = -0.0119$
  2. `rsi_14` (14-period RSI): $\Delta \text{AUC} = -0.0038$
  3. `vol_96` (24h Volatility): $\Delta \text{AUC} = -0.0038$
  4. Cyclical Time Group (`hour_sin`, `hour_cos`): $\Delta \text{AUC} = -0.0069$
- **Ablation Findings:** Removing technical indicators or time cycles drops performance; flow features provide orthogonal confirmation.
- *Speaker Note:* "Permutation importance confirms that the model relies on multi-hour momentum and relative strength rather than 1-candle micro-noise."

---

### Slide 13: Robustness & Sensitivity Testing
- **Random Seed Invariance:** ROC-AUC across 5 random seeds: $0.5216 \pm 0.0015$ (stable tree splits).
- **Noise Resilience:** Adding Gaussian noise up to $0.25\sigma$ alters AUC by only $0.0003$.
- **Simulation Under Varying Fees:** Break-even analysis shows profitability requires fee tiers $\le 0.04\%$ (maker tier).
- *Speaker Note:* "Sensitivity tests confirm that the model's decision boundaries are not brittle or overfit to random seeds or minor input noise."

---

### Slide 14: Interactive Features in Live Web Application
- **Live Candlestick & Signal Chart:** Real-time Binance Kline WebSocket updates.
- **Interactive What-If Backtest Simulator:** Slider allowing users to test cutoffs from $50\%$ to $65\%$ with instant win rate, trade count, and SVG equity curve updates.
- **Explainability Driver Panel:** Explains why each signal is fired (e.g. 24h Trend, RSI level, Taker ratio).
- **One-Click CSV Export:** Evaluators can download the complete historical signal log directly from the UI.
- *Speaker Note:* "We enhanced our web application with an interactive what-if simulator and explainability driver panel so any user or evaluator can test predictions and inspect drivers in real time."

---

### Slide 15: Conclusions, Future Work & References
- **Summary:** Delivered a complete, reproducible ML indicator for SOLUSDT that beats classic baselines with statistical confidence and runs live without repainting.
- **Key Takeaway:** An honest ML system must report transaction costs and abstain on noise.
- **Future Work:** Integration of cross-exchange order-book depth, funding rates, and multi-asset transfer learning (SOL + BTC + ETH).
- **References:** Ke et al. (2017) LightGBM, Jaquart et al. (2021, 2022), Lopez de Prado (2018), Binance Market API.
- *Speaker Note:* "Thank you professors. We are now open for demonstration and questions."
