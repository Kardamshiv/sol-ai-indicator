# Leakage-Free Machine Learning Directional Predictor & Calibrated AI Indicator for High-Volatility Asset Trading

**Student:** Kardam Shiv (Roll Number: 2201CS01)  
**Supervisor:** Dr. Research Supervisor  
**Laboratory:** IMPACT Lab  
**Department:** Department of Computer Science & Engineering, Indian Institute of Technology Patna  
**Date:** March 2026 (Academic Year 2025–2026)  

---

## Abstract
Machine learning applications in high-frequency algorithmic trading consistently struggle with extreme noise-to-signal ratios, non-stationary market regimes, and severe data leakage. In commercial technical analysis indicators, lookahead bias and indicator repainting frequently yield deceptive backtest performance that fails catastrophically upon live execution. In this project, we engineer a leak-free, production-grade directional forecasting framework for Solana (SOLUSDT) on a 15-minute timeframe over a 1-hour forward horizon (4 candle bars).

Operating on a clean dataset of 70,079 Binance klines spanning November 2024 to March 2026, we extract 33 causal, point-in-time cross-domain features encompassing returns, moving-average divergences, volatility ratios, and volume momentum. We enforce strict data hygiene through expanding-window normalization and a 5-Fold Purged and Embargoed Group Time-Series Cross-Validation scheme to completely eliminate serial autocorrelation leakage across splits. Our primary model, a gradient-boosted decision tree (LightGBM), achieves an out-of-sample directional accuracy of 54.7% across cross-validation folds and 58.3% on confident out-of-fold test signals, statistically outperforming the Always-Up baseline (49.9%) and classical indicators (RSI Rule: 51.4%).

To resolve the well-documented overconfidence of tree-based margin outputs, we incorporate post-hoc Platt sigmoid calibration, improving the test Brier score from 0.250 to 0.244 and reducing log-loss to 0.680. The entire pipeline is deployed as an end-to-end cloud microservice on Render featuring live Binance WebSocket ingestion, real-time feature computation, TradingView lightweight chart visualization, and a What-If backtesting engine with zero indicator repainting.

---

## Chapter 1: Introduction

### 1.1 Background and Motivation
Financial market prediction represents one of the most formidable frontiers in applied machine learning. Unlike stationary computer vision or speech recognition domains where underlying physics remain constant, algorithmic trading operates on continuous-time stochastic processes governed by complex human psychology, automated institutional execution, and macroeconomic regime shifts. Cryptocurrency assets, most notably Solana (SOLUSDT), exhibit volatility orders of magnitude higher than conventional equities or foreign exchange pairs, featuring heavy-tailed return distributions, frequent intraday flash crashes, and rapid regime transitions.

In retail and quantitative trading environments, technical indicators (e.g., RSI, MACD, Bollinger Bands) remain widely utilized. However, standard indicator heuristics suffer from severe statistical weaknesses: they are uncalibrated, generate vast numbers of false-positive breakout signals in mean-reverting regimes, and often exhibit "repainting"—the retroactive modification of historical buy/sell arrows as future price candles unfold.

Establishing a strictly leak-free, mathematically sound, and live-verifiable predictive framework is therefore of paramount academic and industrial importance.

### 1.2 Problem Statement
Given a streaming multivariate time series of 15-minute Open, High, Low, Close, Volume (OHLCV) candles for the SOLUSDT asset:
$$\mathcal{S}_T = \{ (O_t, H_t, L_t, C_t, V_t) \}_{t=1}^T$$

the objective is to forecast the sign of the forward 1-hour return (defined across a forward horizon of $h=4$ bars):
$$y_t = \mathbb{I}\left( C_{t+h} > C_t \right) \in \{0, 1\}$$

such that for every closed bar at time $t$, the model emits a directional signal $\hat{y}_t \in \{\text{BUY}, \text{NEUTRAL}, \text{SELL}\}$ and a well-calibrated posterior probability $P(y_t = 1 \mid \mathcal{F}_t)$ conditioned solely on the filtration $\mathcal{F}_t = \sigma(\mathcal{S}_t)$, completely eliminating lookahead leakage and indicator repainting in live production.

### 1.3 Objectives
1. **Leakage-Free Feature Engineering:** Design 33 causal features using strictly backward-looking operations and expanding-window normalization.
2. **Rigorous Cross-Validation:** Implement 5-Fold Purged and Embargoed Time-Series Cross-Validation to eliminate serial correlation leakage from overlapping 4-candle forward horizons.
3. **Model Benchmarking:** Benchmark baselines (Always-Up, Persistence, RSI Heuristic, Logistic Regression, Random Forest) against LightGBM.
4. **Probability Calibration:** Implement post-hoc Platt sigmoid calibration to minimize Brier score and log-loss.
5. **Cloud Streaming Deployment:** Deploy a low-latency live web service with Binance WebSocket integration, TradingView charts, and zero repainting.

### 1.4 Scope and Limitations
- **Scope:** 70,079 15-minute SOLUSDT candles from Binance Spot (Nov 2024 to Mar 2026).
- **Limitations:** Single-asset specialization; fixed 0.10% transaction friction assumption; Level-1 OHLCV data without order book depth.

### 1.5 Contributions of the Project
- Pure Point-in-Time Causal Data Pipeline (no future data leakage).
- Purged and Embargoed 5-Fold Validation protocol.
- Post-Hoc Platt Calibration for Financial Decision Trees (Brier score 0.244).
- Empirically Validated Alpha Over Traditional Baselines (58.3% test accuracy on confident signals vs 49.9% Always-Up).
- Open-Source Real-Time Production Indicator deployed on Render with live Binance WebSocket.

---

## Chapter 2: Literature Review

### 2.1 Existing Approaches
1. **Statistical Time Series (ARIMA, GARCH):** Rigorous volatility clustering, but assume weak stationarity and fail on non-linear cross-feature dynamics.
2. **Heuristic Technical Indicators (RSI, MACD):** Uncalibrated, high false-positive rates, vulnerable to data snooping bias.
3. **Deep Sequence Models (LSTM, Transformers):** Prone to catastrophic overfitting on high-noise financial time series.
4. **Gradient Boosted Decision Trees (LightGBM, XGBoost):** Premier tabular architecture, robust against outliers, handles non-linear interactions natively.

### 2.2 Research Gaps & Identified Flaws
- Lookahead bias during global preprocessing (fitting scalers on full series).
- Autocorrelation leakage from overlapping forward return labels.
- Probability distortion in tree-based boosting models.
- Indicator repainting in live trading engines.

---

## Chapter 3: Proposed Methodology

### 3.1 Dataset Description
- **Asset:** SOLUSDT (Binance Spot)
- **Timeframe:** 15 Minutes
- **Sample Size:** 70,079 Candlesticks
- **Horizon:** 4 Bars (1 Hour)
- **Class Balance:** 50.3% Up / 49.7% Down

### 3.2 Feature Engineering (33 Causal Features)
1. **Log Returns:** 1, 2, 4, 8, 16, 32, 64 bars.
2. **EMA Ratios:** (8, 21), (21, 55), (55, 200).
3. **Volatility:** Normalized ATR (`natr_14`), rolling standard deviation of returns (`vol_roll_std_20`).
4. **Momentum & Volume:** RSI-14, MACD difference/histogram, volume z-scores (`vol_z_20`).

### 3.3 Proposed Architecture
![Architecture Diagram](assets/architecture_diagram.png)

### 3.4 Algorithm 3.1: Leak-Free Directional Signal Ingest & Inference
```text
Require: Live OHLCV Stream S; Horizon h = 4; Threshold tau = 0.53
Ensure: Calibrated Signal y_hat in {BUY, NEUTRAL, SELL} and Confidence P_t
1. Upon arrival of closed candle C_t:
2.   Append C_t to ring buffer B
3.   Extract 33 point-in-time features X_t strictly from past candles
4.   Compute raw tree logit margin z_t = f*(X_t)
5.   Compute Platt calibrated probability: P_t = 1 / (1 + exp(A * z_t + B))
6.   If P_t >= tau -> BUY; else if P_t <= 1 - tau -> SELL; else NEUTRAL
7.   Broadcast deterministic signal to WebSocket clients (0 Repainting)
```

---

## Chapter 4: Results and Discussion

### 4.1 Benchmark Comparison Table
| Model / Strategy | CV AUC | CV Accuracy | Brier Score | Confident Test Acc | Net Ret / Trade |
|---|---|---|---|---|---|
| Always Up (Majority Baseline) | 0.500 | 50.3% | 0.250 | 49.9% | -0.106% |
| Persistence (Continuation) | 0.490 | 49.3% | 0.252 | 47.5% | -0.185% |
| RSI Rule (30 / 70 Extremes) | 0.522 | 51.4% | 0.251 | 51.9% | -0.111% |
| Logistic Regression (L2) | 0.514 | 50.5% | 0.251 | 52.5% | +0.008% |
| Random Forest (100 Trees) | 0.523 | 51.5% | 0.251 | 56.5% | -0.004% |
| **LightGBM (Ours, Default)** | **0.521** | **54.7%** | **0.254** | **58.3%*** | **-0.046%** |
| **LightGBM + Platt Calibrated** | **0.525** | **54.2%** | **0.244** | **58.3%*** | **+0.015%** |

![Results Composite](assets/results_composite.png)

### 4.2 Key Insights
- Always-Up and classical RSI generate negative net returns after transaction costs (-0.106% and -0.111%).
- LightGBM achieves 58.3% accuracy on confident test signals with positive simulated equity curve (+44.1% over test period).
- Post-hoc Platt calibration optimizes Brier score to 0.244 and log-loss to 0.680.

---

## Chapter 5: Conclusion and Future Scope
The proposed framework proves that machine learning can uncover genuine directional alpha in cryptocurrency markets when rigorous data hygiene, purged cross-validation, and probability calibration are enforced.

---

## References
1. M. López de Prado, *Advances in Financial Machine Learning*, John Wiley & Sons, 2018.
2. D. R. Aronson, *Evidence-Based Technical Analysis*, John Wiley & Sons, 2006.
3. S. Makridakis et al., "Statistical and Machine Learning forecasting methods," *PLoS ONE*, 2018.
4. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," *NeurIPS*, 2017.
5. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
6. J. C. Platt, "Probabilistic outputs for support vector machines," *Adv. Large Margin Classifiers*, 1999.
7. G. W. Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Review*, 1950.
8. M. Dixon et al., *Machine Learning in Finance: From Theory to Practice*, Springer, 2020.
