# Literature Survey and Gap (Project Step 2)

Compiled 10 November 2026. Each entry below was found through a web search and checked against its abstract or publisher record. Details marked *to verify* must be confirmed in the full paper before the final report.

## 1. Comparison table

| Study | Dataset | Method | Evaluation | Gap / limitation for our project |
|---|---|---|---|---|
| Fang et al. (2024), *Eur. J. Finance* 30(1):78-100 (arXiv 2003.00803) | Live tick-level data from 8 cryptocurrency pairs; headline result on BTC-USD | Deep learning to predict next-tick mid-price direction; transaction fee used as the labelling threshold | Reports about 78% accuracy on binary mid-price movement on live exchange data | Next-tick horizon and order-book style inputs; not a one-hour candle-level indicator; trading profit after costs *to verify* |
| Jaquart, Dann and Weinhardt (2021), *J. Finance Data Sci.* 7:45-66 | Bitcoin, prediction horizons of 1 to 60 minutes, technical, blockchain, sentiment and asset-based features | Several ML models; recurrent networks and gradient boosting found best | All models beat a random classifier; technical features most relevant; predictability rises with horizon; a long-short strategy earned positive returns before costs but negative returns after costs | Bitcoin only, minute-scale horizons; costs erase the edge, which is exactly the risk for our 1-hour SOLUSDT task |
| Jaquart, Koepke and Weinhardt (2022), *J. Finance Data Sci.* 8:331-352 | 100 largest cryptocurrencies, daily | Gradient boosting, random forest, LSTM, GRU, TCN and ensembles | Average accuracy 52.9% to 54.1%; 57.5% to 59.5% on the 10% most confident predictions per class and day; ensemble long-short Sharpe ratio after costs of about 3.1 to 3.2 vs 1.33 buy-and-hold | Daily horizon, many coins; supports using a confidence filter, but says nothing about intraday SOLUSDT or live non-repainting use |
| Rodrigues and Machado (2025), *Information* 16(4):300 | Minute-step data for ten cryptocurrencies; Bitcoin study uses 30 days | RNNs, ARIMA and conventional regression to forecast price 1 to 4 hours ahead | Forecast accuracy comparison (metrics *to verify*) | Only 30 days of data; forecasts price level, not tradable direction with confidence; no live test |
| Fischer and Krauss (2018), *Eur. J. Oper. Res.* 270(2):654-669 | S&P 500 constituents, 1992 to 2015, daily | LSTM vs random forest, deep neural network and logistic regression for directional movement | LSTM beat the memory-free models; profitability fluctuated around zero after transaction costs from 2010 | Equities and daily data; shows that edges can fade and disappear after costs |
| Lopez de Prado (2018), *Advances in Financial Machine Learning*, Wiley | General financial ML methodology (no single dataset) | Purged k-fold cross-validation with embargo (ch. 7), triple-barrier labelling (ch. 3), hyper-parameter tuning with CV (ch. 9), combinatorial purged CV (ch. 12) | Methodology for leakage-free validation of overlapping labels | Not crypto-specific; we use an embargo and a chronological split, but not yet purging, triple-barrier labels or CPCV |
| Ke et al. (2017), *NeurIPS* 30:3146-3154 (LightGBM) | Public benchmark datasets | Gradient boosting with GOSS and EFB | Up to 20 times faster training than conventional GBDT at almost the same accuracy | General-purpose algorithm, not a finance study; we use it as our main model |

## 2. Gap statement

Among the studies we reviewed, results are reported at tick, minute or daily horizons, mostly for Bitcoin or large baskets of coins, and several note that transaction costs remove the apparent edge. We did **not** find, in this limited search, a study that evaluates a one-hour, candle-level SOLUSDT direction model as a **live, non-repainting indicator** over a fixed window against simple baselines with cost-aware metrics. This project addresses that engineering gap on a small scale; it does not claim a methodological novelty. A broader search (Google Scholar, IEEE Xplore, SSRN) should be done before the final report.

## 3. How the literature shapes our choices

- **Gradient boosting** is among the models the crypto studies found effective, and it trains fast (Ke et al.).
- **Confidence filtering** raised accuracy in Jaquart et al. (2022), which motivates a confidence cutoff and an explicit "No signal" state.
- **Costs and longer horizons:** Jaquart et al. (2021) and Fischer and Krauss both show edges shrinking after costs, so we report profit after costs honestly and test several horizons.
- **Leakage control:** we follow Lopez de Prado in using chronological splits with an embargo; purging and CPCV are listed as future work.

## 4. Verification notes

- Search results were read at abstract or publisher-record level for most entries. Open each paper, confirm the numbers quoted above, and add page numbers before submission.
- A search also returned an arXiv preprint (2508.02356) on neural-network crypto trading; it is **not** included because it could not be verified as peer-reviewed.
- Do not cite any source in the report that you have not opened yourself.

## 5. References

1. Fang, F., Chung, W., Ventre, C., Basios, M., Kanthan, L., Li, L., and Wu, F. (2024). Ascertaining price formation in cryptocurrency markets with machine learning. *The European Journal of Finance*, 30(1), 78-100. Preprint: arXiv:2003.00803, https://doi.org/10.48550/arXiv.2003.00803
2. Jaquart, P., Dann, D., and Weinhardt, C. (2021). Short-term bitcoin market prediction via machine learning. *The Journal of Finance and Data Science*, 7, 45-66. https://doi.org/10.1016/j.jfds.2021.03.001
3. Jaquart, P., Koepke, S., and Weinhardt, C. (2022). Machine learning for cryptocurrency market prediction and trading. *The Journal of Finance and Data Science*, 8, 331-352. https://doi.org/10.1016/j.jfds.2022.12.001
4. Rodrigues, F., and Machado, M. (2025). High-frequency cryptocurrency price forecasting using machine learning models: a comparative study. *Information*, 16(4), 300. https://mdpi.com/2078-2489/16/4/300
5. Fischer, T., and Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669. https://doi.org/10.1016/j.ejor.2017.11.054
6. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. ISBN 978-1-119-48208-6.
7. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*, 3146-3154. https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

Tools and data sources to cite in the report: Binance public market-data API documentation, TradingView Lightweight Charts, scikit-learn, pandas, FastAPI.
