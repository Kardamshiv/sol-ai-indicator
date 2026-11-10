# Dataset card: SOLUSDT 15m candles

| Field | Value |
|---|---|
| Source | Binance public market-data API, klines endpoint (https://data-api.binance.vision) |
| Access date | 2026-11-10 |
| Citation | TO FILL: cite the Binance API documentation page you used, with the access date |
| License / terms | TO FILL: read Binance's market-data terms of use before submission; raw data is not redistributed in the repo (data/ is git-ignored) |
| Unit of observation | One closed 15m candle |
| Period analysed here | 2024-10-28 11:00:00+00:00 to 2026-03-23 17:00:00+00:00 (training portion only) |
| Training rows after cleaning | 48,981 |
| Target | Direction of the close 4 candles ahead; Neutral if the move is inside +/-0.20% |
| Class balance (training) | Down 36.4%, Neutral 26.2%, Up 37.3%; Up share among clear moves 0.506 |
| Features | 18, all from closed candles; see data_dictionary.csv |
| Privacy | Aggregated public market data, no personal data |
| Target proxies | None known: every feature uses only data up to the candle close |

## Audit summary (training portion)
- Duplicate candles: 0; gaps: 0 (largest 1 candles); out-of-order timestamps: 0
- Missing raw values: 0
- Impossible values: high < low: 0, close outside [low, high]: 0, open outside [low, high]: 0, non-positive price: 0, negative volume: 0, zero volume: 0, zero trades: 0, taker_buy > volume: 0
- Outlier candles (robust z > 8 on 1-candle return): 103; kept, because extreme moves are real market events

## Split protocol
Chronological 70% / 15% / 15% with an embargo of 4 candles; boundaries are saved in results/split_boundaries.json.
The test period was not analysed by this script and is reserved for the final one-shot evaluation.

## Known limitations
Single exchange and single asset, 24/7 market with regime changes, no news or order-book data, results depend on the period covered.
