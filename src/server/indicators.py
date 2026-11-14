"""Technical Indicators & Market Intelligence Service.

Computes exact technical indicators from recent closed candles and generates
institutional quantitative interpretations for the Market Intelligence panel.
"""
import numpy as np
import pandas as pd
from src.features.build import rsi


def compute_market_intelligence(candles: list[dict]) -> dict:
    """Calculate institutional market indicators and interpretations on candle history."""
    if len(candles) < 30:
        return {"status": "insufficient_data", "candles": len(candles)}

    df = pd.DataFrame(candles)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # Price & Changes
    cur_price = float(c.iloc[-1])
    ret_1 = float(c.pct_change(1).iloc[-1]) if len(c) > 1 else 0.0
    ret_4 = float(c.pct_change(4).iloc[-1]) if len(c) > 4 else 0.0
    ret_16 = float(c.pct_change(16).iloc[-1]) if len(c) > 16 else 0.0
    ret_96 = float(c.pct_change(96).iloc[-1]) if len(c) > 96 else float(c.iloc[-1] / c.iloc[0] - 1)

    # Moving Averages & EMAs
    ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_diff = (ema20 / ema50 - 1)
    ema_state = "Bullish Alignment (EMA20 > EMA50)" if ema20 > ema50 else "Bearish Alignment (EMA20 < EMA50)"

    # VWAP (Rolling 96 candles / 24 hours)
    vwap_len = min(96, len(df))
    rolling_vol = v.iloc[-vwap_len:].sum()
    rolling_pv = (c * v).iloc[-vwap_len:].sum()
    cur_vwap = float(rolling_pv / (rolling_vol + 1e-12))
    dist_vwap = float(cur_price / cur_vwap - 1)
    vwap_state = "Trading Above VWAP (Institutional Bullish)" if dist_vwap > 0.005 else \
                 "Trading Below VWAP (Institutional Bearish)" if dist_vwap < -0.005 else "At VWAP Equilibrium"

    # RSI (14)
    rsi_series = rsi(c, n=14)
    cur_rsi = float(rsi_series.iloc[-1])
    rsi_state = "Deeply Oversold (< 30)" if cur_rsi < 30 else \
                "Moderately Bearish (30 - 45)" if cur_rsi < 45 else \
                "Neutral Consolidation (45 - 55)" if cur_rsi <= 55 else \
                "Moderately Bullish (55 - 70)" if cur_rsi <= 70 else "Overbought (> 70)"

    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    cur_macd = float(macd_line.iloc[-1])
    cur_sig = float(signal_line.iloc[-1])
    cur_hist = float(macd_hist.iloc[-1])
    macd_state = "Bullish Crossover (Histogram Expanding)" if cur_hist > 0 and cur_hist > float(macd_hist.iloc[-2]) else \
                 "Bullish Regime (Above Signal)" if cur_hist > 0 else \
                 "Bearish Crossover (Histogram Contracting)" if cur_hist < 0 and cur_hist < float(macd_hist.iloc[-2]) else \
                 "Bearish Regime (Below Signal)"

    # Bollinger Bands (20, 2)
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    bb_upper = float(ma20.iloc[-1] + 2 * sd20.iloc[-1])
    bb_lower = float(ma20.iloc[-1] - 2 * sd20.iloc[-1])
    bb_mid = float(ma20.iloc[-1])
    bb_width = float(4 * sd20.iloc[-1] / (bb_mid + 1e-12))
    bb_pct_b = float((cur_price - bb_lower) / (bb_upper - bb_lower + 1e-12))
    bb_state = "Volatility Squeeze" if bb_width < 0.02 else \
               "Band Expansion" if bb_width > 0.06 else "Normal Volatility Regime"

    # ATR (14)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    cur_atr = float(tr.rolling(14).mean().iloc[-1])
    atr_norm = float(cur_atr / cur_price)

    # Order Flow & Taker Buy
    cur_taker = float(df["taker_buy_base"].iloc[-1] / (v.iloc[-1] + 1e-12)) if "taker_buy_base" in df else 0.5
    vol_mean = v.iloc[-min(96, len(v)):].mean()
    vol_std = v.iloc[-min(96, len(v)):].std()
    vol_z = float((v.iloc[-1] - vol_mean) / (vol_std + 1e-12))
    flow_state = "Aggressive Buyer Absorption" if cur_taker > 0.53 else \
                 "Aggressive Seller Liquidation" if cur_taker < 0.47 else "Balanced Two-Way Flow"

    return {
        "status": "ok",
        "current_price": cur_price,
        "indicators": {
            "rsi_14": {
                "name": "RSI (14)",
                "value": round(cur_rsi, 1),
                "state": rsi_state,
                "sentiment": "bullish" if cur_rsi > 55 else "bearish" if cur_rsi < 45 else "neutral",
                "interpretation": f"RSI at {cur_rsi:.1f} indicates {rsi_state.lower()}."
            },
            "macd": {
                "name": "MACD (12, 26, 9)",
                "macd_line": round(cur_macd, 3),
                "signal_line": round(cur_sig, 3),
                "histogram": round(cur_hist, 3),
                "state": macd_state,
                "sentiment": "bullish" if cur_hist > 0 else "bearish",
                "interpretation": f"Histogram at {cur_hist:+.3f}: {macd_state}."
            },
            "ema_trend": {
                "name": "EMA 20 / EMA 50",
                "ema_20": round(ema20, 2),
                "ema_50": round(ema50, 2),
                "spread_pct": round(ema_diff * 100, 2),
                "state": ema_state,
                "sentiment": "bullish" if ema20 > ema50 else "bearish",
                "interpretation": f"EMA 20 is trading {abs(ema_diff)*100:.2f}% {'above' if ema20 > ema50 else 'below'} EMA 50."
            },
            "vwap": {
                "name": "Volume-Weighted Average Price",
                "vwap_value": round(cur_vwap, 2),
                "distance_pct": round(dist_vwap * 100, 2),
                "state": vwap_state,
                "sentiment": "bullish" if dist_vwap > 0.005 else "bearish" if dist_vwap < -0.005 else "neutral",
                "interpretation": f"Price is {dist_vwap:+.2%} relative to 24h rolling VWAP (${cur_vwap:.2f})."
            },
            "bollinger": {
                "name": "Bollinger Bands (20, 2)",
                "upper": round(bb_upper, 2),
                "middle": round(bb_mid, 2),
                "lower": round(bb_lower, 2),
                "bandwidth_pct": round(bb_width * 100, 2),
                "pct_b": round(bb_pct_b, 3),
                "state": bb_state,
                "sentiment": "neutral",
                "interpretation": f"Bandwidth {bb_width*100:.1f}% ({bb_state}). %B position: {bb_pct_b:.2f}."
            },
            "atr": {
                "name": "Average True Range (14)",
                "atr_val": round(cur_atr, 3),
                "atr_pct": round(atr_norm * 100, 2),
                "interpretation": f"Typical 15m candle volatility is ${cur_atr:.2f} ({atr_norm*100:.2f}% of price)."
            },
            "order_flow": {
                "name": "Taker Buy Volume Ratio",
                "taker_ratio_pct": round(cur_taker * 100, 1),
                "volume_z_score": round(vol_z, 2),
                "state": flow_state,
                "sentiment": "bullish" if cur_taker > 0.52 else "bearish" if cur_taker < 0.48 else "neutral",
                "interpretation": f"{cur_taker*100:.1f}% market buys vs sells with volume Z-score {vol_z:+.1f}σ."
            },
            "momentum": {
                "name": "Multi-Horizon Momentum",
                "ret_15m": round(ret_1 * 100, 2),
                "ret_1h": round(ret_4 * 100, 2),
                "ret_4h": round(ret_16 * 100, 2),
                "ret_24h": round(ret_96 * 100, 2),
                "interpretation": f"24h Trend: {ret_96:+.2%} | 1h: {ret_4:+.2%} | 15m: {ret_1:+.2%}."
            }
        }
    }
