"""Multi-Timeframe Candle Service.

Fetches genuine OHLCV candles from Binance for user-selected chart timeframes
(1m, 5m, 15m, 30m, 1h, 4h). Includes short-lived caching to respect exchange rate limits.
No candles are ever fabricated.
"""
import time
import requests
from src.config import BASE_URL, SYMBOL

VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h"}
_tf_cache = {}
CACHE_TTL_SEC = 10


def get_timeframe_candles(interval: str = "15m", limit: int = 500, fallback_candles: list = None) -> list[dict]:
    """Fetch genuine OHLCV candles for any supported timeframe."""
    if interval not in VALID_INTERVALS:
        interval = "15m"

    now = time.time()
    cache_entry = _tf_cache.get(interval)
    if cache_entry and (now - cache_entry["ts"] < CACHE_TTL_SEC) and len(cache_entry["data"]) > 0:
        return cache_entry["data"]

    try:
        url = f"{BASE_URL}/api/v3/klines"
        r = requests.get(url, params={"symbol": SYMBOL, "interval": interval, "limit": min(limit, 1000)}, timeout=10)
        r.raise_for_status()
        rows = r.json()
        formatted = [
            {
                "time": int(k[0]) // 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in rows
        ]
        _tf_cache[interval] = {"ts": now, "data": formatted}
        return formatted
    except Exception as e:
        print(f"Notice: Multi-timeframe fetch error for {interval}:", repr(e))
        if interval == "15m" and fallback_candles:
            return [
                {
                    "time": c["open_time"] // 1000,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c.get("volume", 0)),
                }
                for c in fallback_candles
            ]
        return cache_entry["data"] if cache_entry else []
