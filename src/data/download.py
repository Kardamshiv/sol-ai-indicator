"""Download historical klines from Binance into data/<symbol>_<interval>.csv

Run from repo root:  python -m src.data.download
"""
import time
import pandas as pd
import requests
from src.config import *

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch(symbol, interval, start_ms):
    rows, t = [], start_ms
    while True:
        r = requests.get(f"{BASE_URL}/api/v3/klines", timeout=30,
                         params=dict(symbol=symbol, interval=interval, startTime=t, limit=1000))
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows += data
        t = data[-1][0] + 1
        print(f"\rfetched {len(rows)} candles", end="")
        if len(data) < 1000:
            break
        time.sleep(0.2)
    print()
    return rows


def main():
    DATA_DIR.mkdir(exist_ok=True)
    start = int((time.time() - YEARS * 365 * 86400) * 1000)
    df = pd.DataFrame(fetch(SYMBOL, INTERVAL, start), columns=COLS)
    num = [c for c in COLS if c not in ("ignore",)]
    df[num] = df[num].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[df["close_time"] < time.time() * 1000]          # drop unclosed candle
    df = df.drop_duplicates("open_time").sort_values("open_time").drop(columns="ignore")
    gaps = (df["open_time"].diff().dropna() != df["open_time"].diff().mode()[0]).sum()
    print(f"{len(df)} candles, {df.time.min()} -> {df.time.max()}, gaps: {gaps}")
    df.to_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", index=False)


if __name__ == "__main__":
    main()
