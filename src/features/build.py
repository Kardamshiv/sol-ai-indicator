"""Shared feature code. Used by training AND the live server (prevents train/serve mismatch).
Every feature uses only data up to and including the candle's close -> no leakage."""
import numpy as np
import pandas as pd
from src.config import HORIZON, THRESH


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 4, 16, 96):
        f[f"ret_{n}"] = c.pct_change(n)
    f["vol_16"] = c.pct_change().rolling(16).std()
    f["vol_96"] = c.pct_change().rolling(96).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    f["rsi_14"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = 4 * sd / ma
    f["bb_pos"] = (c - ma) / (2 * sd + 1e-12)
    f["vol_z"] = (v - v.rolling(96).mean()) / (v.rolling(96).std() + 1e-12)
    f["taker_ratio"] = df["taker_buy_base"] / (v + 1e-12)
    f["trades_z"] = (df["trades"] - df["trades"].rolling(96).mean()) / (df["trades"].rolling(96).std() + 1e-12)
    vwap = (c * v).rolling(96).sum() / (v.rolling(96).sum() + 1e-12)
    f["dist_vwap"] = c / vwap - 1
    hour = df["time"].dt.hour + df["time"].dt.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow"] = df["time"].dt.dayofweek
    return f


FEATURES = list(make_features(pd.DataFrame({
    "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0],
    "taker_buy_base": [1.0], "trades": [1.0],
    "time": pd.to_datetime(["2024-01-01"], utc=True)})).columns)


def make_labels(df: pd.DataFrame, horizon=HORIZON, thresh=THRESH):
    """Return (forward return, class). class: 0=Down, 1=Neutral, 2=Up. Training only."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    y = pd.Series(1.0, index=df.index)
    y[fwd > thresh] = 2
    y[fwd < -thresh] = 0
    y[fwd.isna()] = np.nan
    return fwd, y
