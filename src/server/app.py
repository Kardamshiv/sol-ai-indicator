"""Live server. Run from repo root:  uvicorn src.server.app:app --port 8000

- Warms up with recent closed candles, then listens to the Binance kline stream.
- A signal is created ONLY when a candle is closed, and stored once (INSERT OR IGNORE),
  so it can never change afterwards (no repainting).
- After HORIZON candles each signal is scored hit/miss.
"""
import asyncio, json, sqlite3, time
from contextlib import asynccontextmanager
import joblib, pandas as pd, requests, websockets
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.config import *
from src.features.build import make_features

CFG = json.load(open(MODEL_DIR / "config.json"))
MODEL = joblib.load(MODEL_DIR / "model.joblib")
FEATS, H, CUT = CFG["features"], CFG["horizon"], CFG["cutoff"]
IV_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[INTERVAL]
IV_MS = IV_SEC * 1000
DB = DATA_DIR / "signals.db"
candles: list[dict] = []


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS signals(
            open_time INTEGER PRIMARY KEY, direction INTEGER, confidence REAL,
            entry_price REAL, horizon INTEGER, created_at REAL,
            exit_price REAL, ret REAL, hit INTEGER)""")


def rest_row(k):
    return dict(open_time=int(k[0]), open=float(k[1]), high=float(k[2]), low=float(k[3]),
                close=float(k[4]), volume=float(k[5]), close_time=int(k[6]),
                trades=float(k[8]), taker_buy_base=float(k[9]))


def ws_row(k):
    return dict(open_time=k["t"], open=float(k["o"]), high=float(k["h"]), low=float(k["l"]),
                close=float(k["c"]), volume=float(k["v"]), close_time=k["T"],
                trades=float(k["n"]), taker_buy_base=float(k["V"]))


def resolve(row):
    """Score the signal made H candles ago, using this candle's close as the exit."""
    with db() as c:
        s = c.execute("SELECT * FROM signals WHERE open_time=? AND direction!=0 AND exit_price IS NULL",
                      (row["open_time"] - H * IV_MS,)).fetchone()
        if s:
            ret = row["close"] / s["entry_price"] - 1
            c.execute("UPDATE signals SET exit_price=?, ret=?, hit=? WHERE open_time=?",
                      (row["close"], ret, int(ret * s["direction"] > 0), s["open_time"]))


def predict_and_store():
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    p = MODEL.predict_proba(make_features(df)[FEATS].iloc[[-1]])[0]
    k = int(p.argmax()); conf = float(p[k])
    d = 1 if k == 2 else -1 if k == 0 else 0
    if conf < CUT:
        d = 0                                   # not confident enough -> no signal
    last = candles[-1]
    with db() as c:                             # INSERT OR IGNORE: stored once, never edited
        c.execute("INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?,NULL,NULL,NULL)",
                  (last["open_time"], d, conf, last["close"], H, time.time()))


def process(row, live=True):
    if candles and row["open_time"] <= candles[-1]["open_time"]:
        return                                  # already seen
    candles.append(row)
    del candles[:-600]
    resolve(row)
    if live and len(candles) >= 120:
        predict_and_store()


def backfill():
    """Fetch recent closed candles. First call = warm-up only; later calls fill gaps after a reconnect."""
    warm = not candles
    r = requests.get(f"{BASE_URL}/api/v3/klines", timeout=30,
                     params=dict(symbol=SYMBOL, interval=INTERVAL, limit=500))
    r.raise_for_status()
    now = time.time() * 1000
    for k in r.json():
        if k[6] < now:
            process(rest_row(k), live=not warm)


async def stream():
    url = f"{WS_URL}/ws/{SYMBOL.lower()}@kline_{INTERVAL}"
    while True:
        try:
            await asyncio.to_thread(backfill)
            async with websockets.connect(url, ping_interval=20) as ws:
                print("stream connected")
                async for msg in ws:
                    k = json.loads(msg)["k"]
                    if k["x"]:                  # candle is closed
                        process(ws_row(k))
        except Exception as e:
            print("stream error, reconnecting in 5s:", repr(e))
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app):
    init_db()
    task = asyncio.create_task(stream())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/api/meta")
def meta():
    return dict(symbol=SYMBOL, interval=INTERVAL, interval_sec=IV_SEC, horizon=H, cutoff=CUT, cost=COST)


@app.get("/api/candles")
def get_candles():
    return [dict(time=c["open_time"] // 1000, open=c["open"], high=c["high"], low=c["low"], close=c["close"])
            for c in candles]


@app.get("/api/signals")
def get_signals():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM signals ORDER BY open_time")]


@app.get("/api/stats")
def stats():
    with db() as c:
        rows = c.execute("SELECT direction, ret, hit FROM signals WHERE direction!=0 AND hit IS NOT NULL").fetchall()
    n = len(rows)
    if n == 0:
        return dict(n=0)
    net = [r["direction"] * r["ret"] - COST for r in rows]
    return dict(n=n, hit_rate=sum(r["hit"] for r in rows) / n,
                always_up_hit=sum(r["ret"] > 0 for r in rows) / n,
                net_per_trade=sum(net) / n, total_net=sum(net))


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
