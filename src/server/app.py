"""Live server. Run from repo root:  uvicorn src.server.app:app --port 8000

- Warms up with recent closed candles, then listens to the Binance kline stream.
- A signal is created ONLY when a candle is closed, and stored once (INSERT OR IGNORE),
  so it can never change afterwards (no repainting).
- After HORIZON candles each signal is scored hit/miss.
"""
import asyncio, json, math, sqlite3, time
from contextlib import asynccontextmanager
import joblib, numpy as np, pandas as pd, requests, websockets
import io, csv
from fastapi import FastAPI, Response, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from src.config import *
from src.features.build import make_features

CFG = json.load(open(MODEL_DIR / "config.json"))
MODEL = joblib.load(MODEL_DIR / "model.joblib")
FEATS, H, CUT = CFG["features"], CFG["horizon"], CFG["cutoff"]
CAL = json.load(open(MODEL_DIR / "calibration.json")) if (MODEL_DIR / "calibration.json").exists() else None
CUT_SHOWN = CAL["cutoff_calibrated"] if CAL else CUT      # confidence needed to signal, in the units shown to the trader
IV_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[INTERVAL]
IV_MS = IV_SEC * 1000
DB = DATA_DIR / "signals.db"
candles: list[dict] = []
_cached_test_data = None


def get_test_dataset():
    """Lazily load and cache evaluation predictions for fast interactive what-if simulation."""
    global _cached_test_data
    if _cached_test_data is None:
        # First priority: held-out test split from data/SOLUSDT_15m.csv
        try:
            from src.data.split import build_dataset, split_bounds
            csv_path = DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["time"])
                _, d = build_dataset(df)
                b = split_bounds(len(d))
                te = d.iloc[b["test"][0]:b["test"][1]].copy()
                raw_p = MODEL.predict_proba(te[FEATS])
                _cached_test_data = {
                    "source": "held-out test split (10,493 candles)",
                    "raw_p": raw_p,
                    "fwd": te["fwd"].values,
                    "close": te["close"].values,
                    "times": te["time"].dt.strftime("%Y-%m-%d %H:%M").values,
                }
                return _cached_test_data
        except Exception as e:
            print("Notice: could not load from CSV:", repr(e))

        # Second priority: results/oof_predictions.csv (always committed in repository)
        try:
            oof_path = ROOT / "results" / "oof_predictions.csv"
            if oof_path.exists():
                df = pd.read_csv(oof_path)
                # Use latest fold (fold 6, ~8,500 candles) for fast simulation
                late_df = df[df["fold"] >= 5] if "fold" in df.columns else df
                p_up = late_df["p_default"].values
                raw_p = np.column_stack([1 - p_up, np.zeros(len(p_up)), p_up])
                fwd = late_df["fwd"].values
                _cached_test_data = {
                    "source": "validation out-of-fold candles (17,000 candles)",
                    "raw_p": raw_p,
                    "fwd": fwd,
                    "close": np.cumprod(1 + fwd),
                    "times": late_df["time"].values if "time" in late_df else np.arange(len(fwd)),
                }
                return _cached_test_data
        except Exception as e:
            print("Notice: could not load from OOF predictions:", repr(e))

    return _cached_test_data


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
            exit_price REAL, ret REAL, hit INTEGER, lean INTEGER)""")
        try:
            c.execute("ALTER TABLE signals ADD COLUMN lean INTEGER")   # upgrade a database made by an older version
        except sqlite3.OperationalError:
            pass


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


def calibrated_up_prob(pu):
    """Platt scaling fitted on out-of-fold predictions (see src/train/fit_calibration.py)."""
    pu = min(max(pu, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-(CAL["coef"] * math.log(pu / (1 - pu)) + CAL["intercept"])))


def predict_and_store():
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    p = MODEL.predict_proba(make_features(df)[FEATS].iloc[[-1]])[0]
    if CAL:                                     # calibrated direction probability: "Up 55%" means about 55%
        q = calibrated_up_prob(float(p[2] / (p[0] + p[2] + 1e-12)))
        lean = 1 if q > 0.5 else -1
        conf, cut = max(q, 1 - q), CAL["cutoff_calibrated"]
    else:
        k = int(p.argmax()); conf, cut = float(p[k]), CUT
        lean = 1 if k == 2 else -1 if k == 0 else 0
    d = lean if conf >= cut else 0              # not confident enough -> no signal
    last = candles[-1]
    with db() as c:                             # INSERT OR IGNORE: stored once, never edited
        c.execute("INSERT OR IGNORE INTO signals(open_time,direction,confidence,entry_price,horizon,created_at,lean) "
                  "VALUES(?,?,?,?,?,?,?)",
                  (last["open_time"], d, conf, last["close"], H, time.time(), lean))


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
    if warm and len(candles) >= 120:
        predict_and_store()


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
    return dict(symbol=SYMBOL, interval=INTERVAL, interval_sec=IV_SEC, horizon=H, cutoff=CUT_SHOWN, calibrated=bool(CAL), cost=COST)


@app.get("/api/candles")
def get_candles():
    return [dict(time=c["open_time"] // 1000, open=c["open"], high=c["high"], low=c["low"], close=c["close"])
            for c in candles]


@app.get("/api/signals")
def get_signals():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM signals ORDER BY open_time")]


@app.get("/api/reference")
def reference():
    """Backtest numbers (test period, NOT live). Created by: python -m src.backtest.backtest"""
    p = MODEL_DIR / "reference.json"
    return json.load(open(p)) if p.exists() else {}


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


@app.get("/api/health")
def health():
    """Used by hosting checks: if last_close_age_sec grows past ~2 candles, the stream has stalled."""
    last = candles[-1]["close_time"] / 1000 if candles else None
    return dict(ok=bool(candles), candles=len(candles),
                last_close_age_sec=None if last is None else round(time.time() - last))


@app.get("/api/predict/latest")
def predict_latest():
    """Detailed prediction breakdown on the latest closed candle with explainability drivers."""
    if len(candles) < 120:
        return dict(status="warming_up", candles=len(candles), needed=120)

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    feat_df = make_features(df)
    last_row = feat_df[FEATS].iloc[[-1]]
    p = MODEL.predict_proba(last_row)[0]

    raw_up = float(p[2] / (p[0] + p[2] + 1e-12))
    if CAL:
        q = calibrated_up_prob(raw_up)
        lean = 1 if q > 0.5 else -1
        conf, cut = max(q, 1 - q), CAL["cutoff_calibrated"]
    else:
        k = int(p.argmax())
        conf, cut = float(p[k]), CUT
        lean = 1 if k == 2 else -1 if k == 0 else 0
        q = float(p[2])

    d = lean if conf >= cut else 0
    last_candle = candles[-1]

    # Feature inspection and human-readable drivers
    r = last_row.iloc[0]
    drivers = []
    ret_96_val = float(r.get("ret_96", 0))
    rsi_val = float(r.get("rsi_14", 50))
    vol_val = float(r.get("vol_16", 0))
    taker_val = float(r.get("taker_ratio", 0.5))
    vwap_val = float(r.get("dist_vwap", 0))

    if ret_96_val > 0.01:
        drivers.append(dict(name="24h Trend (ret_96)", value=f"{ret_96_val:+.2%}", impact="bullish", note="Positive 24h momentum"))
    elif ret_96_val < -0.01:
        drivers.append(dict(name="24h Trend (ret_96)", value=f"{ret_96_val:+.2%}", impact="bearish", note="Negative 24h momentum"))

    if rsi_val > 60:
        drivers.append(dict(name="RSI (14)", value=f"{rsi_val:.1f}", impact="bullish", note="Strong buying pressure"))
    elif rsi_val < 40:
        drivers.append(dict(name="RSI (14)", value=f"{rsi_val:.1f}", impact="bearish", note="Strong selling pressure"))
    else:
        drivers.append(dict(name="RSI (14)", value=f"{rsi_val:.1f}", impact="neutral", note="Neutral momentum zone"))

    if taker_val > 0.52:
        drivers.append(dict(name="Taker Buy Ratio", value=f"{taker_val:.1%}", impact="bullish", note="Aggressive buyer market orders"))
    elif taker_val < 0.48:
        drivers.append(dict(name="Taker Buy Ratio", value=f"{taker_val:.1%}", impact="bearish", note="Aggressive seller market orders"))

    if vwap_val > 0.005:
        drivers.append(dict(name="VWAP Distance", value=f"{vwap_val:+.2%}", impact="bullish", note="Trading above rolling VWAP"))
    elif vwap_val < -0.005:
        drivers.append(dict(name="VWAP Distance", value=f"{vwap_val:+.2%}", impact="bearish", note="Trading below rolling VWAP"))

    return dict(
        status="ok",
        candle_open_time=last_candle["open_time"],
        close_price=last_candle["close"],
        direction=d,
        direction_label="Up" if d == 1 else "Down" if d == -1 else "No clear signal",
        lean=lean,
        lean_label="Up" if lean == 1 else "Down",
        confidence=conf,
        confidence_pct=round(conf * 100, 2),
        cutoff=cut,
        cutoff_pct=round(cut * 100, 2),
        is_signal=bool(d != 0),
        raw_prob_up=round(raw_up, 4),
        calibrated_prob_up=round(q, 4),
        drivers=drivers,
        key_features={
            "ret_1": round(float(r.get("ret_1", 0)), 4),
            "ret_4": round(float(r.get("ret_4", 0)), 4),
            "ret_16": round(float(r.get("ret_16", 0)), 4),
            "ret_96": round(ret_96_val, 4),
            "rsi_14": round(rsi_val, 2),
            "taker_ratio": round(taker_val, 4),
            "dist_vwap": round(vwap_val, 4),
            "vol_16": round(vol_val, 5),
        }
    )


FEATURE_GROUPS = {
    "ret_1": "momentum", "ret_4": "momentum", "ret_16": "momentum", "ret_96": "momentum",
    "vol_16": "volatility", "vol_96": "volatility", "atr_14": "volatility", "bb_width": "volatility",
    "bb_pos": "oscillator", "rsi_14": "oscillator", "macd_hist": "oscillator", "dist_vwap": "oscillator",
    "vol_z": "order flow", "trades_z": "order flow", "taker_ratio": "order flow",
    "hour_sin": "diurnal", "hour_cos": "diurnal", "dow": "diurnal"
}


@app.get("/api/features/importance")
def feature_importance():
    """Return permutation importance sorted descending by impact with category tags."""
    p = ROOT / "results" / "analysis" / "permutation_importance.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["group"] = df["feature"].map(FEATURE_GROUPS).fillna("indicator")
        df["drop_mean"] = df["mean"]
        df = df.sort_values(by="mean", ascending=False)
        return df.to_dict(orient="records")
    return []


@app.get("/api/backtest/simulate")
def backtest_simulate(cutoff: float = Query(default=0.5346), cost: float = Query(default=0.001)):
    """Interactive backtest simulation over the test period across user-tuned cutoffs and costs."""
    data = get_test_dataset()
    if not data:
        return dict(error="Test dataset not available on server")

    raw_p = data["raw_p"]
    fwd = data["fwd"]
    close = data["close"]
    n_candles = len(fwd)

    if CAL:
        pu = raw_p[:, 2] / (raw_p[:, 0] + raw_p[:, 2] + 1e-12)
        pu = np.clip(pu, 1e-6, 1 - 1e-6)
        log_odds = np.log(pu / (1 - pu))
        q = 1 / (1 + np.exp(-(CAL["coef"] * log_odds + CAL["intercept"])))
        conf = np.maximum(q, 1 - q)
        pred = np.where(q > 0.5, 1, -1)
    else:
        conf = raw_p.max(axis=1)
        k = raw_p.argmax(axis=1)
        pred = np.where(k == 2, 1, np.where(k == 0, -1, 0))

    signals_arr = np.where(conf >= cutoff, pred, 0)

    rets = []
    trades = []
    eq = [1.0]
    i = 0
    while i < n_candles:
        if signals_arr[i] != 0:
            direction = signals_arr[i]
            r = direction * fwd[i] - cost
            rets.append(r)
            eq.append(eq[-1] * (1 + r))
            trades.append(dict(idx=i, direction=int(direction), ret=float(fwd[i]), net=float(r)))
            i += H
        else:
            i += 1

    total_trades = len(rets)
    if total_trades == 0:
        return dict(
            cutoff=cutoff,
            cost=cost,
            total_trades=0,
            coverage_pct=0.0,
            win_rate=0.0,
            always_up_win_rate=round(float((fwd > 0).mean()) * 100, 2),
            net_per_trade_pct=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            buy_and_hold_pct=round(float(close[-1] / close[0] - 1) * 100, 2),
            curve=[1.0],
        )

    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float((eq_arr / peak - 1).min())
    hits = sum(1 for t in trades if t["direction"] * t["ret"] > 0)
    win_rate = hits / total_trades
    always_up = float((fwd > 0).mean())

    step = max(1, len(eq) // 40)
    curve_sampled = [round(float(eq[idx]), 4) for idx in range(0, len(eq), step)]
    if curve_sampled[-1] != round(float(eq[-1]), 4):
        curve_sampled.append(round(float(eq[-1]), 4))

    return dict(
        cutoff=cutoff,
        cost=cost,
        total_trades=total_trades,
        coverage_pct=round(total_trades * H / n_candles * 100, 2),
        win_rate=round(win_rate * 100, 2),
        always_up_win_rate=round(always_up * 100, 2),
        net_per_trade_pct=round(float(np.mean(rets)) * 100, 3),
        total_return_pct=round(float(eq[-1] - 1) * 100, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        buy_and_hold_pct=round(float(close[-1] / close[0] - 1) * 100, 2),
        curve=curve_sampled,
    )


@app.get("/api/signals/export")
def export_signals():
    """Export the immutable SQLite signal log as a downloadable CSV."""
    with db() as c:
        rows = c.execute("SELECT * FROM signals ORDER BY open_time").fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["open_time_ms", "datetime_utc", "direction", "direction_label", "confidence", "entry_price", "exit_price", "return_pct", "hit", "lean"])
    for r in rows:
        d = r["direction"]
        d_lbl = "Up" if d == 1 else "Down" if d == -1 else "No Signal"
        dt_str = pd.to_datetime(r["open_time"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")
        ret_val = f"{r['ret'] * 100:.3f}%" if r["ret"] is not None else ""
        writer.writerow([r["open_time"], dt_str, d, d_lbl, r["confidence"], r["entry_price"], r["exit_price"], ret_val, r["hit"], r["lean"]])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={SYMBOL}_signals_log.csv"},
    )


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
