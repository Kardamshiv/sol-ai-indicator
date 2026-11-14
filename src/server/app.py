"""Live Quant Trading Server.

Professional FastAPI server for SOLUSDT AI Directional Trading System.
Includes Binance WebSocket market stream, zero-repainting SQLite signal logger,
real-time Tree SHAP explainability, multi-timeframe candles, institutional
market intelligence, paper trading simulation, and interactive backtesting.
"""
import asyncio, csv, io, json, math, sqlite3, time
from contextlib import asynccontextmanager
from pathlib import Path
import joblib, numpy as np, pandas as pd, requests, websockets
from fastapi import FastAPI, Response, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.config import *
from src.features.build import make_features
from src.server.analytics import get_model_analytics, get_model_comparison
from src.server.explainability import build_explainability_payload
from src.server.indicators import compute_market_intelligence
from src.server.paper_trading import get_paper_trading_state, STARTING_CAPITAL
from src.server.multi_tf import get_timeframe_candles

# Load configuration and models
CFG = json.load(open(MODEL_DIR / "config.json"))
MODEL = joblib.load(MODEL_DIR / "model.joblib")
FEATS, H, CUT = CFG["features"], CFG["horizon"], CFG["cutoff"]
CAL = json.load(open(MODEL_DIR / "calibration.json")) if (MODEL_DIR / "calibration.json").exists() else None
CUT_SHOWN = CAL["cutoff_calibrated"] if CAL else CUT
IV_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[INTERVAL]
IV_MS = IV_SEC * 1000
DB = DATA_DIR / "signals.db"

# Server runtime state
candles: list[dict] = []
_cached_test_data = None
server_start_time = time.time()
ws_status = "CONNECTING"
ws_reconnect_count = 0
last_inference_latency_ms = 0.0
last_inference_timestamp = None


def get_test_dataset():
    """Lazily load and cache evaluation predictions for fast interactive what-if simulation."""
    global _cached_test_data
    if _cached_test_data is None:
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

        try:
            oof_path = ROOT / "results" / "oof_predictions.csv"
            if oof_path.exists():
                df = pd.read_csv(oof_path)
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
            exit_price REAL, ret REAL, hit INTEGER, lean INTEGER,
            raw_prob_up REAL, calibrated_prob_up REAL)""")
        for col, typ in [("lean", "INTEGER"), ("raw_prob_up", "REAL"), ("calibrated_prob_up", "REAL")]:
            try:
                c.execute(f"ALTER TABLE signals ADD COLUMN {col} {typ}")
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
    """Platt scaling fitted on out-of-fold predictions."""
    pu = min(max(pu, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-(CAL["coef"] * math.log(pu / (1 - pu)) + CAL["intercept"])))


def predict_and_store():
    global last_inference_latency_ms, last_inference_timestamp
    t0 = time.perf_counter()
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    p = MODEL.predict_proba(make_features(df)[FEATS].iloc[[-1]])[0]
    raw_up = float(p[2] / (p[0] + p[2] + 1e-12))

    if CAL:
        q = calibrated_up_prob(raw_up)
        lean = 1 if q > 0.5 else -1
        conf, cut = max(q, 1 - q), CAL["cutoff_calibrated"]
    else:
        k = int(p.argmax()); conf, cut = float(p[k]), CUT
        lean = 1 if k == 2 else -1 if k == 0 else 0
        q = float(p[2])

    d = lean if conf >= cut else 0
    last = candles[-1]
    with db() as c:
        c.execute("INSERT OR IGNORE INTO signals(open_time,direction,confidence,entry_price,horizon,created_at,lean,raw_prob_up,calibrated_prob_up) "
                  "VALUES(?,?,?,?,?,?,?,?,?)",
                  (last["open_time"], d, conf, last["close"], H, time.time(), lean, raw_up, q))

    last_inference_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    last_inference_timestamp = pd.to_datetime(last["open_time"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def process(row, live=True):
    if candles and row["open_time"] <= candles[-1]["open_time"]:
        return
    candles.append(row)
    del candles[:-600]
    resolve(row)
    if live and len(candles) >= 120:
        predict_and_store()


def backfill():
    """Fetch recent closed candles. First call = warm-up; later calls fill gaps after reconnect."""
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
    global ws_status, ws_reconnect_count
    url = f"{WS_URL}/ws/{SYMBOL.lower()}@kline_{INTERVAL}"
    backoff = 1
    while True:
        try:
            ws_status = "CONNECTING"
            await asyncio.to_thread(backfill)
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                ws_status = "CONNECTED"
                backoff = 1
                print(f"Binance kline stream connected for {SYMBOL} {INTERVAL}")
                async for msg in ws:
                    k = json.loads(msg)["k"]
                    if k["x"]:
                        process(ws_row(k))
        except Exception as e:
            ws_reconnect_count += 1
            ws_status = "RECONNECTING"
            print(f"WebSocket error ({repr(e)}). Reconnecting in {backoff}s (attempt {ws_reconnect_count})...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20)


@asynccontextmanager
async def lifespan(app):
    init_db()
    task = asyncio.create_task(stream())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


# --------------------------------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------------------------------

@app.get("/api/meta")
def meta():
    return dict(
        project_name="AI Quant Trading Intelligence",
        symbol=SYMBOL,
        interval=INTERVAL,
        interval_sec=IV_SEC,
        horizon=H,
        cutoff=CUT_SHOWN,
        calibrated=bool(CAL),
        cost=COST,
        starting_capital=STARTING_CAPITAL,
        model_algorithm="LightGBM Directional Classifier",
        model_calibration="Platt Scaling (Sigmoid)",
    )


@app.get("/api/candles")
def get_candles_endpoint(interval: str = Query(default="15m"), limit: int = Query(default=500)):
    """Return genuine OHLCV candles for the requested timeframe."""
    if interval == "15m" and candles:
        return [
            dict(time=c["open_time"] // 1000, open=c["open"], high=c["high"], low=c["low"], close=c["close"], volume=c.get("volume", 0))
            for c in candles
        ]
    return get_timeframe_candles(interval=interval, limit=limit, fallback_candles=candles)


@app.get("/api/signals")
def get_signals():
    """Return all recorded non-repainting signals with unique deterministic IDs."""
    with db() as c:
        rows = c.execute("SELECT * FROM signals ORDER BY open_time").fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["id"] = f"SIG-{SYMBOL}-{d['open_time']}"
        d["datetime_utc"] = pd.to_datetime(d["open_time"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")
        d["status"] = (
            "HIT" if d.get("hit") == 1 else
            "MISS" if d.get("hit") == 0 else
            "PENDING" if d.get("direction", 0) != 0 and d.get("exit_price") is None else
            "ABSTAINED"
        )
        results.append(d)
    return results


@app.get("/api/reference")
def reference():
    """Validation benchmark metrics."""
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
    return dict(
        n=n,
        hit_rate=sum(r["hit"] for r in rows) / n,
        always_up_hit=sum(r["ret"] > 0 for r in rows) / n,
        net_per_trade=sum(net) / n,
        total_net=sum(net),
    )


@app.get("/api/health")
def health():
    """System health and diagnostic metrics."""
    now = time.time()
    last = candles[-1]["close_time"] / 1000 if candles else None
    age = None if last is None else round(now - last)
    
    # Connection state determination
    if not candles:
        status = "CONNECTING"
    elif age is not None and age > IV_SEC * 2.5:
        status = "DATA_DELAYED"
    elif ws_status == "CONNECTED":
        status = "CONNECTED"
    else:
        status = ws_status

    return dict(
        ok=bool(candles and status in ("CONNECTED", "DATA_DELAYED")),
        status=status,
        candles_loaded=len(candles),
        last_close_age_sec=age,
        exchange_connection=ws_status,
        websocket_url=f"{WS_URL}/ws/{SYMBOL.lower()}@kline_{INTERVAL}",
        reconnect_count=ws_reconnect_count,
        server_uptime_sec=round(now - server_start_time),
        last_inference_latency_ms=last_inference_latency_ms,
        last_inference_timestamp=last_inference_timestamp,
        model_loaded=bool(MODEL is not None),
        model_algorithm="LightGBM Directional Classifier",
        calibration_method="Platt Scaling (Sigmoid)",
        data_quality="PRISTINE" if (age is not None and age <= IV_SEC * 1.5) else "ACCEPTABLE",
    )


@app.get("/api/predict/latest")
def predict_latest():
    """Detailed prediction breakdown with Tree SHAP explainability and institutional rationale."""
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

    # Generate Tree SHAP explainability payload
    explain = build_explainability_payload(
        model=MODEL,
        feat_row=last_row,
        feats=FEATS,
        conf=conf,
        cutoff=cut,
        direction=d,
        lean=lean,
        cal_prob=q,
    )

    return dict(
        status="ok",
        candle_open_time=last_candle["open_time"],
        candle_datetime_utc=pd.to_datetime(last_candle["open_time"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"),
        close_price=last_candle["close"],
        direction=d,
        direction_label="UP" if d == 1 else "DOWN" if d == -1 else "NO CLEAR SIGNAL",
        lean=lean,
        lean_label="UP" if lean == 1 else "DOWN",
        confidence=conf,
        confidence_pct=round(conf * 100, 2),
        cutoff=cut,
        cutoff_pct=round(cut * 100, 2),
        is_signal=bool(d != 0),
        raw_prob_up=round(raw_up, 4),
        raw_prob_down=round(1 - raw_up, 4),
        calibrated_prob_up=round(q, 4),
        calibrated_prob_down=round(1 - q, 4),
        drivers=explain["top_drivers"],
        all_contributions=explain["all_contributions"],
        rationale=explain["rationale"],
        tree_shap_base=explain["shap_base_value"],
        model_name="LightGBM Directional Classifier",
        calibration="Platt Scaling (Sigmoid)",
        horizon="1 Hour (4 x 15m candles)",
    )


@app.get("/api/market/intelligence")
def market_intelligence():
    """Return technical indicators and quantitative market intelligence."""
    return compute_market_intelligence(candles)


@app.get("/api/analytics/model")
def analytics_model():
    """Return comprehensive out-of-sample ML metrics, confusion matrix, and calibration curve."""
    return get_model_analytics()


@app.get("/api/analytics/comparison")
def analytics_comparison():
    """Return academic baseline comparison table across all evaluated algorithms."""
    return get_model_comparison()


@app.get("/api/papertrading")
def papertrading():
    """Return live paper trading portfolio state, open positions, and trade history."""
    with db() as c:
        rows = c.execute("SELECT * FROM signals ORDER BY open_time").fetchall()
    sig_list = [dict(r) for r in rows]
    cur_p = candles[-1]["close"] if candles else 0.0
    return get_paper_trading_state(sig_list, cur_p)


@app.get("/api/features/importance")
def feature_importance():
    """Return permutation feature importance sorted descending."""
    p = ROOT / "results" / "analysis" / "permutation_importance.csv"
    if p.exists():
        df = pd.read_csv(p)
        FEATURE_GROUPS = {
            "ret_1": "Momentum", "ret_4": "Momentum", "ret_16": "Momentum", "ret_96": "Momentum",
            "vol_16": "Volatility", "vol_96": "Volatility", "atr_14": "Volatility", "bb_width": "Volatility",
            "bb_pos": "Oscillator", "rsi_14": "Oscillator", "macd_hist": "Oscillator", "dist_vwap": "Oscillator",
            "vol_z": "Order Flow", "trades_z": "Order Flow", "taker_ratio": "Order Flow",
            "hour_sin": "Diurnal", "hour_cos": "Diurnal", "dow": "Diurnal"
        }
        df["group"] = df["feature"].map(FEATURE_GROUPS).fillna("Indicator")
        df["drop_mean"] = df["mean"]
        df = df.sort_values(by="mean", ascending=False)
        return df.to_dict(orient="records")
    return []


@app.get("/api/backtest/simulate")
def backtest_simulate(cutoff: float = 0.5346, cost: float = 0.001):
    """Full quantitative strategy simulation on untouched test split with institutional analytics."""
    data = get_test_dataset()
    if not data:
        return dict(error="Test dataset not available on server")

    raw_p = data["raw_p"]
    fwd = data["fwd"]
    close = data["close"]
    times = data.get("times")
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
    drawdown_series = [0.0]
    peak = 1.0

    i = 0
    while i < n_candles:
        if signals_arr[i] != 0:
            direction = signals_arr[i]
            r = float(direction * fwd[i] - cost)
            rets.append(r)
            new_eq = eq[-1] * (1 + r)
            eq.append(new_eq)
            if new_eq > peak:
                peak = new_eq
            dd_val = float(new_eq / peak - 1)
            drawdown_series.append(dd_val)

            t_val = str(times[i]) if times is not None else f"Candle #{i}"
            trades.append(dict(
                time=t_val,
                direction=int(direction),
                gross_ret=float(fwd[i] * direction),
                net_ret=r,
                hit=int(direction * fwd[i] > 0)
            ))
            i += H
        else:
            i += 1

    total_trades = len(rets)
    if total_trades == 0:
        return dict(
            cutoff=cutoff,
            cost=cost,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            coverage_pct=0.0,
            win_rate=0.0,
            always_up_win_rate=round(float((fwd > 0).mean()) * 100, 2),
            net_per_trade_pct=0.0,
            total_return_pct=0.0,
            gross_profit_pct=0.0,
            gross_loss_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            buy_and_hold_pct=round(float(close[-1] / close[0] - 1) * 100, 2),
            curve=[1.0],
            drawdown_curve=[0.0],
            monthly_performance=[],
            return_distribution=[],
        )

    eq_arr = np.array(eq)
    peak_arr = np.maximum.accumulate(eq_arr)
    max_dd = float((eq_arr / peak_arr - 1).min())

    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_profit = float(sum(wins))
    gross_loss = abs(float(sum(losses)))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    win_rate = len(wins) / total_trades

    # Sharpe & Sortino (annualized: 365 * 24 / 4 periods = 2190)
    ann_factor = np.sqrt(2190)
    std_rets = float(np.std(rets))
    sharpe = float(np.mean(rets) / (std_rets + 1e-12) * ann_factor) if std_rets > 0 else 0.0

    downside_losses = [r for r in rets if r < 0]
    downside_std = float(np.std(downside_losses)) if downside_losses else 1e-12
    sortino = float(np.mean(rets) / downside_std * ann_factor)

    # Downsampled curves for UI rendering
    step = max(1, len(eq) // 50)
    curve_sampled = [round(float(eq[idx]), 4) for idx in range(0, len(eq), step)]
    if curve_sampled[-1] != round(float(eq[-1]), 4):
        curve_sampled.append(round(float(eq[-1]), 4))

    dd_sampled = [round(float(drawdown_series[idx]) * 100, 2) for idx in range(0, len(drawdown_series), step)]
    if dd_sampled[-1] != round(float(drawdown_series[-1]) * 100, 2):
        dd_sampled.append(round(float(drawdown_series[-1]) * 100, 2))

    # Monthly performance breakdown
    monthly_dict = {}
    for t in trades:
        m_key = t["time"][:7] if len(t["time"]) >= 7 else "Test Period"
        if m_key not in monthly_dict:
            monthly_dict[m_key] = {"trades": 0, "wins": 0, "net_sum": 0.0}
        monthly_dict[m_key]["trades"] += 1
        if t["net_ret"] > 0:
            monthly_dict[m_key]["wins"] += 1
        monthly_dict[m_key]["net_sum"] += t["net_ret"]

    monthly_performance = [
        {
            "month": m,
            "trades": v["trades"],
            "win_rate_pct": round(v["wins"] / v["trades"] * 100, 1),
            "net_return_pct": round(v["net_sum"] * 100, 2),
        }
        for m, v in monthly_dict.items()
    ]

    # Return distribution histogram bins
    bins = [
        {"bin": "<-2.0%", "min": -999, "max": -0.02, "count": 0},
        {"bin": "-2% to -1%", "min": -0.02, "max": -0.01, "count": 0},
        {"bin": "-1% to 0%", "min": -0.01, "max": 0.0, "count": 0},
        {"bin": "0% to +1%", "min": 0.0, "max": 0.01, "count": 0},
        {"bin": "+1% to +2%", "min": 0.01, "max": 0.02, "count": 0},
        {"bin": ">+2.0%", "min": 0.02, "max": 999, "count": 0},
    ]
    for r in rets:
        for b in bins:
            if b["min"] < r <= b["max"]:
                b["count"] += 1
                break

    return dict(
        cutoff=cutoff,
        cost=cost,
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        coverage_pct=round(total_trades * H / n_candles * 100, 2),
        win_rate=round(win_rate * 100, 2),
        always_up_win_rate=round(float((fwd > 0).mean()) * 100, 2),
        net_per_trade_pct=round(float(np.mean(rets)) * 100, 3),
        total_return_pct=round(float(eq[-1] - 1) * 100, 2),
        gross_profit_pct=round(gross_profit * 100, 2),
        gross_loss_pct=round(gross_loss * 100, 2),
        profit_factor=round(pf, 2),
        avg_win_pct=round(float(np.mean(wins)) * 100, 3) if wins else 0.0,
        avg_loss_pct=round(float(np.mean(losses)) * 100, 3) if losses else 0.0,
        win_loss_ratio=round(abs(float(np.mean(wins)) / (np.mean(losses) + 1e-12)), 2) if wins and losses else 0.0,
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        buy_and_hold_pct=round(float(close[-1] / close[0] - 1) * 100, 2),
        curve=curve_sampled,
        drawdown_curve=dd_sampled,
        monthly_performance=monthly_performance,
        return_distribution=bins,
    )


@app.get("/api/signals/export")
def export_signals():
    """Export the immutable SQLite signal log as a downloadable CSV."""
    with db() as c:
        rows = c.execute("SELECT * FROM signals ORDER BY open_time").fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "signal_id", "open_time_ms", "datetime_utc", "asset", "direction",
        "direction_label", "confidence", "threshold", "raw_prob_up", "calibrated_prob_up",
        "entry_price", "exit_price", "return_pct", "outcome", "status"
    ])
    for r in rows:
        d = r["direction"]
        d_lbl = "UP" if d == 1 else "DOWN" if d == -1 else "NO_SIGNAL"
        dt_str = pd.to_datetime(r["open_time"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
        ret_val = f"{r['ret'] * 100:.3f}%" if r["ret"] is not None else ""
        outcome = "HIT" if r["hit"] == 1 else "MISS" if r["hit"] == 0 else "PENDING" if d != 0 else "ABSTAINED"
        sig_id = f"SIG-{SYMBOL}-{r['open_time']}"
        writer.writerow([
            sig_id, r["open_time"], dt_str, SYMBOL, d, d_lbl,
            r["confidence"], CUT_SHOWN, r["raw_prob_up"], r["calibrated_prob_up"],
            r["entry_price"], r["exit_price"], ret_val, outcome, "FINALIZED" if r["exit_price"] else "OPEN"
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={SYMBOL}_institutional_signals.csv"},
    )


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
