"""Institutional Paper Trading Portfolio Simulator.

Tracks a simulated paper portfolio starting at $10,000 USDT.
Executes paper trades strictly when high-conviction signals fire (confidence >= 53.5%).
Records mark-to-market unrealized P&L on open trades and realized P&L upon settlement.
Strictly paper trading: zero real capital at risk.
"""
import time
import numpy as np
import pandas as pd
from src.config import *

STARTING_CAPITAL = 10000.0   # $10,000 USDT initial simulated capital
POSITION_SIZE = 1000.0       # $1,000 USDT per paper trade (10% allocation)


def get_paper_trading_state(signals: list[dict], current_price: float) -> dict:
    """Calculate live virtual portfolio state, equity curve, open and closed paper positions."""
    cash_balance = STARTING_CAPITAL
    closed_trades = []
    open_positions = []
    equity_points = [{"time": "Start", "equity": STARTING_CAPITAL}]

    # Filter signals that actually triggered a position
    fired_signals = [s for s in signals if s.get("direction", 0) != 0]

    for s in fired_signals:
        d = s["direction"]
        entry = float(s["entry_price"])
        exit_p = float(s["exit_price"]) if s.get("exit_price") is not None else None
        open_time = s["open_time"]
        dt_str = pd.to_datetime(open_time, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")
        pos_id = f"POS-{open_time}"

        if exit_p is not None:
            # Trade is closed
            gross_ret = d * (exit_p / entry - 1)
            fee = POSITION_SIZE * COST
            net_pnl = POSITION_SIZE * gross_ret - fee
            cash_balance += net_pnl
            closed_trades.append({
                "id": pos_id,
                "open_time": open_time,
                "datetime": dt_str,
                "asset": f"{SYMBOL}",
                "direction": "LONG (UP)" if d > 0 else "SHORT (DOWN)",
                "direction_num": d,
                "entry_price": round(entry, 2),
                "exit_price": round(exit_p, 2),
                "position_size_usd": POSITION_SIZE,
                "gross_return_pct": round(gross_ret * 100, 2),
                "net_pnl_usd": round(net_pnl, 2),
                "fee_usd": round(fee, 2),
                "hit": int(gross_ret > 0),
                "confidence_pct": round(s["confidence"] * 100, 1),
            })
            equity_points.append({"time": dt_str[-9:], "equity": round(cash_balance, 2)})
        else:
            # Trade is currently open
            gross_ret = d * (current_price / entry - 1)
            fee = POSITION_SIZE * COST
            unrealized_pnl = POSITION_SIZE * gross_ret - fee
            settle_time_ms = open_time + (s.get("horizon", 4) + 1) * 900 * 1000
            settle_str = pd.to_datetime(settle_time_ms, unit="ms", utc=True).strftime("%H:%M UTC")

            # Progress calculation
            now_ms = time.time() * 1000
            total_duration = settle_time_ms - open_time
            elapsed = max(0, min(total_duration, now_ms - open_time))
            progress_pct = round((elapsed / total_duration) * 100, 1)

            open_positions.append({
                "id": pos_id,
                "open_time": open_time,
                "datetime": dt_str,
                "asset": f"{SYMBOL}",
                "direction": "LONG (UP)" if d > 0 else "SHORT (DOWN)",
                "direction_num": d,
                "entry_price": round(entry, 2),
                "current_price": round(current_price, 2),
                "position_size_usd": POSITION_SIZE,
                "unrealized_pnl_usd": round(unrealized_pnl, 2),
                "unrealized_return_pct": round(gross_ret * 100, 2),
                "settle_time": settle_str,
                "progress_pct": progress_pct,
                "confidence_pct": round(s["confidence"] * 100, 1),
            })

    total_unrealized = sum(p["unrealized_pnl_usd"] for p in open_positions)
    current_equity = cash_balance + total_unrealized
    total_return_usd = current_equity - STARTING_CAPITAL
    total_return_pct = (total_return_usd / STARTING_CAPITAL) * 100

    # Win rate and statistics on closed trades
    n_closed = len(closed_trades)
    wins = [t for t in closed_trades if t["net_pnl_usd"] > 0]
    win_rate = (len(wins) / n_closed * 100) if n_closed > 0 else None
    realized_pnl_usd = cash_balance - STARTING_CAPITAL

    # Max Drawdown calculation
    eq_vals = [pt["equity"] for pt in equity_points] + [current_equity]
    peak = eq_vals[0]
    max_dd_pct = 0.0
    for eq in eq_vals:
        if eq > peak:
            peak = eq
        dd = (eq / peak - 1) * 100
        if dd < max_dd_pct:
            max_dd_pct = dd

    return {
        "status": "ok",
        "disclaimer": "PAPER TRADING ONLY — Academic simulation. No real money or exchange execution.",
        "starting_capital": STARTING_CAPITAL,
        "cash_balance": round(cash_balance, 2),
        "current_equity": round(current_equity, 2),
        "total_return_usd": round(total_return_usd, 2),
        "total_return_pct": round(total_return_pct, 2),
        "realized_pnl_usd": round(realized_pnl_usd, 2),
        "unrealized_pnl_usd": round(total_unrealized, 2),
        "total_trades": len(fired_signals),
        "closed_trades_count": n_closed,
        "open_trades_count": len(open_positions),
        "win_rate_pct": round(win_rate, 2) if win_rate is not None else None,
        "max_drawdown_pct": round(abs(max_dd_pct), 2),
        "open_positions": open_positions,
        "closed_trades": closed_trades[-20:],
        "equity_curve": equity_points[-25:],
    }
