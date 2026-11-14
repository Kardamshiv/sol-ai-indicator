"""AI Model Explainability Engine.

Computes genuine Tree SHAP local feature contributions via LightGBM's pred_contrib,
ranks input drivers by directional log-odds impact, and generates institutional
rationales for each prediction or abstention decision.
"""
import numpy as np
import pandas as pd
from src.config import *


FEATURE_META = {
    "ret_1": {"name": "1-Candle Return (15m)", "category": "Momentum", "unit": "%"},
    "ret_4": {"name": "1-Hour Trend (ret_4)", "category": "Momentum", "unit": "%"},
    "ret_16": {"name": "4-Hour Trend (ret_16)", "category": "Momentum", "unit": "%"},
    "ret_96": {"name": "24-Hour Momentum (ret_96)", "category": "Momentum", "unit": "%"},
    "vol_16": {"name": "Short-Term Volatility (vol_16)", "category": "Volatility", "unit": "std"},
    "vol_96": {"name": "Daily Volatility (vol_96)", "category": "Volatility", "unit": "std"},
    "atr_14": {"name": "ATR (14) Normalised", "category": "Volatility", "unit": "ratio"},
    "rsi_14": {"name": "Relative Strength Index (RSI 14)", "category": "Oscillator", "unit": "pts"},
    "macd_hist": {"name": "MACD Histogram Normalised", "category": "Oscillator", "unit": "ratio"},
    "bb_width": {"name": "Bollinger Bandwidth", "category": "Volatility", "unit": "ratio"},
    "bb_pos": {"name": "Bollinger Band %B Position", "category": "Oscillator", "unit": "ratio"},
    "vol_z": {"name": "Volume Z-Score", "category": "Order Flow", "unit": "z"},
    "trades_z": {"name": "Trade Count Z-Score", "category": "Order Flow", "unit": "z"},
    "taker_ratio": {"name": "Taker Buy Volume Ratio", "category": "Order Flow", "unit": "%"},
    "dist_vwap": {"name": "Distance from Rolling VWAP", "category": "Oscillator", "unit": "%"},
    "hour_sin": {"name": "Intraday Cycle (Sin)", "category": "Diurnal", "unit": "rad"},
    "hour_cos": {"name": "Intraday Cycle (Cos)", "category": "Diurnal", "unit": "rad"},
    "dow": {"name": "Day of Week", "category": "Calendar", "unit": "day"},
}


def compute_tree_shap(model, feat_row: pd.DataFrame, feats: list):
    """Compute local Tree SHAP contributions using LightGBM booster."""
    try:
        booster = getattr(model, "m_", None)
        booster_obj = getattr(booster, "booster_", None) if booster else getattr(model, "booster_", None)
        if booster_obj:
            contribs = booster_obj.predict(feat_row[feats], pred_contrib=True)[0]
            base_value = float(contribs[-1])
            shap_dict = {f: float(c) for f, c in zip(feats, contribs[:-1])}
            return shap_dict, base_value
    except Exception as e:
        print("Notice: Tree SHAP calculation error:", repr(e))
    return {f: 0.0 for f in feats}, 0.0


def format_feature_value(feat_name, val):
    meta = FEATURE_META.get(feat_name, {})
    unit = meta.get("unit", "")
    if unit == "%":
        return f"{val:+.2%}"
    elif unit == "pts":
        return f"{val:.1f}"
    elif unit == "z":
        return f"{val:+.2f}σ"
    elif unit == "ratio":
        return f"{val:.4f}"
    return f"{val:.4f}"


def build_explainability_payload(model, feat_row: pd.DataFrame, feats: list, conf: float, cutoff: float, direction: int, lean: int, cal_prob: float):
    """Construct full explainability payload including Tree SHAP and institutional rationale."""
    shap_dict, base_val = compute_tree_shap(model, feat_row, feats)
    r = feat_row.iloc[0]

    drivers = []
    for f in feats:
        c = shap_dict.get(f, 0.0)
        v = float(r.get(f, 0.0))
        meta = FEATURE_META.get(f, {"name": f, "category": "Indicator"})
        impact = "bullish" if c > 0 else "bearish" if c < 0 else "neutral"
        drivers.append({
            "feature": f,
            "display_name": meta["name"],
            "category": meta["category"],
            "raw_value": round(v, 5),
            "formatted_value": format_feature_value(f, v),
            "shap_contribution": round(c, 5),
            "abs_contribution": abs(c),
            "impact": impact,
            "direction_supported": "UP" if c > 0 else "DOWN",
        })

    drivers.sort(key=lambda x: x["abs_contribution"], reverse=True)
    top_bullish = [d for d in drivers if d["impact"] == "bullish"][:3]
    top_bearish = [d for d in drivers if d["impact"] == "bearish"][:3]

    # Generate institutional rationale
    top_lead = drivers[0] if drivers else None
    top_second = drivers[1] if len(drivers) > 1 else None

    if direction == 1:
        rationale = (
            f"The model generated a high-conviction BULLISH prediction ({conf * 100:.1f}% confidence, exceeding "
            f"the {cutoff * 100:.1f}% cutoff). Upward momentum is driven primarily by {top_lead['display_name']} "
            f"({top_lead['formatted_value']}) contributing +{top_lead['abs_contribution']:.4f} log-odds to the upside. "
            f"This is reinforced by {top_second['display_name'] if top_second else 'order flow aggression'}."
        )
    elif direction == -1:
        rationale = (
            f"The model generated a high-conviction BEARISH prediction ({conf * 100:.1f}% confidence, exceeding "
            f"the {cutoff * 100:.1f}% cutoff). Downward pressure is led by {top_lead['display_name']} "
            f"({top_lead['formatted_value']}) exerting a -{top_lead['abs_contribution']:.4f} log-odds bearish pull. "
            f"This is reinforced by {top_second['display_name'] if top_second else 'selling flow imbalance'}."
        )
    else:
        lead_bull = top_bullish[0] if top_bullish else None
        lead_bear = top_bearish[0] if top_bearish else None
        bull_desc = f"{lead_bull['display_name']} ({lead_bull['formatted_value']})" if lead_bull else "bullish momentum"
        bear_desc = f"{lead_bear['display_name']} ({lead_bear['formatted_value']})" if lead_bear else "bearish momentum"
        rationale = (
            f"The model deliberately ABSTAINS ('No Clear Signal'). Although {bull_desc} shows positive potential, "
            f"conflicting signals from {bear_desc} restrain calibrated confidence to {conf * 100:.1f}% "
            f"(below the required {cutoff * 100:.1f}% threshold). Capital is preserved to avoid fee drag during market indecision."
        )

    return {
        "status": "ok",
        "direction": direction,
        "direction_label": "UP" if direction == 1 else "DOWN" if direction == -1 else "NO CLEAR SIGNAL",
        "lean_label": "UP" if lean == 1 else "DOWN",
        "confidence_pct": round(conf * 100, 2),
        "cutoff_pct": round(cutoff * 100, 2),
        "calibrated_prob_up": round(cal_prob, 4),
        "calibrated_prob_down": round(1 - cal_prob, 4),
        "shap_base_value": round(base_val, 4),
        "top_drivers": drivers[:8],
        "all_contributions": drivers,
        "rationale": rationale,
    }
