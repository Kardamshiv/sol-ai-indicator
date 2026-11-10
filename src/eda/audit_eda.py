"""Dataset audit + EDA on TRAINING DATA ONLY (project steps 3 and 5).

Run from repo root:   python -m src.eda.audit_eda
Writes to results/eda/ and results/split_boundaries.json.

Leakage rules followed here
  * Rows after the training period (validation and test) are dropped BEFORE any statistic is computed.
  * The last HORIZON training rows are also dropped, because their labels look into the validation period.
  * Nothing here fits or tunes a model, so the test period stays untouched for the final evaluation.
"""
import datetime, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import *
from src.features.build import FEATURES
from src.data.split import build_dataset, split_bounds, save_boundaries

OUT = ROOT / "results" / "eda"
IV_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}[INTERVAL]
CLASS = {0: "Down", 1: "Neutral", 2: "Up"}
RAW = ["open", "high", "low", "close", "volume", "trades", "taker_buy_base"]
sections = []


def py(v):
    return int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig)


def add(fig, title, facts):
    sections.append(f"## {title}\n\n![{title}]({fig})\n\n" + "\n".join(f"- {f}" for f in facts) +
                    "\n\n**Interpretation (write 1-2 sentences after viewing the figure):** _TODO_\n")


DICT = {
 "open_time": ("int, ms since epoch (UTC)", "Candle open time"),
 "time": ("datetime, UTC", "Candle open time as a timestamp"),
 "open": ("USDT", "First trade price in the candle"), "high": ("USDT", "Highest price in the candle"),
 "low": ("USDT", "Lowest price in the candle"), "close": ("USDT", "Last trade price (known only when the candle closes)"),
 "volume": ("SOL (base asset)", "Traded volume in the candle"), "close_time": ("int, ms since epoch (UTC)", "Candle close time"),
 "quote_volume": ("USDT", "Traded value in the candle"), "trades": ("count", "Number of trades in the candle"),
 "taker_buy_base": ("SOL", "Volume bought by aggressive (taker) buyers"), "taker_buy_quote": ("USDT", "Value bought by takers"),
 "ret_1": ("fraction", "Close-to-close return over 1 candle"), "ret_4": ("fraction", "Return over 4 candles (1 hour)"),
 "ret_16": ("fraction", "Return over 16 candles (4 hours)"), "ret_96": ("fraction", "Return over 96 candles (1 day)"),
 "vol_16": ("fraction", "Std of 1-candle returns over the last 16 candles"), "vol_96": ("fraction", "Std of 1-candle returns over the last 96 candles"),
 "atr_14": ("fraction of close", "14-candle average true range divided by close"), "rsi_14": ("0 to 100", "Relative strength index, 14 candles"),
 "macd_hist": ("fraction of close", "MACD(12,26) minus its 9-candle signal line, divided by close"),
 "bb_width": ("fraction", "Bollinger band width (4 x std over 20 candles) relative to the 20-candle mean"),
 "bb_pos": ("std units", "Close minus 20-candle mean, in units of 2 x std"), "vol_z": ("z-score", "Volume vs its 96-candle mean and std"),
 "taker_ratio": ("0 to 1", "Taker-buy volume divided by total volume (buying pressure)"),
 "trades_z": ("z-score", "Trade count vs its 96-candle mean and std"), "dist_vwap": ("fraction", "Close divided by 96-candle VWAP, minus 1"),
 "hour_sin": ("-1 to 1", "Sine of UTC time of day (cyclical)"), "hour_cos": ("-1 to 1", "Cosine of UTC time of day (cyclical)"),
 "dow": ("0 to 6", "UTC day of week, 0 = Monday"),
 "fwd": ("fraction", "TARGET SOURCE: close HORIZON candles ahead divided by current close, minus 1 (training only)"),
 "y": ("0/1/2", "TARGET: 0 = Down, 1 = Neutral (inside +/-THRESH), 2 = Up (training only)"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", parse_dates=["time"])
    X, d = build_dataset(df)
    b = split_bounds(len(d))
    bounds = save_boundaries(d)
    train_end = d["time"].iloc[b["train"][1] - 1]

    # ---- keep ONLY the training period from here on
    raw = df[df["time"] <= train_end].copy()
    Xtr = X.loc[raw.index]
    tr = d.iloc[:b["train"][1] - HORIZON].copy()          # drop last HORIZON rows (labels look into validation)
    del df, X, d
    assert raw["time"].max() <= train_end and tr["time"].max() <= train_end
    n_tr = len(tr)

    # ================= AUDIT (step 3) =================
    dt = raw["open_time"].diff().dropna()
    imp = {"high < low": (raw.high < raw.low).sum(),
           "close outside [low, high]": ((raw.close < raw.low) | (raw.close > raw.high)).sum(),
           "open outside [low, high]": ((raw.open < raw.low) | (raw.open > raw.high)).sum(),
           "non-positive price": (raw[["open", "high", "low", "close"]] <= 0).any(axis=1).sum(),
           "negative volume": (raw.volume < 0).sum(), "zero volume": (raw.volume == 0).sum(),
           "zero trades": (raw.trades == 0).sum(), "taker_buy > volume": (raw.taker_buy_base > raw.volume + 1e-9).sum()}
    r1 = raw.close.pct_change().dropna()
    mad = (r1 - r1.median()).abs().median()
    outl = int(((r1 - r1.median()).abs() / (1.4826 * mad + 1e-12) > 8).sum())
    a = dict(rows=len(raw), start=str(raw.time.min()), end=str(raw.time.max()),
             duplicate_open_time=int(raw.open_time.duplicated().sum()), gaps=int((dt > IV_MS).sum()),
             largest_gap_candles=int(dt.max() // IV_MS) if len(dt) else 0, out_of_order=int((dt <= 0).sum()),
             missing_raw={c: int(raw[c].isna().sum()) for c in RAW}, impossible_values={k: int(v) for k, v in imp.items()},
             outlier_candles_robust_z_gt_8=outl)
    json.dump(a, open(OUT / "audit_summary.json", "w"), indent=2)
    miss = pd.DataFrame({"raw_missing": raw[RAW].isna().sum()}).reindex(RAW + FEATURES).fillna(0).astype(int)
    miss["feature_nan_warmup"] = Xtr.isna().sum().reindex(miss.index).fillna(0).astype(int)
    miss.to_csv(OUT / "missing_summary.csv")
    cols = [c for c in raw.columns if c in DICT] + [f for f in FEATURES] + ["fwd", "y"]
    pd.DataFrame([(c,) + DICT.get(c, ("", "")) for c in cols], columns=["name", "unit_or_type", "description"]).to_csv(OUT / "data_dictionary.csv", index=False)
    undocumented = [f for f in FEATURES if f not in DICT]
    if undocumented: print("WARNING: undocumented features:", undocumented)

    # ================= EDA (step 5), training rows only =================
    cnt = tr["y"].astype(int).value_counts().reindex([0, 1, 2]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([CLASS[i] for i in cnt.index], cnt.values, color=["#d9534f", "#999999", "#2a9d8f"])
    for i, v in enumerate(cnt.values): ax.text(i, v, f"{v:,}\n{v / cnt.sum():.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, cnt.max() * 1.2); ax.set_ylabel("Candles (training period)")
    ax.set_title(f"Target classes: {HORIZON}-candle horizon, dead zone +/-{THRESH:.2%}")
    save(fig, "01_target_classes.png")
    clear = tr[tr.y != 1]
    add("01_target_classes.png", "Target class counts",
        [f"Training candles analysed: {n_tr:,}", f"Share per class: Down {cnt[0] / cnt.sum():.1%}, Neutral {cnt[1] / cnt.sum():.1%}, Up {cnt[2] / cnt.sum():.1%}",
         f"Clear moves (Up or Down): {len(clear) / n_tr:.1%} of candles; Up share among them: {(clear.y == 2).mean():.3f} (this is the 'always Up' direction baseline)"])

    lo, hi = tr.fwd.quantile([.005, .995])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(tr.fwd.clip(lo, hi), bins=80, color="#4c72b0")
    for s in (-THRESH, THRESH): ax.axvline(s, color="k", ls="--", lw=1)
    ax.set_xlabel(f"{HORIZON}-candle forward return (clipped to the 0.5-99.5 percentile)"); ax.set_ylabel("Candles")
    ax.set_title("Distribution of the forward return (dashed = dead zone)")
    save(fig, "02_forward_return_hist.png")
    add("02_forward_return_hist.png", "Forward return distribution",
        [f"Mean {tr.fwd.mean():+.4%}, std {tr.fwd.std():.4%}, skew {tr.fwd.skew():.2f}, excess kurtosis {tr.fwd.kurt():.2f}",
         f"Inside the dead zone (Neutral): {(tr.fwd.abs() <= THRESH).mean():.1%}; 0.5th / 99.5th percentile: {lo:+.3%} / {hi:+.3%}"])

    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    heavy = []
    for ax, f in zip(axes.ravel(), FEATURES):
        v = tr[f]; l, h = v.quantile([.01, .99])
        ax.hist(v.clip(l, h), bins=50, color="#55a868"); ax.set_title(f, fontsize=10); ax.tick_params(labelsize=7)
        if abs(v.skew()) > 2: heavy.append(f)
    for ax in axes.ravel()[len(FEATURES):]: ax.axis("off")
    fig.suptitle("Feature distributions (training period, clipped to the 1-99 percentile)")
    save(fig, "03_feature_distributions.png")
    add("03_feature_distributions.png", "Feature distributions",
        [f"Features analysed: {len(FEATURES)}; features with |skew| > 2 (heavy tails): {', '.join(heavy) if heavy else 'none'}",
         "Tree models need no scaling; the logistic-regression baseline standardises features inside the pipeline."])

    cm = tr[FEATURES].rank().corr()
    cm.to_csv(OUT / "feature_correlations.csv")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=8)
    fig.colorbar(im, label="Spearman correlation"); ax.set_title("Feature-feature correlation (training period)")
    save(fig, "04_feature_correlation.png")
    pairs = sorted(((abs(cm.iloc[i, j]), FEATURES[i], FEATURES[j], cm.iloc[i, j]) for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))), reverse=True)[:5]
    add("04_feature_correlation.png", "Feature correlations (redundancy)",
        ["Most correlated pairs: " + "; ".join(f"{x} and {y} ({c:+.2f})" for _, x, y, c in pairs)])

    ct = tr[FEATURES].rank().corrwith(tr["fwd"].rank()).sort_values()
    ct.to_csv(OUT / "feature_target_correlation.csv", header=["spearman_with_forward_return"])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.barh(ct.index, ct.values, color=np.where(ct.values >= 0, "#2a9d8f", "#d9534f"))
    ax.set_xlabel("Spearman correlation with the forward return"); ax.set_title("Feature vs target (training period)")
    save(fig, "05_feature_target_correlation.png")
    add("05_feature_target_correlation.png", "Feature vs target relationship",
        [f"Strongest: {ct.abs().idxmax()} ({ct[ct.abs().idxmax()]:+.3f}); median |correlation| {ct.abs().median():.3f}",
         "Values near zero are expected for short-horizon returns; a tree model can still use non-linear structure. Correlation is not causation."])

    mon = tr["time"].dt.strftime("%Y-%m")
    up_m = tr[tr.y != 1].groupby(mon[tr.y != 1])["y"].apply(lambda s: (s == 2).mean())
    vol_m, tk_m, ne_m = tr.groupby(mon)["vol_96"].mean(), tr.groupby(mon)["taker_ratio"].mean(), tr.groupby(mon)["y"].apply(lambda s: (s == 1).mean())
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(vol_m.index, vol_m.values, marker="o"); axes[0].set_ylabel("Mean vol_96")
    axes[1].plot(up_m.index, up_m.values, marker="o", color="#2a9d8f"); axes[1].axhline(0.5, color="k", ls="--", lw=1); axes[1].set_ylabel("Up share (clear moves)")
    axes[2].plot(ne_m.index, ne_m.values, marker="o", color="#999999"); axes[2].set_ylabel("Neutral share")
    axes[2].tick_params(axis="x", rotation=90); axes[0].set_title("Drift by month (training period)")
    save(fig, "06_drift_by_month.png")
    add("06_drift_by_month.png", "Drift over time",
        [f"Months covered: {len(vol_m)}; mean vol_96 ranges {vol_m.min():.4f} to {vol_m.max():.4f}",
         f"Up share among clear moves ranges {up_m.min():.3f} to {up_m.max():.3f} (0.5 = balanced)", f"Neutral share ranges {ne_m.min():.3f} to {ne_m.max():.3f}",
         "Changing volatility or balance across months means the market regime shifts; this limits how far results generalise."])

    hr, wd = tr["time"].dt.hour, tr["time"].dt.dayofweek
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    facts = []
    for ax, key, name in ((axes[0], hr, "UTC hour of day"), (axes[1], wd, "UTC weekday (0 = Monday)")):
        up = (clear.y == 2).groupby(key[clear.index]).mean(); ab = tr.fwd.abs().groupby(key).mean(); n = clear.groupby(key[clear.index]).size()
        ax.bar(up.index, up.values, color="#4c72b0"); ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_ylim(max(0, up.min() - 0.05), up.max() + 0.05); ax.set_xlabel(name); ax.set_ylabel("Up share among clear moves")
        t = ax.twinx(); t.plot(ab.index, ab.values, color="#c44e52", marker="o"); t.set_ylabel(f"Mean |{HORIZON}-candle return|")
        facts.append(f"By {name}: highest Up share {up.idxmax()} ({up.max():.3f}), lowest {up.idxmin()} ({up.min():.3f}); smallest group has {n.min():,} clear moves "
                     f"(standard error about {0.5 / np.sqrt(n.min()):.3f}); most volatile group {ab.idxmax()}")
    fig.suptitle("Subgroup differences (training period)")
    save(fig, "07_subgroups_hour_weekday.png")
    add("07_subgroups_hour_weekday.png", "Subgroup differences by time", facts + ["Differences smaller than about two standard errors are likely noise."])

    # ================= REPORT + DATASET CARD =================
    (OUT / "eda_report.md").write_text("# EDA report (training period only)\n\nGenerated by `python -m src.eda.audit_eda`. "
        "Validation and test periods were not analysed.\n\n" + "\n".join(sections), encoding="utf-8")
    card = f"""# Dataset card: {SYMBOL} {INTERVAL} candles

| Field | Value |
|---|---|
| Source | Binance public market-data API, klines endpoint ({BASE_URL}) |
| Access date | {datetime.date.today().isoformat()} |
| Citation | TO FILL: cite the Binance API documentation page you used, with the access date |
| License / terms | TO FILL: read Binance's market-data terms of use before submission; raw data is not redistributed in the repo (data/ is git-ignored) |
| Unit of observation | One closed {INTERVAL} candle |
| Period analysed here | {a['start']} to {a['end']} (training portion only) |
| Training rows after cleaning | {n_tr:,} |
| Target | Direction of the close {HORIZON} candles ahead; Neutral if the move is inside +/-{THRESH:.2%} |
| Class balance (training) | Down {cnt[0] / cnt.sum():.1%}, Neutral {cnt[1] / cnt.sum():.1%}, Up {cnt[2] / cnt.sum():.1%}; Up share among clear moves {(clear.y == 2).mean():.3f} |
| Features | {len(FEATURES)}, all from closed candles; see data_dictionary.csv |
| Privacy | Aggregated public market data, no personal data |
| Target proxies | None known: every feature uses only data up to the candle close |

## Audit summary (training portion)
- Duplicate candles: {a['duplicate_open_time']}; gaps: {a['gaps']} (largest {a['largest_gap_candles']} candles); out-of-order timestamps: {a['out_of_order']}
- Missing raw values: {sum(a['missing_raw'].values())}
- Impossible values: {', '.join(f'{k}: {v}' for k, v in a['impossible_values'].items())}
- Outlier candles (robust z > 8 on 1-candle return): {a['outlier_candles_robust_z_gt_8']}; kept, because extreme moves are real market events

## Split protocol
Chronological 70% / 15% / 15% with an embargo of {HORIZON} candles; boundaries are saved in results/split_boundaries.json.
The test period was not analysed by this script and is reserved for the final one-shot evaluation.

## Known limitations
Single exchange and single asset, 24/7 market with regime changes, no news or order-book data, results depend on the period covered.
"""
    (OUT / "dataset_card.md").write_text(card, encoding="utf-8")

    print("Training rows analysed:", f"{n_tr:,}", "| period:", a["start"], "->", a["end"])
    print("Frozen split:", {k: (bounds[k]["rows"], bounds[k]["start_time"][:10], bounds[k]["end_time"][:10]) for k in ("train", "val", "test")})
    print("Audit:", {k: v for k, v in a.items() if k in ("duplicate_open_time", "gaps", "out_of_order", "outlier_candles_robust_z_gt_8")})
    print("Saved figures, tables, dataset_card.md and eda_report.md to", OUT)


if __name__ == "__main__":
    main()
