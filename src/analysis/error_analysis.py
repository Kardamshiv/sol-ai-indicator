"""Error analysis, explainability and robustness (PDF step 11).  TRAINING + VALIDATION data only.

Run from repo root, after tune.py and fit_calibration:     python -m src.analysis.error_analysis
Outputs -> results/analysis/ (csv tables, 2 figures, error_analysis_report.md)

What it does
  1. Errors of the LIVE signals (calibrated cutoff) by time of day, weekday, volatility, trend, predicted side and period
  2. Representative correct and wrong signals, and the confusion matrix on clear moves
  3. Permutation importance (which features the model really uses)
  4. Ablation by feature group, sensitivity to the random seed, sensitivity to noise in the inputs
No test data is read: rows after the validation period are dropped before anything runs.
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from src.config import *
from src.features.build import FEATURES
from src.data.split import build_dataset, split_bounds
from src.models import DirectionModel
from src.train.fit_calibration import logit

OUT = ROOT / "results" / "analysis"
N_FOLDS = 6
GROUPS = {"time": ["hour_sin", "hour_cos", "dow"], "flow": ["vol_z", "taker_ratio", "trades_z"],
          "classic indicators": ["rsi_14", "macd_hist", "bb_width", "bb_pos", "dist_vwap"],
          "volatility": ["vol_16", "vol_96", "atr_14"], "returns": ["ret_1", "ret_4", "ret_16", "ret_96"]}
sections = []


def auc(te, p):
    k = (te["y"] != 1).values; yt = (te["y"].values[k] == 2).astype(int)
    return roc_auc_score(yt, p[k]) if 0 < yt.sum() < len(yt) else np.nan


def fit_folds(dev, folds, feats=FEATURES, params=None):
    out = []
    for a, b in folds:
        tr, te = dev.iloc[a], dev.iloc[b]
        out.append((DirectionModel("lgbm", params).fit(tr[feats], tr["y"].astype(int)), tr, te))
    return out


def md_table(df):
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values])


def add(title, body, figure=None):
    sections.append(f"## {title}\n\n" + (f"![{title}]({figure})\n\n" if figure else "") + body +
                    "\n\n**Interpretation (write 1-2 sentences):** _TODO_\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv", parse_dates=["time"])
    _, d = build_dataset(df); b = split_bounds(len(d)); test_start = d["time"].iloc[b["test"][0]]
    dev = d.iloc[:b["val"][1]].copy(); del df, d
    assert dev["time"].max() < test_start
    print(f"Using {len(dev):,} rows up to {dev['time'].iloc[-1]}; the test period is not used.")
    folds = list(TimeSeriesSplit(n_splits=N_FOLDS, gap=HORIZON).split(dev)); folds5 = folds[1:]   # fold 1 trains on too little data
    rng = np.random.default_rng(SEED)

    # ---------- 1. subgroup errors of the live signals
    cal = json.load(open(MODEL_DIR / "calibration.json"))
    oof = pd.read_csv(ROOT / "results" / "oof_predictions.csv", parse_dates=["time"]); oof = oof[oof.fold >= 2]
    oof = oof.merge(dev[["time", "vol_96", "ret_96"]], on="time", how="left")
    q = 1 / (1 + np.exp(-(cal["coef"] * logit(oof.p_default.values) + cal["intercept"])))
    oof["conf"], oof["dirn"] = np.maximum(q, 1 - q), np.where(q > 0.5, 1, -1)
    oof["sig"] = np.where(oof.conf >= cal["cutoff_calibrated"], oof.dirn, 0)
    oof["hit"] = (np.sign(oof.fwd) == oof.dirn).astype(float); oof["net"] = oof.dirn * oof.fwd - COST
    S = oof[oof.sig != 0].copy()
    lv = {"UTC hours": pd.cut(oof.time.dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str),
          "Weekday": np.where(oof.time.dt.day_of_week >= 5, "weekend", "weekday"),
          "Volatility (vol_96)": pd.qcut(oof.vol_96, 3, labels=["low", "mid", "high"]).astype(str),
          "Trend over last day": np.where(oof.ret_96 > 0, "after a rise", "after a fall"),
          "Predicted side": np.where(oof.sig > 0, "Up", "Down"), "Period (fold)": oof.fold.astype(str)}
    rows = []
    for g, v in lv.items():
        v = pd.Series(np.asarray(v), index=oof.index).loc[S.index]
        for name, grp in S.groupby(v):
            h = grp.hit.mean(); rows.append(dict(group=g, level=name, n=len(grp), hit=round(h, 3), se=round(float(np.sqrt(h * (1 - h) / len(grp))), 3), net_per_trade=round(grp.net.mean(), 5)))
    sub = pd.DataFrame(rows); sub.to_csv(OUT / "subgroup_accuracy.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7)); facts = [f"Live-rule signals analysed: {len(S):,}; overall hit rate {S.hit.mean():.3f}"]
    for ax, (g, t) in zip(axes.ravel(), sub.groupby("group", sort=False)):
        ax.bar(t.level, t.hit, yerr=1.96 * t.se, capsize=3, color="#4c72b0"); ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_ylim(0.35, 0.75); ax.set_title(g, fontsize=10); ax.set_ylabel("Hit rate"); ax.tick_params(axis="x", labelsize=8)
        lo, hi = t.loc[t.hit.idxmin()], t.loc[t.hit.idxmax()]
        facts.append(f"{g}: lowest {lo.level} ({lo.hit:.3f}, n={lo.n}), highest {hi.level} ({hi.hit:.3f}, n={hi.n})")
    fig.suptitle("Hit rate of live-rule signals by subgroup (bars = 95% range; dashed = coin flip)"); fig.tight_layout()
    fig.savefig(OUT / "errors_by_subgroup.png", dpi=130); plt.close(fig)
    add("Errors by subgroup", "\n".join(f"- {f}" for f in facts) + "\n- With six groups and several levels, extremes are partly chance; only gaps clearly larger than the bars are worth acting on.", "errors_by_subgroup.png")

    # ---------- 2. examples and confusion
    Sc = S[S.y != 1]
    cm = pd.crosstab(np.where(Sc.dirn > 0, "predicted Up", "predicted Down"), np.where(Sc.y == 2, "actual Up", "actual Down"))
    json.dump({r: {c: int(cm.loc[r, c]) for c in cm.columns} for r in cm.index}, open(OUT / "confusion_at_cutoff.json", "w"), indent=2)
    ex = S.assign(direction=np.where(S.dirn > 0, "Up", "Down"), confidence=S.conf.round(3), move=(S.fwd * 100).round(2).astype(str) + "%", time=S.time.dt.strftime("%Y-%m-%d %H:%M"))
    good = ex[ex.hit == 1].nlargest(5, "conf")[["time", "direction", "confidence", "move"]]; bad = ex[ex.hit == 0].nlargest(5, "conf")[["time", "direction", "confidence", "move"]]
    good.to_csv(OUT / "examples_correct.csv", index=False); bad.to_csv(OUT / "examples_wrong.csv", index=False)
    add("Representative cases and confusion", f"Confusion matrix (clear moves only, signals):\n\n{md_table(cm.reset_index())}\n\n"
        f"Share of signals followed by a Neutral move (inside +/-{THRESH:.2%}): {(S.y == 1).mean():.1%}\n\nMost confident CORRECT signals:\n\n{md_table(good)}\n\nMost confident WRONG signals:\n\n{md_table(bad)}")

    # ---------- 3. permutation importance (folds 2-6)
    fits = fit_folds(dev, folds5); base = [auc(te, m.predict_proba(te[FEATURES])[:, 2]) for m, _, te in fits]
    rows = []
    for (m, _, te), b0 in zip(fits, base):
        for f in FEATURES:
            drops = []
            for _ in range(5):
                Xp = te[FEATURES].copy(); Xp[f] = rng.permutation(Xp[f].values); drops.append(b0 - auc(te, m.predict_proba(Xp)[:, 2]))
            rows.append(dict(feature=f, auc_drop=np.mean(drops)))
    imp = pd.DataFrame(rows).groupby("feature").auc_drop.agg(["mean", "std"]).sort_values("mean"); imp.round(5).to_csv(OUT / "permutation_importance.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.barh(imp.index, imp["mean"], xerr=imp["std"], color="#55a868", capsize=2)
    ax.axvline(0, color="k", lw=1); ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled (mean over folds 2-6)"); ax.set_title("Permutation importance")
    fig.tight_layout(); fig.savefig(OUT / "permutation_importance.png", dpi=130); plt.close(fig)
    top = imp.sort_values("mean", ascending=False).head(3)
    add("Permutation importance", f"Baseline ROC-AUC (folds 2-6): {np.mean(base):.4f}. Largest drops: " + "; ".join(f"{i} ({r['mean']:.4f})" for i, r in top.iterrows()) +
        ".\n\nCaution: correlated features share credit (for example vol_16 and atr_14), so single-feature drops understate their importance; and importance is not causation.", "permutation_importance.png")

    # ---------- 4. ablation, seeds, noise
    res = {"all features": base}
    for g, cols in GROUPS.items():
        fe = [f for f in FEATURES if f not in cols]
        res[f"without {g}"] = [auc(te, m.predict_proba(te[fe])[:, 2]) for m, _, te in fit_folds(dev, folds5, fe)]
    ab = pd.DataFrame({k: dict(auc_mean=np.mean(v), auc_std=np.std(v), delta_vs_all=np.mean(v) - np.mean(base)) for k, v in res.items()}).T.round(4); ab.to_csv(OUT / "ablation.csv")
    seeds = {SEED: np.mean(base)}
    for s in (1, 2, 3, 4): seeds[s] = np.mean([auc(te, m.predict_proba(te[FEATURES])[:, 2]) for m, _, te in fit_folds(dev, folds5, params={"random_state": s})])
    sd = pd.Series(seeds); sd.to_csv(OUT / "seed_sensitivity.csv", header=["auc_mean_over_folds"])
    noise = {}
    for lvl in (0, 0.1, 0.25, 0.5, 1.0):
        noise[lvl] = np.mean([auc(te, m.predict_proba(te[FEATURES] + lvl * tr[FEATURES].std().values * rng.standard_normal((len(te), len(FEATURES))))[:, 2]) for m, tr, te in fits])
    nz = pd.Series(noise); nz.to_csv(OUT / "noise_sensitivity.csv", header=["auc_mean_over_folds"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(ab.index[::-1], ab.auc_mean[::-1], xerr=ab.auc_std[::-1], color="#c44e52", capsize=2); axes[0].set_xlim(0.49, 0.54); axes[0].axvline(0.5, color="k", ls="--", lw=1)
    axes[0].set_xlabel("ROC-AUC (folds 2-6)"); axes[0].set_title("Ablation by feature group")
    axes[1].plot(list(noise), list(noise.values()), "o-"); axes[1].axhline(0.5, color="k", ls="--", lw=1); axes[1].set_xlabel("Noise added to inputs (multiples of each feature's std)")
    axes[1].set_ylabel("ROC-AUC (folds 2-6)"); axes[1].set_title("Sensitivity to noise")
    fig.tight_layout(); fig.savefig(OUT / "robustness.png", dpi=130); plt.close(fig)
    worst = ab.drop("all features").delta_vs_all.idxmin()
    add("Ablation, seed and noise robustness", f"{md_table(ab.reset_index().rename(columns={'index': 'variant'}))}\n\nLargest AUC loss when removing a group: {worst} ({ab.loc[worst, 'delta_vs_all']:+.4f}).\n\n"
        f"Random seed: ROC-AUC {sd.mean():.4f} +/- {sd.std():.4f} over 5 seeds (range {sd.min():.4f} to {sd.max():.4f}).\n\n"
        f"Noise: ROC-AUC {nz[0]:.4f} with clean inputs, {nz[0.25]:.4f} at 0.25 std of noise, {nz[1.0]:.4f} at 1.0 std.", "robustness.png")

    (OUT / "error_analysis_report.md").write_text("# Error analysis and robustness (training + validation data only)\n\nGenerated by `python -m src.analysis.error_analysis`.\n\n" + "\n".join(sections), encoding="utf-8")
    print("Largest permutation drops:", {i: round(float(r['mean']), 4) for i, r in top.iterrows()})
    print("Ablation (AUC):", ab.auc_mean.to_dict()); print(f"Seeds: {sd.mean():.4f} +/- {sd.std():.4f} | noise 0 / 0.25 / 1.0: {nz[0]:.4f} / {nz[0.25]:.4f} / {nz[1.0]:.4f}")
    print("Saved tables, figures and error_analysis_report.md to", OUT)


if __name__ == "__main__":
    main()
