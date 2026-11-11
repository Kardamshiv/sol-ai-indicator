# SOL AI Trading Indicator & Predictor

An end-to-end, scientifically validated machine learning indicator and live trading application for **SOLUSDT** (Option B: Python server + TradingView Lightweight Charts).

Developed according to the **IIT Patna IMPACT Lab Machine Learning Project Guidelines** (Steps 1–12) and the **Student Project Brief** (100-mark evaluation rubric).

---

## Key Features

1. **Strictly Non-Repainting Signals:** Signals are created strictly upon candle close and logged immutably into SQLite with `INSERT OR IGNORE`. Past arrows and confidence bars never move or change.
2. **Leakage-Free Protocol:** 70/15/15 chronological split with an embargo of 4 candles ($H=4$); all scaling and tuning restricted exclusively to training/validation data.
3. **Platt Probability Calibration:** Raw LightGBM leaf outputs are calibrated on out-of-fold predictions so that confidence directly corresponds to empirical win rate.
4. **Abstention & Dead Zone:** Predicts only on non-noise candles ($|\text{return}| > 0.20\%$), displaying *"No Clear Signal"* on ambiguous consolidation to avoid fee drag.
5. **Interactive Web Application:**
   - Real-time Binance WebSocket Kline streaming + REST backfill.
   - **Feature Explainability:** Live breakdown of top driving factors (RSI, 24h momentum, taker-buy order flow, VWAP distance).
   - **What-If Backtest Simulator:** Interactive confidence cutoff slider simulating trade counts, win rate vs Always-Up, and drawing a real-time equity curve.
   - **Signal Filtering & Export:** Filter recent signals by Up, Down, Hits, Misses, and download the full verified SQLite log as CSV.

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Reproduce Pipeline from Scratch
```bash
python -m src.data.download           # 1. Download 2-year 15m dataset (data/SOLUSDT_15m.csv)
python -m src.eda.audit_eda           # 2. Run data audit & EDA plots (results/eda/)
python -m src.train.tune              # 3. Expanding-window CV, tuning, baseline comparison
python -m src.train.fit_calibration   # 4. Fit out-of-fold Platt scaling (models/calibration.json)
python -m src.train.make_reference    # 5. Generate validation reference metrics (models/reference.json)
python -m src.analysis.error_analysis # 6. Run error analysis, subgroup breakdown, and ablations
```

### 3. Run the Live Web App
```bash
uvicorn src.server.app:app --port 8000
```
Open **http://localhost:8000** in your browser.

---

## Project Structure
```
sol-ai-indicator/
├── data/              # SOLUSDT_15m.csv and live SQLite log (signals.db)
├── docs/              # Problem definition, literature review, deployment & user guide
├── models/            # Saved model artifact (model.joblib), config.json, calibration.json
├── notebooks/         # 01_eda_and_data_audit.ipynb (EDA only, following guidelines)
├── reports/           # PROJECT_REPORT.md, PRESENTATION_SLIDES.md, backtest.png
├── results/           # Model comparison CSV, experiment logs, EDA and error analysis
├── src/               # Data, features, models, training, server
└── web/               # Lightweight Charts web application (index.html)
```

---

## Roadmap & Deliverables
- [x] Step 1: 1-Page Problem Definition & Hypothesis (`docs/problem_definition.md`)
- [x] Step 2: Literature Review with Comparative Table (`docs/literature_review.md`)
- [x] Step 3 & 5: Dataset Audit, Dataset Card, & EDA Notebook (`notebooks/01_eda_and_data_audit.ipynb`)
- [x] Step 4 & 6: Leakage-free Chronological Split & Causal Features (`src/features/build.py`)
- [x] Step 7 & 8: Baselines, Expanding-Window CV Tuning (`results/model_comparison.csv`)
- [x] Step 9 & 10: Calibration, Final Held-Out Evaluation, Cost Accounting
- [x] Step 11: Error Analysis, Permutation Importance, Ablations (`results/analysis/`)
- [x] Step 12: Comprehensive Final Report (`reports/PROJECT_REPORT.md`), Presentation Deck (`reports/PRESENTATION_SLIDES.md`), & Dockerfile
