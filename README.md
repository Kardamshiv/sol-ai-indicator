# BTC AI Trading Indicator

Model-driven live indicator for BTCUSDT (Option B: Python server + Lightweight Charts web app).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce (run from the repo root)
```bash
python -m src.data.download      # 1. downloads data/BTCUSDT_15m.csv
python -m src.train.train        # 2. trains, validates, saves models/
```
Settings (asset, timeframe, horizon, threshold, costs) live in `src/config.py`.

## Roadmap
- [x] Data download
- [x] Features, labels, baselines, walk-forward validation
- [ ] Probability calibration + ablation experiments
- [ ] FastAPI live server + signal log (SQLite)
- [ ] Lightweight Charts frontend
- [ ] Deployment, report, user guide
