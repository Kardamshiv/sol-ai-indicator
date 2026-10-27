"""Single place for every experiment setting. Change here, rerun, document the result."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

SYMBOL = "SOLUSDT"
INTERVAL = "15m"
YEARS = 2                 # history length (compare 0.25 / 1 / 2 in your report)
HORIZON = 4               # candles ahead (4 x 15m = 1 hour)
THRESH = 0.002           # +-0.1% dead zone -> "neutral / no signal"
COST = 0.001              # round-trip fee + slippage assumption
SEED = 42

# data-api.binance.vision works when api.binance.com is blocked
BASE_URL = "https://data-api.binance.vision"

# Live market-data websocket (market data only, no API key)
WS_URL = "wss://data-stream.binance.vision"
