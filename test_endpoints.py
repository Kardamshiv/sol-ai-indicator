from src.server.app import meta, health, analytics_model, analytics_comparison, papertrading, backtest_simulate
from src.server.app import candles

print("meta:", meta()["symbol"])
print("health:", health()["status"])
bs = backtest_simulate(cutoff=0.5346, cost=0.001)
print("backtest OK:", bs["total_trades"], "trades, Sharpe:", bs["sharpe_ratio"], "PF:", bs["profit_factor"])
am = analytics_model()
print("analytics_model OK, Accuracy:", am["classification_metrics"]["accuracy"])
ac = analytics_comparison()
print("analytics_comparison OK, models:", len(ac))
pt = papertrading()
print("papertrading OK, capital:", pt["starting_capital"])
print("candles loaded:", len(candles))
print("ALL BACKEND TESTS PASSED")
