from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import BacktestConfig, Backtester
from strategies import SMACrossoverStrategy


def make_sample_data(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    returns = rng.normal(loc=0.0005, scale=0.015, size=n_days)
    close = 100.0 * np.cumprod(1.0 + returns)

    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0005, 0.01, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0005, 0.01, size=n_days))
    volume = rng.integers(100_000, 1_000_000, size=n_days)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    df = make_sample_data()

    strategy = SMACrossoverStrategy(short_window=20, long_window=60)
    config = BacktestConfig(
        initial_capital=10_000,
        fee_rate=0.001,
        stop_loss_pct=0.05,
        take_profit_pct=0.12,
    )

    backtester = Backtester(strategy=strategy, config=config)
    result = backtester.run(df)

    print("=== Metrics ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    print("\n=== Trade Log (head) ===")
    print(result["trade_log"].head(10).to_string(index=False))

    print("\n=== Equity Curve (tail) ===")
    print(result["equity_curve"].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
