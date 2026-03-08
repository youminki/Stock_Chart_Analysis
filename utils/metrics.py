from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_performance_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    initial_capital: float,
) -> dict[str, Any]:
    if equity_curve.empty:
        raise ValueError("equity_curve가 비어 있습니다.")

    eq = equity_curve.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")

    final_equity = float(eq["equity"].iloc[-1])
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0

    elapsed_days = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days, 1)
    years = elapsed_days / 365.0
    cagr = (final_equity / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily_returns = eq["equity"].pct_change().fillna(0.0)
    daily_std = float(daily_returns.std(ddof=1))
    sharpe = float(np.sqrt(252) * daily_returns.mean() / daily_std) if daily_std > 0 else 0.0

    running_max = eq["equity"].cummax()
    drawdown = eq["equity"] / running_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)

    num_trades = int(len(trade_log))
    if num_trades == 0:
        win_rate = 0.0
        profit_factor = 0.0
        avg_holding = 0.0
    else:
        wins = trade_log[trade_log["net_pnl"] > 0]
        losses = trade_log[trade_log["net_pnl"] < 0]

        win_rate = float(len(wins) / num_trades * 100.0)

        gross_profit = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
        gross_loss = float(losses["net_pnl"].sum()) if not losses.empty else 0.0
        if gross_loss < 0:
            profit_factor = gross_profit / abs(gross_loss)
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        avg_holding = float(trade_log["holding_period"].mean())

    return {
        "total_return_pct": total_return_pct,
        "cagr": float(cagr),
        "win_rate_pct": win_rate,
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe,
        "num_trades": num_trades,
        "avg_holding_period_days": avg_holding,
    }
