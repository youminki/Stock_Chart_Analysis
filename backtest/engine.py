from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from strategies.base import BaseStrategy
from utils.metrics import calculate_performance_metrics
from utils.validation import validate_ohlcv_dataframe


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001  # 0.1%
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital은 0보다 커야 합니다.")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate는 0 이상 1 미만이어야 합니다.")
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct는 0보다 커야 합니다.")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct는 0보다 커야 합니다.")


class Backtester:
    """룰 기반 롱온리 시계열 백테스터."""

    def __init__(self, strategy: BaseStrategy, config: BacktestConfig | None = None) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        data = validate_ohlcv_dataframe(df)
        data = self.strategy.generate_signals(data)

        if "signal" not in data.columns:
            raise ValueError("전략 출력에 'signal' 컬럼이 없습니다.")

        cash = self.config.initial_capital
        shares = 0.0
        in_position = False

        entry_price = 0.0
        entry_date = None
        entry_equity = 0.0
        trade_id = 0

        trades: list[dict[str, Any]] = []
        equity_points: list[dict[str, Any]] = []

        for row in data.itertuples(index=False):
            date = row.date
            close_price = float(row.close)
            signal = int(row.signal)

            exit_price = None
            exit_reason = None

            if in_position:
                stop_price = (
                    entry_price * (1.0 - self.config.stop_loss_pct)
                    if self.config.stop_loss_pct is not None
                    else None
                )
                take_price = (
                    entry_price * (1.0 + self.config.take_profit_pct)
                    if self.config.take_profit_pct is not None
                    else None
                )

                # 동일 봉에서 손절/익절이 동시에 충족될 수 있으므로 보수적으로 손절 우선 처리.
                if stop_price is not None and float(row.low) <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                elif take_price is not None and float(row.high) >= take_price:
                    exit_price = take_price
                    exit_reason = "take_profit"
                elif signal == -1:
                    exit_price = close_price
                    exit_reason = "signal_exit"

                if exit_price is not None:
                    gross_value = shares * exit_price
                    exit_fee = gross_value * self.config.fee_rate
                    cash = gross_value - exit_fee

                    gross_pnl = gross_value - entry_equity
                    net_pnl = cash - entry_equity
                    holding_period = (pd.Timestamp(date) - pd.Timestamp(entry_date)).days

                    trade_id += 1
                    trades.append(
                        {
                            "trade_id": trade_id,
                            "entry_date": pd.Timestamp(entry_date),
                            "entry_price": entry_price,
                            "exit_date": pd.Timestamp(date),
                            "exit_price": float(exit_price),
                            "shares": shares,
                            "gross_pnl": float(gross_pnl),
                            "net_pnl": float(net_pnl),
                            "return_pct": float((cash / entry_equity - 1.0) * 100.0),
                            "holding_period": int(holding_period),
                            "exit_reason": exit_reason,
                        }
                    )

                    shares = 0.0
                    in_position = False
                    entry_price = 0.0
                    entry_date = None
                    entry_equity = 0.0

            if (not in_position) and signal == 1:
                buy_fee = cash * self.config.fee_rate
                investable = cash - buy_fee
                if investable > 0:
                    shares = investable / close_price
                    entry_price = close_price
                    entry_date = pd.Timestamp(date)
                    entry_equity = cash
                    cash = 0.0
                    in_position = True

            equity = shares * close_price if in_position else cash
            equity_points.append(
                {
                    "date": pd.Timestamp(date),
                    "equity": float(equity),
                    "cash": float(cash),
                    "position_size": float(shares),
                }
            )

        if in_position:
            last_row = data.iloc[-1]
            last_date = pd.Timestamp(last_row["date"])
            last_close = float(last_row["close"])

            gross_value = shares * last_close
            exit_fee = gross_value * self.config.fee_rate
            cash = gross_value - exit_fee

            gross_pnl = gross_value - entry_equity
            net_pnl = cash - entry_equity
            holding_period = (last_date - pd.Timestamp(entry_date)).days

            trade_id += 1
            trades.append(
                {
                    "trade_id": trade_id,
                    "entry_date": pd.Timestamp(entry_date),
                    "entry_price": entry_price,
                    "exit_date": last_date,
                    "exit_price": last_close,
                    "shares": shares,
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "return_pct": float((cash / entry_equity - 1.0) * 100.0),
                    "holding_period": int(holding_period),
                    "exit_reason": "force_exit_end",
                }
            )

            equity_points[-1]["equity"] = float(cash)
            equity_points[-1]["cash"] = float(cash)
            equity_points[-1]["position_size"] = 0.0

        trade_log = pd.DataFrame(trades)
        equity_curve = pd.DataFrame(equity_points)
        metrics = calculate_performance_metrics(
            equity_curve=equity_curve,
            trade_log=trade_log,
            initial_capital=self.config.initial_capital,
        )

        return {
            "trade_log": trade_log,
            "equity_curve": equity_curve,
            "metrics": metrics,
        }
