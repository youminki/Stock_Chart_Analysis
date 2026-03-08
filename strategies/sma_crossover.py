from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    """단순 이동평균선 골든/데드 크로스 전략."""

    def __init__(self, short_window: int = 20, long_window: int = 60) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("short_window와 long_window는 1 이상이어야 합니다.")
        if short_window >= long_window:
            raise ValueError("short_window는 long_window보다 작아야 합니다.")

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["sma_short"] = out["close"].rolling(window=self.short_window, min_periods=self.short_window).mean()
        out["sma_long"] = out["close"].rolling(window=self.long_window, min_periods=self.long_window).mean()

        out["signal"] = 0

        prev_short = out["sma_short"].shift(1)
        prev_long = out["sma_long"].shift(1)

        buy_cond = (out["sma_short"] > out["sma_long"]) & (prev_short <= prev_long)
        sell_cond = (out["sma_short"] < out["sma_long"]) & (prev_short >= prev_long)

        out.loc[buy_cond, "signal"] = 1
        out.loc[sell_cond, "signal"] = -1

        return out
