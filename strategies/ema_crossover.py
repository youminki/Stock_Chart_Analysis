from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """지수이동평균 골든/데드 크로스 전략."""

    def __init__(self, short_span: int = 12, long_span: int = 26) -> None:
        if short_span <= 0 or long_span <= 0:
            raise ValueError("short_span과 long_span은 1 이상이어야 합니다.")
        if short_span >= long_span:
            raise ValueError("short_span은 long_span보다 작아야 합니다.")
        self.short_span = short_span
        self.long_span = long_span

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema_short"] = out["close"].ewm(span=self.short_span, adjust=False).mean()
        out["ema_long"] = out["close"].ewm(span=self.long_span, adjust=False).mean()

        prev_short = out["ema_short"].shift(1)
        prev_long = out["ema_long"].shift(1)

        out["signal"] = 0
        buy_cond = (out["ema_short"] > out["ema_long"]) & (prev_short <= prev_long)
        sell_cond = (out["ema_short"] < out["ema_long"]) & (prev_short >= prev_long)

        out.loc[buy_cond, "signal"] = 1
        out.loc[sell_cond, "signal"] = -1
        return out
