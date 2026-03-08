from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class RSIReversionStrategy(BaseStrategy):
    """RSI 과매도 매수 / 과매수 매도 역추세 전략."""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        if period <= 1:
            raise ValueError("period는 2 이상이어야 합니다.")
        if not (0 < oversold < overbought < 100):
            raise ValueError("oversold < overbought 조건과 0~100 범위를 만족해야 합니다.")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        delta = out["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))

        out["rsi"] = rsi
        out["signal"] = 0

        out.loc[out["rsi"] <= self.oversold, "signal"] = 1
        out.loc[out["rsi"] >= self.overbought, "signal"] = -1
        out["signal"] = out["signal"].fillna(0).astype(int)
        return out
