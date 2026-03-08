from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class BollingerReversionStrategy(BaseStrategy):
    """볼린저 밴드 하단 이탈 매수 / 상단 돌파 매도 역추세 전략."""

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        if window <= 1:
            raise ValueError("window는 2 이상이어야 합니다.")
        if num_std <= 0:
            raise ValueError("num_std는 0보다 커야 합니다.")
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        ma = out["close"].rolling(window=self.window, min_periods=self.window).mean()
        std = out["close"].rolling(window=self.window, min_periods=self.window).std(ddof=0)

        upper = ma + self.num_std * std
        lower = ma - self.num_std * std

        out["bb_mid"] = ma
        out["bb_upper"] = upper
        out["bb_lower"] = lower
        out["signal"] = 0

        out.loc[out["close"] <= out["bb_lower"], "signal"] = 1
        out.loc[out["close"] >= out["bb_upper"], "signal"] = -1

        out["signal"] = out["signal"].fillna(0).astype(int)
        return out
