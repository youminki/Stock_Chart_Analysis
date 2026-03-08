from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """모든 전략이 구현해야 하는 최소 인터페이스."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력 OHLCV DataFrame을 받아 'signal' 컬럼이 포함된 DataFrame을 반환한다.

        signal 규칙:
        - 1: 매수
        - -1: 매도
        - 0: 관망
        """
        raise NotImplementedError
