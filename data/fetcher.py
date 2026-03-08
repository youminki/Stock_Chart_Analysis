from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class YahooFinanceDataFetcher:
    """Yahoo Finance(yfinance) 기반 무료 데이터 수집기."""

    def _import_yfinance(self):
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance가 설치되어 있지 않습니다. `pip install yfinance` 후 다시 실행하세요."
            ) from exc
        return yf

    def _normalize_download_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        return out

    def fetch_daily_ohlcv(self, ticker: str) -> pd.DataFrame:
        yf = self._import_yfinance()

        raw = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False, actions=False)
        if raw.empty:
            raise ValueError(f"티커 '{ticker}'의 일봉 데이터를 가져오지 못했습니다.")

        out = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        out = out[["date", "open", "high", "low", "close", "volume"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        out = out.sort_values("date").reset_index(drop=True)
        return out

    def fetch_prices_for_interval(
        self,
        ticker: str,
        interval: str = "1h",
        period: str | None = None,
        prepost: bool = False,
    ) -> pd.DataFrame:
        """
        변동성 계산용 종가 시계열 조회.
        Yahoo 제약:
        - intraday(예: 1h)는 최대 약 730일
        - 1d는 period=max 가능
        """
        yf = self._import_yfinance()

        if period is None:
            period = "730d" if interval.endswith("m") or interval.endswith("h") else "max"

        raw = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            prepost=prepost,
        )
        raw = self._normalize_download_columns(raw)

        if raw.empty or "Close" not in raw.columns:
            raise ValueError(
                f"티커 '{ticker}' interval='{interval}' 데이터 조회 실패. "
                "interval/period 조합을 확인하세요."
            )

        out = raw.reset_index()
        dt_col = "Datetime" if "Datetime" in out.columns else "Date"

        out = out.rename(columns={dt_col: "date", "Close": "close"})[["date", "close"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        out = out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        return out

    def fetch_ohlcv_for_interval(
        self,
        ticker: str,
        interval: str = "1h",
        period: str | None = None,
        prepost: bool = False,
    ) -> pd.DataFrame:
        yf = self._import_yfinance()

        if period is None:
            period = "730d" if interval.endswith("m") or interval.endswith("h") else "max"

        raw = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            prepost=prepost,
        )
        raw = self._normalize_download_columns(raw)

        if raw.empty:
            raise ValueError(
                f"티커 '{ticker}' interval='{interval}' 데이터 조회 실패. interval/period 조합을 확인하세요."
            )

        out = raw.reset_index()
        dt_col = "Datetime" if "Datetime" in out.columns else "Date"
        rename_map = {
            dt_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        out = out.rename(columns=rename_map)
        need = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in out.columns]
        if missing:
            raise ValueError(f"OHLCV 컬럼 누락: {missing}")

        out = out[need].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        out = out.dropna().sort_values("date").reset_index(drop=True)
        return out
