from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass
class TwelveDataFetcher:
    """Twelve Data 기반 데이터 수집기."""

    api_key: str | None = None
    base_url: str = "https://api.twelvedata.com"
    max_retries: int = 5
    retry_wait_sec: int = 65

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("TWELVE_DATA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Twelve Data API 키가 필요합니다. "
                "--api-key 옵션 또는 TWELVE_DATA_API_KEY 환경변수를 설정하세요."
            )

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {**params, "apikey": self.api_key}
        url = f"{self.base_url}/{endpoint}"

        last_error = None
        for attempt in range(self.max_retries + 1):
            resp = requests.get(url, params=query, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            if isinstance(payload, dict) and payload.get("status") == "error":
                code = str(payload.get("code", ""))
                msg = str(payload.get("message", "Unknown error"))

                if code == "429" and attempt < self.max_retries:
                    wait_sec = self._extract_wait_seconds(msg)
                    time.sleep(wait_sec)
                    continue

                last_error = ValueError(f"Twelve Data API 오류(code={code}): {msg}")
                break

            return payload

        if last_error is not None:
            raise last_error
        raise RuntimeError("Twelve Data 요청 실패: 알 수 없는 오류")

    def _extract_wait_seconds(self, message: str) -> int:
        lower = message.lower()
        if "next minute" in lower:
            return self.retry_wait_sec

        match = re.search(r"(\d+)\s*second", lower)
        if match:
            return max(int(match.group(1)), 1)

        return self.retry_wait_sec

    def fetch_daily_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self.fetch_ohlcv_for_interval(ticker=ticker, interval="1day")

    def fetch_prices_for_interval(
        self,
        ticker: str,
        interval: str = "1h",
        period: str | None = None,
    ) -> pd.DataFrame:
        _ = period
        ohlcv = self.fetch_ohlcv_for_interval(ticker=ticker, interval=interval)
        return ohlcv[["date", "close"]].copy()

    def fetch_ohlcv_for_interval(self, ticker: str, interval: str = "1h") -> pd.DataFrame:
        payload = self._request(
            "time_series",
            {
                "symbol": ticker.upper(),
                "interval": interval,
                "outputsize": 5000,
                "order": "asc",
                "format": "JSON",
            },
        )

        values = payload.get("values", [])
        if not values:
            raise ValueError(f"티커 '{ticker}' interval='{interval}' 데이터를 가져오지 못했습니다.")

        df = pd.DataFrame(values)
        need_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"응답에 필요한 OHLCV 컬럼이 없습니다: {missing}")

        for col in need_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={"datetime": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        out = df[["date", "open", "high", "low", "close", "volume"]].dropna()
        out = out.sort_values("date").reset_index(drop=True)
        return out

    def fetch_ohlcv_range(
        self,
        ticker: str,
        interval: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        chunk_days: int = 90,
    ) -> pd.DataFrame:
        """
        start_date~end_date 구간을 여러 청크로 나눠 조회한다.
        intraday에서 outputsize=5000 제한을 우회하기 위한 유틸.
        """
        if start_date >= end_date:
            raise ValueError("start_date는 end_date보다 이전이어야 합니다.")
        if chunk_days <= 0:
            raise ValueError("chunk_days는 1 이상이어야 합니다.")

        parts: list[pd.DataFrame] = []
        cur = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        while cur <= end:
            chunk_end = min(cur + pd.Timedelta(days=chunk_days), end)
            payload = self._request(
                "time_series",
                {
                    "symbol": ticker.upper(),
                    "interval": interval,
                    "start_date": cur.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_date": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "outputsize": 5000,
                    "order": "asc",
                    "format": "JSON",
                },
            )

            values = payload.get("values", [])
            if values:
                df = pd.DataFrame(values)
                need_cols = ["open", "high", "low", "close", "volume"]
                missing = [c for c in need_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"응답에 필요한 OHLCV 컬럼이 없습니다: {missing}")

                for col in need_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.rename(columns={"datetime": "date"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
                parts.append(df)

            cur = chunk_end + pd.Timedelta(minutes=1)

        if not parts:
            raise ValueError(
                f"티커 '{ticker}' interval='{interval}' 범위 조회 결과가 비어 있습니다. "
                f"(start={start_date}, end={end_date})"
            )

        out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["date"])
        out = out.sort_values("date").reset_index(drop=True)
        return out
