from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    """입력 데이터 형식/품질 검증 실패 예외."""


def validate_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    백테스트 입력용 OHLCV DataFrame 검증 후 안전한 사본을 반환.
    - 필수 컬럼 존재
    - 날짜 오름차순 정렬
    - 수치 컬럼 결측 없음
    - 가격 컬럼 양수
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("입력 데이터는 pandas DataFrame이어야 합니다.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"필수 컬럼이 없습니다: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if out["date"].isna().any():
        raise DataValidationError("date 컬럼에 파싱 불가능한 값이 있습니다.")

    if out["date"].duplicated().any():
        raise DataValidationError("date 컬럼에 중복 값이 있습니다.")

    if not out["date"].is_monotonic_increasing:
        raise DataValidationError("데이터는 date 오름차순으로 정렬되어 있어야 합니다.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    if out[numeric_cols].isna().any().any():
        raise DataValidationError("OHLCV 컬럼에 결측치가 있습니다.")

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("가격(open/high/low/close)은 0보다 커야 합니다.")

    if (out["volume"] < 0).any():
        raise DataValidationError("volume은 음수가 될 수 없습니다.")

    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError("high 값이 open/close/low보다 작은 행이 있습니다.")

    if (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise DataValidationError("low 값이 open/close/high보다 큰 행이 있습니다.")

    return out
