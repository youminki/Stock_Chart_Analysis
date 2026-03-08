from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import TwelveDataFetcher
from scripts.m7_weighted_1h_analysis import (
    ForecastParams,
    M7_TICKERS,
    build_feature_frame,
    build_strategy_signals,
    ema,
    eval_baseline,
    eval_forecast,
    predict_lr,
    tune_params,
    weight_strategies,
    weighted_signal,
    backtest_long_only,
)


def get_prev_trading_day(dates: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp | None:
    idx = None
    for i, d in enumerate(dates):
        if d == target:
            idx = i
            break
    if idx is None or idx == 0:
        return None
    return dates[idx - 1]


def build_row_now(df_hourly: pd.DataFrame, w_sig_series: pd.Series) -> pd.Series:
    close = df_hourly["close"]
    lr = np.log(close / close.shift(1)).dropna()
    e20 = ema(close, 20)
    rev20 = (close - e20) / e20.replace(0, np.nan)

    return pd.Series(
        {
            "mu5": float(lr.tail(5).mean()),
            "mu20": float(lr.tail(20).mean()),
            "vol20": float(lr.tail(20).std(ddof=1)),
            "rsi14": float(
                (100 - (100 / (1 + (close.diff().clip(lower=0).rolling(14, min_periods=14).mean() /
                    (-close.diff().clip(upper=0)).rolling(14, min_periods=14).mean().replace(0, np.nan))))).fillna(50.0).iloc[-1]
            ),
            "rev20": float(rev20.iloc[-1]),
            "w_sig": float(w_sig_series.iloc[-1]),
        }
    )


def fit_ticker_model(hourly_df: pd.DataFrame) -> tuple[dict[str, float], ForecastParams]:
    signals = build_strategy_signals(hourly_df)
    split_train = int(len(hourly_df) * 0.6)

    metric_map = {
        name: backtest_long_only(hourly_df.loc[: split_train - 1, "close"], sig.loc[: split_train - 1])
        for name, sig in signals.items()
    }
    weights = weight_strategies(metric_map)

    w_sig_series = weighted_signal(signals, weights)
    feat = build_feature_frame(hourly_df, w_sig=w_sig_series, horizon=7)
    f_valid = feat.iloc[int(len(feat) * 0.6) : int(len(feat) * 0.8)].copy()

    valid_mid = max(1, int(len(f_valid) * 0.5))
    f_tune = f_valid.iloc[:valid_mid].copy()
    f_select = f_valid.iloc[valid_mid:].copy()
    if f_select.empty:
        f_select = f_tune.copy()

    # 빠른 재학습을 위해 f_valid를 샘플링해 튜닝 계산량 축소
    step = max(1, len(f_tune) // 120)
    f_tune_fast = f_tune.iloc[::step].reset_index(drop=True)
    step2 = max(1, len(f_select) // 120)
    f_select_fast = f_select.iloc[::step2].reset_index(drop=True)

    params = tune_params(f_tune_fast, f_select_fast)
    model_sel = eval_forecast(f_select, params)
    base_sel = eval_baseline(f_select)

    # selection 구간에서 baseline보다 0.2%p 이상 개선되지 않으면 baseline 강제
    if model_sel["mape_pct"] > base_sel["mape_pct"] - 0.002:
        params = ForecastParams(blend=0.0, c_mu5=0.25, c_mu20=0.25, c_wsig_vol=0.0, c_rsi_vol=0.0, c_rev20=0.0)
    return weights, params


def main() -> None:
    load_dotenv(ROOT / ".env")
    fetcher = TwelveDataFetcher(max_retries=8, retry_wait_sec=65)

    asof_today = pd.Timestamp(datetime.now().date())
    start_lookback = asof_today - pd.Timedelta(days=420)

    pred_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for ticker in M7_TICKERS:
        hourly = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1h")
        hourly = hourly[(hourly["date"] >= start_lookback) & (hourly["date"] <= asof_today + pd.Timedelta(days=1))].copy()
        hourly = hourly.sort_values("date").reset_index(drop=True)

        daily = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1day")
        daily = daily.sort_values("date").reset_index(drop=True)

        if len(hourly) < 1200 or len(daily) < 100:
            raise ValueError(f"{ticker}: 데이터 부족 (hourly={len(hourly)}, daily={len(daily)})")

        latest_daily = pd.Timestamp(daily["date"].max()).normalize()
        eval_start = latest_daily - pd.Timedelta(days=30)

        eval_days = [pd.Timestamp(d).normalize() for d in daily["date"] if pd.Timestamp(d).normalize() >= eval_start]
        eval_days = sorted(set(eval_days))

        daily_dates = [pd.Timestamp(d).normalize() for d in daily["date"]]

        cached_weights = None
        cached_params = None

        for i, pred_day in enumerate(eval_days):
            prev_day = get_prev_trading_day(daily_dates, pred_day)
            if prev_day is None:
                continue

            ctx_start = prev_day - pd.Timedelta(days=365)
            ctx_hourly = hourly[(hourly["date"] >= ctx_start) & (hourly["date"] <= prev_day + pd.Timedelta(days=1))].copy()
            if len(ctx_hourly) < 700:
                continue

            # 워크포워드: 5거래일마다 재학습, 그 사이 파라미터 재사용
            if cached_weights is None or cached_params is None or (i % 5 == 0):
                cached_weights, cached_params = fit_ticker_model(ctx_hourly)
            weights, params = cached_weights, cached_params

            signals_ctx = build_strategy_signals(ctx_hourly)
            w_sig_ctx = weighted_signal(signals_ctx, weights)
            row_now = build_row_now(ctx_hourly, w_sig_ctx)
            pred_lr = predict_lr(row_now, params)
            last_close = float(ctx_hourly["close"].iloc[-1])
            pred_close = float(last_close * np.exp(pred_lr))

            actual_row = daily[daily["date"].dt.normalize() == pred_day]
            if actual_row.empty:
                continue
            actual_close = float(actual_row["close"].iloc[0])

            abs_error_pct = abs(pred_close - actual_close) / actual_close * 100.0

            pred_rows.append(
                {
                    "ticker": ticker,
                    "pred_date": str(pred_day.date()),
                    "asof_prev_day": str(prev_day.date()),
                    "blend": params.blend,
                    "pred_close": pred_close,
                    "actual_close": actual_close,
                    "abs_error_pct": abs_error_pct,
                }
            )

        ticker_df = pd.DataFrame([r for r in pred_rows if r["ticker"] == ticker])
        summary_rows.append(
            {
                "ticker": ticker,
                "n_preds": int(len(ticker_df)),
                "mean_abs_error_pct": float(ticker_df["abs_error_pct"].mean()),
                "median_abs_error_pct": float(ticker_df["abs_error_pct"].median()),
                "max_abs_error_pct": float(ticker_df["abs_error_pct"].max()),
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_abs_error_pct")

    overall = {
        "ticker": "ALL",
        "n_preds": int(len(pred_df)),
        "mean_abs_error_pct": float(pred_df["abs_error_pct"].mean()),
        "median_abs_error_pct": float(pred_df["abs_error_pct"].median()),
        "max_abs_error_pct": float(pred_df["abs_error_pct"].max()),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof_today.date())

    pred_path = out_dir / f"m7_1h_last1m_daily_close_predictions_{tag}.csv"
    summary_path = out_dir / f"m7_1h_last1m_daily_close_error_summary_{tag}.csv"

    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=== Last 1M Daily Close Prediction Error (M7) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {pred_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
