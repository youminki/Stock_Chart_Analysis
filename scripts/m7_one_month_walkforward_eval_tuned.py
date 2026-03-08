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
    M7_TICKERS,
    ForecastParams,
    backtest_long_only,
    build_feature_frame,
    build_strategy_signals,
    ema,
    eval_baseline,
    eval_forecast,
    predict_lr,
    tune_params,
    weight_strategies,
    weighted_signal,
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


def fit_once(hourly_df: pd.DataFrame) -> tuple[dict[str, float], ForecastParams]:
    signals = build_strategy_signals(hourly_df)
    split_train = int(len(hourly_df) * 0.6)

    metric_map = {
        name: backtest_long_only(hourly_df.loc[: split_train - 1, "close"], sig.loc[: split_train - 1])
        for name, sig in signals.items()
    }
    weights = weight_strategies(metric_map)

    w_sig = weighted_signal(signals, weights)
    feat = build_feature_frame(hourly_df, w_sig=w_sig, horizon=7)

    f_valid = feat.iloc[int(len(feat) * 0.6) : int(len(feat) * 0.8)].copy()
    mid = max(1, int(len(f_valid) * 0.5))
    f_tune = f_valid.iloc[:mid].copy()
    f_select = f_valid.iloc[mid:].copy()
    if f_select.empty:
        f_select = f_tune.copy()

    params = tune_params(f_tune, f_select)

    # baseline 대비 개선이 약하면 보수적으로 blend 축소
    m_sel = eval_forecast(f_select, params)
    b_sel = eval_baseline(f_select)
    if m_sel["mape_pct"] >= b_sel["mape_pct"]:
        params = ForecastParams(
            blend=0.0,
            c_mu5=params.c_mu5,
            c_mu20=params.c_mu20,
            c_wsig_vol=params.c_wsig_vol,
            c_rsi_vol=params.c_rsi_vol,
            c_rev20=params.c_rev20,
        )

    return weights, params


def build_row_now(df_hourly: pd.DataFrame, w_sig_series: pd.Series) -> pd.Series:
    close = df_hourly["close"]
    lr = np.log(close / close.shift(1)).dropna()
    e20 = ema(close, 20)
    rev20 = (close - e20) / e20.replace(0, np.nan)

    rsi14 = (
        100
        - (
            100
            / (
                1
                + (
                    close.diff().clip(lower=0).rolling(14, min_periods=14).mean()
                    / (-close.diff().clip(upper=0)).rolling(14, min_periods=14).mean().replace(0, np.nan)
                )
            )
        )
    ).fillna(50.0)

    return pd.Series(
        {
            "mu5": float(lr.tail(5).mean()),
            "mu20": float(lr.tail(20).mean()),
            "vol20": float(lr.tail(20).std(ddof=1)),
            "rsi14": float(rsi14.iloc[-1]),
            "rev20": float(rev20.iloc[-1]),
            "w_sig": float(w_sig_series.iloc[-1]),
        }
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    fetcher = TwelveDataFetcher(max_retries=8, retry_wait_sec=65)

    asof_today = pd.Timestamp(datetime.now().date())
    start_lookback = asof_today - pd.Timedelta(days=420)

    pred_rows = []
    summary_rows = []

    for ticker in M7_TICKERS:
        hourly = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1h")
        hourly = hourly[(hourly["date"] >= start_lookback) & (hourly["date"] <= asof_today + pd.Timedelta(days=1))].copy()
        hourly = hourly.sort_values("date").reset_index(drop=True)

        daily = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1day")
        daily = daily.sort_values("date").reset_index(drop=True)

        latest_daily = pd.Timestamp(daily["date"].max()).normalize()
        eval_start = latest_daily - pd.Timedelta(days=30)
        eval_days = sorted(set(pd.Timestamp(d).normalize() for d in daily["date"] if pd.Timestamp(d).normalize() >= eval_start))
        daily_dates = [pd.Timestamp(d).normalize() for d in daily["date"]]

        train_end = eval_days[0] - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=365)
        train_hourly = hourly[(hourly["date"] >= train_start) & (hourly["date"] <= train_end + pd.Timedelta(days=1))].copy()
        weights, params = fit_once(train_hourly)

        # 온라인 편향 보정: 직전 오차의 EWMA (alpha=0.25)
        bias_ewma = 0.0

        for pred_day in eval_days:
            prev_day = get_prev_trading_day(daily_dates, pred_day)
            if prev_day is None:
                continue

            ctx_start = prev_day - pd.Timedelta(days=365)
            ctx_hourly = hourly[(hourly["date"] >= ctx_start) & (hourly["date"] <= prev_day + pd.Timedelta(days=1))].copy()
            if len(ctx_hourly) < 500:
                continue

            signals_ctx = build_strategy_signals(ctx_hourly)
            w_sig_ctx = weighted_signal(signals_ctx, weights)
            row_now = build_row_now(ctx_hourly, w_sig_ctx)

            base_lr = predict_lr(row_now, params)
            # 극단값 억제: 최근 변동성 기준 동적 캡
            vol20 = float(row_now["vol20"])
            cap = max(0.006, min(0.03, 2.2 * vol20))
            pred_lr = float(np.clip(base_lr + bias_ewma, -cap, cap))

            last_close = float(ctx_hourly["close"].iloc[-1])
            pred_close = float(last_close * np.exp(pred_lr))

            actual_row = daily[daily["date"].dt.normalize() == pred_day]
            if actual_row.empty:
                continue
            actual_close = float(actual_row["close"].iloc[0])

            signed_err = (pred_close - actual_close) / actual_close
            abs_error_pct = abs(signed_err) * 100.0

            # 다음 날 예측에 사용할 bias 업데이트 (전날까지 정보만 사용)
            bias_ewma = 0.25 * (-signed_err) + 0.75 * bias_ewma

            pred_rows.append(
                {
                    "ticker": ticker,
                    "pred_date": str(pred_day.date()),
                    "asof_prev_day": str(prev_day.date()),
                    "pred_close": pred_close,
                    "actual_close": actual_close,
                    "abs_error_pct": abs_error_pct,
                    "bias_ewma": bias_ewma,
                }
            )

        tdf = pd.DataFrame([r for r in pred_rows if r["ticker"] == ticker])
        summary_rows.append(
            {
                "ticker": ticker,
                "n_preds": int(len(tdf)),
                "mean_abs_error_pct": float(tdf["abs_error_pct"].mean()),
                "median_abs_error_pct": float(tdf["abs_error_pct"].median()),
                "max_abs_error_pct": float(tdf["abs_error_pct"].max()),
                "blend": float(params.blend),
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
        "blend": np.nan,
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof_today.date())

    pred_path = out_dir / f"m7_1h_last1m_daily_close_predictions_tuned_{tag}.csv"
    summary_path = out_dir / f"m7_1h_last1m_daily_close_error_summary_tuned_{tag}.csv"

    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=== Last 1M Daily Close Prediction Error (M7, Tuned) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {pred_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
