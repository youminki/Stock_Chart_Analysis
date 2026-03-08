from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import YahooFinanceDataFetcher
from scripts.m7_1y_100usd_reference_backtest import BacktestCfg, M7_TICKERS, build_signals, ema
from scripts.m7_last1m_close_accuracy_reference import (
    build_weights,
    prev_trading_day,
    weighted_signal,
)

HORIZON_DAYS = 5


def add_trading_days(dates: list[pd.Timestamp], start_day: pd.Timestamp, n: int) -> pd.Timestamp | None:
    idx = None
    for i, d in enumerate(dates):
        if d == start_day:
            idx = i
            break
    if idx is None:
        return None
    tgt = idx + n
    if tgt >= len(dates):
        return None
    return dates[tgt]


def build_pred_lr_5d(last_close: float, e200: float, wsig: float, vol20: float, c: float) -> float:
    # 1일 추정치를 기반으로 5일로 스케일
    lr_1d = c * wsig * vol20
    if last_close < e200 and lr_1d > 0:
        lr_1d *= 0.4
    cap_1d = max(0.004, min(0.03, 2.2 * vol20))
    lr_1d = float(np.clip(lr_1d, -cap_1d, cap_1d))

    lr_5d = lr_1d * math.sqrt(HORIZON_DAYS)
    cap_5d = min(0.12, cap_1d * math.sqrt(HORIZON_DAYS) * 1.5)
    return float(np.clip(lr_5d, -cap_5d, cap_5d))


def tune_c_5d(ctx_hourly: pd.DataFrame, daily_df: pd.DataFrame, eval_start: pd.Timestamp, weights: dict[str, float]) -> float:
    d = daily_df[daily_df["date"] < eval_start].copy().reset_index(drop=True)
    if len(d) < 100:
        return 1.0

    calib_days = [pd.Timestamp(x).normalize() for x in d["date"].tail(80)]
    all_days = [pd.Timestamp(x).normalize() for x in d["date"]]

    sigs = build_signals(ctx_hourly)
    ws_all = weighted_signal(sigs, weights)
    e200_all = ema(ctx_hourly["close"], 200)
    lr_all = np.log(ctx_hourly["close"] / ctx_hourly["close"].shift(1))
    vol20_all = lr_all.rolling(20, min_periods=20).std(ddof=1)

    candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    best_c, best_mape = 1.0, float("inf")
    for c in candidates:
        errs = []
        for day in calib_days:
            pday = prev_trading_day(all_days, day)
            if pday is None:
                continue
            tday = add_trading_days(all_days, day, HORIZON_DAYS)
            if tday is None:
                continue

            ctx = ctx_hourly[ctx_hourly["date"] <= pday + pd.Timedelta(days=1)]
            if len(ctx) < 400:
                continue
            i = len(ctx) - 1
            last_close = float(ctx["close"].iloc[-1])
            e200 = float(e200_all.iloc[i])
            wsig = float(ws_all.iloc[i])
            vol20 = float(vol20_all.iloc[i]) if not np.isnan(vol20_all.iloc[i]) else 0.004
            pred_lr = build_pred_lr_5d(last_close, e200, wsig, vol20, c)
            pred_close = last_close * math.exp(pred_lr)

            row = daily_df[daily_df["date"].dt.normalize() == tday]
            if row.empty:
                continue
            actual = float(row["close"].iloc[0])
            errs.append(abs(pred_close - actual) / actual * 100.0)

        if errs:
            mape = float(np.mean(errs))
            if mape < best_mape:
                best_mape, best_c = mape, c
    return best_c


def main() -> None:
    load_dotenv(ROOT / ".env")
    fetcher = YahooFinanceDataFetcher()
    cfg = BacktestCfg()

    asof = pd.Timestamp(datetime.now().date())
    start_hourly = asof - pd.Timedelta(days=500)

    pred_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for ticker in M7_TICKERS:
        hourly = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1h", period="730d", prepost=True)
        hourly = hourly[(hourly["date"] >= start_hourly) & (hourly["date"] <= asof + pd.Timedelta(days=1))].copy()
        hourly = hourly.sort_values("date").reset_index(drop=True)

        daily = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1d")
        daily = daily.sort_values("date").reset_index(drop=True)
        if len(hourly) < 900 or len(daily) < 150:
            continue

        latest_daily = pd.Timestamp(daily["date"].max()).normalize()
        eval_start = latest_daily - pd.Timedelta(days=30)
        all_days = [pd.Timestamp(x).normalize() for x in daily["date"]]

        eval_days = [d for d in all_days if d >= eval_start]
        eval_days = [d for d in eval_days if add_trading_days(all_days, d, HORIZON_DAYS) is not None]

        # 평가 직전 1년으로 가중치 고정 (누수 방지)
        ctx_end = eval_start - pd.Timedelta(days=1)
        ctx_start = ctx_end - pd.Timedelta(days=365)
        hourly_ctx = hourly[(hourly["date"] >= ctx_start) & (hourly["date"] <= ctx_end + pd.Timedelta(days=1))].copy()
        weights = build_weights(hourly_ctx, cfg)
        c_scale = tune_c_5d(hourly_ctx, daily, eval_start, weights)

        for day in eval_days:
            pday = prev_trading_day(all_days, day)
            if pday is None:
                continue
            target_day = add_trading_days(all_days, day, HORIZON_DAYS)
            if target_day is None:
                continue

            ctx = hourly[
                (hourly["date"] >= pday - pd.Timedelta(days=365)) & (hourly["date"] <= pday + pd.Timedelta(days=1))
            ].copy()
            if len(ctx) < 400:
                continue

            sigs = build_signals(ctx)
            ws = weighted_signal(sigs, weights)
            e200 = ema(ctx["close"], 200)
            lr = np.log(ctx["close"] / ctx["close"].shift(1))
            vol20 = lr.rolling(20, min_periods=20).std(ddof=1)

            base_close = float(ctx["close"].iloc[-1])
            pred_lr = build_pred_lr_5d(
                last_close=base_close,
                e200=float(e200.iloc[-1]),
                wsig=float(ws.iloc[-1]),
                vol20=float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else 0.004,
                c=c_scale,
            )
            pred_close = float(base_close * math.exp(pred_lr))

            target_row = daily[daily["date"].dt.normalize() == target_day]
            if target_row.empty:
                continue
            actual_close = float(target_row["close"].iloc[0])

            abs_error_pct = abs(pred_close - actual_close) / actual_close * 100.0
            pred_ret = pred_close / base_close - 1.0
            actual_ret = actual_close / base_close - 1.0
            hit = int(np.sign(pred_ret) == np.sign(actual_ret))

            pred_rows.append(
                {
                    "ticker": ticker,
                    "pred_date": str(day.date()),
                    "asof_prev_day": str(pday.date()),
                    "target_date_5d": str(target_day.date()),
                    "base_close": base_close,
                    "pred_close_5d": pred_close,
                    "actual_close_5d": actual_close,
                    "abs_error_pct": abs_error_pct,
                    "direction_hit": hit,
                    "c_scale": c_scale,
                }
            )

        tdf = pd.DataFrame([x for x in pred_rows if x["ticker"] == ticker])
        if tdf.empty:
            continue
        summary_rows.append(
            {
                "ticker": ticker,
                "n_preds": int(len(tdf)),
                "mean_abs_error_pct": float(tdf["abs_error_pct"].mean()),
                "median_abs_error_pct": float(tdf["abs_error_pct"].median()),
                "max_abs_error_pct": float(tdf["abs_error_pct"].max()),
                "directional_accuracy_pct": float(tdf["direction_hit"].mean() * 100.0),
                "c_scale": c_scale,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_abs_error_pct").reset_index(drop=True)
    if not pred_df.empty:
        overall = pd.DataFrame(
            [
                {
                    "ticker": "ALL",
                    "n_preds": int(len(pred_df)),
                    "mean_abs_error_pct": float(pred_df["abs_error_pct"].mean()),
                    "median_abs_error_pct": float(pred_df["abs_error_pct"].median()),
                    "max_abs_error_pct": float(pred_df["abs_error_pct"].max()),
                    "directional_accuracy_pct": float(pred_df["direction_hit"].mean() * 100.0),
                    "c_scale": np.nan,
                }
            ]
        )
        summary_df = pd.concat([summary_df, overall], ignore_index=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof.date())
    pred_path = out_dir / f"m7_1h_last1m_5d_close_predictions_reference_{tag}.csv"
    summary_path = out_dir / f"m7_1h_last1m_5d_close_error_summary_reference_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=== Last 1M 5D-Ahead Close Prediction Error (Reference Strategy) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {pred_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
