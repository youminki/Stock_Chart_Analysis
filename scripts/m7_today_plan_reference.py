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
    build_pred_lr,
    build_weights,
    prev_trading_day,
    tune_c,
    weighted_signal,
)


def action_from_weighted_signal(w_sig: float, threshold: float = 0.15) -> str:
    if w_sig >= threshold:
        return "BUY"
    if w_sig <= -threshold:
        return "SELL/REDUCE"
    return "HOLD"


def suggest_levels(last_close: float, action: str, vol20: float) -> tuple[float, float, float, float, float]:
    horizon = 7  # 1h bar 기준 대략 1거래일
    sigma_h = vol20 * math.sqrt(horizon)
    target_pct = min(max(1.8 * sigma_h, 0.01), 0.08)
    stop_pct = min(max(1.2 * sigma_h, 0.008), 0.05)

    if action == "BUY":
        entry = last_close * 0.998
        target = entry * (1.0 + target_pct)
        stop = entry * (1.0 - stop_pct)
    elif action == "SELL/REDUCE":
        entry = last_close * 1.002
        target = entry * (1.0 - target_pct)
        stop = entry * (1.0 + stop_pct)
    else:
        entry = last_close
        target = float("nan")
        stop = float("nan")
    return entry, target, stop, target_pct * 100.0, stop_pct * 100.0


def main() -> None:
    load_dotenv(ROOT / ".env")
    fetcher = YahooFinanceDataFetcher()
    cfg = BacktestCfg()

    asof = pd.Timestamp(datetime.now().date())
    start_hourly = asof - pd.Timedelta(days=460)
    rows: list[dict[str, object]] = []

    for ticker in M7_TICKERS:
        try:
            hourly = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1h", period="730d", prepost=True)
            hourly = hourly[(hourly["date"] >= start_hourly) & (hourly["date"] <= asof + pd.Timedelta(days=1))].copy()
            hourly = hourly.sort_values("date").reset_index(drop=True)

            daily = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1d")
            daily = daily.sort_values("date").reset_index(drop=True)
            if len(hourly) < 800 or len(daily) < 120:
                print(f"[WARN] {ticker}: insufficient rows hourly={len(hourly)} daily={len(daily)}")
                continue

            latest_daily = pd.Timestamp(daily["date"].max()).normalize()
            today_pred_date = asof.normalize()

            # 가중치/스케일은 최신 기준 사전 추정
            ctx_end = latest_daily
            ctx_start = ctx_end - pd.Timedelta(days=365)
            hourly_ctx = hourly[
                (hourly["date"] >= ctx_start) & (hourly["date"] <= ctx_end + pd.Timedelta(days=1))
            ].copy()
            weights = build_weights(hourly_ctx, cfg)

            eval_start = latest_daily - pd.Timedelta(days=30)
            c_scale = tune_c(hourly_ctx, daily, eval_start, weights)

            # 오늘 종가 예측은 전일 데이터까지만 사용 (no lookahead)
            all_days = [pd.Timestamp(x).normalize() for x in daily["date"]]
            if latest_daily < today_pred_date:
                pday = latest_daily
            else:
                pday = prev_trading_day(all_days, latest_daily)
            if pday is None:
                continue
            pred_ctx = hourly[
                (hourly["date"] >= pday - pd.Timedelta(days=365)) & (hourly["date"] <= pday + pd.Timedelta(days=1))
            ]
            if pred_ctx.empty:
                continue
            pred_sigs = build_signals(pred_ctx)
            pred_wsig = weighted_signal(pred_sigs, weights)
            pred_e200 = ema(pred_ctx["close"], 200)
            pred_lr_series = np.log(pred_ctx["close"] / pred_ctx["close"].shift(1))
            pred_vol20 = pred_lr_series.rolling(20, min_periods=20).std(ddof=1)

            prev_close = float(pred_ctx["close"].iloc[-1])
            pred_lr = build_pred_lr(
                last_close=prev_close,
                e200=float(pred_e200.iloc[-1]),
                wsig=float(pred_wsig.iloc[-1]),
                vol20=float(pred_vol20.iloc[-1]) if not np.isnan(pred_vol20.iloc[-1]) else 0.004,
                c=c_scale,
            )
            predicted_today_close = float(prev_close * math.exp(pred_lr))

            # 매매 플랜은 최신 1h 기준
            now_ctx = hourly[
                (hourly["date"] >= asof - pd.Timedelta(days=365)) & (hourly["date"] <= asof + pd.Timedelta(days=1))
            ]
            now_sigs = build_signals(now_ctx)
            now_wsig = weighted_signal(now_sigs, weights)
            latest_close = float(now_ctx["close"].iloc[-1])
            lr_now = np.log(now_ctx["close"] / now_ctx["close"].shift(1))
            vol20_now = float(lr_now.rolling(20, min_periods=20).std(ddof=1).iloc[-1])

            opinion = action_from_weighted_signal(float(now_wsig.iloc[-1]), threshold=0.15)
            entry, target, stop, target_pct, stop_pct = suggest_levels(latest_close, opinion, vol20_now)

            rows.append(
                {
                    "ticker": ticker,
                    "latest_daily_date": str(latest_daily.date()),
                    "last_close": latest_close,
                    "predicted_today_date": str(today_pred_date.date()),
                    "predicted_today_close": predicted_today_close,
                    "weighted_signal": float(now_wsig.iloc[-1]),
                    "opinion": opinion,
                    "suggested_buy_price": entry if opinion == "BUY" else np.nan,
                    "suggested_sell_price": entry if opinion == "SELL/REDUCE" else np.nan,
                    "target_return_pct": target_pct if opinion != "HOLD" else np.nan,
                    "target_price": target if opinion != "HOLD" else np.nan,
                    "stop_loss_pct": stop_pct if opinion != "HOLD" else np.nan,
                    "stop_loss_price": stop if opinion != "HOLD" else np.nan,
                    "sell_timing_rule": "목표가 도달 또는 손절가 이탈 또는 앙상블 신호 -1",
                    "strategy_version": "reference_v1_regime_voltarget",
                    "c_scale": c_scale,
                }
            )
        except Exception as exc:
            print(f"[WARN] {ticker} failed: {exc}")
            continue

    if not rows:
        raise ValueError("No rows generated for today plan. Check data provider status.")

    plan_df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof.date())
    out_path = out_dir / f"m7_today_plan_reference_{tag}.csv"
    plan_df.to_csv(out_path, index=False)

    print("=== M7 Today Plan (Reference Strategy) ===")
    print(plan_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
