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
from scripts.m7_1y_100usd_reference_backtest import (
    BacktestCfg,
    M7_TICKERS,
    backtest_with_regime_and_vol_target,
    build_signals,
    ema,
    score_strategy,
)


def prev_trading_day(dates: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp | None:
    pos = None
    for i, d in enumerate(dates):
        if d == target:
            pos = i
            break
    if pos is None or pos == 0:
        return None
    return dates[pos - 1]


def weighted_signal(signals: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    out = pd.Series(0.0, index=next(iter(signals.values())).index)
    for k, s in signals.items():
        out += weights.get(k, 0.0) * s.astype(float)
    return out


def build_weights(ctx_hourly: pd.DataFrame, cfg: BacktestCfg) -> dict[str, float]:
    split = int(len(ctx_hourly) * 0.6)
    sigs = build_signals(ctx_hourly)
    scores = {}
    for name, sig in sigs.items():
        m = backtest_with_regime_and_vol_target(ctx_hourly.iloc[:split], sig.iloc[:split], cfg)
        scores[name] = score_strategy(
            total_return=float(m["total_return"]),
            sharpe=float(m["sharpe"]),
            mdd=float(m["max_drawdown"]),
            win_rate=float(m["win_rate"]),
        )
    pos = {k: v for k, v in scores.items() if v > 0}
    if pos:
        tot = sum(pos.values())
        return {k: pos.get(k, 0.0) / tot for k in scores}
    best = max(scores.items(), key=lambda x: x[1])[0]
    return {k: (1.0 if k == best else 0.0) for k in scores}


def build_pred_lr(last_close: float, e200: float, wsig: float, vol20: float, c: float) -> float:
    lr = c * wsig * vol20
    # 약세 레짐에선 상승 예측을 보수적으로 제한
    if last_close < e200 and lr > 0:
        lr *= 0.4
    cap = max(0.004, min(0.03, 2.2 * vol20))
    return float(np.clip(lr, -cap, cap))


def tune_c(ctx_hourly: pd.DataFrame, daily_df: pd.DataFrame, eval_start: pd.Timestamp, weights: dict[str, float]) -> float:
    # eval 직전 60거래일로 c를 고정 튜닝 (누수 방지)
    d = daily_df[daily_df["date"] < eval_start].copy().reset_index(drop=True)
    if len(d) < 80:
        return 1.0
    calib_days = [pd.Timestamp(x).normalize() for x in d["date"].tail(60)]
    all_days = [pd.Timestamp(x).normalize() for x in d["date"]]

    sigs_all = build_signals(ctx_hourly)
    ws_all = weighted_signal(sigs_all, weights)
    e200_all = ema(ctx_hourly["close"], 200)
    lr_all = np.log(ctx_hourly["close"] / ctx_hourly["close"].shift(1))
    vol20_all = lr_all.rolling(20, min_periods=20).std(ddof=1)

    candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    best_c = 1.0
    best_mape = float("inf")
    for c in candidates:
        errs = []
        for day in calib_days:
            pday = prev_trading_day(all_days, day)
            if pday is None:
                continue
            ctx = ctx_hourly[ctx_hourly["date"] <= pday + pd.Timedelta(days=1)]
            if len(ctx) < 400:
                continue
            i = len(ctx) - 1
            last_close = float(ctx["close"].iloc[-1])
            e200 = float(e200_all.iloc[i])
            wsig = float(ws_all.iloc[i])
            vol20 = float(vol20_all.iloc[i]) if not np.isnan(vol20_all.iloc[i]) else 0.004
            pred_lr = build_pred_lr(last_close, e200, wsig, vol20, c)
            pred_close = last_close * math.exp(pred_lr)
            row = daily_df[daily_df["date"].dt.normalize() == day]
            if row.empty:
                continue
            actual = float(row["close"].iloc[0])
            errs.append(abs(pred_close - actual) / actual * 100.0)
        if errs:
            mape = float(np.mean(errs))
            if mape < best_mape:
                best_mape = mape
                best_c = c
    return best_c


def main() -> None:
    load_dotenv(ROOT / ".env")
    fetcher = YahooFinanceDataFetcher()
    cfg = BacktestCfg()

    asof = pd.Timestamp(datetime.now().date())
    start_hourly = asof - pd.Timedelta(days=460)
    pred_rows = []
    summary_rows = []

    for ticker in M7_TICKERS:
        hourly = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1h", period="730d", prepost=True)
        hourly = hourly[(hourly["date"] >= start_hourly) & (hourly["date"] <= asof + pd.Timedelta(days=1))].copy()
        hourly = hourly.sort_values("date").reset_index(drop=True)

        daily = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1d")
        daily = daily.sort_values("date").reset_index(drop=True)
        if len(hourly) < 800 or len(daily) < 120:
            continue

        latest_daily = pd.Timestamp(daily["date"].max()).normalize()
        eval_start = latest_daily - pd.Timedelta(days=30)
        eval_days = sorted(set(pd.Timestamp(x).normalize() for x in daily["date"] if pd.Timestamp(x).normalize() >= eval_start))
        all_days = [pd.Timestamp(x).normalize() for x in daily["date"]]

        # 평가 시작 직전 컨텍스트로 가중치 고정
        ctx_end = eval_start - pd.Timedelta(days=1)
        ctx_start = ctx_end - pd.Timedelta(days=365)
        hourly_ctx_for_weight = hourly[(hourly["date"] >= ctx_start) & (hourly["date"] <= ctx_end + pd.Timedelta(days=1))]
        weights = build_weights(hourly_ctx_for_weight, cfg)

        # 누수 없이 스케일 c만 사전 튜닝
        c = tune_c(hourly_ctx_for_weight, daily, eval_start, weights)

        for day in eval_days:
            pday = prev_trading_day(all_days, day)
            if pday is None:
                continue
            ctx = hourly[(hourly["date"] <= pday + pd.Timedelta(days=1)) & (hourly["date"] >= pday - pd.Timedelta(days=365))]
            if len(ctx) < 400:
                continue

            sigs = build_signals(ctx)
            ws = weighted_signal(sigs, weights)
            e200 = ema(ctx["close"], 200)
            lr = np.log(ctx["close"] / ctx["close"].shift(1))
            vol20 = lr.rolling(20, min_periods=20).std(ddof=1)

            last_close = float(ctx["close"].iloc[-1])
            pred_lr = build_pred_lr(
                last_close=last_close,
                e200=float(e200.iloc[-1]),
                wsig=float(ws.iloc[-1]),
                vol20=float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else 0.004,
                c=c,
            )
            pred_close = float(last_close * math.exp(pred_lr))

            row = daily[daily["date"].dt.normalize() == day]
            if row.empty:
                continue
            actual_close = float(row["close"].iloc[0])
            abs_error_pct = abs(pred_close - actual_close) / actual_close * 100.0

            pred_rows.append(
                {
                    "ticker": ticker,
                    "pred_date": str(day.date()),
                    "asof_prev_day": str(pday.date()),
                    "c_scale": c,
                    "pred_close": pred_close,
                    "actual_close": actual_close,
                    "abs_error_pct": abs_error_pct,
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
                "c_scale": c,
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
                    "c_scale": np.nan,
                }
            ]
        )
        summary_df = pd.concat([summary_df, overall], ignore_index=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof.date())
    pred_path = out_dir / f"m7_1h_last1m_daily_close_predictions_reference_{tag}.csv"
    summary_path = out_dir / f"m7_1h_last1m_daily_close_error_summary_reference_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=== Last 1M Daily Close Prediction Error (Reference Strategy) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {pred_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
