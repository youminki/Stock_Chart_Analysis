from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.tseries.offsets import BDay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import YahooFinanceDataFetcher
from scripts.m7_1y_100usd_reference_backtest import BacktestCfg, M7_TICKERS, build_signals, ema
from scripts.m7_last1m_close_accuracy_reference import build_pred_lr, build_weights, tune_c, weighted_signal


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
                continue

            latest_daily = pd.Timestamp(daily["date"].max()).normalize()
            ctx_end = latest_daily
            ctx_start = ctx_end - pd.Timedelta(days=365)
            hourly_ctx = hourly[
                (hourly["date"] >= ctx_start) & (hourly["date"] <= ctx_end + pd.Timedelta(days=1))
            ].copy()

            weights = build_weights(hourly_ctx, cfg)
            eval_start = latest_daily - pd.Timedelta(days=30)
            c_scale = tune_c(hourly_ctx, daily, eval_start, weights)

            sigs = build_signals(hourly_ctx)
            ws = weighted_signal(sigs, weights)
            e200 = ema(hourly_ctx["close"], 200)
            lr = np.log(hourly_ctx["close"] / hourly_ctx["close"].shift(1))
            vol20 = lr.rolling(20, min_periods=20).std(ddof=1)

            base_close = float(hourly_ctx["close"].iloc[-1])
            pred_lr_1d = build_pred_lr(
                last_close=base_close,
                e200=float(e200.iloc[-1]),
                wsig=float(ws.iloc[-1]),
                vol20=float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else 0.004,
                c=c_scale,
            )

            for k in range(1, 6):
                # 일간 기대 로그수익률을 선형 누적해 k거래일 후 종가 예측
                pred_close_k = float(base_close * math.exp(pred_lr_1d * k))
                pred_day = latest_daily + BDay(k)
                rows.append(
                    {
                        "ticker": ticker,
                        "base_date": str(latest_daily.date()),
                        "base_close": base_close,
                        "forecast_horizon_days": k,
                        "forecast_date": str(pred_day.date()),
                        "predicted_close": pred_close_k,
                        "predicted_return_pct_from_base": (pred_close_k / base_close - 1.0) * 100.0,
                        "weighted_signal": float(ws.iloc[-1]),
                        "c_scale": float(c_scale),
                        "strategy_version": "reference_v1_regime_voltarget",
                    }
                )
        except Exception as exc:
            print(f"[WARN] {ticker} failed: {exc}")
            continue

    if not rows:
        raise ValueError("No forecast rows generated.")

    out_df = pd.DataFrame(rows).sort_values(["ticker", "forecast_horizon_days"]).reset_index(drop=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof.date())
    out_path = out_dir / f"m7_next5d_close_forecast_reference_{tag}.csv"
    out_df.to_csv(out_path, index=False)

    print("=== M7 Next 5 Trading Days Close Forecast (Reference Strategy) ===")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
