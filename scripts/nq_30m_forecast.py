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

from data import YahooFinanceDataFetcher

TICKER = "NQ=F"
INTERVAL = "30m"


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame(
        {
            "date": df["date"],
            "close": close,
            "mu3": lr.rolling(3).mean(),
            "mu12": lr.rolling(12).mean(),
            "vol12": lr.rolling(12).std(ddof=1),
            "ema_gap": (ema(close, 10) - ema(close, 30)) / ema(close, 30).replace(0, np.nan),
            "rsi14": rsi(close, 14),
            "target_lr_30m": np.log(close.shift(-1) / close),
            "target_close_30m": close.shift(-1),
        }
    )
    return feat.dropna().reset_index(drop=True)


def predict_lr(row: pd.Series, params: dict[str, float]) -> float:
    rsi_term = (float(row["rsi14"]) - 50.0) / 50.0
    x = (
        params["blend"]
        * (
            params["c_mu3"] * float(row["mu3"])
            + params["c_mu12"] * float(row["mu12"])
            + params["c_vol"] * float(row["vol12"])
            + params["c_ema"] * float(row["ema_gap"])
            + params["c_rsi"] * rsi_term * float(row["vol12"])
        )
    )
    return float(np.clip(x, -0.02, 0.02))


def eval_set(feat: pd.DataFrame, params: dict[str, float]) -> dict[str, float]:
    pred_lr = feat.apply(lambda r: predict_lr(r, params), axis=1)
    actual_lr = feat["target_lr_30m"].astype(float)

    mae_lr = float(np.mean(np.abs(pred_lr - actual_lr)))
    rmse_lr = float(np.sqrt(np.mean((pred_lr - actual_lr) ** 2)))

    pred_close = feat["close"].astype(float) * np.exp(pred_lr)
    actual_close = feat["target_close_30m"].astype(float)
    mape = float(np.mean(np.abs((pred_close - actual_close) / actual_close)) * 100.0)

    dir_acc = float((np.sign(pred_lr) == np.sign(actual_lr)).mean() * 100.0)
    return {
        "mape_pct": mape,
        "mae_lr": mae_lr,
        "rmse_lr": rmse_lr,
        "directional_acc_pct": dir_acc,
    }


def eval_baseline(feat: pd.DataFrame) -> dict[str, float]:
    pred_lr = pd.Series(0.0, index=feat.index)
    actual_lr = feat["target_lr_30m"].astype(float)

    mae_lr = float(np.mean(np.abs(pred_lr - actual_lr)))
    rmse_lr = float(np.sqrt(np.mean((pred_lr - actual_lr) ** 2)))

    pred_close = feat["close"].astype(float)
    actual_close = feat["target_close_30m"].astype(float)
    mape = float(np.mean(np.abs((pred_close - actual_close) / actual_close)) * 100.0)

    dir_acc = float((np.sign(pred_lr) == np.sign(actual_lr)).mean() * 100.0)
    return {
        "mape_pct": mape,
        "mae_lr": mae_lr,
        "rmse_lr": rmse_lr,
        "directional_acc_pct": dir_acc,
    }


def tune_params(valid_df: pd.DataFrame) -> dict[str, float]:
    best = None
    for blend in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0):
        for c_mu3 in (0.5, 1.0, 1.5, 2.0):
            for c_mu12 in (0.5, 1.0, 1.5):
                for c_vol in (-0.5, 0.0, 0.5):
                    for c_ema in (0.5, 1.0, 1.5, 2.0):
                        for c_rsi in (0.0, 0.5, 1.0):
                            p = {
                                "blend": blend,
                                "c_mu3": c_mu3,
                                "c_mu12": c_mu12,
                                "c_vol": c_vol,
                                "c_ema": c_ema,
                                "c_rsi": c_rsi,
                            }
                            m = eval_set(valid_df, p)
                            key = (m["mape_pct"], -m["directional_acc_pct"], m["rmse_lr"])
                            if best is None or key < best["key"]:
                                best = {"params": p, "key": key}

    if best is None:
        return {
            "blend": 0.0,
            "c_mu3": 1.0,
            "c_mu12": 1.0,
            "c_vol": 0.0,
            "c_ema": 1.0,
            "c_rsi": 0.0,
        }
    return best["params"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    fetcher = YahooFinanceDataFetcher()

    now = pd.Timestamp(datetime.now())
    start = now - pd.Timedelta(days=365)

    used_interval = INTERVAL
    try:
        raw = fetcher.fetch_ohlcv_for_interval(
            ticker=TICKER,
            interval=INTERVAL,
            period="1y",
            prepost=True,
        )
    except Exception:
        used_interval = "60m"
        raw = fetcher.fetch_ohlcv_for_interval(
            ticker=TICKER,
            interval=used_interval,
            period="1y",
            prepost=True,
        )
    raw = raw[(raw["date"] >= start) & (raw["date"] <= now + pd.Timedelta(days=1))].copy()

    feat = build_features(raw)
    if len(feat) < 1200:
        raise ValueError(f"데이터가 충분하지 않습니다. feat_rows={len(feat)}")

    n = len(feat)
    i_train = int(n * 0.6)
    i_valid = int(n * 0.8)

    train_df = feat.iloc[:i_train].copy()
    valid_df = feat.iloc[i_train:i_valid].copy()
    test_df = feat.iloc[i_valid:].copy()

    params = tune_params(valid_df)

    valid_model = eval_set(valid_df, params)
    test_model = eval_set(test_df, params)
    valid_base = eval_baseline(valid_df)
    test_base = eval_baseline(test_df)

    # 현재 시점 다음 30분 종가 예측
    last_row = feat.iloc[-1]
    pred_lr = predict_lr(last_row, params)
    last_close = float(last_row["close"])
    if used_interval == "30m":
        pred_next_close = float(last_close * np.exp(pred_lr))
        pred_ret_pct = pred_lr * 100.0
    else:
        # 60m 예측을 30m로 근사 스케일링
        pred_next_close = float(last_close * np.exp(pred_lr * 0.5))
        pred_ret_pct = pred_lr * 0.5 * 100.0

    # 간단 매매의견
    if pred_lr > 0.0008:
        opinion = "BUY"
    elif pred_lr < -0.0008:
        opinion = "SELL/REDUCE"
    else:
        opinion = "HOLD"

    out_eval = pd.DataFrame(
        [
            {
                "set": "valid",
                "model_mape_pct": valid_model["mape_pct"],
                "baseline_mape_pct": valid_base["mape_pct"],
                "mape_improvement_pct": valid_base["mape_pct"] - valid_model["mape_pct"],
                "model_directional_acc_pct": valid_model["directional_acc_pct"],
                "baseline_directional_acc_pct": valid_base["directional_acc_pct"],
            },
            {
                "set": "test",
                "model_mape_pct": test_model["mape_pct"],
                "baseline_mape_pct": test_base["mape_pct"],
                "mape_improvement_pct": test_base["mape_pct"] - test_model["mape_pct"],
                "model_directional_acc_pct": test_model["directional_acc_pct"],
                "baseline_directional_acc_pct": test_base["directional_acc_pct"],
            },
        ]
    )

    out_pred = pd.DataFrame(
        [
            {
                "ticker": TICKER,
                "interval": used_interval,
                "last_bar_time": str(raw["date"].iloc[-1]),
                "last_close": last_close,
                "predicted_next_30m_close": pred_next_close,
                "predicted_return_pct": pred_ret_pct,
                "opinion": opinion,
            }
        ]
    )

    out_params = pd.DataFrame([params])

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(now.date())
    eval_path = out_dir / f"nq_30m_backtest_eval_{tag}.csv"
    pred_path = out_dir / f"nq_30m_next_forecast_{tag}.csv"
    params_path = out_dir / f"nq_30m_params_{tag}.csv"

    out_eval.to_csv(eval_path, index=False)
    out_pred.to_csv(pred_path, index=False)
    out_params.to_csv(params_path, index=False)

    print("=== NQ 30m Backtest Eval ===")
    print(out_eval.to_string(index=False))
    print("\n=== NQ Next 30m Forecast ===")
    print(out_pred.to_string(index=False))
    print("\n=== Tuned Params ===")
    print(out_params.to_string(index=False))
    print(f"\nSaved: {eval_path}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {params_path}")


if __name__ == "__main__":
    main()
