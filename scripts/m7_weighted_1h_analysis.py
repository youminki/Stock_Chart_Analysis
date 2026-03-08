from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.tseries.offsets import BDay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import TwelveDataFetcher, YahooFinanceDataFetcher

M7_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
HORIZON_BARS = 7  # 1시간봉 기준 대략 1거래일


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M7 1시간봉 멀티전략 가중 분석 + 종가 예측")
    parser.add_argument("--provider", choices=["twelve", "yahoo"], default="twelve", help="데이터 제공자")
    parser.add_argument(
        "--extended-hours",
        action="store_true",
        help="프리마켓/애프터마켓 포함(현재 yahoo provider에서만 적용)",
    )
    parser.add_argument(
        "--asof",
        default=None,
        help="분석 기준일(YYYY-MM-DD). 기본값: 오늘",
    )
    parser.add_argument(
        "--attach-actual-next-close",
        action="store_true",
        help="예측 대상일 실제 종가를 붙여 오차를 계산",
    )
    return parser.parse_args()


@dataclass
class ForecastParams:
    blend: float
    c_mu5: float
    c_mu20: float
    c_wsig_vol: float
    c_rsi_vol: float
    c_rev20: float


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


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = true_range(df)
    atr = tr.rolling(window=period, min_periods=period).mean()

    plus_di = 100 * (plus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.rolling(period, min_periods=period).mean()

    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx_val.fillna(0.0)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


# --- Strategy signals ---
def signal_ema_crossover(df: pd.DataFrame) -> pd.Series:
    e_fast = ema(df["close"], 20)
    e_slow = ema(df["close"], 50)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[(e_fast > e_slow) & (e_fast.shift(1) <= e_slow.shift(1))] = 1
    out.loc[(e_fast < e_slow) & (e_fast.shift(1) >= e_slow.shift(1))] = -1
    return out


def signal_macd(df: pd.DataFrame) -> pd.Series:
    macd_line = ema(df["close"], 12) - ema(df["close"], 26)
    sig_line = ema(macd_line, 9)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[(macd_line > sig_line) & (macd_line.shift(1) <= sig_line.shift(1))] = 1
    out.loc[(macd_line < sig_line) & (macd_line.shift(1) >= sig_line.shift(1))] = -1
    return out


def signal_rsi_reversion(df: pd.DataFrame) -> pd.Series:
    r = rsi(df["close"], 14)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[r <= 30] = 1
    out.loc[r >= 70] = -1
    return out


def signal_bollinger_reversion(df: pd.DataFrame) -> pd.Series:
    _, upper, lower = bollinger(df["close"], 20, 2.0)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[df["close"] <= lower] = 1
    out.loc[df["close"] >= upper] = -1
    return out


def signal_donchian_breakout(df: pd.DataFrame) -> pd.Series:
    high_n = df["high"].rolling(20, min_periods=20).max()
    low_n = df["low"].rolling(20, min_periods=20).min()
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[df["close"] > high_n.shift(1)] = 1
    out.loc[df["close"] < low_n.shift(1)] = -1
    return out


def signal_adx_trend(df: pd.DataFrame) -> pd.Series:
    plus_di, minus_di, adx_val = adx(df, 14)
    out = pd.Series(0, index=df.index, dtype=int)
    strong = adx_val >= 20
    out.loc[strong & (plus_di > minus_di)] = 1
    out.loc[strong & (plus_di < minus_di)] = -1
    return out


def signal_obv_trend(df: pd.DataFrame) -> pd.Series:
    o = obv(df)
    o_fast = ema(o, 10)
    o_slow = ema(o, 30)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[(o_fast > o_slow) & (o_fast.shift(1) <= o_slow.shift(1))] = 1
    out.loc[(o_fast < o_slow) & (o_fast.shift(1) >= o_slow.shift(1))] = -1
    return out


def signal_volume_momentum(df: pd.DataFrame) -> pd.Series:
    vol_ma = df["volume"].rolling(20, min_periods=20).mean()
    e20 = ema(df["close"], 20)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[(df["close"] > e20) & (df["volume"] > 1.5 * vol_ma)] = 1
    out.loc[(df["close"] < e20) & (df["volume"] > 1.5 * vol_ma)] = -1
    return out


def build_strategy_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ema_crossover": signal_ema_crossover(df),
        "macd": signal_macd(df),
        "rsi_reversion": signal_rsi_reversion(df),
        "bollinger_reversion": signal_bollinger_reversion(df),
        "donchian_breakout": signal_donchian_breakout(df),
        "adx_trend": signal_adx_trend(df),
        "obv_trend": signal_obv_trend(df),
        "volume_momentum": signal_volume_momentum(df),
    }


def backtest_long_only(close: pd.Series, signal: pd.Series, fee_rate: float = 0.001) -> dict[str, float | int]:
    cash = 1.0
    shares = 0.0
    in_pos = False
    entry_equity = 1.0
    trades = []
    equity = []

    for idx in close.index:
        price = float(close.loc[idx])
        sig = int(signal.loc[idx])

        if in_pos and sig == -1:
            gross = shares * price
            cash = gross * (1 - fee_rate)
            trades.append(cash / entry_equity - 1.0)
            shares = 0.0
            in_pos = False

        if (not in_pos) and sig == 1:
            investable = cash * (1 - fee_rate)
            shares = investable / price
            entry_equity = cash
            cash = 0.0
            in_pos = True

        eq = shares * price if in_pos else cash
        equity.append(eq)

    if in_pos:
        price = float(close.iloc[-1])
        gross = shares * price
        cash = gross * (1 - fee_rate)
        trades.append(cash / entry_equity - 1.0)
        equity[-1] = cash

    eq_series = pd.Series(equity, index=close.index)
    rets = eq_series.pct_change().fillna(0.0)
    std = float(rets.std(ddof=1))
    sharpe = float(np.sqrt(252 * 6.5) * rets.mean() / std) if std > 0 else 0.0

    return {
        "total_return": float(eq_series.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float((eq_series / eq_series.cummax() - 1.0).min()),
        "win_rate": float((np.array(trades) > 0).mean()) if trades else 0.0,
        "num_trades": int(len(trades)),
    }


def backtest_long_only_session_aware(
    close: pd.Series,
    signal: pd.Series,
    dates: pd.Series,
    fee_rate: float = 0.001,
) -> dict[str, float | int]:
    cash = 1.0
    shares = 0.0
    in_pos = False
    entry_equity = 1.0
    trades = []
    equity = []
    prev_sig = 0

    for i in range(len(close)):
        price = float(close.iloc[i])
        sig = int(signal.iloc[i])
        ts = pd.Timestamp(dates.iloc[i])
        session = classify_session(ts)

        eff_fee = fee_rate if session == "regular" else fee_rate * 1.5
        entry_fraction = 1.0 if session == "regular" else 0.5

        exec_sig = sig
        if session != "regular" and sig != 0 and sig != prev_sig:
            exec_sig = 0

        if in_pos and exec_sig == -1:
            gross = shares * price
            cash = cash + gross * (1 - eff_fee)
            trades.append((cash / entry_equity) - 1.0)
            shares = 0.0
            in_pos = False

        if (not in_pos) and exec_sig == 1:
            investable = cash * entry_fraction
            buy_fee = investable * eff_fee
            net = investable - buy_fee
            if net > 0:
                shares = net / price
                cash = cash - investable
                entry_equity = cash + shares * price
                in_pos = True

        eq = cash + shares * price
        equity.append(eq)
        prev_sig = sig

    if in_pos:
        price = float(close.iloc[-1])
        gross = shares * price
        cash = cash + gross * (1 - fee_rate)
        trades.append((cash / entry_equity) - 1.0)
        equity[-1] = cash

    eq_series = pd.Series(equity, index=close.index)
    rets = eq_series.pct_change().fillna(0.0)
    std = float(rets.std(ddof=1))
    sharpe = float(np.sqrt(252 * 6.5) * rets.mean() / std) if std > 0 else 0.0

    return {
        "total_return": float(eq_series.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float((eq_series / eq_series.cummax() - 1.0).min()),
        "win_rate": float((np.array(trades) > 0).mean()) if trades else 0.0,
        "num_trades": int(len(trades)),
    }


def strategy_score(metrics: dict[str, float | int]) -> float:
    total_return = float(metrics["total_return"])
    sharpe = float(metrics["sharpe"])
    mdd = abs(float(metrics["max_drawdown"]))
    wr = float(metrics["win_rate"])
    if total_return <= 0:
        return 0.0
    return max(sharpe, 0.0) * total_return * (1.0 + wr) / (1.0 + mdd)


def weight_strategies(metric_map: dict[str, dict[str, float | int]]) -> dict[str, float]:
    scores = {name: strategy_score(m) for name, m in metric_map.items()}
    positive = {k: v for k, v in scores.items() if v > 0}

    if not positive:
        best = max(metric_map.items(), key=lambda kv: float(kv[1]["sharpe"]))[0]
        return {k: (1.0 if k == best else 0.0) for k in metric_map}

    total = sum(positive.values())
    return {k: positive.get(k, 0.0) / total for k in metric_map}


def weighted_signal(signals: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    out = pd.Series(0.0, index=next(iter(signals.values())).index)
    for name, sig in signals.items():
        out += weights.get(name, 0.0) * sig.astype(float)
    return out


def weighted_signal_session_aware(
    signals: dict[str, pd.Series],
    weights: dict[str, float],
    dates: pd.Series,
) -> pd.Series:
    raw = weighted_signal(signals, weights)
    out = raw.copy()

    prev_sign = 0
    for i in range(len(out)):
        ts = pd.Timestamp(dates.iloc[i])
        session = classify_session(ts)
        v = float(raw.iloc[i])
        sign = 1 if v > 0 else (-1 if v < 0 else 0)

        if session != "regular":
            # 시간외는 신호 강도 축소
            v *= 0.7
            # 2연속 동일 방향 확인 전에는 신호 무효
            if sign != 0 and sign != prev_sign:
                v = 0.0

        out.iloc[i] = v
        prev_sign = sign

    return out


def build_feature_frame(df: pd.DataFrame, w_sig: pd.Series, horizon: int) -> pd.DataFrame:
    close = df["close"]
    log_r = np.log(close / close.shift(1))
    e20 = ema(close, 20)
    rev20 = (close - e20) / e20.replace(0, np.nan)

    feat = pd.DataFrame({
        "date": df["date"],
        "close": close,
        "mu5": log_r.rolling(5).mean(),
        "mu20": log_r.rolling(20).mean(),
        "vol20": log_r.rolling(20).std(ddof=1),
        "rsi14": rsi(close, 14),
        "rev20": rev20,
        "w_sig": w_sig,
    })

    fut_lr = np.log(close.shift(-horizon) / close)
    feat["target_lr"] = fut_lr
    feat["target_close"] = close.shift(-horizon)

    return feat.dropna().reset_index(drop=True)


def predict_lr_raw(row: pd.Series, p: ForecastParams) -> float:
    rsi_term = (float(row["rsi14"]) - 50.0) / 50.0
    trend_component = (
        p.c_mu5 * float(row["mu5"])
        + p.c_mu20 * float(row["mu20"])
        + p.c_wsig_vol * float(row["w_sig"]) * float(row["vol20"])
        + p.c_rsi_vol * rsi_term * float(row["vol20"])
        - p.c_rev20 * float(row["rev20"]) * float(row["vol20"])
    )
    return float(np.clip(trend_component, -0.08, 0.08))


def predict_lr(row: pd.Series, p: ForecastParams) -> float:
    raw = predict_lr_raw(row, p)
    # blend=0이면 naive(변화 없음), 1이면 full model
    return float(np.clip(p.blend * raw, -0.08, 0.08))


def eval_forecast(feat: pd.DataFrame, p: ForecastParams) -> dict[str, float]:
    pred_lr = feat.apply(lambda r: predict_lr(r, p), axis=1)
    actual_lr = feat["target_lr"].astype(float)

    mae_lr = float(np.mean(np.abs(pred_lr - actual_lr)))
    rmse_lr = float(np.sqrt(np.mean((pred_lr - actual_lr) ** 2)))

    pred_close = feat["close"].astype(float) * np.exp(pred_lr)
    actual_close = feat["target_close"].astype(float)
    mape = float(np.mean(np.abs((pred_close - actual_close) / actual_close)) * 100.0)

    return {"mae_lr": mae_lr, "rmse_lr": rmse_lr, "mape_pct": mape}


def eval_baseline(feat: pd.DataFrame) -> dict[str, float]:
    pred_lr = pd.Series(0.0, index=feat.index)
    actual_lr = feat["target_lr"].astype(float)

    mae_lr = float(np.mean(np.abs(pred_lr - actual_lr)))
    rmse_lr = float(np.sqrt(np.mean((pred_lr - actual_lr) ** 2)))

    pred_close = feat["close"].astype(float)
    actual_close = feat["target_close"].astype(float)
    mape = float(np.mean(np.abs((pred_close - actual_close) / actual_close)) * 100.0)

    return {"mae_lr": mae_lr, "rmse_lr": rmse_lr, "mape_pct": mape}


def tune_params(tune_feat: pd.DataFrame, select_feat: pd.DataFrame) -> ForecastParams:
    best = None
    for blend in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0):
        for c_mu5 in (0.25, 0.5, 1.0, 1.5):
            for c_mu20 in (0.25, 0.5, 1.0, 1.5):
                for c_wsig in (0.0, 0.5, 1.0, 1.5, 2.0):
                    for c_rsi in (0.0, 0.25, 0.5, 1.0):
                        for c_rev20 in (0.0, 0.5, 1.0):
                            p = ForecastParams(blend, c_mu5, c_mu20, c_wsig, c_rsi, c_rev20)
                            m_tune = eval_forecast(tune_feat, p)
                            m = eval_forecast(select_feat, p)
                            key = (m["mape_pct"], m["rmse_lr"])
                            # selection set 우선, 동률이면 tuning set 성능으로 tie-break
                            tiebreak = (m_tune["mape_pct"], m_tune["rmse_lr"])
                            if best is None or (key, tiebreak) < (best["key"], best["tiebreak"]):
                                best = {"p": p, "key": key, "tiebreak": tiebreak}
    if best is None:
        return ForecastParams(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    return best["p"]


def action_from_weighted_signal(w_sig: float, threshold: float = 0.15) -> str:
    if w_sig >= threshold:
        return "BUY"
    if w_sig <= -threshold:
        return "SELL/REDUCE"
    return "HOLD"


def classify_session(ts: pd.Timestamp) -> str:
    hm = ts.hour * 60 + ts.minute
    if hm < 9 * 60 + 30:
        return "premarket"
    if hm <= 16 * 60:
        return "regular"
    return "afterhours"


def session_adjusted_action(w_sig: float, session: str) -> str:
    # 시간외는 노이즈가 크므로 임계값을 높여 보수적으로 진입/청산
    if session == "regular":
        threshold = 0.15
    else:
        threshold = 0.25
    return action_from_weighted_signal(w_sig, threshold=threshold)


def suggest_levels(last_close: float, action: str, vol20: float) -> tuple[float, float, float, float, float]:
    horizon = 6
    sigma_h = vol20 * math.sqrt(horizon)
    target_pct = min(max(1.8 * sigma_h, 0.01), 0.08)
    stop_pct = min(max(1.2 * sigma_h, 0.008), 0.05)

    if action == "BUY":
        entry = last_close * 0.998
        target = entry * (1 + target_pct)
        stop = entry * (1 - stop_pct)
    elif action == "SELL/REDUCE":
        entry = last_close * 1.002
        target = entry * (1 - target_pct)
        stop = entry * (1 + stop_pct)
    else:
        entry = last_close
        target = float("nan")
        stop = float("nan")

    return entry, target, stop, target_pct * 100.0, stop_pct * 100.0


def suggest_levels_session_aware(
    last_close: float, action: str, vol20: float, session: str
) -> tuple[float, float, float, float, float]:
    entry, target, stop, target_pct, stop_pct = suggest_levels(last_close, action, vol20)
    if session != "regular" and action != "HOLD":
        # 시간외는 갭 리스크 반영: 손절/목표 폭 확장
        target_pct *= 1.15
        stop_pct *= 1.25
        if action == "BUY":
            target = entry * (1 + target_pct / 100.0)
            stop = entry * (1 - stop_pct / 100.0)
        else:
            target = entry * (1 - target_pct / 100.0)
            stop = entry * (1 + stop_pct / 100.0)
    return entry, target, stop, target_pct, stop_pct


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    if args.provider == "twelve":
        if args.extended_hours:
            print("[WARN] twelve provider에서는 extended-hours 옵션을 별도 지원하지 않아 정규 세션 중심 데이터가 사용됩니다.")
        fetcher = TwelveDataFetcher(max_retries=8, retry_wait_sec=65)
    else:
        fetcher = YahooFinanceDataFetcher()
    if args.asof:
        asof = pd.Timestamp(args.asof)
    else:
        asof = pd.Timestamp(datetime.now().date())
    start = asof - pd.Timedelta(days=365)
    predicted_for_date = pd.Timestamp(asof + BDay(1)).date()

    summary_rows: list[dict[str, object]] = []
    strat_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []

    per_ticker_ctx: dict[str, dict[str, object]] = {}

    for ticker in M7_TICKERS:
        if args.provider == "yahoo":
            raw = fetcher.fetch_ohlcv_for_interval(
                ticker=ticker, interval="1h", period="730d", prepost=args.extended_hours
            )
        else:
            raw = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1h")
        df = raw[(raw["date"] >= start) & (raw["date"] <= asof + pd.Timedelta(days=1))].copy().reset_index(drop=True)

        if len(df) < 800:
            raise ValueError(f"{ticker}: 1시간봉 데이터가 부족합니다. rows={len(df)}")

        split_train = int(len(df) * 0.6)
        signals = build_strategy_signals(df)

        # 전략 가중치는 train 구간으로 고정(누수 방지)
        train_close = df.loc[: split_train - 1, "close"]
        train_dates = df.loc[: split_train - 1, "date"]
        metric_map = {}
        for name, sig in signals.items():
            train_sig = sig.loc[: split_train - 1]
            if args.provider == "yahoo" and args.extended_hours:
                metric_map[name] = backtest_long_only_session_aware(
                    close=train_close, signal=train_sig, dates=train_dates
                )
            else:
                metric_map[name] = backtest_long_only(train_close, train_sig)
        weights = weight_strategies(metric_map)

        for name, metrics in metric_map.items():
            strat_rows.append(
                {
                    "ticker": ticker,
                    "strategy": name,
                    "weight": weights[name],
                    "train_total_return_pct": float(metrics["total_return"]) * 100.0,
                    "train_sharpe": float(metrics["sharpe"]),
                    "train_max_drawdown_pct": float(metrics["max_drawdown"]) * 100.0,
                    "train_win_rate_pct": float(metrics["win_rate"]) * 100.0,
                    "train_num_trades": int(metrics["num_trades"]),
                }
            )

        if args.provider == "yahoo" and args.extended_hours:
            w_sig = weighted_signal_session_aware(signals, weights, df["date"])
        else:
            w_sig = weighted_signal(signals, weights)
        feat = build_feature_frame(df, w_sig=w_sig, horizon=HORIZON_BARS)

        if len(feat) < 200:
            raise ValueError(f"{ticker}: 예측 피처 데이터가 부족합니다. rows={len(feat)}")

        # feat도 시간순 split
        f_valid = feat.iloc[int(len(feat) * 0.6) : int(len(feat) * 0.8)].copy()
        f_test = feat.iloc[int(len(feat) * 0.8) :].copy()

        per_ticker_ctx[ticker] = {
            "df": df,
            "weights": weights,
            "w_sig": w_sig,
            "feat_valid": f_valid,
            "feat_test": f_test,
        }

    for ticker in M7_TICKERS:
        ctx = per_ticker_ctx[ticker]
        df = ctx["df"]
        weights = ctx["weights"]
        w_sig = ctx["w_sig"]
        f_valid = ctx["feat_valid"]
        f_test = ctx["feat_test"]

        valid_mid = max(1, int(len(f_valid) * 0.5))
        f_tune = f_valid.iloc[:valid_mid].copy()
        f_select = f_valid.iloc[valid_mid:].copy()
        if f_select.empty:
            f_select = f_tune.copy()

        best_params = tune_params(f_tune, f_select)
        valid_metrics = eval_forecast(f_valid, best_params)
        test_metrics = eval_forecast(f_test, best_params)
        valid_base = eval_baseline(f_valid)
        test_base = eval_baseline(f_test)

        eval_rows.append(
            {
                "ticker": ticker,
                "set": "valid",
                "model_mape_pct": valid_metrics["mape_pct"],
                "model_rmse_lr": valid_metrics["rmse_lr"],
                "baseline_mape_pct": valid_base["mape_pct"],
                "baseline_rmse_lr": valid_base["rmse_lr"],
                "mape_improvement_pct": valid_base["mape_pct"] - valid_metrics["mape_pct"],
                "blend": best_params.blend,
                "c_mu5": best_params.c_mu5,
                "c_mu20": best_params.c_mu20,
                "c_wsig_vol": best_params.c_wsig_vol,
                "c_rsi_vol": best_params.c_rsi_vol,
                "c_rev20": best_params.c_rev20,
            }
        )
        eval_rows.append(
            {
                "ticker": ticker,
                "set": "test",
                "model_mape_pct": test_metrics["mape_pct"],
                "model_rmse_lr": test_metrics["rmse_lr"],
                "baseline_mape_pct": test_base["mape_pct"],
                "baseline_rmse_lr": test_base["rmse_lr"],
                "mape_improvement_pct": test_base["mape_pct"] - test_metrics["mape_pct"],
                "blend": best_params.blend,
                "c_mu5": best_params.c_mu5,
                "c_mu20": best_params.c_mu20,
                "c_wsig_vol": best_params.c_wsig_vol,
                "c_rsi_vol": best_params.c_rsi_vol,
                "c_rev20": best_params.c_rev20,
            }
        )

        close = df["close"]
        last_close = float(close.iloc[-1])
        first_close = float(close.iloc[0])
        log_r = np.log(close / close.shift(1)).dropna()
        vol20 = float(log_r.tail(20).std(ddof=1))

        row_now = pd.Series(
            {
                "mu5": float(log_r.tail(5).mean()),
                "mu20": float(log_r.tail(20).mean()),
                "vol20": vol20,
                "rsi14": float(rsi(close, 14).iloc[-1]),
                "rev20": float(((close - ema(close, 20)) / ema(close, 20).replace(0, np.nan)).iloc[-1]),
                "w_sig": float(w_sig.iloc[-1]),
            }
        )

        pred_lr = predict_lr(row_now, best_params)
        pred_close = last_close * math.exp(pred_lr)

        latest_ts = pd.Timestamp(df["date"].iloc[-1])
        latest_session = classify_session(latest_ts)
        action = session_adjusted_action(float(w_sig.iloc[-1]), latest_session)
        entry, target, stop, target_pct, stop_pct = suggest_levels_session_aware(
            last_close, action, vol20, latest_session
        )

        summary_rows.append(
            {
                "ticker": ticker,
                "data_start": str(df["date"].iloc[0]),
                "data_end": str(df["date"].iloc[-1]),
                "bars_1h": int(len(df)),
                "return_1y_pct": (last_close / first_close - 1.0) * 100.0,
                "last_close": last_close,
                "predicted_for_date": str(predicted_for_date),
                "predicted_next_day_close": pred_close,
                "latest_bar_time": str(latest_ts),
                "latest_session": latest_session,
                "weighted_signal": float(w_sig.iloc[-1]),
                "opinion": action,
                "entry_price": entry,
                "target_return_pct": target_pct if action != "HOLD" else np.nan,
                "target_price": target if action != "HOLD" else np.nan,
                "stop_loss_pct": stop_pct if action != "HOLD" else np.nan,
                "stop_loss_price": stop if action != "HOLD" else np.nan,
                "forecast_blend": best_params.blend,
                "weights_json": json.dumps({k: round(v, 4) for k, v in weights.items()}, ensure_ascii=False),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    strat_df = pd.DataFrame(strat_rows)

    if args.attach_actual_next_close:
        actual_rows = []
        for ticker in M7_TICKERS:
            day_df = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1day")
            actual = day_df.loc[day_df["date"].dt.date == predicted_for_date, "close"]
            actual_close = float(actual.iloc[0]) if not actual.empty else np.nan
            actual_rows.append({"ticker": ticker, "actual_next_close": actual_close})

        actual_df = pd.DataFrame(actual_rows)
        summary_df = summary_df.merge(actual_df, on="ticker", how="left")
        summary_df["abs_error_pct"] = (
            (summary_df["predicted_next_day_close"] - summary_df["actual_next_close"]).abs()
            / summary_df["actual_next_close"]
            * 100.0
        )

    forecast_eval_df = pd.DataFrame(eval_rows)
    forecast_eval_summary_df = (
        forecast_eval_df.groupby("set", as_index=False)[
            ["model_mape_pct", "model_rmse_lr", "baseline_mape_pct", "baseline_rmse_lr", "mape_improvement_pct"]
        ]
        .mean()
        .sort_values("set")
    )

    params_df = forecast_eval_df[
        ["ticker", "blend", "c_mu5", "c_mu20", "c_wsig_vol", "c_rsi_vol", "c_rev20"]
    ].drop_duplicates().reset_index(drop=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof.date())

    summary_path = out_dir / f"m7_1h_weighted_summary_{tag}.csv"
    strat_path = out_dir / f"m7_1h_weighted_strategy_metrics_{tag}.csv"
    eval_path = out_dir / f"m7_1h_forecast_backtest_{tag}.csv"
    eval_summary_path = out_dir / f"m7_1h_forecast_backtest_summary_{tag}.csv"
    param_path = out_dir / f"m7_1h_forecast_params_{tag}.csv"

    summary_df.to_csv(summary_path, index=False)
    strat_df.to_csv(strat_path, index=False)
    forecast_eval_df.to_csv(eval_path, index=False)
    forecast_eval_summary_df.to_csv(eval_summary_path, index=False)
    params_df.to_csv(param_path, index=False)

    print("=== Forecast Params (Per Ticker) ===")
    print(params_df.to_string(index=False))
    print("\n=== Forecast Backtest (Avg) ===")
    print(forecast_eval_summary_df.to_string(index=False))
    print("\n=== M7 Weighted 1H Summary ===")
    display_cols = [
        "ticker",
        "latest_bar_time",
        "latest_session",
        "predicted_for_date",
        "return_1y_pct",
        "last_close",
        "predicted_next_day_close",
        "weighted_signal",
        "opinion",
        "entry_price",
        "target_return_pct",
        "target_price",
        "stop_loss_pct",
        "stop_loss_price",
    ]
    if "actual_next_close" in summary_df.columns:
        display_cols += ["actual_next_close", "abs_error_pct"]
    print(summary_df[display_cols].to_string(index=False))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {strat_path}")
    print(f"Saved: {eval_path}")
    print(f"Saved: {eval_summary_path}")
    print(f"Saved: {param_path}")


if __name__ == "__main__":
    main()
