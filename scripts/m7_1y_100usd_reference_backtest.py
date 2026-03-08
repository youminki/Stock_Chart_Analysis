from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import YahooFinanceDataFetcher

M7_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]


@dataclass
class BacktestCfg:
    fee_rate: float = 0.001
    target_annual_vol: float = 0.20
    max_leverage: float = 1.0


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


def signal_tsmom(df: pd.DataFrame) -> pd.Series:
    # 63시간 전 대비 모멘텀(대략 2주 거래시간)
    mom = df["close"].pct_change(63)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[mom > 0] = 1
    out.loc[mom < 0] = -1
    return out


def signal_volume_momentum(df: pd.DataFrame) -> pd.Series:
    vol_ma = df["volume"].rolling(20, min_periods=20).mean()
    e20 = ema(df["close"], 20)
    out = pd.Series(0, index=df.index, dtype=int)
    out.loc[(df["close"] > e20) & (df["volume"] > 1.5 * vol_ma)] = 1
    out.loc[(df["close"] < e20) & (df["volume"] > 1.5 * vol_ma)] = -1
    return out


def build_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ema_crossover": signal_ema_crossover(df),
        "macd": signal_macd(df),
        "rsi_reversion": signal_rsi_reversion(df),
        "bollinger_reversion": signal_bollinger_reversion(df),
        "donchian_breakout": signal_donchian_breakout(df),
        "adx_trend": signal_adx_trend(df),
        "obv_trend": signal_obv_trend(df),
        "tsmom_63h": signal_tsmom(df),
        "volume_momentum": signal_volume_momentum(df),
    }


def score_strategy(total_return: float, sharpe: float, mdd: float, win_rate: float) -> float:
    if total_return <= 0:
        return 0.0
    return max(sharpe, 0.0) * total_return * (1.0 + win_rate) / (1.0 + abs(mdd))


def backtest_with_regime_and_vol_target(
    df: pd.DataFrame,
    signal: pd.Series,
    cfg: BacktestCfg,
) -> dict[str, float]:
    close = df["close"].astype(float)
    log_r = np.log(close / close.shift(1)).fillna(0.0)
    ann_vol = (log_r.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252 * 6.5)).shift(1)

    regime_ok = (close > ema(close, 200)).fillna(False)

    cash = 1.0
    shares = 0.0
    in_pos = False
    entry_eq = 1.0
    trades = []
    equity = []

    for i in range(len(df)):
        price = float(close.iloc[i])
        sig = int(signal.iloc[i])

        # Regime filter: 약세(regime off)에서는 신규 매수 금지, 보유 시 청산 우선
        if not bool(regime_ok.iloc[i]):
            if in_pos:
                sig = -1
            else:
                sig = 0

        if in_pos and sig == -1:
            gross = shares * price
            cash = cash + gross * (1.0 - cfg.fee_rate)
            trades.append((cash / entry_eq) - 1.0)
            shares = 0.0
            in_pos = False

        if (not in_pos) and sig == 1:
            realized_vol = float(ann_vol.iloc[i]) if not np.isnan(ann_vol.iloc[i]) else 0.0
            if realized_vol <= 0:
                size = 0.25
            else:
                size = min(cfg.max_leverage, cfg.target_annual_vol / realized_vol)
            investable = cash * max(0.0, min(1.0, size))
            net = investable * (1.0 - cfg.fee_rate)
            if net > 0:
                shares = net / price
                cash = cash - investable
                entry_eq = cash + shares * price
                in_pos = True

        eq = cash + shares * price
        equity.append(eq)

    if in_pos:
        price = float(close.iloc[-1])
        gross = shares * price
        cash = cash + gross * (1.0 - cfg.fee_rate)
        trades.append((cash / entry_eq) - 1.0)
        equity[-1] = cash

    eq = pd.Series(equity, index=df.index)
    rets = eq.pct_change().fillna(0.0)
    std = float(rets.std(ddof=1))
    sharpe = float(np.sqrt(252 * 6.5) * rets.mean() / std) if std > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1.0).min())
    win_rate = float((np.array(trades) > 0).mean()) if trades else 0.0

    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "num_trades": float(len(trades)),
    }


def weighted_signal(signals: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    out = pd.Series(0.0, index=next(iter(signals.values())).index)
    for name, s in signals.items():
        out += weights.get(name, 0.0) * s.astype(float)
    return out


def run_one_ticker(df: pd.DataFrame, cfg: BacktestCfg) -> tuple[pd.DataFrame, dict[str, float]]:
    split = int(len(df) * 0.6)
    train = df.iloc[:split].copy()
    full = df.copy()

    sig_full = build_signals(full)
    metric_map: dict[str, dict[str, float]] = {}
    for name, sig in sig_full.items():
        metric_map[name] = backtest_with_regime_and_vol_target(train, sig.iloc[:split], cfg)

    scores = {
        name: score_strategy(
            total_return=m["total_return"],
            sharpe=m["sharpe"],
            mdd=m["max_drawdown"],
            win_rate=m["win_rate"],
        )
        for name, m in metric_map.items()
    }
    pos_scores = {k: v for k, v in scores.items() if v > 0}
    if pos_scores:
        tot = sum(pos_scores.values())
        weights = {k: pos_scores.get(k, 0.0) / tot for k in scores}
    else:
        best = max(metric_map.items(), key=lambda kv: kv[1]["sharpe"])[0]
        weights = {k: (1.0 if k == best else 0.0) for k in scores}

    wsig = weighted_signal(sig_full, weights)
    combined_sig = pd.Series(0, index=df.index, dtype=int)
    combined_sig.loc[wsig >= 0.15] = 1
    combined_sig.loc[wsig <= -0.15] = -1

    total = backtest_with_regime_and_vol_target(full, combined_sig, cfg)

    strat_rows = []
    for name, m in metric_map.items():
        strat_rows.append(
            {
                "strategy": name,
                "weight": float(weights[name]),
                "train_total_return_pct": float(m["total_return"] * 100.0),
                "train_sharpe": float(m["sharpe"]),
                "train_max_drawdown_pct": float(m["max_drawdown"] * 100.0),
                "train_win_rate_pct": float(m["win_rate"] * 100.0),
                "train_num_trades": int(m["num_trades"]),
            }
        )
    return pd.DataFrame(strat_rows), total


def main() -> None:
    load_dotenv(ROOT / ".env")
    fetcher = YahooFinanceDataFetcher()
    cfg = BacktestCfg()

    asof = pd.Timestamp(datetime.now().date())
    start = asof - pd.Timedelta(days=365)

    result_rows = []
    weight_rows = []
    for ticker in M7_TICKERS:
        raw = fetcher.fetch_ohlcv_for_interval(ticker=ticker, interval="1h", period="730d", prepost=True)
        df = raw[(raw["date"] >= start) & (raw["date"] <= asof + pd.Timedelta(days=1))].copy().reset_index(drop=True)
        if len(df) < 500:
            continue

        strat_df, total = run_one_ticker(df, cfg)
        for _, row in strat_df.iterrows():
            weight_rows.append({"ticker": ticker, **row.to_dict()})

        start_cap = 100.0
        end_cap = start_cap * (1.0 + float(total["total_return"]))
        result_rows.append(
            {
                "ticker": ticker,
                "bars_1h": int(len(df)),
                "start_capital_usd": start_cap,
                "end_capital_usd": end_cap,
                "return_pct": (end_cap / start_cap - 1.0) * 100.0,
                "sharpe": float(total["sharpe"]),
                "max_drawdown_pct": float(total["max_drawdown"] * 100.0),
                "win_rate_pct": float(total["win_rate"] * 100.0),
                "num_trades": int(total["num_trades"]),
                "data_start": str(df["date"].iloc[0]),
                "data_end": str(df["date"].iloc[-1]),
            }
        )

    result_df = pd.DataFrame(result_rows).sort_values("ticker").reset_index(drop=True)
    weights_df = pd.DataFrame(weight_rows).sort_values(["ticker", "weight"], ascending=[True, False]).reset_index(
        drop=True
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(asof.date())
    result_path = out_dir / f"m7_1y_100usd_reference_backtest_{tag}.csv"
    weight_path = out_dir / f"m7_1y_reference_strategy_weights_{tag}.csv"
    result_df.to_csv(result_path, index=False)
    weights_df.to_csv(weight_path, index=False)

    portfolio_start = float(result_df["start_capital_usd"].sum())
    portfolio_end = float(result_df["end_capital_usd"].sum())
    portfolio_ret = (portfolio_end / portfolio_start - 1.0) * 100.0 if portfolio_start > 0 else 0.0

    print("=== M7 1Y Reference Backtest ($100 each) ===")
    print(result_df.to_string(index=False))
    print(
        f"\nPortfolio Start=${portfolio_start:.2f}, End=${portfolio_end:.2f}, Return={portfolio_ret:.2f}% "
        f"(as of {asof.date()})"
    )
    print(f"\nSaved: {result_path}")
    print(f"Saved: {weight_path}")


if __name__ == "__main__":
    main()
