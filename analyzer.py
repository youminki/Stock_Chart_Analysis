from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest import BacktestConfig, Backtester
from strategies import (
    BaseStrategy,
    BollingerReversionStrategy,
    EMACrossoverStrategy,
    RSIReversionStrategy,
    SMACrossoverStrategy,
)

DEFAULT_M7_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]


class SignalStrategy(BaseStrategy):
    """이미 계산된 signal 컬럼을 그대로 사용하는 전략 어댑터."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "signal" not in df.columns:
            raise ValueError("SignalStrategy 입력 데이터에 signal 컬럼이 필요합니다.")
        out = df.copy()
        out["signal"] = out["signal"].astype(int)
        return out


@dataclass
class AnalysisService:
    fetcher: Any

    def analyze_ticker(
        self,
        ticker: str,
        short_window: int = 20,
        long_window: int = 60,
        fee_rate: float = 0.001,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        initial_capital: float = 10_000.0,
    ) -> dict[str, Any]:
        daily_df = self.fetcher.fetch_daily_ohlcv(ticker)

        strategy = SMACrossoverStrategy(short_window=short_window, long_window=long_window)
        config = BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        result = Backtester(strategy=strategy, config=config).run(daily_df)

        first_date = daily_df["date"].iloc[0].date()
        last_date = daily_df["date"].iloc[-1].date()
        buy_and_hold_return = (daily_df["close"].iloc[-1] / daily_df["close"].iloc[0] - 1.0) * 100.0

        return {
            "ticker": ticker.upper(),
            "date_range": (str(first_date), str(last_date)),
            "rows": int(len(daily_df)),
            "buy_and_hold_return_pct": float(buy_and_hold_return),
            "backtest": result,
        }

    def _slice_period(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        out = df[(df["date"] >= s) & (df["date"] <= e)].copy()
        return out.reset_index(drop=True)

    def _strategy_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "sma",
                "factory": lambda p: SMACrossoverStrategy(
                    short_window=p["short_window"], long_window=p["long_window"]
                ),
                "params": [
                    {"short_window": s, "long_window": l}
                    for s in (10, 15, 20, 30)
                    for l in (40, 60, 90, 120)
                    if s < l
                ],
            },
            {
                "name": "ema",
                "factory": lambda p: EMACrossoverStrategy(short_span=p["short_span"], long_span=p["long_span"]),
                "params": [
                    {"short_span": s, "long_span": l}
                    for s in (8, 12, 16, 24)
                    for l in (26, 40, 60, 90)
                    if s < l
                ],
            },
            {
                "name": "rsi_reversion",
                "factory": lambda p: RSIReversionStrategy(
                    period=p["period"], oversold=p["oversold"], overbought=p["overbought"]
                ),
                "params": [
                    {"period": 14, "oversold": 25.0, "overbought": 70.0},
                    {"period": 14, "oversold": 30.0, "overbought": 70.0},
                    {"period": 21, "oversold": 30.0, "overbought": 75.0},
                ],
            },
            {
                "name": "bollinger_reversion",
                "factory": lambda p: BollingerReversionStrategy(window=p["window"], num_std=p["num_std"]),
                "params": [
                    {"window": 20, "num_std": 2.0},
                    {"window": 20, "num_std": 2.5},
                    {"window": 30, "num_std": 2.0},
                ],
            },
        ]

    def _run_backtest(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        fee_rate: float,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
        initial_capital: float,
    ) -> dict[str, Any]:
        config = BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        return Backtester(strategy=strategy, config=config).run(df)

    def _strategy_selection_and_weights(
        self,
        ticker_df: pd.DataFrame,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fee_rate: float,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
        initial_capital: float,
    ) -> dict[str, Any]:
        selected_strategies: list[dict[str, Any]] = []

        for spec in self._strategy_catalog():
            best = None
            leaderboard: list[dict[str, Any]] = []

            for params in spec["params"]:
                strategy = spec["factory"](params)

                train_result = self._run_backtest(
                    train_df,
                    strategy,
                    fee_rate,
                    stop_loss_pct,
                    take_profit_pct,
                    initial_capital,
                )
                valid_result = self._run_backtest(
                    valid_df,
                    strategy,
                    fee_rate,
                    stop_loss_pct,
                    take_profit_pct,
                    initial_capital,
                )

                valid_sharpe = float(valid_result["metrics"]["sharpe_ratio"])
                valid_return = float(valid_result["metrics"]["total_return_pct"])
                score = max(valid_sharpe, 0.0) * max(valid_return, 0.0)

                leaderboard.append(
                    {
                        "strategy": spec["name"],
                        "params": str(params),
                        "train_total_return_pct": float(train_result["metrics"]["total_return_pct"]),
                        "valid_total_return_pct": valid_return,
                        "valid_sharpe": valid_sharpe,
                        "score": score,
                    }
                )

                if best is None or (score, valid_sharpe, valid_return) > (
                    best["score"],
                    best["valid_sharpe"],
                    best["valid_return"],
                ):
                    best = {
                        "strategy_name": spec["name"],
                        "params": params,
                        "score": score,
                        "valid_sharpe": valid_sharpe,
                        "valid_return": valid_return,
                        "train_result": train_result,
                        "valid_result": valid_result,
                        "strategy_obj": strategy,
                    }

            if best is None:
                continue

            test_result = self._run_backtest(
                test_df,
                best["strategy_obj"],
                fee_rate,
                stop_loss_pct,
                take_profit_pct,
                initial_capital,
            )

            full_signal = best["strategy_obj"].generate_signals(ticker_df.copy())[["date", "signal"]].copy()
            test_signal = full_signal[
                (full_signal["date"] >= test_df["date"].iloc[0])
                & (full_signal["date"] <= test_df["date"].iloc[-1])
            ].reset_index(drop=True)

            selected_strategies.append(
                {
                    "strategy_name": best["strategy_name"],
                    "best_params": best["params"],
                    "score": best["score"],
                    "train_metrics": best["train_result"]["metrics"],
                    "valid_metrics": best["valid_result"]["metrics"],
                    "test_metrics": test_result["metrics"],
                    "test_signals": test_signal,
                    "leaderboard": pd.DataFrame(leaderboard).sort_values(
                        ["score", "valid_sharpe", "valid_total_return_pct"], ascending=False
                    ),
                }
            )

        if not selected_strategies:
            return {"selected": [], "weights": {}}

        positive = [s for s in selected_strategies if s["score"] > 0]
        if not positive:
            positive = sorted(selected_strategies, key=lambda x: x["valid_metrics"]["sharpe_ratio"], reverse=True)[:1]

        total_score = sum(s["score"] for s in positive)
        if total_score <= 0:
            weights = {s["strategy_name"]: 1.0 / len(positive) for s in positive}
        else:
            weights = {s["strategy_name"]: s["score"] / total_score for s in positive}

        for s in selected_strategies:
            s["weight"] = float(weights.get(s["strategy_name"], 0.0))

        return {
            "selected": selected_strategies,
            "weights": weights,
        }

    def _daily_ensemble_signals(
        self,
        test_df: pd.DataFrame,
        selected: list[dict[str, Any]],
        threshold: float = 0.2,
    ) -> pd.DataFrame:
        merged = test_df[["date", "open", "high", "low", "close", "volume"]].copy()
        merged["weighted_signal"] = 0.0

        for s in selected:
            w = float(s.get("weight", 0.0))
            if w <= 0:
                continue

            sig = s["test_signals"].rename(columns={"signal": f"signal_{s['strategy_name']}"})
            merged = merged.merge(sig, on="date", how="left")
            merged[f"signal_{s['strategy_name']}"] = merged[f"signal_{s['strategy_name']}"] .fillna(0).astype(int)
            merged["weighted_signal"] += w * merged[f"signal_{s['strategy_name']}"]

        merged["signal"] = 0
        merged.loc[merged["weighted_signal"] >= threshold, "signal"] = 1
        merged.loc[merged["weighted_signal"] <= -threshold, "signal"] = -1

        return merged

    def _daily_opinion_log(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        out = signal_df[["date", "close", "weighted_signal", "signal"]].copy()
        out["action"] = "HOLD"
        out["opinion"] = "관망"

        position = 0
        actions = []
        opinions = []
        next_pos_list = []

        for row in out.itertuples(index=False):
            action = "HOLD"
            opinion = "관망"
            next_position = position

            if position == 0 and int(row.signal) == 1:
                action = "BUY"
                opinion = "매수"
                next_position = 1
            elif position == 1 and int(row.signal) == -1:
                action = "SELL"
                opinion = "매도"
                next_position = 0

            actions.append(action)
            opinions.append(opinion)
            next_pos_list.append(next_position)
            position = next_position

        out["action"] = actions
        out["opinion"] = opinions
        out["position"] = next_pos_list
        return out

    def _weekly_returns_from_equity(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        eq = equity_curve[["date", "equity"]].copy().set_index("date").sort_index()
        weekly = eq.resample("W-FRI").last().dropna().reset_index()
        weekly["weekly_return_pct"] = weekly["equity"].pct_change().fillna(0.0) * 100.0
        weekly["cum_return_pct"] = (weekly["equity"] / weekly["equity"].iloc[0] - 1.0) * 100.0
        return weekly

    def _reliability_assessment(
        self,
        valid_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        valid_ret = float(valid_metrics["total_return_pct"])
        test_ret = float(test_metrics["total_return_pct"])
        valid_sharpe = float(valid_metrics["sharpe_ratio"])
        test_sharpe = float(test_metrics["sharpe_ratio"])

        overfit_risk = valid_ret > 0 and test_ret < 0

        if test_ret > 0 and test_sharpe > 0 and not overfit_risk:
            grade = "높음"
        elif test_ret > 0:
            grade = "보통"
        else:
            grade = "낮음"

        return {
            "confidence_grade": grade,
            "overfit_risk": overfit_risk,
            "valid_return_pct": valid_ret,
            "test_return_pct": test_ret,
            "valid_sharpe": valid_sharpe,
            "test_sharpe": test_sharpe,
        }

    def evaluate_m7_multi_strategy(
        self,
        train_start: str = "2015-01-01",
        train_end: str = "2020-12-31",
        valid_start: str = "2021-01-01",
        valid_end: str = "2021-12-31",
        test_start: str = "2022-01-01",
        test_end: str = "2024-12-31",
        fee_rate: float = 0.001,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        initial_capital: float = 10_000.0,
        threshold: float = 0.2,
    ) -> dict[str, Any]:
        per_ticker: dict[str, Any] = {}
        all_weekly: list[pd.DataFrame] = []

        for ticker in DEFAULT_M7_TICKERS:
            df = self.fetcher.fetch_daily_ohlcv(ticker)

            train_df = self._slice_period(df, train_start, train_end)
            valid_df = self._slice_period(df, valid_start, valid_end)
            test_df = self._slice_period(df, test_start, test_end)

            if train_df.empty or valid_df.empty or test_df.empty:
                per_ticker[ticker] = {"error": "기간 분할 후 데이터가 비어 있습니다."}
                continue

            selection = self._strategy_selection_and_weights(
                ticker_df=df,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                fee_rate=fee_rate,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                initial_capital=initial_capital,
            )

            selected = selection["selected"]
            if not selected:
                per_ticker[ticker] = {"error": "전략 선택 실패"}
                continue

            ensemble_signal_df = self._daily_ensemble_signals(test_df=test_df, selected=selected, threshold=threshold)
            ensemble_result = self._run_backtest(
                ensemble_signal_df,
                strategy=SignalStrategy(),
                fee_rate=fee_rate,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                initial_capital=initial_capital,
            )

            daily_opinions = self._daily_opinion_log(ensemble_signal_df)
            weekly_returns = self._weekly_returns_from_equity(ensemble_result["equity_curve"])
            weekly_returns.insert(0, "ticker", ticker)
            all_weekly.append(weekly_returns)

            valid_weighted_return = float(sum(s["weight"] * s["valid_metrics"]["total_return_pct"] for s in selected))
            valid_weighted_sharpe = float(sum(s["weight"] * s["valid_metrics"]["sharpe_ratio"] for s in selected))

            reliability = self._reliability_assessment(
                valid_metrics={"total_return_pct": valid_weighted_return, "sharpe_ratio": valid_weighted_sharpe},
                test_metrics=ensemble_result["metrics"],
            )

            per_ticker[ticker] = {
                "selected_strategies": [
                    {
                        "strategy_name": s["strategy_name"],
                        "best_params": s["best_params"],
                        "weight": s["weight"],
                        "train_metrics": s["train_metrics"],
                        "valid_metrics": s["valid_metrics"],
                        "test_metrics": s["test_metrics"],
                    }
                    for s in selected
                ],
                "weights": selection["weights"],
                "ensemble_test_metrics": ensemble_result["metrics"],
                "reliability": reliability,
                "daily_test_opinions": daily_opinions,
                "weekly_test_returns": weekly_returns,
                "leaderboards": {s["strategy_name"]: s["leaderboard"].head(5) for s in selected},
            }

        weekly_all = pd.concat(all_weekly, ignore_index=True) if all_weekly else pd.DataFrame()

        return {
            "tickers": DEFAULT_M7_TICKERS,
            "periods": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
            "per_ticker": per_ticker,
            "weekly_test_all_tickers": weekly_all,
        }

    def estimate_future_volatility(
        self,
        ticker: str,
        interval: str = "1h",
        horizon_steps: int = 24,
        lookback_points: int = 200,
    ) -> dict[str, Any]:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps는 1 이상이어야 합니다.")
        if lookback_points < 30:
            raise ValueError("lookback_points는 최소 30 이상을 권장합니다.")

        prices = self.fetcher.fetch_prices_for_interval(ticker=ticker, interval=interval)
        if len(prices) < lookback_points + 1:
            raise ValueError(
                f"변동성 계산에 필요한 데이터가 부족합니다. (필요: {lookback_points + 1}, 실제: {len(prices)})"
            )

        close = prices["close"].tail(lookback_points + 1)
        log_returns = np.log(close / close.shift(1)).dropna()
        vol_per_step = float(log_returns.std(ddof=1))
        expected_vol = vol_per_step * np.sqrt(horizon_steps)

        return {
            "ticker": ticker.upper(),
            "interval": interval,
            "horizon_steps": horizon_steps,
            "lookback_points": lookback_points,
            "vol_per_step": vol_per_step,
            "expected_vol_for_horizon": expected_vol,
            "expected_vol_for_horizon_pct": expected_vol * 100.0,
            "data_start": str(prices["date"].iloc[0]),
            "data_end": str(prices["date"].iloc[-1]),
        }
