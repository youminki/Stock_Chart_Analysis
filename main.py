from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from analyzer import AnalysisService
from data import TwelveDataFetcher, YahooFinanceDataFetcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="룰 기반 주식 백테스트/변동성 분석 CLI")
    parser.add_argument("--provider", choices=["yahoo", "twelve"], default="yahoo", help="데이터 제공자")
    parser.add_argument("--api-key", default=None, help="Twelve Data API 키 (provider=twelve일 때 사용)")

    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="티커 일봉 전체 기간 백테스트 분석")
    analyze.add_argument("--ticker", required=True, help="예: AAPL")
    analyze.add_argument("--short-window", type=int, default=20)
    analyze.add_argument("--long-window", type=int, default=60)
    analyze.add_argument("--fee-rate", type=float, default=0.001)
    analyze.add_argument("--stop-loss-pct", type=float, default=None)
    analyze.add_argument("--take-profit-pct", type=float, default=None)
    analyze.add_argument("--initial-capital", type=float, default=10_000)

    m7 = sub.add_parser("m7-ensemble", help="M7 다중 전략 평가 + 가중 앙상블 백테스트")
    m7.add_argument("--train-start", default="2015-01-01")
    m7.add_argument("--train-end", default="2020-12-31")
    m7.add_argument("--valid-start", default="2021-01-01")
    m7.add_argument("--valid-end", default="2021-12-31")
    m7.add_argument("--test-start", default="2022-01-01")
    m7.add_argument("--test-end", default="2024-12-31")
    m7.add_argument("--fee-rate", type=float, default=0.001)
    m7.add_argument("--stop-loss-pct", type=float, default=None)
    m7.add_argument("--take-profit-pct", type=float, default=None)
    m7.add_argument("--initial-capital", type=float, default=10_000)
    m7.add_argument("--threshold", type=float, default=0.2, help="가중 신호 임계값")
    m7.add_argument("--save-dir", default="results", help="결과 CSV 저장 폴더")

    vol = sub.add_parser("volatility", help="시간 단위 기반 미래 변동성 추정")
    vol.add_argument("--ticker", required=True, help="예: AAPL")
    vol.add_argument("--interval", default="1h", help="예: 1h, 1d")
    vol.add_argument("--horizon-steps", type=int, default=24, help="미래 예측 스텝 수")
    vol.add_argument("--lookback-points", type=int, default=200, help="최근 수익률 계산 길이")

    return parser


def build_fetcher(provider: str, api_key: str | None):
    if provider == "yahoo":
        return YahooFinanceDataFetcher()
    if provider == "twelve":
        return TwelveDataFetcher(api_key=api_key)
    raise ValueError(f"지원하지 않는 provider: {provider}")


def main() -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    parser = build_parser()
    args = parser.parse_args()

    service = AnalysisService(fetcher=build_fetcher(provider=args.provider, api_key=args.api_key))

    if args.command == "analyze":
        result = service.analyze_ticker(
            ticker=args.ticker,
            short_window=args.short_window,
            long_window=args.long_window,
            fee_rate=args.fee_rate,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            initial_capital=args.initial_capital,
        )

        print(f"[TICKER] {result['ticker']}")
        print(f"[DATE RANGE] {result['date_range'][0]} ~ {result['date_range'][1]}")
        print(f"[ROWS] {result['rows']}")
        print(f"[BUY&HOLD RETURN %] {result['buy_and_hold_return_pct']:.4f}")
        print("[BACKTEST METRICS]")
        print(json.dumps(result["backtest"]["metrics"], indent=2, ensure_ascii=False))

    elif args.command == "m7-ensemble":
        result = service.evaluate_m7_multi_strategy(
            train_start=args.train_start,
            train_end=args.train_end,
            valid_start=args.valid_start,
            valid_end=args.valid_end,
            test_start=args.test_start,
            test_end=args.test_end,
            fee_rate=args.fee_rate,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            initial_capital=args.initial_capital,
            threshold=args.threshold,
        )

        print("[M7 MULTI-STRATEGY ENSEMBLE]")
        print(json.dumps(result["periods"], ensure_ascii=False))

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for ticker, item in result["per_ticker"].items():
            print(f"\n=== {ticker} ===")
            if "error" in item:
                print(f"ERROR: {item['error']}")
                continue

            print("[weights]")
            print(json.dumps(item["weights"], ensure_ascii=False))
            print("[ensemble_test_metrics]")
            print(json.dumps(item["ensemble_test_metrics"], ensure_ascii=False))
            print("[reliability]")
            print(json.dumps(item["reliability"], ensure_ascii=False))

            selected = item["selected_strategies"]
            for s in selected:
                print(
                    f"- {s['strategy_name']} | weight={s['weight']:.4f} | params={s['best_params']} "
                    f"| valid_ret={s['valid_metrics']['total_return_pct']:.4f} "
                    f"| test_ret={s['test_metrics']['total_return_pct']:.4f}"
                )

            weekly = item["weekly_test_returns"]
            daily = item["daily_test_opinions"]
            weekly.to_csv(save_dir / f"weekly_{ticker}.csv", index=False)
            daily.to_csv(save_dir / f"daily_opinion_{ticker}.csv", index=False)

            for strat_name, lb in item["leaderboards"].items():
                lb.to_csv(save_dir / f"leaderboard_{ticker}_{strat_name}.csv", index=False)

        weekly_all = result["weekly_test_all_tickers"]
        if not weekly_all.empty:
            weekly_all.to_csv(save_dir / "weekly_all_m7.csv", index=False)
            print(f"\n[Saved] {save_dir / 'weekly_all_m7.csv'}")

    elif args.command == "volatility":
        result = service.estimate_future_volatility(
            ticker=args.ticker,
            interval=args.interval,
            horizon_steps=args.horizon_steps,
            lookback_points=args.lookback_points,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
