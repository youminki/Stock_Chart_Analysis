from __future__ import annotations

import os
import sys
import pathlib
import pandas as pd

# Ensure project root is on sys.path so imports work when running from scripts/
proj_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

from analyzer import AnalysisService, DEFAULT_M7_TICKERS
from data import YahooFinanceDataFetcher


def main():
    tickers = DEFAULT_M7_TICKERS
    service = AnalysisService(fetcher=YahooFinanceDataFetcher())

    rows = []
    for t in tickers:
        try:
            print(f"Analyzing {t}...")
            res = service.analyze_ticker(ticker=t)
            metrics = res["backtest"]["metrics"]
            row = {
                "ticker": res["ticker"],
                "date_start": res["date_range"][0],
                "date_end": res["date_range"][1],
                "rows": res["rows"],
                "buy_and_hold_return_pct": res["buy_and_hold_return_pct"],
            }
            for k, v in metrics.items():
                row[k] = v
            rows.append(row)
        except Exception as e:
            print(f"Error analyzing {t}: {e}")
            rows.append({"ticker": t, "error": str(e)})

    df = pd.DataFrame(rows)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ranking_analysis.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved ranking to {out_path}")


if __name__ == "__main__":
    main()
