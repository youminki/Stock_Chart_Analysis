from __future__ import annotations

import sys
import pathlib
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd

# allow running from scripts/
proj_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

from analyzer import DEFAULT_M7_TICKERS
from data.fetcher import YahooFinanceDataFetcher


def plot_price_with_sma(df: pd.DataFrame, ticker: str, out_dir: str) -> str:
    df = df.copy().set_index("date").sort_index()
    short = df["close"].rolling(window=20).mean()
    long = df["close"].rolling(window=60).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["close"], label="close")
    plt.plot(short.index, short, label="SMA20")
    plt.plot(long.index, long, label="SMA60")
    plt.title(f"{ticker} - Close with SMA20 / SMA60")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_price_sma.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def main(days: int = 365):
    out_dir = os.path.join(proj_root, "results", "plots")
    fetcher = YahooFinanceDataFetcher()

    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=days)

    created = []
    for t in DEFAULT_M7_TICKERS:
        try:
            print(f"Fetching {t}...")
            df = fetcher.fetch_daily_ohlcv(t)
            recent = df[(df["date"] >= start) & (df["date"] <= end)].copy()
            if recent.empty:
                print(f"No recent data for {t}, skipping")
                continue

            path = plot_price_with_sma(recent, t, out_dir)
            created.append(path)
            print(f"Saved plot: {path}")
        except Exception as e:
            print(f"Error for {t}: {e}")

    if created:
        print("Plots created:")
        for p in created:
            print(p)
    else:
        print("No plots created.")


if __name__ == "__main__":
    main()
