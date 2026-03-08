from __future__ import annotations

import sys
import pathlib
import os

import pandas as pd
import matplotlib.pyplot as plt

# ensure project root on path
proj_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

# tickers with high confidence from previous run
TOP_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA"]


def plot_for_ticker(ticker: str, out_dir: str) -> str:
    daily_path = os.path.join(proj_root, "results", f"daily_opinion_{ticker}.csv")
    weekly_path = os.path.join(proj_root, "results", f"weekly_{ticker}.csv")

    if not os.path.exists(daily_path) or not os.path.exists(weekly_path):
        raise FileNotFoundError("Missing daily or weekly CSV for " + ticker)

    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").sort_index()
    weekly = pd.read_csv(weekly_path, parse_dates=["date"]).set_index("date").sort_index()

    buys = daily[daily["action"] == "BUY"]
    sells = daily[daily["action"] == "SELL"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(daily.index, daily["close"], label="Close", color="tab:blue")
    ax1.scatter(buys.index, buys["close"], marker="^", color="green", label="BUY", zorder=5)
    ax1.scatter(sells.index, sells["close"], marker="v", color="red", label="SELL", zorder=5)
    ax1.set_title(f"{ticker} Price with BUY/SELL (ensemble signals)")
    ax1.set_ylabel("Price")
    ax1.legend()

    ax2.plot(weekly.index, weekly["equity"], label="Equity (weekly)", color="tab:orange")
    ax2.set_ylabel("Equity")
    ax2.set_xlabel("Date")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_ensemble_plot.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    out_dir = os.path.join(proj_root, "results", "plots", "ensemble")
    created = []
    for t in TOP_TICKERS:
        try:
            print(f"Plotting ensemble results for {t}...")
            p = plot_for_ticker(t, out_dir)
            created.append(p)
            print(f"Saved: {p}")
        except Exception as e:
            print(f"Error plotting {t}: {e}")

    if created:
        print("Created plots:")
        for c in created:
            print(c)
    else:
        print("No plots created.")


if __name__ == "__main__":
    main()
