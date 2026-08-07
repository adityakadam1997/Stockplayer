#!/usr/bin/env python3
"""CLI entry point for the strategy backtest.

Usage:
    python scripts/run_backtest.py                          # full watchlist, full cache
    python scripts/run_backtest.py --symbols RELIANCE,SBIN   # override the watchlist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import report
from backtest.costs import load_cost_config
from backtest.simulator import run_backtest
from signals.config import load_signals_config
from strategy.base import load_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VWAP Wave System backtest.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols to use instead of the config watchlist."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open() as f:
        raw_config = yaml.safe_load(f)

    cache_dir = REPO_ROOT / raw_config["data"]["cache_dir"]
    interval_minutes = raw_config["data"]["interval_minutes"]
    watchlist = args.symbols.split(",") if args.symbols else raw_config["watchlist"]

    signals_config = load_signals_config(config_path)
    strategy_config = load_strategy_config(config_path)
    cost_config = load_cost_config(config_path)
    capital = raw_config["backtest"]["capital"]
    risk_pct = raw_config["backtest"]["risk_pct"]

    print(f"Running backtest over {len(watchlist)} symbols: {', '.join(watchlist)}")
    records = run_backtest(
        watchlist, cache_dir, interval_minutes, signals_config, strategy_config, cost_config, capital, risk_pct
    )
    print(f"{len(records)} trades generated.\n")

    trades = report.trades_to_dataframe(records)
    summary = report.build_summary(trades, capital)
    report.print_report(summary, capital)

    results_dir = REPO_ROOT / "backtest" / "results"
    report.save_results(trades, summary, results_dir)
    print(f"\nSaved {results_dir / 'trades.csv'} and {results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
