#!/usr/bin/env python3
"""CLI entry point for the strategy/backtest.

Usage:
    python scripts/run_backtest.py                          # full watchlist, default config
    python scripts/run_backtest.py --symbols RELIANCE,SBIN   # override watchlist
    python scripts/run_backtest.py --no-save                 # skip writing backtest/results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs, report, simulator
from data import store
from signals import condition, vwap
from signals.config import load_signals_config
from strategy.base import load_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VWAP Wave System backtest.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols to use instead of the config watchlist."
    )
    parser.add_argument("--no-save", action="store_true", help="Don't write backtest/results/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    import yaml

    with config_path.open() as f:
        config = yaml.safe_load(f)

    cache_dir = REPO_ROOT / config["data"]["cache_dir"]
    interval_minutes = config["data"]["interval_minutes"]
    watchlist = args.symbols.split(",") if args.symbols else config["watchlist"]

    signals_cfg = load_signals_config(config_path)
    strategy_cfg = load_strategy_config(config_path)
    cost_cfg = costs.load_cost_config(config_path)
    sim_cfg = simulator.load_simulator_config(config_path)

    symbol_data = {}
    for symbol in watchlist:
        df = store.read_symbol(symbol, interval_minutes, cache_dir)
        if df is None or df.empty:
            print(f"[backtest] {symbol}: no cached data, skipping.")
            continue
        df = vwap.compute_session_vwap(df, deviation_bands=signals_cfg.deviation_bands)
        df = condition.compute_condition(
            df,
            acceptance_candles=signals_cfg.acceptance_candles,
            value_area_band=signals_cfg.value_area_band,
        )
        symbol_data[symbol] = df
        print(f"[backtest] {symbol}: {len(df)} candles, {df['timestamp'].dt.date.nunique()} sessions.")

    if not symbol_data:
        print("No cached data to backtest. Run scripts/download_data.py first.")
        return

    trades = simulator.run_backtest(symbol_data, strategy_cfg, cost_cfg, sim_cfg)
    print(f"\n[backtest] {len(trades)} total trades across {len(symbol_data)} symbols.")

    rpt = report.build_report(trades)
    report.print_report(rpt)

    if not args.no_save:
        out_dir = REPO_ROOT / "backtest" / "results"
        report.save_report(rpt, out_dir)
        print(f"\n[backtest] wrote {out_dir / 'summary.csv'} and {out_dir / 'trades.csv'}")


if __name__ == "__main__":
    main()
