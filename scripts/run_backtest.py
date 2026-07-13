#!/usr/bin/env python3
"""CLI entry point for the strategy/backtest.

Usage:
    python scripts/run_backtest.py                          # full watchlist, default config
    python scripts/run_backtest.py --symbols RELIANCE,SBIN   # override watchlist
    python scripts/run_backtest.py --no-save                 # skip writing backtest/results/

    # In-sample / out-of-sample split (inclusive on both ends, filtered by
    # calendar day before any session-aware indicator is computed, so vwap /
    # atr / condition never bleed across the split boundary):
    python scripts/run_backtest.py --start-date 2025-07-07 --end-date 2026-02-28
    python scripts/run_backtest.py --start-date 2026-03-01 --end-date 2026-07-10

    # Stress test: inflate slippage and brokerage on top of config.yaml's costs:
    python scripts/run_backtest.py --start-date 2026-03-01 --end-date 2026-07-10 \\
        --slippage-multiplier 2.0 --brokerage-multiplier 1.5
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs, report, simulator
from data import store
from signals import condition, vwap
from signals.config import load_signals_config
from strategy.base import compute_atr, load_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VWAP Wave System backtest.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols to use instead of the config watchlist."
    )
    parser.add_argument("--start-date", default=None, help="ISO date (YYYY-MM-DD), inclusive.")
    parser.add_argument("--end-date", default=None, help="ISO date (YYYY-MM-DD), inclusive.")
    parser.add_argument(
        "--slippage-multiplier",
        type=float,
        default=1.0,
        help="Stress-test override: multiplies slippage_ticks, slippage_pct, and stop_slippage_pct.",
    )
    parser.add_argument(
        "--brokerage-multiplier",
        type=float,
        default=1.0,
        help="Stress-test override: multiplies brokerage_per_order.",
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

    if args.slippage_multiplier != 1.0 or args.brokerage_multiplier != 1.0:
        cost_cfg = dataclasses.replace(
            cost_cfg,
            slippage_ticks=cost_cfg.slippage_ticks * args.slippage_multiplier,
            slippage_pct=cost_cfg.slippage_pct * args.slippage_multiplier,
            stop_slippage_pct=cost_cfg.stop_slippage_pct * args.slippage_multiplier,
            brokerage_per_order=cost_cfg.brokerage_per_order * args.brokerage_multiplier,
        )
        print(
            f"[backtest] STRESS TEST: slippage x{args.slippage_multiplier}, "
            f"brokerage x{args.brokerage_multiplier} -> {cost_cfg}"
        )

    start_date = pd.Timestamp(args.start_date).date() if args.start_date else None
    end_date = pd.Timestamp(args.end_date).date() if args.end_date else None

    symbol_data = {}
    for symbol in watchlist:
        df = store.read_symbol(symbol, interval_minutes, cache_dir)
        if df is None or df.empty:
            print(f"[backtest] {symbol}: no cached data, skipping.")
            continue

        if start_date is not None or end_date is not None:
            day = df["timestamp"].dt.date
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= day >= start_date
            if end_date is not None:
                mask &= day <= end_date
            df = df.loc[mask].reset_index(drop=True)
        if df.empty:
            print(f"[backtest] {symbol}: no candles in the selected date range, skipping.")
            continue

        df = vwap.compute_session_vwap(df, deviation_bands=signals_cfg.deviation_bands)
        df = condition.compute_condition(
            df,
            acceptance_candles=signals_cfg.acceptance_candles,
            value_area_band=signals_cfg.value_area_band,
        )
        df = compute_atr(df)
        symbol_data[symbol] = df
        print(f"[backtest] {symbol}: {len(df)} candles, {df['timestamp'].dt.date.nunique()} sessions.")

    if not symbol_data:
        print("No cached data to backtest. Run scripts/download_data.py first.")
        return

    trades = simulator.run_backtest(symbol_data, strategy_cfg, cost_cfg, sim_cfg)
    print(f"\n[backtest] {len(trades)} total trades across {len(symbol_data)} symbols.")

    rpt = report.build_report(trades)
    report.print_report(rpt)

    diagnostics = report.compute_candle_range_diagnostics(symbol_data)
    report.print_candle_range_diagnostics(diagnostics)

    if not args.no_save:
        out_dir = REPO_ROOT / "backtest" / "results"
        report.save_report(rpt, out_dir, candle_range_diagnostics=diagnostics)
        print(f"\n[backtest] wrote {out_dir / 'summary.csv'}, {out_dir / 'trades.csv'}, and diagnostics.")


if __name__ == "__main__":
    main()
