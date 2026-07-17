#!/usr/bin/env python3
"""CLI entry point for Cycle 3's daily-bar swing backtest.

Usage:
    python scripts/run_swing_backtest.py
    python scripts/run_swing_backtest.py --symbols RELIANCE,SBIN
    python scripts/run_swing_backtest.py --start-date 2021-07-09 --end-date 2025-01-08
    python scripts/run_swing_backtest.py --start-date 2025-01-09 --end-date 2026-07-10

    # Stress test: inflate slippage and all fees on top of config.yaml's costs_delivery:
    python scripts/run_swing_backtest.py --start-date 2025-01-09 --end-date 2026-07-10 \\
        --slippage-multiplier 2.0 --fee-multiplier 1.25
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs_delivery, swing_report, swing_simulator
from data import store
from signals import condition, vwap
from signals.config import load_signals_config
from strategy import swing_engine
from strategy.base import compute_atr, load_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEFRAME = "swing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cycle 3's daily-bar swing backtest.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols to use instead of the config watchlist."
    )
    parser.add_argument("--start-date", default=None, help="ISO date (YYYY-MM-DD), inclusive.")
    parser.add_argument("--end-date", default=None, help="ISO date (YYYY-MM-DD), inclusive.")
    parser.add_argument(
        "--slippage-multiplier", type=float, default=1.0, help="Stress-test override: multiplies slippage_pct."
    )
    parser.add_argument(
        "--fee-multiplier",
        type=float,
        default=1.0,
        help="Stress-test override: multiplies STT/txn/SEBI/stamp-duty/DP percentages "
        "(brokerage is already Rs0, unaffected).",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't write backtest/results/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    swing_overrides = config.get("timeframes", {}).get(TIMEFRAME, {})
    cache_dir = REPO_ROOT / config["data"]["cache_dir"]
    interval_minutes = swing_overrides.get("data", {}).get("interval_minutes", 1440)
    watchlist = args.symbols.split(",") if args.symbols else config["watchlist"]

    signals_cfg = load_signals_config(config_path, timeframe=TIMEFRAME)
    strategy_cfg = load_strategy_config(config_path, timeframe=TIMEFRAME)
    cost_cfg = costs_delivery.load_delivery_cost_config(config_path)
    portfolio_cfg = swing_simulator.load_swing_portfolio_config(config_path)

    if args.slippage_multiplier != 1.0 or args.fee_multiplier != 1.0:
        cost_cfg = dataclasses.replace(
            cost_cfg,
            slippage_pct=cost_cfg.slippage_pct * args.slippage_multiplier,
            stt_pct=cost_cfg.stt_pct * args.fee_multiplier,
            txn_charge_pct=cost_cfg.txn_charge_pct * args.fee_multiplier,
            sebi_fee_pct=cost_cfg.sebi_fee_pct * args.fee_multiplier,
            stamp_duty_pct=cost_cfg.stamp_duty_pct * args.fee_multiplier,
            dp_charge_rs=cost_cfg.dp_charge_rs * args.fee_multiplier,
        )
        print(
            f"[swing] STRESS TEST: slippage x{args.slippage_multiplier}, fees x{args.fee_multiplier} -> {cost_cfg}"
        )

    start_date = pd.Timestamp(args.start_date).date() if args.start_date else None
    end_date = pd.Timestamp(args.end_date).date() if args.end_date else None

    symbol_data = {}
    for symbol in watchlist:
        df = store.read_symbol(symbol, interval_minutes, cache_dir, interval_label="1d")
        if df is None or df.empty:
            print(f"[swing] {symbol}: no cached daily data, skipping.")
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
            print(f"[swing] {symbol}: no candles in the selected date range, skipping.")
            continue

        df = vwap.compute_weekly_vwap(df, deviation_bands=signals_cfg.deviation_bands)
        monthly = vwap.compute_monthly_vwap(
            df[["timestamp", "open", "high", "low", "close", "volume"]], deviation_bands=[1]
        )
        df["monthly_vwap"] = monthly["vwap"]
        week_period = df["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN")
        df = condition.compute_condition_periodic(
            df,
            period_key=week_period,
            acceptance_candles=signals_cfg.acceptance_candles,
            value_area_band=signals_cfg.value_area_band,
        )
        df = compute_atr(df, period=14)
        df[swing_engine.ROLLING_HIGH_COLUMN] = swing_engine.compute_prior_n_day_high(df)
        symbol_data[symbol] = df
        print(f"[swing] {symbol}: {len(df)} daily candles, {df['timestamp'].dt.date.min()} to {df['timestamp'].dt.date.max()}.")

    if not symbol_data:
        print("No cached daily data to backtest. Populate cache/<SYMBOL>_1d.parquet first.")
        return

    trades = swing_simulator.run_portfolio(symbol_data, strategy_cfg, cost_cfg, portfolio_cfg)
    print(f"\n[swing] {len(trades)} total trades across {len(symbol_data)} symbols.")

    rpt = swing_report.build_report(trades)
    swing_report.print_report(rpt)

    max_dd_pct = swing_report.compute_max_drawdown_pct_of_capital(rpt, strategy_cfg.capital)
    print(f"\nmax drawdown as % of capital (Rs {strategy_cfg.capital:,.0f}): {max_dd_pct:.2f}%")

    executed_by_symbol: dict[str, int] = {}
    for trade in trades:
        executed_by_symbol[trade.symbol] = executed_by_symbol.get(trade.symbol, 0) + 1
    funnel_table = swing_report.compute_funnel_table(symbol_data, strategy_cfg, cost_cfg, executed_by_symbol)
    swing_report.print_funnel_table(funnel_table)

    rr_distribution = swing_report.compute_rr_distribution(symbol_data, strategy_cfg, cost_cfg)
    swing_report.print_rr_distribution(rr_distribution, strategy_cfg.min_rr)

    cost_risk_distribution = swing_report.compute_cost_risk_ratio_distribution(rpt.trades_df)
    swing_report.print_cost_risk_ratio_distribution(cost_risk_distribution)

    stop_distance_table = swing_report.compute_stop_distance_table(rpt.trades_df)
    swing_report.print_stop_distance_table(stop_distance_table)

    holding_period_table = swing_report.compute_holding_period_table(rpt.trades_df)
    swing_report.print_holding_period_table(holding_period_table)

    all_dates = pd.concat([df["timestamp"] for df in symbol_data.values()])
    n_months = max((all_dates.max() - all_dates.min()).days / 30.44, 1e-9)
    trades_per_month_table = swing_report.compute_trades_per_month_table(rpt.trades_df, n_months, len(symbol_data))
    swing_report.print_trades_per_month_table(trades_per_month_table)

    if not args.no_save:
        out_dir = REPO_ROOT / "backtest" / "results_swing"
        swing_report.save_report(rpt, out_dir)
        funnel_table.to_csv(out_dir / "funnel.csv", index=False)
        stop_distance_table.to_csv(out_dir / "stop_distance.csv", index=False)
        holding_period_table.to_csv(out_dir / "holding_period.csv", index=False)
        trades_per_month_table.to_csv(out_dir / "trades_per_month.csv", index=False)
        print(f"\n[swing] wrote {out_dir / 'summary.csv'}, {out_dir / 'trades.csv'}, and diagnostics.")


if __name__ == "__main__":
    main()
