#!/usr/bin/env python3
"""Phase 1's pass/fail instrument: re-runs the BACKTEST engine
(``backtest.swing_simulator.run_portfolio``) over the paper period's data,
scoped to start its portfolio walk exactly at ``paper/state.json``'s
``paper_start_date`` (so it replays with the same clean, empty starting
book paper trading actually had), and diffs the result against the real,
git-committed ``paper/trades.csv`` row by row. Every signal, fill price,
exit, and R multiple must match exactly.

    python scripts/verify_fidelity.py
    python scripts/verify_fidelity.py --no-telegram

Exit code 0 = 100% match (or nothing to compare yet). Exit code 1 = at
least one mismatch -- the pass/fail signal this Phase 1 criterion is built on.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs_delivery, swing_simulator
from data import store
from paper import journal as journal_module
from paper import telegram as telegram_module
from paper.state import load_state
from signals import condition, vwap
from strategy import swing_engine
from strategy.base import compute_atr, load_strategy_config
from signals.config import load_signals_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEFRAME = "swing"
INTERVAL_LABEL = "1d"

_COMPARE_FIELDS_EXACT = ["setup_id", "direction", "exit_reason"]
_COMPARE_FIELDS_TIMESTAMP = ["signal_timestamp", "entry_timestamp", "exit_timestamp"]
_COMPARE_FIELDS_FLOAT = [
    "entry_fill_price",
    "stop_price",
    "target_price",
    "exit_fill_price",
    "r_multiple",
    "net_pnl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff paper/trades.csv against a fresh backtest-engine replay.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--paper-dir", default=str(REPO_ROOT / "paper"))
    parser.add_argument("--no-telegram", action="store_true")
    return parser.parse_args()


def _compute_indicators(df: pd.DataFrame, signals_cfg) -> pd.DataFrame:
    df = vwap.compute_weekly_vwap(df, deviation_bands=signals_cfg.deviation_bands)
    monthly = vwap.compute_monthly_vwap(df[["timestamp", "open", "high", "low", "close", "volume"]], deviation_bands=[1])
    df["monthly_vwap"] = monthly["vwap"]
    week_period = df["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN")
    df = condition.compute_condition_periodic(
        df, period_key=week_period, acceptance_candles=signals_cfg.acceptance_candles, value_area_band=signals_cfg.value_area_band
    )
    df = compute_atr(df, period=14)
    df[swing_engine.ROLLING_HIGH_COLUMN] = swing_engine.compute_prior_n_day_high(df)
    return df


def _trade_key(symbol: str, entry_timestamp: str) -> tuple[str, str]:
    return (symbol, entry_timestamp)


def diff_trades(ground_truth_rows: list[dict], paper_rows: list[dict]) -> list[str]:
    """Both args are lists of dicts with the same shape as
    ``paper.journal.trade_record_to_row``'s output (ground truth rows come
    from ``swing_simulator.SwingTradeRecord``s converted the same way).
    Returns a list of human-readable mismatch descriptions -- empty means
    100% fidelity."""
    ground_truth_by_key = {_trade_key(r["symbol"], r["entry_timestamp"]): r for r in ground_truth_rows if r["exit_reason"] != "end_of_data"}
    paper_by_key = {_trade_key(r["symbol"], r["entry_timestamp"]): r for r in paper_rows}

    mismatches = []

    for key, paper_row in paper_by_key.items():
        gt_row = ground_truth_by_key.get(key)
        if gt_row is None:
            mismatches.append(f"{key}: present in paper/trades.csv but the backtest replay never produced this trade")
            continue
        for field in _COMPARE_FIELDS_EXACT + _COMPARE_FIELDS_TIMESTAMP:
            if str(paper_row[field]) != str(gt_row[field]):
                mismatches.append(f"{key}: field '{field}' differs -- paper={paper_row[field]!r} vs backtest={gt_row[field]!r}")
        for field in _COMPARE_FIELDS_FLOAT:
            paper_val, gt_val = float(paper_row[field]), float(gt_row[field])
            if abs(paper_val - gt_val) > max(1e-6, abs(gt_val) * 1e-6):
                mismatches.append(f"{key}: field '{field}' differs -- paper={paper_val} vs backtest={gt_val}")

    for key, gt_row in ground_truth_by_key.items():
        if key not in paper_by_key:
            mismatches.append(f"{key}: the backtest replay produced this trade but it's missing from paper/trades.csv")

    return mismatches


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    paper_dir = Path(args.paper_dir)

    with config_path.open() as f:
        config = yaml.safe_load(f)
    cache_dir = REPO_ROOT / config["data"]["cache_dir"]
    watchlist = config["watchlist"]

    signals_cfg = load_signals_config(config_path, timeframe=TIMEFRAME)
    strategy_cfg = load_strategy_config(config_path, timeframe=TIMEFRAME)
    cost_cfg = costs_delivery.load_delivery_cost_config(config_path)
    portfolio_cfg = swing_simulator.load_swing_portfolio_config(config_path)

    state = load_state(paper_dir / "state.json", capital=strategy_cfg.capital)
    if state.paper_start_date is None:
        print("[fidelity] paper trading hasn't started yet (no state.json / paper_start_date). Nothing to verify.")
        return 0

    symbol_data = {}
    for symbol in watchlist:
        df = store.read_symbol(symbol, 1440, cache_dir, interval_label=INTERVAL_LABEL)
        if df is None or df.empty:
            continue
        symbol_data[symbol] = _compute_indicators(df, signals_cfg)

    ground_truth_trades = swing_simulator.run_portfolio(
        symbol_data, strategy_cfg, cost_cfg, portfolio_cfg, walk_start_date=state.paper_start_date
    )
    ground_truth_rows = [journal_module.trade_record_to_row(t) for t in ground_truth_trades]

    paper_rows = journal_module.read_trades_csv(paper_dir / "trades.csv")

    mismatches = diff_trades(ground_truth_rows, paper_rows)
    n_compared = len(paper_rows)

    print(f"[fidelity] paper_start_date={state.paper_start_date}, paper trades={len(paper_rows)}, "
          f"backtest-replay closed trades={len([r for r in ground_truth_rows if r['exit_reason'] != 'end_of_data'])}")

    if mismatches:
        print(f"[fidelity] FAIL -- {len(mismatches)} mismatch(es):")
        for m in mismatches:
            print(f"  {m}")
        message = telegram_module.format_fidelity_alert(mismatches, n_compared)
        if not args.no_telegram:
            telegram_module.send_message(message, os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"))
        return 1

    print(f"[fidelity] PASS -- 100% match ({n_compared} trades compared).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
