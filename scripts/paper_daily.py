#!/usr/bin/env python3
"""Phase 1: the daily paper-trading job. Idempotent -- safe to re-run any
number of times on the same day; only the first run on a genuinely new
trading day does anything.

    python scripts/paper_daily.py
    python scripts/paper_daily.py --no-telegram   # for local/manual runs
    python scripts/paper_daily.py --symbols RELIANCE,SBIN
    python scripts/paper_daily.py --test-telegram # sends a short connectivity test message and exits;
                                                    # does not touch data/state/journal at all

Steps (see paper/ package docstrings for the detailed design rationale):
1. Incrementally update daily candles for the watchlist.
2. Fill pending orders from the previous session at today's open.
3. Manage open positions against today's completed bar (stop/target/time-stop).
4. Generate new proposals from today's close -> pending orders for tomorrow.
5. Journal everything to paper/journal.csv + paper/trades.csv + paper/state.json.
6. Notify Telegram (silent on a no-activity day).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs_delivery, swing_simulator
from data import downloader, instrument_fallback, store
from paper import journal as journal_module
from paper import telegram as telegram_module
from paper.pipeline import run_daily_step
from paper.state import load_state, save_state
from signals import condition, vwap
from signals.config import load_signals_config
from strategy import swing_engine
from strategy.base import compute_atr, load_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEFRAME = "swing"
INTERVAL_LABEL = "1d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one day of Phase 1 paper trading.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--symbols", default=None, help="Comma-separated override of the watchlist.")
    parser.add_argument("--no-telegram", action="store_true", help="Don't send a Telegram message even if configured.")
    parser.add_argument("--paper-dir", default=str(REPO_ROOT / "paper"), help="Where journal.csv/trades.csv/state.json live.")
    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Send a short 'connection OK' test message using the configured TELEGRAM_BOT_TOKEN/"
        "TELEGRAM_CHAT_ID and exit -- does not download data, touch state/journal, or run any part "
        "of the daily job.",
    )
    return parser.parse_args()


TEST_TELEGRAM_MESSAGE = "Stockplayer test -- connection OK"


def _send_test_telegram() -> int:
    sent = telegram_module.send_message(
        TEST_TELEGRAM_MESSAGE, os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    )
    if sent:
        print(f"[paper] test Telegram message sent: {TEST_TELEGRAM_MESSAGE!r}")
        return 0
    print("[paper] TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID not set -- nothing sent.")
    return 1


def _update_symbol_data(symbol: str, instrument_key: str, cache_dir: Path, dl_cfg: downloader.DownloaderConfig) -> pd.DataFrame | None:
    existing = store.read_symbol(symbol, 1440, cache_dir, interval_label=INTERVAL_LABEL)
    if existing is None or existing.empty:
        df = downloader.download_symbol_daily_history(instrument_key, lookback_years=6, config=dl_cfg)
    else:
        since = existing["timestamp"].max()
        new_rows = downloader.download_incremental_daily(instrument_key, since=since, config=dl_cfg)
        df = new_rows
    if df is not None and not df.empty:
        return store.write_symbol(symbol, df, 1440, cache_dir, interval_label=INTERVAL_LABEL)
    return store.read_symbol(symbol, 1440, cache_dir, interval_label=INTERVAL_LABEL)


def _should_skip(state, candidate_today) -> bool:
    """True if there's no new trading day to process -- either an NSE
    holiday (no symbol got a fresher candle than what's already cached) or
    the job has already fully processed ``candidate_today`` earlier today.
    Both manifest identically: the latest available date across every
    symbol is not strictly newer than ``state.last_processed_date``."""
    return state.last_processed_date is not None and candidate_today <= state.last_processed_date


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


def main() -> int:
    args = parse_args()

    if args.test_telegram:
        return _send_test_telegram()

    config_path = Path(args.config)
    paper_dir = Path(args.paper_dir)

    # Guarantee journal.csv/trades.csv/run_log.csv exist (header-only if
    # nothing's happened yet) before anything else runs, including before
    # any FATAL-exit path below -- a zero-activity day (the common case,
    # and always true on the very first run) would otherwise leave
    # trades.csv absent from disk entirely, which is exactly what broke the
    # workflow's `git add paper/trades.csv` step on its first real run.
    journal_module.ensure_files(paper_dir)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    cache_dir = REPO_ROOT / config["data"]["cache_dir"]
    watchlist = args.symbols.split(",") if args.symbols else config["watchlist"]

    signals_cfg = load_signals_config(config_path, timeframe=TIMEFRAME)
    strategy_cfg = load_strategy_config(config_path, timeframe=TIMEFRAME)
    cost_cfg = costs_delivery.load_delivery_cost_config(config_path)
    portfolio_cfg = swing_simulator.load_swing_portfolio_config(config_path)

    dl_raw = config.get("data", {})
    dl_cfg = downloader.DownloaderConfig(
        interval_minutes=1440,
        request_chunk_days=dl_raw.get("request_chunk_days", 28),
        request_sleep_seconds=dl_raw.get("request_sleep_seconds", 0.35),
        max_retries=dl_raw.get("max_retries", 5),
        access_token=os.environ.get("UPSTOX_ACCESS_TOKEN"),
    )

    print(f"[paper] resolving instrument keys for {len(watchlist)} symbols...")
    instrument_keys = instrument_fallback.resolve_symbols_with_fallback(watchlist, cache_dir, strict=False)
    if not instrument_keys:
        print("[paper] FATAL: could not resolve ANY watchlist symbol to an instrument key.")
        return 1

    symbol_data: dict[str, pd.DataFrame] = {}
    for symbol in instrument_keys:
        try:
            df = _update_symbol_data(symbol, instrument_keys[symbol], cache_dir, dl_cfg)
        except Exception as exc:
            print(f"[paper] {symbol}: download failed ({exc}), using whatever is cached.")
            df = store.read_symbol(symbol, 1440, cache_dir, interval_label=INTERVAL_LABEL)
        if df is None or df.empty:
            print(f"[paper] {symbol}: no data at all, skipping.")
            continue
        symbol_data[symbol] = _compute_indicators(df, signals_cfg)

    if not symbol_data:
        print("[paper] FATAL: no symbol data available at all.")
        return 1

    state_path = paper_dir / "state.json"
    state = load_state(state_path, capital=strategy_cfg.capital)

    latest_dates = [df["timestamp"].max().date() for df in symbol_data.values()]
    candidate_today = min(latest_dates)

    if _should_skip(state, candidate_today):
        print(
            f"[paper] no new trading day beyond {state.last_processed_date} "
            f"(latest available: {candidate_today}) -- market closed or already processed today. No-op."
        )
        return 0

    today = candidate_today
    if state.paper_start_date is None:
        state.paper_start_date = today
        print(f"[paper] first-ever run -- paper trading starts {today}.")

    print(f"[paper] processing {today}...")
    result = run_daily_step(symbol_data, state, strategy_cfg, cost_cfg, portfolio_cfg, today, sorted(watchlist))

    journal_module.append_trade_rows(paper_dir / "trades.csv", result.trades)
    journal_module.append_journal_rows(paper_dir / "journal.csv", today.isoformat(), result.journal_trace)
    journal_module.append_run_log(paper_dir / "run_log.csv", today.isoformat())
    save_state(result.state, state_path)

    print(f"[paper] {len(result.fills)} fills, {len(result.trades)} exits, {len(result.new_pending)} new pending orders.")
    for t in result.trades:
        print(f"  EXIT  {t.symbol} {t.exit_reason} @ {t.exit_fill_price:.2f} -> {t.r_multiple:+.2f}R (Rs{t.net_pnl:+,.0f})")
    for f in result.fills:
        print(f"  FILL  {f['symbol']} {f['direction']} x{f['quantity']} @ Rs{f['entry_fill_price']:.2f}")
    for p in result.new_pending:
        print(f"  PENDING  {p['symbol']} {p['direction']} {p['setup_id']} R:R {p['rr_ratio']:.2f} -> fills next session")
    print(f"[paper] equity: Rs{result.state.equity:,.0f}")

    message = telegram_module.format_daily_message(
        run_date=today.isoformat(),
        fills=result.fills,
        exits=[journal_module.trade_record_to_row(t) for t in result.trades],
        new_pending=result.new_pending,
        equity=result.state.equity,
    )
    if message and not args.no_telegram:
        sent = telegram_module.send_message(
            message, os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        )
        print(f"[paper] Telegram message sent: {sent}")
    elif message:
        print("[paper] --no-telegram set; message that would have been sent:")
        print(message)
    else:
        print("[paper] no activity today -- no Telegram message.")

    if today.weekday() == 4:  # Friday
        _run_weekly_summary(paper_dir, today, args.no_telegram)

    return 0


def _run_weekly_summary(paper_dir: Path, today, no_telegram: bool) -> None:
    print("[paper] Friday -- running weekly summary + fidelity check...")
    all_trades = journal_module.read_trades_csv(paper_dir / "trades.csv")
    week_start = today - pd.Timedelta(days=today.weekday())
    trades_this_week = [
        t for t in all_trades if week_start <= pd.Timestamp(t["exit_timestamp"]).date() <= today
    ]

    if all_trades:
        cumulative_expectancy_r = sum(float(t["r_multiple"]) for t in all_trades) / len(all_trades)
    else:
        cumulative_expectancy_r = 0.0

    all_journal_rows = journal_module.read_journal_csv(paper_dir / "journal.csv")
    funnel_totals = journal_module.compute_funnel_totals(all_journal_rows)

    run_log = journal_module.read_run_log(paper_dir / "run_log.csv")
    days_run = len(run_log)
    run_dates = [pd.Timestamp(r["run_date"]).date() for r in run_log]
    days_expected = len(pd.bdate_range(min(run_dates), max(run_dates))) if run_dates else 0

    equity_state = load_state(paper_dir / "state.json")

    summary = telegram_module.format_weekly_summary(
        week_label=f"week of {week_start.isoformat()}",
        trades_this_week=trades_this_week,
        cumulative_expectancy_r=cumulative_expectancy_r,
        cumulative_trade_count=len(all_trades),
        equity=equity_state.equity,
        funnel_totals=funnel_totals,
        days_run=days_run,
        days_expected=days_expected,
    )
    print(summary)
    if not no_telegram:
        telegram_module.send_message(summary, os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"))

    fidelity_script = REPO_ROOT / "scripts" / "verify_fidelity.py"
    result = subprocess.run(
        [sys.executable, str(fidelity_script)] + (["--no-telegram"] if no_telegram else []),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[paper] FIDELITY CHECK FAILED -- see output above.")
    else:
        print("[paper] fidelity check passed.")


if __name__ == "__main__":
    sys.exit(main())
