#!/usr/bin/env python3
"""Phase 1: the daily paper-trading job. Idempotent -- safe to re-run any
number of times on the same day; only genuinely new trading days do
anything. If more than one trading day's candle became available since the
last run (Upstox's EOD data publishes with a lag that isn't always exactly
one day), every pending day is walked through IN ORDER, one at a time --
never just the latest -- so no day's fills/exits/signals are silently
skipped. See ``_pending_trading_dates``.

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

Weekly summary: decided every run (not just on days a new candle was
processed) against the real current Asia/Kolkata calendar date -- fires on
Friday, or as a catch-up if it's been >= WEEKLY_SUMMARY_CATCHUP_DAYS since
the last one sent, so a holiday, a data-lag no-op, or a failed Telegram
send on a given Friday can never permanently lose that week's report. See
_weekly_summary_decision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs_delivery, swing_simulator
from data import downloader, instrument_fallback, store
from paper import journal as journal_module
from paper import telegram as telegram_module
from paper.pipeline import run_daily_step
from paper.state import PaperState, load_state, save_state
from signals import condition, vwap
from signals.config import load_signals_config
from strategy import swing_engine
from strategy.base import compute_atr, load_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEFRAME = "swing"
INTERVAL_LABEL = "1d"
IST = ZoneInfo("Asia/Kolkata")
WEEKLY_SUMMARY_CATCHUP_DAYS = 7


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


def _pending_trading_dates(symbol_data: dict[str, pd.DataFrame], state: PaperState) -> list[dt.date]:
    """Every trading date that still needs to be walked through
    ``run_daily_step``, in ascending order -- NOT just the single latest
    one. Upstox's EOD data publishes with a lag (empirically confirmed:
    every real cron run so far has processed the PRIOR day's candle, not
    the day it actually ran on), and that lag isn't constant -- some runs
    see it catch up by more than one day at once. A single-date design
    (``today = min(latest_dates)``) silently SKIPS any earlier day that
    became available in the same fetch: exactly what happened to
    2026-07-22 in production, whose candle arrived bundled together with
    2026-07-23's on the very next run, and was never walked through on its
    own -- no fills/exits were ever checked against its bar, and it's
    missing from run_log.csv entirely. Processing every pending date here,
    oldest first, is what ``scripts/verify_fidelity.py``'s day-by-day-vs-
    batch-replay guarantee actually requires: the day-by-day walk must
    visit every trading day in order, with nothing skipped, or it stops
    being equivalent to the batch backtest engine by construction.

    Dates only count once EVERY symbol in ``symbol_data`` has a candle for
    them (mirrors the existing conservative min-across-symbols philosophy,
    now applied to a range instead of a single date) -- a lone symbol
    lagging behind (a transient per-symbol download failure) holds back
    only the dates it's actually missing, not ones it already has.

    On the very first-ever run (``state.last_processed_date is None``),
    paper trading intentionally starts with a clean, empty book on the
    latest available date -- it does not replay all of history as
    individual days. See ``paper/pipeline.py``'s module docstring for why."""
    if not symbol_data:
        return []
    common_dates = set.intersection(*(set(df["timestamp"].dt.date) for df in symbol_data.values()))
    if not common_dates:
        return []
    if state.last_processed_date is None:
        return [max(common_dates)]
    return sorted(d for d in common_dates if d > state.last_processed_date)


def _ist_today() -> dt.date:
    """The real, current calendar date in Asia/Kolkata -- deliberately NOT
    derived from candle data or the runner's local/UTC clock. The weekly
    summary's Friday check and >7-day catch-up window both need to reason
    about actual elapsed wall-clock time, including on a day when no new
    trading candle was processed at all (an NSE holiday, a data-lag no-op,
    or any other reason the daily pipeline was a no-op) -- tying either
    check to candle-derived dates would make them silently stop firing
    whenever the daily pipeline itself doesn't advance."""
    return dt.datetime.now(IST).date()


def _weekly_summary_decision(state, ist_today: dt.date) -> tuple[bool, str]:
    """Decides whether to send/re-check the weekly summary on this run, and
    why -- pure function, no I/O, so it's directly testable. Two firing
    conditions, evaluated against the real IST calendar date so this works
    identically whether or not today's run processed a new trading candle:

    1. It's Friday (Asia/Kolkata) and a summary hasn't already gone out
       today (same-day re-runs must not resend).
    2. Catch-up: it's been >= WEEKLY_SUMMARY_CATCHUP_DAYS days since the
       last summary was sent (or since paper trading started, if none has
       ever been sent) -- so a Friday that's silently skipped (a holiday,
       a data-availability lag, a prior Telegram failure, a missed cron
       firing) can never permanently lose that week's report; the very
       next run of any kind catches it up.
    """
    last = state.last_weekly_summary_date
    if last == ist_today:
        return False, f"already sent today ({ist_today.isoformat()})"

    if ist_today.weekday() == 4:
        return True, "Friday (Asia/Kolkata)"

    baseline = last or state.paper_start_date
    if baseline is None:
        return False, "not due yet (paper trading hasn't started)"

    days_since = (ist_today - baseline).days
    if days_since >= WEEKLY_SUMMARY_CATCHUP_DAYS:
        last_label = last.isoformat() if last else "never sent"
        return True, f"catch-up -- {days_since} days since last summary ({last_label})"

    return False, f"not due yet ({days_since} days since last summary/start; next Friday or day-{WEEKLY_SUMMARY_CATCHUP_DAYS} catch-up)"


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


def _process_one_day(
    symbol_data: dict[str, pd.DataFrame],
    state: PaperState,
    strategy_cfg,
    cost_cfg,
    portfolio_cfg,
    watchlist: list[str],
    paper_dir: Path,
    today: dt.date,
    no_telegram: bool,
) -> PaperState:
    """Runs exactly one trading day through the pipeline and journals it --
    factored out of ``main()`` so a run with multiple pending days (a
    backfill after a data-lag gap) can call this once per day, in order,
    instead of skipping straight to the latest."""
    if state.paper_start_date is None:
        state.paper_start_date = today
        print(f"[paper] first-ever run -- paper trading starts {today}.")

    print(f"[paper] processing {today}...")
    result = run_daily_step(symbol_data, state, strategy_cfg, cost_cfg, portfolio_cfg, today, watchlist)
    state = result.state

    journal_module.append_trade_rows(paper_dir / "trades.csv", result.trades)
    journal_module.append_journal_rows(paper_dir / "journal.csv", today.isoformat(), result.journal_trace)
    journal_module.append_run_log(paper_dir / "run_log.csv", today.isoformat())

    print(f"[paper] {len(result.fills)} fills, {len(result.trades)} exits, {len(result.new_pending)} new pending orders.")
    for t in result.trades:
        print(f"  EXIT  {t.symbol} {t.exit_reason} @ {t.exit_fill_price:.2f} -> {t.r_multiple:+.2f}R (Rs{t.net_pnl:+,.0f})")
    for f in result.fills:
        print(f"  FILL  {f['symbol']} {f['direction']} x{f['quantity']} @ Rs{f['entry_fill_price']:.2f}")
    for p in result.new_pending:
        print(f"  PENDING  {p['symbol']} {p['direction']} {p['setup_id']} R:R {p['rr_ratio']:.2f} -> fills next session")
    print(f"[paper] equity: Rs{state.equity:,.0f}")

    message = telegram_module.format_daily_message(
        run_date=today.isoformat(),
        fills=result.fills,
        exits=[journal_module.trade_record_to_row(t) for t in result.trades],
        new_pending=result.new_pending,
        equity=state.equity,
    )
    if message and not no_telegram:
        sent = telegram_module.send_message(
            message, os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        )
        print(f"[paper] Telegram daily message: {'sent' if sent else 'FAILED (see [telegram] log line above, or no secrets configured)'}")
    elif message:
        print("[paper] --no-telegram set; message that would have been sent:")
        print(message)
    else:
        print("[paper] no activity today -- no Telegram message.")

    return state


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
    ist_today = _ist_today()

    pending_dates = _pending_trading_dates(symbol_data, state)

    if not pending_dates:
        print(
            f"[paper] no new trading day beyond {state.last_processed_date} -- "
            "market closed or already processed today. No-op."
        )
    else:
        if len(pending_dates) > 1:
            print(
                f"[paper] {len(pending_dates)} trading days pending ({pending_dates[0]} through "
                f"{pending_dates[-1]}) -- backfilling each one in order, not just the latest."
            )
        for today in pending_dates:
            state = _process_one_day(
                symbol_data, state, strategy_cfg, cost_cfg, portfolio_cfg, sorted(watchlist), paper_dir, today, args.no_telegram
            )

    # Weekly summary + fidelity check: evaluated on EVERY run, including a
    # no-op day above, and decided by the real IST calendar date rather than
    # by whether a new trading candle happened to be processed today -- see
    # _weekly_summary_decision's docstring for why (this is exactly the gap
    # that let a Friday's summary silently never fire).
    should_send, reason = _weekly_summary_decision(state, ist_today)
    if should_send:
        print(f"[paper] weekly summary: SENT ({reason})")
        state = _run_weekly_summary(paper_dir, ist_today, args.no_telegram, state)
    else:
        print(f"[paper] weekly summary: skipped ({reason})")

    save_state(state, state_path)
    return 0


def _run_weekly_summary(paper_dir: Path, ist_today: dt.date, no_telegram: bool, state: PaperState) -> PaperState:
    """Sends the weekly summary and runs the fidelity check. Returns
    ``state`` with ``last_weekly_summary_date`` updated to ``ist_today`` --
    callers must persist the returned state (this function doesn't save it
    itself, so ``main()`` can do a single ``save_state`` at the end)."""
    print("[paper] running weekly summary + fidelity check...")
    all_trades = journal_module.read_trades_csv(paper_dir / "trades.csv")
    week_start = ist_today - dt.timedelta(days=ist_today.weekday())
    trades_this_week = [
        t for t in all_trades if week_start <= pd.Timestamp(t["exit_timestamp"]).date() <= ist_today
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

    summary = telegram_module.format_weekly_summary(
        week_label=f"week of {week_start.isoformat()}",
        trades_this_week=trades_this_week,
        cumulative_expectancy_r=cumulative_expectancy_r,
        cumulative_trade_count=len(all_trades),
        equity=state.equity,
        funnel_totals=funnel_totals,
        days_run=days_run,
        days_expected=days_expected,
    )
    print(summary)
    if not no_telegram:
        sent = telegram_module.send_message(
            summary, os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        )
        print(f"[paper] weekly summary Telegram send: {'sent' if sent else 'FAILED (see [telegram] log line above, or no secrets configured)'}")
    else:
        print("[paper] --no-telegram set; weekly summary above would have been sent.")

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

    state.last_weekly_summary_date = ist_today
    return state


if __name__ == "__main__":
    sys.exit(main())
