#!/usr/bin/env python3
"""CLI entry point for the historical data downloader.

Usage:
    python scripts/download_data.py             # full download per config.yaml
    python scripts/download_data.py --update     # incremental update only
    python scripts/download_data.py --report     # quality/coverage report only
    python scripts/download_data.py --symbols RELIANCE,SBIN   # override watchlist
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import downloader, instruments, quality, store

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and validate NSE intraday candle data.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true", help="Incremental update only.")
    mode.add_argument("--report", action="store_true", help="Print quality/coverage report only.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols to use instead of the config watchlist."
    )
    return parser.parse_args()


def run_full_download(
    watchlist: list[str],
    symbol_keys: dict[str, str],
    cache_dir: Path,
    dl_config: downloader.DownloaderConfig,
    lookback_months: int,
) -> None:
    for symbol in watchlist:
        print(f"[download] {symbol}: fetching up to {lookback_months} months of history...")
        df = downloader.download_symbol_history(symbol_keys[symbol], lookback_months, dl_config)
        merged = store.write_symbol(symbol, df, dl_config.interval_minutes, cache_dir)
        print(f"[download] {symbol}: fetched {len(df)} new candles, {len(merged)} total cached.")


def run_incremental_update(
    watchlist: list[str],
    symbol_keys: dict[str, str],
    cache_dir: Path,
    dl_config: downloader.DownloaderConfig,
    lookback_months: int,
) -> None:
    for symbol in watchlist:
        last_ts = store.last_timestamp(symbol, dl_config.interval_minutes, cache_dir)
        if last_ts is None:
            print(f"[update] {symbol}: no cache found, running full download instead.")
            df = downloader.download_symbol_history(symbol_keys[symbol], lookback_months, dl_config)
        else:
            print(f"[update] {symbol}: fetching candles newer than {last_ts}...")
            df = downloader.download_incremental(symbol_keys[symbol], last_ts, dl_config)
        merged = store.write_symbol(symbol, df, dl_config.interval_minutes, cache_dir)
        print(f"[update] {symbol}: fetched {len(df)} new candles, {len(merged)} total cached.")


def run_report(watchlist: list[str], cache_dir: Path, interval_minutes: int) -> None:
    reports = []
    for symbol in watchlist:
        df = store.read_symbol(symbol, interval_minutes, cache_dir)
        if df is None:
            print(f"[report] {symbol}: no cached data.")
            continue
        report = quality.build_report(symbol, df, interval_minutes)
        reports.append(report)
        if report.partial_days:
            for day, actual, expected in report.partial_days:
                print(f"[report] {symbol}: gap on {day} ({actual}/{expected} candles)")

    if not reports:
        print("No cached data to report on.")
        return

    table = quality.summary_table(reports)
    print()
    print(table.to_string(index=False))


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    cache_dir = REPO_ROOT / config["data"]["cache_dir"]
    interval_minutes = config["data"]["interval_minutes"]
    lookback_months = config["data"]["lookback_months"]
    watchlist = args.symbols.split(",") if args.symbols else config["watchlist"]

    dl_config = downloader.DownloaderConfig(
        interval_minutes=interval_minutes,
        request_chunk_days=config["data"].get("request_chunk_days", 30),
        request_sleep_seconds=config["data"].get("request_sleep_seconds", 0.35),
        max_retries=config["data"].get("max_retries", 5),
        access_token=os.environ.get("UPSTOX_ACCESS_TOKEN"),
    )

    if args.report:
        run_report(watchlist, cache_dir, interval_minutes)
        return

    symbol_keys = instruments.resolve_symbols(watchlist, cache_dir)

    if args.update:
        run_incremental_update(watchlist, symbol_keys, cache_dir, dl_config, lookback_months)
    else:
        run_full_download(watchlist, symbol_keys, cache_dir, dl_config, lookback_months)

    run_report(watchlist, cache_dir, interval_minutes)


if __name__ == "__main__":
    main()
