#!/usr/bin/env python3
"""CLI entry point for resampling cached candles to a coarser interval.

Usage:
    python scripts/resample_data.py                              # 5min -> 15min, full watchlist
    python scripts/resample_data.py --symbols RELIANCE,SBIN
    python scripts/resample_data.py --source-interval 5 --target-interval 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import resample

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample cached candles to a coarser interval.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols to use instead of the config watchlist."
    )
    parser.add_argument("--source-interval", type=int, default=5, help="Source candle interval in minutes.")
    parser.add_argument("--target-interval", type=int, default=15, help="Target candle interval in minutes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    cache_dir = REPO_ROOT / config["data"]["cache_dir"]
    watchlist = args.symbols.split(",") if args.symbols else config["watchlist"]

    for symbol in watchlist:
        try:
            merged = resample.resample_and_cache(symbol, args.source_interval, args.target_interval, cache_dir)
        except ValueError as exc:
            print(f"[resample] {symbol}: {exc}")
            continue
        print(f"[resample] {symbol}: {len(merged)} {args.target_interval}-min candles cached.")


if __name__ == "__main__":
    main()
