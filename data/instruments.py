"""Resolve NSE trading symbols to Upstox instrument keys.

Upstox publishes a daily-refreshed JSON instrument master per exchange segment at
https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz. This module
downloads and caches that file, then builds a ``trading_symbol -> instrument_key`` map
for the equity segment (e.g. ``RELIANCE -> NSE_EQ|INE002A01018``).
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import requests

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
INSTRUMENTS_MAX_AGE_SECONDS = 24 * 60 * 60  # Upstox refreshes this file ~daily

# Field names have varied across Upstox instrument file revisions; try each in order.
_SYMBOL_KEYS = ("trading_symbol", "tradingsymbol", "symbol")
_SEGMENT_KEYS = ("segment",)
_TYPE_KEYS = ("instrument_type",)
_KEY_KEYS = ("instrument_key",)


class InstrumentResolutionError(RuntimeError):
    """Raised when one or more watchlist symbols cannot be resolved to instrument keys."""


def _first(d: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def download_instruments(cache_dir: Path, force_refresh: bool = False) -> Path:
    """Download (or reuse a fresh cached copy of) the NSE instrument master.

    Returns the path to the cached, decompressed JSON file.
    """
    instruments_dir = cache_dir / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)
    dest = instruments_dir / "NSE.json"

    if not force_refresh and dest.exists():
        age = time.time() - dest.stat().st_mtime
        if age < INSTRUMENTS_MAX_AGE_SECONDS:
            return dest

    response = requests.get(INSTRUMENTS_URL, timeout=60)
    response.raise_for_status()
    raw = gzip.decompress(response.content)
    dest.write_bytes(raw)
    return dest


def load_instrument_map(cache_dir: Path, force_refresh: bool = False) -> dict[str, str]:
    """Build a ``trading_symbol -> instrument_key`` map for NSE equity instruments."""
    path = download_instruments(cache_dir, force_refresh=force_refresh)
    records = json.loads(path.read_text())

    symbol_map: dict[str, str] = {}
    for record in records:
        segment = _first(record, _SEGMENT_KEYS)
        instrument_type = _first(record, _TYPE_KEYS)
        if segment != "NSE_EQ" or instrument_type != "EQ":
            continue
        symbol = _first(record, _SYMBOL_KEYS)
        instrument_key = _first(record, _KEY_KEYS)
        if symbol and instrument_key:
            symbol_map[symbol] = instrument_key
    return symbol_map


def resolve_symbols(
    watchlist: list[str], cache_dir: Path, force_refresh: bool = False
) -> dict[str, str]:
    """Resolve every symbol in ``watchlist`` to its Upstox instrument key.

    Raises :class:`InstrumentResolutionError` naming every symbol that could not
    be resolved, so a bad watchlist entry fails loudly instead of silently
    downloading a partial dataset.
    """
    symbol_map = load_instrument_map(cache_dir, force_refresh=force_refresh)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for symbol in watchlist:
        instrument_key = symbol_map.get(symbol)
        if instrument_key is None:
            missing.append(symbol)
        else:
            resolved[symbol] = instrument_key

    if missing:
        raise InstrumentResolutionError(
            f"Could not resolve {len(missing)} watchlist symbol(s) to an NSE_EQ "
            f"instrument key: {', '.join(missing)}. Check spelling against the "
            f"cached instrument file at {cache_dir / 'instruments' / 'NSE.json'}, "
            "or delete it to force a fresh download."
        )
    return resolved
