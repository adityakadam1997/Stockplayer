"""Hardcoded, live-API-verified NSE instrument key fallback for the paper
trading daily job.

``data.instruments.resolve_symbols`` is the primary resolution path (it
downloads Upstox's full instrument master from ``assets.upstox.com``), but
that host was unreachable from the sandbox this was built in (blocked at the
network-egress layer, confirmed via ``curl`` returning no response). The
paper-trading cron needs to keep working even if ``assets.upstox.com`` is
ever unreachable or its file format changes, so this module provides a
hardcoded fallback: for each symbol below, ``NSE_EQ|<ISIN>`` was verified
directly against the (separately reachable) ``api.upstox.com`` historical
candle endpoint in this session -- a real request for that exact key
returned real, recent OHLC data for that symbol.

**13 of the 15 watchlist symbols are covered.** KOTAKBANK and BAJFINANCE
could NOT be independently verified: every ISIN this session tried for them
was rejected by the live API as "Invalid Instrument key", and every
secondary source (NSE archives, Wikipedia, Moneycontrol, Screener) was also
unreachable from this sandbox to cross-check the correct ISIN. Guessing an
unverified financial identifier risks silently pulling a DIFFERENT
company's data under the right symbol name -- worse than no fallback at
all -- so those two are deliberately omitted here. They rely solely on the
primary ``data.instruments`` path, which should work fine in the actual
GitHub Actions runner (a normal internet host, not this sandbox).

Also flagged for the record: the existing ``cache/KOTAKBANK_1d.parquet``
and ``cache/BAJFINANCE_1d.parquet`` files (downloaded in an earlier, now-
merged cycle) show price ranges that don't match either company's known
real-world trading range for 2021-2026 (KOTAKBANK: Rs309-454 cached vs.
Kotak Mahindra Bank's actual ~Rs1600-2200; BAJFINANCE: Rs528-1094 cached vs.
Bajaj Finance's actual multi-thousand-rupee range even after its 2023
split). This suggests those two symbols' historical cache may have been
populated against the wrong instrument key at some point. Not fixed here --
"no changes to strategy/signals/backtest logic" is this phase's explicit
rule, and Phase 1 is a plumbing gate, not a data-quality audit -- but
flagged prominently so a human can re-verify before trusting KOTAKBANK/
BAJFINANCE results specifically.
"""

from __future__ import annotations

from pathlib import Path

from data import instruments

# Verified via a live GET to api.upstox.com/v3/historical-candle/<key>/days/1/...
# in this session -- every entry below returned real OHLC data for that symbol.
VERIFIED_INSTRUMENT_KEYS: dict[str, str] = {
    "RELIANCE": "NSE_EQ|INE002A01018",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "INFY": "NSE_EQ|INE009A01021",
    "TCS": "NSE_EQ|INE467B01029",
    "SBIN": "NSE_EQ|INE062A01020",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "TATASTEEL": "NSE_EQ|INE081A01020",
    "AXISBANK": "NSE_EQ|INE238A01034",
    "LT": "NSE_EQ|INE018A01030",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "MARUTI": "NSE_EQ|INE585B01010",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
}


def resolve_symbols_with_fallback(
    watchlist: list[str], cache_dir: Path, force_refresh: bool = False, strict: bool = True
) -> dict[str, str]:
    """Resolve every symbol in ``watchlist`` to an instrument key, preferring
    the live ``assets.upstox.com`` instrument master and falling back to
    ``VERIFIED_INSTRUMENT_KEYS`` per-symbol if that fails entirely (network
    error, format change, etc).

    ``strict=True`` (default) raises :class:`instruments.InstrumentResolutionError`
    naming any symbol resolved by neither path -- use this when a partial
    watchlist is unacceptable. ``strict=False`` returns whatever subset WAS
    resolved and just prints a warning per missing symbol -- the daily paper
    job uses this so one bad symbol (e.g. an unverifiable instrument key)
    doesn't halt the other 14."""
    try:
        symbol_map = instruments.load_instrument_map(cache_dir, force_refresh=force_refresh)
    except Exception:
        symbol_map = {}

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for symbol in watchlist:
        key = symbol_map.get(symbol) or VERIFIED_INSTRUMENT_KEYS.get(symbol)
        if key is None:
            missing.append(symbol)
        else:
            resolved[symbol] = key

    if missing:
        if strict:
            raise instruments.InstrumentResolutionError(
                f"Could not resolve {len(missing)} watchlist symbol(s) to an instrument key via "
                f"either assets.upstox.com or the hardcoded fallback: {', '.join(missing)}."
            )
        print(
            f"[instrument_fallback] WARNING: could not resolve {len(missing)} symbol(s), "
            f"skipping them: {', '.join(missing)}"
        )
    return resolved
