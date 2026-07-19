"""Unit tests for data/instrument_fallback.py -- no network access required.

The central assertion (test_fallback_map_covers_full_watchlist) is what the
paper-trading job actually depends on: if assets.upstox.com is ever
unreachable (as it was in the sandbox this was built in), the hardcoded
VERIFIED_INSTRUMENT_KEYS map must be able to resolve every single watchlist
symbol on its own, not just most of them -- a partial fallback silently
freezes whichever symbols it's missing (exactly what happened to KOTAKBANK/
BAJFINANCE before their post-split ISINs were added).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import instrument_fallback, instruments

REPO_ROOT = Path(__file__).resolve().parent.parent


def _watchlist() -> list[str]:
    with (REPO_ROOT / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    return config["watchlist"]


def test_fallback_map_covers_full_watchlist():
    watchlist = _watchlist()
    missing = [s for s in watchlist if s not in instrument_fallback.VERIFIED_INSTRUMENT_KEYS]
    assert missing == [], f"VERIFIED_INSTRUMENT_KEYS is missing: {missing}"


def test_all_fallback_keys_are_well_formed_nse_eq_isins():
    # NSE_EQ|<12-char ISIN> -- IN (country) + E (India-specific security
    # type marker used by Upstox's NSE_EQ segment) + 9 alphanumeric + 1
    # checksum digit. Catches an obviously malformed entry (typo, wrong
    # segment prefix) without needing network access to verify correctness.
    pattern = re.compile(r"^NSE_EQ\|[A-Z0-9]{12}$")
    for symbol, key in instrument_fallback.VERIFIED_INSTRUMENT_KEYS.items():
        assert pattern.match(key), f"{symbol}: {key!r} doesn't look like a well-formed NSE_EQ instrument key"


def test_kotakbank_and_bajfinance_use_their_post_split_isins():
    # Regression guard for the specific bug this was fixed for: both were
    # previously absent from the map entirely, freezing their cached data.
    assert instrument_fallback.VERIFIED_INSTRUMENT_KEYS["KOTAKBANK"] == "NSE_EQ|INE237A01036"
    assert instrument_fallback.VERIFIED_INSTRUMENT_KEYS["BAJFINANCE"] == "NSE_EQ|INE296A01032"


def test_resolve_symbols_with_fallback_uses_hardcoded_map_when_primary_fails(tmp_path, monkeypatch):
    # Simulates assets.upstox.com being unreachable (load_instrument_map
    # raising) -- every watchlist symbol must still resolve via the
    # hardcoded fallback alone.
    monkeypatch.setattr(
        instruments, "load_instrument_map", lambda cache_dir, force_refresh=False: (_ for _ in ()).throw(RuntimeError("network down"))
    )

    watchlist = _watchlist()
    resolved = instrument_fallback.resolve_symbols_with_fallback(watchlist, tmp_path, strict=True)

    assert set(resolved.keys()) == set(watchlist)
    for symbol in watchlist:
        assert resolved[symbol] == instrument_fallback.VERIFIED_INSTRUMENT_KEYS[symbol]


def test_resolve_symbols_with_fallback_prefers_primary_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        instruments,
        "load_instrument_map",
        lambda cache_dir, force_refresh=False: {"RELIANCE": "NSE_EQ|SOME_PRIMARY_KEY"},
    )

    resolved = instrument_fallback.resolve_symbols_with_fallback(["RELIANCE"], tmp_path, strict=True)
    assert resolved["RELIANCE"] == "NSE_EQ|SOME_PRIMARY_KEY"  # primary wins over the hardcoded fallback


def test_resolve_symbols_with_fallback_strict_raises_on_unresolvable_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(instruments, "load_instrument_map", lambda cache_dir, force_refresh=False: {})

    with pytest.raises(instruments.InstrumentResolutionError):
        instrument_fallback.resolve_symbols_with_fallback(["NOT_A_REAL_SYMBOL"], tmp_path, strict=True)


def test_resolve_symbols_with_fallback_non_strict_skips_unresolvable_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(instruments, "load_instrument_map", lambda cache_dir, force_refresh=False: {})

    resolved = instrument_fallback.resolve_symbols_with_fallback(
        ["RELIANCE", "NOT_A_REAL_SYMBOL"], tmp_path, strict=False
    )
    assert resolved == {"RELIANCE": instrument_fallback.VERIFIED_INSTRUMENT_KEYS["RELIANCE"]}
