"""Unit tests for data/store.py -- no network access required."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import store


def _candles(timestamps: list[str], start_price: float = 100.0) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps).tz_localize("Asia/Kolkata")
    n = len(ts)
    prices = [start_price + i for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * n,
        }
    )


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    df = _candles(["2026-01-05 09:15", "2026-01-05 09:20", "2026-01-05 09:25"])
    store.write_symbol("TEST", df, interval_minutes=5, cache_dir=tmp_path)

    result = store.read_symbol("TEST", interval_minutes=5, cache_dir=tmp_path)
    assert result is not None
    assert len(result) == 3
    assert list(result.columns) == store.SCHEMA_COLUMNS
    assert result["timestamp"].dt.tz is not None


def test_read_missing_symbol_returns_none(tmp_path: Path) -> None:
    assert store.read_symbol("NOPE", interval_minutes=5, cache_dir=tmp_path) is None


def test_write_appends_and_dedupes(tmp_path: Path) -> None:
    first = _candles(["2026-01-05 09:15", "2026-01-05 09:20"])
    store.write_symbol("TEST", first, interval_minutes=5, cache_dir=tmp_path)

    # Overlaps on 09:20 (should be deduped) and adds a new candle at 09:25.
    second = _candles(["2026-01-05 09:20", "2026-01-05 09:25"], start_price=200.0)
    merged = store.write_symbol("TEST", second, interval_minutes=5, cache_dir=tmp_path)

    assert len(merged) == 3
    assert merged["timestamp"].is_monotonic_increasing
    assert merged["timestamp"].duplicated().sum() == 0
    # The later write should win for the overlapping timestamp.
    overlapping_row = merged[merged["timestamp"] == pd.Timestamp("2026-01-05 09:20", tz="Asia/Kolkata")]
    assert overlapping_row.iloc[0]["open"] == 200.0


def test_last_timestamp(tmp_path: Path) -> None:
    assert store.last_timestamp("TEST", interval_minutes=5, cache_dir=tmp_path) is None

    df = _candles(["2026-01-05 09:15", "2026-01-05 09:20"])
    store.write_symbol("TEST", df, interval_minutes=5, cache_dir=tmp_path)

    last = store.last_timestamp("TEST", interval_minutes=5, cache_dir=tmp_path)
    assert last == pd.Timestamp("2026-01-05 09:20", tz="Asia/Kolkata")


def test_cached_symbols(tmp_path: Path) -> None:
    store.write_symbol("AAA", _candles(["2026-01-05 09:15"]), interval_minutes=5, cache_dir=tmp_path)
    store.write_symbol("BBB", _candles(["2026-01-05 09:15"]), interval_minutes=5, cache_dir=tmp_path)
    store.write_symbol("AAA", _candles(["2026-01-05 09:15"]), interval_minutes=15, cache_dir=tmp_path)

    assert store.cached_symbols(5, tmp_path) == ["AAA", "BBB"]
    assert store.cached_symbols(15, tmp_path) == ["AAA"]
