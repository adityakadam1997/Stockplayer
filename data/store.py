"""Parquet-backed storage for per-symbol candle data.

One file per symbol per interval: ``cache/{SYMBOL}_{interval}min.parquet`` with
columns ``timestamp`` (tz-aware Asia/Kolkata), ``open``, ``high``, ``low``,
``close``, ``volume``. Writes are always merge-append-dedupe so re-running a
download is always safe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IST = "Asia/Kolkata"
SCHEMA_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parquet_path(symbol: str, interval_minutes: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{interval_minutes}min.parquet"


def _localize_ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df["timestamp"] = ts
    return df


def read_symbol(symbol: str, interval_minutes: int, cache_dir: Path) -> pd.DataFrame | None:
    """Return the cached DataFrame for ``symbol``, or ``None`` if nothing is cached."""
    path = parquet_path(symbol, interval_minutes, cache_dir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return _localize_ist(df)[SCHEMA_COLUMNS]


def write_symbol(
    symbol: str, df: pd.DataFrame, interval_minutes: int, cache_dir: Path
) -> pd.DataFrame:
    """Merge ``df`` into whatever is already cached for ``symbol`` and persist it.

    Appends new rows, drops duplicate timestamps (keeping the newest fetch), and
    sorts ascending. Returns the merged DataFrame that was written.
    """
    path = parquet_path(symbol, interval_minutes, cache_dir)
    incoming = _localize_ist(df)[SCHEMA_COLUMNS] if not df.empty else df.reindex(columns=SCHEMA_COLUMNS)

    existing = read_symbol(symbol, interval_minutes, cache_dir)
    if existing is not None and not existing.empty:
        merged = pd.concat([existing, incoming], ignore_index=True)
    else:
        merged = incoming

    merged = merged.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    merged = merged.reset_index(drop=True)
    merged.to_parquet(path, index=False)
    return merged


def last_timestamp(symbol: str, interval_minutes: int, cache_dir: Path) -> pd.Timestamp | None:
    """Return the most recent cached timestamp for ``symbol``, or ``None`` if empty."""
    df = read_symbol(symbol, interval_minutes, cache_dir)
    if df is None or df.empty:
        return None
    return df["timestamp"].max()


def cached_symbols(interval_minutes: int, cache_dir: Path) -> list[str]:
    """List symbols that currently have a cached parquet file for this interval."""
    if not cache_dir.exists():
        return []
    suffix = f"_{interval_minutes}min.parquet"
    return sorted(
        p.name[: -len(suffix)] for p in cache_dir.glob(f"*{suffix}") if p.name.endswith(suffix)
    )
