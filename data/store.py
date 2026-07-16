"""Parquet-backed storage for per-symbol candle data.

One file per symbol per interval: ``cache/{SYMBOL}_{interval}min.parquet`` with
columns ``timestamp`` (tz-aware Asia/Kolkata), ``open``, ``high``, ``low``,
``close``, ``volume``. Writes are always merge-append-dedupe so re-running a
download is always safe.

``interval_label`` overrides the ``"{interval_minutes}min"`` filename suffix
(e.g. ``"1d"`` for daily candles, used by Cycle 3's swing backtest) --
``interval_minutes`` remains required everywhere since some callers still key
off it, but is otherwise ignored when a label is given.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IST = "Asia/Kolkata"
SCHEMA_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parquet_path(symbol: str, interval_minutes: int, cache_dir: Path, interval_label: str | None = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    label = interval_label if interval_label is not None else f"{interval_minutes}min"
    return cache_dir / f"{symbol}_{label}.parquet"


def _localize_ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df["timestamp"] = ts
    return df


def read_symbol(
    symbol: str, interval_minutes: int, cache_dir: Path, interval_label: str | None = None
) -> pd.DataFrame | None:
    """Return the cached DataFrame for ``symbol``, or ``None`` if nothing is cached."""
    path = parquet_path(symbol, interval_minutes, cache_dir, interval_label=interval_label)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return _localize_ist(df)[SCHEMA_COLUMNS]


def write_symbol(
    symbol: str, df: pd.DataFrame, interval_minutes: int, cache_dir: Path, interval_label: str | None = None
) -> pd.DataFrame:
    """Merge ``df`` into whatever is already cached for ``symbol`` and persist it.

    Appends new rows, drops duplicate timestamps (keeping the newest fetch), and
    sorts ascending. Returns the merged DataFrame that was written.
    """
    path = parquet_path(symbol, interval_minutes, cache_dir, interval_label=interval_label)
    incoming = _localize_ist(df)[SCHEMA_COLUMNS] if not df.empty else df.reindex(columns=SCHEMA_COLUMNS)

    existing = read_symbol(symbol, interval_minutes, cache_dir, interval_label=interval_label)
    if existing is not None and not existing.empty:
        merged = pd.concat([existing, incoming], ignore_index=True)
    else:
        merged = incoming

    merged = merged.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    merged = merged.reset_index(drop=True)
    merged.to_parquet(path, index=False)
    return merged


def last_timestamp(
    symbol: str, interval_minutes: int, cache_dir: Path, interval_label: str | None = None
) -> pd.Timestamp | None:
    """Return the most recent cached timestamp for ``symbol``, or ``None`` if empty."""
    df = read_symbol(symbol, interval_minutes, cache_dir, interval_label=interval_label)
    if df is None or df.empty:
        return None
    return df["timestamp"].max()


def cached_symbols(interval_minutes: int, cache_dir: Path, interval_label: str | None = None) -> list[str]:
    """List symbols that currently have a cached parquet file for this interval."""
    if not cache_dir.exists():
        return []
    label = interval_label if interval_label is not None else f"{interval_minutes}min"
    suffix = f"_{label}.parquet"
    return sorted(
        p.name[: -len(suffix)] for p in cache_dir.glob(f"*{suffix}") if p.name.endswith(suffix)
    )
