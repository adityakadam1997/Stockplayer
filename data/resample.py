"""Resample cached candles to a coarser interval, session-aware.

Bars are anchored to the 09:15 IST session open *of each calendar day
independently* -- bin edges are computed as offsets from that day's own
09:15, never from the actual first candle present. That's what makes this
"session-aware" in the sense the brief means it: a bar can never straddle
the 09:15 open (the first bin is always exactly [09:15, 09:15+interval)),
and a special session with unusual hours or a partial day (the Diwali
Muhurat session, the Sunday Feb 1 2026 Budget session) just produces fewer
or differently-populated bins from whatever candles that day actually has --
no special-casing needed.

Nothing here re-downloads or re-validates data; it only aggregates candles
already in ``data.store``'s cache. A day's bar count naturally reflects
however many source candles that day had (see ``data.quality.gap_report``
for whether that count is itself suspicious).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import store

SCHEMA_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def resample_to_interval(df: pd.DataFrame, target_minutes: int, session_open: str = "09:15") -> pd.DataFrame:
    """Aggregate ``df`` (the standard 5-min-candle schema) into
    ``target_minutes``-wide session-anchored bars: open=first, high=max,
    low=min, close=last, volume=sum. The output timestamp is each bar's
    *start* time, matching ``data.store``'s existing convention."""
    if df.empty:
        return df.reindex(columns=SCHEMA_COLUMNS)

    df = df.sort_values("timestamp").reset_index(drop=True)
    hour, minute = (int(part) for part in session_open.split(":"))
    session_open_ts = df["timestamp"].dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)

    minutes_since_open = (df["timestamp"] - session_open_ts).dt.total_seconds() / 60.0
    bin_index = np.floor(minutes_since_open / target_minutes).astype("int64")
    bar_start = session_open_ts + pd.to_timedelta(bin_index * target_minutes, unit="m")

    grouped = df.assign(_bar_start=bar_start).groupby("_bar_start", sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    out = out.reset_index().rename(columns={"_bar_start": "timestamp"})
    return out[SCHEMA_COLUMNS]


def resample_and_cache(
    symbol: str,
    source_interval_minutes: int,
    target_interval_minutes: int,
    cache_dir: Path,
    session_open: str = "09:15",
) -> pd.DataFrame:
    """Read ``symbol``'s cached ``source_interval_minutes`` candles, resample
    to ``target_interval_minutes``, and write the result to its own cache
    file (e.g. ``cache/RELIANCE_15min.parquet``) via ``data.store``."""
    source = store.read_symbol(symbol, source_interval_minutes, cache_dir)
    if source is None or source.empty:
        raise ValueError(
            f"No cached {source_interval_minutes}-min data for {symbol} in {cache_dir}; "
            "run scripts/download_data.py first."
        )
    resampled = resample_to_interval(source, target_interval_minutes, session_open=session_open)
    return store.write_symbol(symbol, resampled, target_interval_minutes, cache_dir)
