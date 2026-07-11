"""Fetch and cache historical candles from the Upstox V3 Historical Candle API.

Endpoint (V3, supports arbitrary minute intervals unlike V2's fixed 1/30 minute
buckets):

    GET https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}

``unit`` is one of ``minutes``, ``hours``, ``days``, ``weeks``, ``months``. Minute
and hour candles are only available from 2022-01-01 onwards. The endpoint appears
to work without an Authorization header; if ``UPSTOX_ACCESS_TOKEN`` is set it is
sent as a bearer token, but it is never required.

Candles are returned newest-first as arrays of
``[timestamp, open, high, low, close, volume, open_interest]``.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import pandas as pd
import requests

API_BASE = "https://api.upstox.com/v3/historical-candle"
MINUTE_DATA_FLOOR = dt.date(2022, 1, 1)  # Upstox minute/hour data starts here
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]


class UpstoxAPIError(RuntimeError):
    """Raised when the Upstox API returns a non-retryable error."""


@dataclass
class DownloaderConfig:
    interval_minutes: int = 5
    request_chunk_days: int = 30
    request_sleep_seconds: float = 0.35
    max_retries: int = 5
    access_token: str | None = None


def _headers(access_token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _request_chunk(
    instrument_key: str,
    unit: str,
    interval: int,
    from_date: dt.date,
    to_date: dt.date,
    config: DownloaderConfig,
) -> list[list]:
    """Fetch one from/to window, retrying on HTTP 429 with exponential backoff."""
    url = f"{API_BASE}/{instrument_key}/{unit}/{interval}/{to_date.isoformat()}/{from_date.isoformat()}"

    backoff = 1.0
    for attempt in range(config.max_retries + 1):
        response = requests.get(url, headers=_headers(config.access_token), timeout=30)
        if response.status_code == 429:
            if attempt == config.max_retries:
                raise UpstoxAPIError(f"Rate limited repeatedly fetching {url}")
            time.sleep(backoff)
            backoff *= 2
            continue
        if not response.ok:
            raise UpstoxAPIError(
                f"Upstox API error {response.status_code} for {url}: {response.text[:500]}"
            )
        payload = response.json()
        if payload.get("status") != "success":
            raise UpstoxAPIError(f"Upstox API returned non-success payload for {url}: {payload}")
        return payload.get("data", {}).get("candles", [])

    return []  # unreachable, satisfies type checkers


def _candles_to_frame(candles: list[list]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS[:6]).astype(
            {"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "int64"}
        )
    df = pd.DataFrame(candles, columns=CANDLE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.drop(columns=["open_interest"])
    return df


def download_symbol_history(
    instrument_key: str,
    lookback_months: int,
    config: DownloaderConfig,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    """Page backwards through history for one symbol until ``lookback_months`` is
    covered or the API has no more data (floored at the minute-data start date).

    Returns a DataFrame sorted ascending by timestamp, deduplicated.
    """
    end_date = end_date or dt.date.today()
    start_date = max(
        end_date - dt.timedelta(days=lookback_months * 31),
        MINUTE_DATA_FLOOR,
    )

    frames: list[pd.DataFrame] = []
    window_end = end_date
    while window_end >= start_date:
        window_start = max(window_end - dt.timedelta(days=config.request_chunk_days - 1), start_date)

        candles = _request_chunk(
            instrument_key, "minutes", config.interval_minutes, window_start, window_end, config
        )
        frames.append(_candles_to_frame(candles))

        window_end = window_start - dt.timedelta(days=1)
        time.sleep(config.request_sleep_seconds)

    if not frames:
        return _candles_to_frame([])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return combined.reset_index(drop=True)


def download_incremental(
    instrument_key: str,
    since: pd.Timestamp,
    config: DownloaderConfig,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    """Fetch candles strictly newer than ``since`` up to ``end_date`` (default today)."""
    end_date = end_date or dt.date.today()
    start_date = since.date()
    if start_date > end_date:
        return _candles_to_frame([])

    frames: list[pd.DataFrame] = []
    window_end = end_date
    while window_end >= start_date:
        window_start = max(window_end - dt.timedelta(days=config.request_chunk_days - 1), start_date)

        candles = _request_chunk(
            instrument_key, "minutes", config.interval_minutes, window_start, window_end, config
        )
        frames.append(_candles_to_frame(candles))

        window_end = window_start - dt.timedelta(days=1)
        time.sleep(config.request_sleep_seconds)

    if not frames:
        return _candles_to_frame([])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["timestamp"] > since]
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return combined.reset_index(drop=True)
