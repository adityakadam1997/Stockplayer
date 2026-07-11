"""Unit tests for data/quality.py -- no network access required."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import quality


def _clean_day(date: str) -> pd.DataFrame:
    """A full, valid trading day of 5-minute candles (09:15-15:25, 75 candles)."""
    ts = pd.date_range(f"{date} 09:15", f"{date} 15:25", freq="5min", tz="Asia/Kolkata")
    n = len(ts)
    price = [100.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": price,
            "high": [p + 0.5 for p in price],
            "low": [p - 0.5 for p in price],
            "close": price,
            "volume": [1000] * n,
        }
    )


def test_check_monotonic_unique_clean() -> None:
    df = _clean_day("2026-01-05")
    dupes, non_increasing = quality.check_monotonic_unique(df)
    assert dupes == 0
    assert non_increasing == 0


def test_check_monotonic_unique_detects_duplicate() -> None:
    df = _clean_day("2026-01-05")
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dupes, _ = quality.check_monotonic_unique(df)
    assert dupes == 1


def test_check_market_hours_clean() -> None:
    df = _clean_day("2026-01-05")  # Monday
    assert quality.check_market_hours(df) == 0


def test_check_market_hours_flags_outside_window() -> None:
    df = _clean_day("2026-01-05")
    bad_row = df.iloc[[0]].copy()
    bad_row["timestamp"] = pd.Timestamp("2026-01-05 16:00", tz="Asia/Kolkata")
    df = pd.concat([df, bad_row], ignore_index=True)
    assert quality.check_market_hours(df) == 1


def test_check_market_hours_flags_weekend() -> None:
    df = _clean_day("2026-01-10")  # Saturday
    assert quality.check_market_hours(df) == len(df)


def test_check_ohlc_clean() -> None:
    df = _clean_day("2026-01-05")
    assert quality.check_ohlc(df) == 0


def test_check_ohlc_detects_violation() -> None:
    df = _clean_day("2026-01-05")
    df.loc[0, "high"] = df.loc[0, "low"] - 1  # high below low: invalid
    assert quality.check_ohlc(df) == 1


def test_gap_report_full_holiday_not_flagged() -> None:
    # Two consecutive trading days with a weekday gap (holiday) in between.
    monday = _clean_day("2026-01-05")
    wednesday = _clean_day("2026-01-07")
    df = pd.concat([monday, wednesday], ignore_index=True)

    holidays, partial_days = quality.gap_report(df, interval_minutes=5)
    assert holidays == 1  # Tuesday 2026-01-06 has zero candles
    assert partial_days == []


def test_gap_report_flags_partial_day() -> None:
    monday = _clean_day("2026-01-05")
    tuesday_partial = _clean_day("2026-01-06").iloc[:40]  # half a day
    df = pd.concat([monday, tuesday_partial], ignore_index=True)

    holidays, partial_days = quality.gap_report(df, interval_minutes=5)
    assert holidays == 0
    assert len(partial_days) == 1
    date, actual, expected = partial_days[0]
    assert date == "2026-01-06"
    assert actual == 40
    assert expected == 75


def test_build_report_clean_data_is_clean() -> None:
    df = _clean_day("2026-01-05")
    report = quality.build_report("TEST", df, interval_minutes=5)
    assert report.is_clean
    assert report.total_candles == len(df)


def test_summary_table_shape() -> None:
    df = _clean_day("2026-01-05")
    report = quality.build_report("TEST", df, interval_minutes=5)
    table = quality.summary_table([report])
    assert len(table) == 1
    assert table.iloc[0]["symbol"] == "TEST"
    assert bool(table.iloc[0]["clean"]) is True
