"""Data quality checks for cached candle data.

All functions operate on a single symbol's candle DataFrame (columns
``timestamp, open, high, low, close, volume`` with a tz-aware IST timestamp) and
are pure/side-effect free, so they're easy to unit test and reuse from the CLI
``--report`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MARKET_OPEN = pd.Timestamp("09:15:00").time()
MARKET_CLOSE = pd.Timestamp("15:30:00").time()


@dataclass
class SymbolQualityReport:
    symbol: str
    total_candles: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    duplicate_timestamps: int
    non_increasing: int
    outside_market_hours: int
    ohlc_violations: int
    holiday_days: int
    partial_days: list[tuple[str, int, int]] = field(default_factory=list)  # (date, actual, expected)

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicate_timestamps == 0
            and self.non_increasing == 0
            and self.outside_market_hours == 0
            and self.ohlc_violations == 0
            and len(self.partial_days) == 0
        )


def check_monotonic_unique(df: pd.DataFrame) -> tuple[int, int]:
    """Return (duplicate_count, non_increasing_count) for the timestamp column."""
    if df.empty:
        return 0, 0
    ts = df["timestamp"]
    duplicates = int(ts.duplicated().sum())
    non_increasing = int((ts.diff().dropna() <= pd.Timedelta(0)).sum())
    return duplicates, non_increasing


def check_market_hours(df: pd.DataFrame) -> int:
    """Count timestamps that fall outside 09:15-15:30 IST or on a weekend."""
    if df.empty:
        return 0
    ts = df["timestamp"]
    times = ts.dt.time
    within_hours = (times >= MARKET_OPEN) & (times < MARKET_CLOSE)
    weekday = ts.dt.weekday < 5  # Mon=0 .. Sun=6
    return int((~(within_hours & weekday)).sum())


def check_ohlc(df: pd.DataFrame) -> int:
    """Count rows violating high >= max(open, close) and low <= min(open, close)."""
    if df.empty:
        return 0
    high_ok = df["high"] >= df[["open", "close"]].max(axis=1)
    low_ok = df["low"] <= df[["open", "close"]].min(axis=1)
    return int((~(high_ok & low_ok)).sum())


def gap_report(df: pd.DataFrame, interval_minutes: int) -> tuple[int, list[tuple[str, int, int]]]:
    """Identify trading-day gaps.

    Returns ``(holiday_days, partial_days)`` where ``holiday_days`` is the count of
    weekdays in range with zero candles (assumed exchange holidays, not flagged),
    and ``partial_days`` lists ``(date, actual_count, expected_count)`` for weekdays
    with some but fewer than the expected number of candles (real gaps).
    """
    if df.empty:
        return 0, []

    expected_per_day = (375 // interval_minutes)  # 09:15-15:30 = 375 minutes
    counts = df.groupby(df["timestamp"].dt.date).size()

    first_day = df["timestamp"].dt.date.min()
    last_day = df["timestamp"].dt.date.max()
    all_weekdays = pd.bdate_range(first_day, last_day).date

    holiday_days = 0
    partial_days: list[tuple[str, int, int]] = []
    for day in all_weekdays:
        actual = int(counts.get(day, 0))
        if actual == 0:
            holiday_days += 1
        elif actual < expected_per_day:
            partial_days.append((day.isoformat(), actual, expected_per_day))

    return holiday_days, partial_days


def build_report(symbol: str, df: pd.DataFrame, interval_minutes: int) -> SymbolQualityReport:
    """Run all quality checks for one symbol and return a structured report."""
    duplicates, non_increasing = check_monotonic_unique(df)
    outside_hours = check_market_hours(df)
    ohlc_violations = check_ohlc(df)
    holiday_days, partial_days = gap_report(df, interval_minutes)

    return SymbolQualityReport(
        symbol=symbol,
        total_candles=len(df),
        first_timestamp=df["timestamp"].min() if not df.empty else None,
        last_timestamp=df["timestamp"].max() if not df.empty else None,
        duplicate_timestamps=duplicates,
        non_increasing=non_increasing,
        outside_market_hours=outside_hours,
        ohlc_violations=ohlc_violations,
        holiday_days=holiday_days,
        partial_days=partial_days,
    )


def summary_table(reports: list[SymbolQualityReport]) -> pd.DataFrame:
    """Build a printable coverage/quality summary table across symbols."""
    rows = [
        {
            "symbol": r.symbol,
            "first_date": r.first_timestamp.date().isoformat() if r.first_timestamp is not None else "-",
            "last_date": r.last_timestamp.date().isoformat() if r.last_timestamp is not None else "-",
            "candles": r.total_candles,
            "dupes": r.duplicate_timestamps,
            "non_increasing": r.non_increasing,
            "outside_hours": r.outside_market_hours,
            "ohlc_violations": r.ohlc_violations,
            "holidays": r.holiday_days,
            "partial_days": len(r.partial_days),
            "clean": r.is_clean,
        }
        for r in reports
    ]
    return pd.DataFrame(rows)
