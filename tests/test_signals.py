"""Unit tests for signals/vwap.py and signals/condition.py -- no network access,
no dependency on the local cache/ directory. Synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals import condition, vwap


def _candles(date: str, times: list[str], opens, highs, lows, closes, volumes) -> pd.DataFrame:
    ts = pd.to_datetime([f"{date} {t}" for t in times]).tz_localize("Asia/Kolkata")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


# ---------------------------------------------------------------------------
# vwap.py
# ---------------------------------------------------------------------------


def test_vwap_nan_on_first_candle_then_populated() -> None:
    df = _candles(
        "2026-01-05",
        ["09:15", "09:20", "09:25"],
        opens=[100, 101, 102],
        highs=[101, 102, 103],
        lows=[99, 100, 101],
        closes=[100, 101, 102],
        volumes=[1000, 1000, 1000],
    )
    out = vwap.compute_session_vwap(df)

    assert pd.isna(out.loc[0, "vwap"])
    assert pd.isna(out.loc[0, "band_upper_1"])
    assert pd.isna(out.loc[0, "band_lower_1"])
    assert not pd.isna(out.loc[1, "vwap"])
    assert not pd.isna(out.loc[2, "vwap"])

    # Candle 2's VWAP should be the volume-weighted typical price of candles 1-2.
    tp1 = (101 + 99 + 100) / 3.0
    tp2 = (102 + 100 + 101) / 3.0
    expected_vwap_2 = (tp1 * 1000 + tp2 * 1000) / 2000
    assert out.loc[1, "vwap"] == pytest.approx(expected_vwap_2)


def test_vwap_resets_at_new_session() -> None:
    day1 = _candles(
        "2026-01-05",
        ["09:15", "09:20"],
        opens=[100, 200],
        highs=[100, 200],
        lows=[100, 200],
        closes=[100, 200],
        volumes=[1000, 1000],
    )
    day2 = _candles(
        "2026-01-06",
        ["09:15", "09:20"],
        opens=[500, 500],
        highs=[500, 500],
        lows=[500, 500],
        closes=[500, 500],
        volumes=[1000, 1000],
    )
    df = pd.concat([day1, day2], ignore_index=True)
    out = vwap.compute_session_vwap(df)

    # Day 2's first candle is a fresh session -> NaN, not a continuation of day 1.
    assert pd.isna(out.loc[2, "vwap"])
    # Day 2's second candle should only reflect day-2 data (flat at 500), not
    # any influence from day 1's much lower prices.
    assert out.loc[3, "vwap"] == pytest.approx(500.0)


def test_deviation_bands_symmetric_around_vwap() -> None:
    df = _candles(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100, 105, 95, 110],
        highs=[101, 106, 96, 111],
        lows=[99, 104, 94, 109],
        closes=[100, 105, 95, 110],
        volumes=[1000, 1000, 1000, 1000],
    )
    out = vwap.compute_session_vwap(df)
    populated = out.iloc[1:]

    for k in (1, 2):
        upper_gap = populated[f"band_upper_{k}"] - populated["vwap"]
        lower_gap = populated["vwap"] - populated[f"band_lower_{k}"]
        assert (upper_gap - lower_gap).abs().lt(1e-9).all()

    # Band 2 should be strictly wider than band 1 wherever std > 0.
    has_spread = populated["band_upper_2"] > populated["vwap"]
    assert (
        populated.loc[has_spread, "band_upper_2"] > populated.loc[has_spread, "band_upper_1"]
    ).all()


def test_deviation_bands_widen_with_higher_volatility() -> None:
    calm = _candles(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100.0, 100.1, 99.9, 100.0],
        highs=[100.1, 100.2, 100.0, 100.1],
        lows=[99.9, 100.0, 99.8, 99.9],
        closes=[100.0, 100.1, 99.9, 100.0],
        volumes=[1000, 1000, 1000, 1000],
    )
    volatile = _candles(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100.0, 110.0, 90.0, 115.0],
        highs=[101.0, 112.0, 92.0, 117.0],
        lows=[99.0, 108.0, 88.0, 113.0],
        closes=[100.0, 110.0, 90.0, 115.0],
        volumes=[1000, 1000, 1000, 1000],
    )
    calm_out = vwap.compute_session_vwap(calm)
    volatile_out = vwap.compute_session_vwap(volatile)

    calm_width = (calm_out.loc[3, "band_upper_1"] - calm_out.loc[3, "band_lower_1"])
    volatile_width = (volatile_out.loc[3, "band_upper_1"] - volatile_out.loc[3, "band_lower_1"])
    assert volatile_width > calm_width


# ---------------------------------------------------------------------------
# condition.py
# ---------------------------------------------------------------------------


def _band_frame(date: str, n: int, close, high=None, low=None) -> pd.DataFrame:
    """A synthetic single-session frame with fixed, hand-picked band levels
    (vwap=100, band_1=[95,105], band_2=[90,110]) so the price path controls
    the classification exactly, independent of signals.vwap's own math."""
    times = [f"{9 + i // 12:02d}:{(15 + (i % 12) * 5) % 60:02d}" for i in range(n)]
    high = high if high is not None else close
    low = low if low is not None else close
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([f"{date} {t}" for t in times]).tz_localize("Asia/Kolkata"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "vwap": [100.0] * n,
            "band_upper_1": [105.0] * n,
            "band_lower_1": [95.0] * n,
            "band_upper_2": [110.0] * n,
            "band_lower_2": [90.0] * n,
        }
    )


def test_condition_inside_value() -> None:
    df = _band_frame("2026-01-05", 3, close=[100, 102, 98])
    out = condition.compute_condition(df, acceptance_candles=3)
    assert (out["condition"] == condition.INSIDE_VALUE).all()
    assert (out["acceptance_streak"] == 0).all()


def test_condition_requires_persistence_not_single_touch() -> None:
    # Pokes above the band once, then immediately reverts -- should NOT be
    # classified as accepted, per "a single candle poking above the band and
    # immediately reverting is NOT acceptance".
    df = _band_frame("2026-01-05", 3, close=[100, 106, 100])
    out = condition.compute_condition(df, acceptance_candles=3)
    assert out.loc[1, "condition"] == condition.INSIDE_VALUE
    assert out.loc[1, "acceptance_streak"] == 1
    assert out.loc[2, "condition"] == condition.INSIDE_VALUE
    assert out.loc[2, "acceptance_streak"] == 0


def test_condition_accepted_above_after_n_consecutive() -> None:
    df = _band_frame("2026-01-05", 4, close=[100, 106, 107, 108])
    out = condition.compute_condition(df, acceptance_candles=3)
    assert list(out["condition"]) == [
        condition.INSIDE_VALUE,
        condition.INSIDE_VALUE,
        condition.INSIDE_VALUE,
        condition.ACCEPTED_ABOVE,
    ]
    assert list(out["acceptance_streak"]) == [0, 1, 2, 3]


def test_condition_accepted_below_after_n_consecutive() -> None:
    df = _band_frame("2026-01-05", 4, close=[100, 94, 93, 92])
    out = condition.compute_condition(df, acceptance_candles=3)
    assert list(out["condition"]) == [
        condition.INSIDE_VALUE,
        condition.INSIDE_VALUE,
        condition.INSIDE_VALUE,
        condition.ACCEPTED_BELOW,
    ]
    assert list(out["acceptance_streak"]) == [0, -1, -2, -3]


def test_condition_accepted_immediately_on_extreme_touch() -> None:
    # First excursion candle's high spikes through band_2 (110) even though
    # its close is only just above band_1 -- acceptance should trigger right
    # away, without waiting for N consecutive closes.
    df = _band_frame("2026-01-05", 2, close=[100, 106], high=[100, 111])
    out = condition.compute_condition(df, acceptance_candles=3)
    assert out.loc[1, "condition"] == condition.ACCEPTED_ABOVE
    assert out.loc[1, "acceptance_streak"] == 1


def test_condition_streak_resets_when_price_returns_to_value() -> None:
    df = _band_frame("2026-01-05", 5, close=[100, 106, 107, 108, 100])
    out = condition.compute_condition(df, acceptance_candles=3)
    assert out.loc[3, "condition"] == condition.ACCEPTED_ABOVE
    assert out.loc[4, "condition"] == condition.INSIDE_VALUE
    assert out.loc[4, "acceptance_streak"] == 0


def test_condition_first_candle_of_session_has_no_bands_is_inside_value() -> None:
    df = _band_frame("2026-01-05", 2, close=[106, 106])
    df.loc[0, ["vwap", "band_upper_1", "band_lower_1", "band_upper_2", "band_lower_2"]] = float("nan")
    out = condition.compute_condition(df, acceptance_candles=3)
    assert out.loc[0, "condition"] == condition.INSIDE_VALUE
    assert out.loc[0, "acceptance_streak"] == 0
