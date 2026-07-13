"""Unit tests for data/resample.py -- no network access, no cache/ dependency,
synthetic data only.

Also covers the config.yaml ``timeframes:`` profile-merge mechanism
(Cycle 2 / Weekend 5) and a regression guard that the base 5-min profile's
values -- the ones Weekend 4's NO-GO verdict was computed against -- are
unchanged by this weekend's additions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.resample import resample_to_interval
from signals.condition import ACCEPTED_ABOVE, INSIDE_VALUE
from signals.config import load_signals_config
from strategy import engine
from strategy.base import StrategyConfig, load_strategy_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _candles(date: str, times: list[str], opens, highs, lows, closes, volumes) -> pd.DataFrame:
    ts = pd.to_datetime([f"{date} {t}" for t in times]).tz_localize("Asia/Kolkata")
    return pd.DataFrame(
        {"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


# ---------------------------------------------------------------------------
# Candle alignment / OHLCV aggregation
# ---------------------------------------------------------------------------


def test_resample_aggregates_ohlcv_correctly_for_one_full_bar():
    # Three 5-min candles making up the first 15-min bar (09:15-09:30).
    df = _candles(
        "2026-01-05",
        ["09:15", "09:20", "09:25"],
        opens=[100, 102, 101],
        highs=[103, 105, 104],
        lows=[99, 101, 100],
        closes=[102, 101, 103],
        volumes=[1000, 1500, 1200],
    )
    out = resample_to_interval(df, target_minutes=15)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")
    assert row["open"] == 100  # first candle's open
    assert row["high"] == 105  # max high
    assert row["low"] == 99  # min low
    assert row["close"] == 103  # last candle's close
    assert row["volume"] == 3700  # sum


def test_first_bar_is_0915_to_0930():
    df = _candles(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "09:35", "09:40"],
        opens=[100] * 6,
        highs=[101] * 6,
        lows=[99] * 6,
        closes=[100] * 6,
        volumes=[1000] * 6,
    )
    out = resample_to_interval(df, target_minutes=15)

    assert len(out) == 2
    assert out.loc[0, "timestamp"] == pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")
    assert out.loc[1, "timestamp"] == pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")


def test_bars_do_not_span_session_boundary():
    # Last candle of day 1 (15:25) and first candle of day 2 (09:15) must not
    # be merged into the same bar even though only 5 minutes of *data*
    # separate their timestamps' bin arithmetic would otherwise use.
    day1 = _candles("2026-01-05", ["15:20", "15:25"], [100, 101], [102, 103], [99, 100], [101, 102], [1000, 1000])
    day2 = _candles("2026-01-06", ["09:15", "09:20"], [200, 201], [202, 203], [199, 200], [201, 202], [1000, 1000])
    df = pd.concat([day1, day2], ignore_index=True)

    out = resample_to_interval(df, target_minutes=15)

    day1_bars = out[out["timestamp"].dt.date == pd.Timestamp("2026-01-05").date()]
    day2_bars = out[out["timestamp"].dt.date == pd.Timestamp("2026-01-06").date()]
    assert len(day1_bars) == 1
    assert len(day2_bars) == 1
    assert day2_bars.iloc[0]["timestamp"] == pd.Timestamp("2026-01-06 09:15", tz="Asia/Kolkata")
    # No bar mixes day1 and day2 prices.
    assert day1_bars.iloc[0]["close"] == 102
    assert day2_bars.iloc[0]["open"] == 200


# ---------------------------------------------------------------------------
# Volume conservation
# ---------------------------------------------------------------------------


def test_volume_conserved_per_day():
    day1 = _candles(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "09:35"],
        opens=[100] * 5,
        highs=[101] * 5,
        lows=[99] * 5,
        closes=[100] * 5,
        volumes=[111, 222, 333, 444, 555],
    )
    day2 = _candles(
        "2026-01-06",
        ["09:15", "09:20", "09:25"],
        opens=[100] * 3,
        highs=[101] * 3,
        lows=[99] * 3,
        closes=[100] * 3,
        volumes=[10, 20, 30],
    )
    df = pd.concat([day1, day2], ignore_index=True)
    out = resample_to_interval(df, target_minutes=15)

    for day, expected_total in ((pd.Timestamp("2026-01-05").date(), 111 + 222 + 333 + 444 + 555), (pd.Timestamp("2026-01-06").date(), 10 + 20 + 30)):
        assert out[out["timestamp"].dt.date == day]["volume"].sum() == expected_total

    assert out["volume"].sum() == df["volume"].sum()


# ---------------------------------------------------------------------------
# Special / partial sessions
# ---------------------------------------------------------------------------


def test_partial_muhurat_style_session_resamples_without_special_casing():
    # A short special session starting well after the normal open (like the
    # Diwali Muhurat session, 13:45-14:40) -- still anchored to that day's
    # 09:15, landing cleanly on the 15-min grid: 13:45/13:50/13:55 -> bin
    # 13:45 (270 min after 09:15, exactly divisible by 15); 14:00 -> its own
    # bin; 14:35 -> the 14:30 bin (320 min after 09:15 -> floor(320/15)=21).
    df = _candles(
        "2025-10-21",
        ["13:45", "13:50", "13:55", "14:00", "14:35"],
        opens=[100, 101, 102, 103, 104],
        highs=[105, 106, 107, 108, 109],
        lows=[95, 96, 97, 98, 99],
        closes=[101, 102, 103, 104, 105],
        volumes=[10, 20, 30, 40, 50],
    )
    out = resample_to_interval(df, target_minutes=15)

    assert len(out) == 3
    assert list(out["timestamp"]) == [
        pd.Timestamp("2025-10-21 13:45", tz="Asia/Kolkata"),
        pd.Timestamp("2025-10-21 14:00", tz="Asia/Kolkata"),
        pd.Timestamp("2025-10-21 14:30", tz="Asia/Kolkata"),
    ]
    assert out.loc[0, "volume"] == 10 + 20 + 30
    assert out.loc[1, "volume"] == 40
    assert out.loc[2, "volume"] == 50
    assert out["volume"].sum() == df["volume"].sum()


def test_special_session_with_irregular_minute_offsets_still_bins_from_0915():
    # A hypothetical special session whose candles fall mid-bin relative to
    # the normal grid -- resample must not crash or silently drop candles;
    # it just bins whatever exists relative to 09:15 of that calendar day.
    df = _candles(
        "2026-02-01",
        ["09:15", "09:22", "09:31", "09:44"],
        opens=[100, 101, 102, 103],
        highs=[105, 106, 107, 108],
        lows=[95, 96, 97, 98],
        closes=[101, 102, 103, 104],
        volumes=[10, 20, 30, 40],
    )
    out = resample_to_interval(df, target_minutes=15)

    # 09:15 -> bin0 [09:15,09:30); 09:22 -> bin0; 09:31 -> bin1 [09:30,09:45); 09:44 -> bin1.
    assert len(out) == 2
    assert out.loc[0, "timestamp"] == pd.Timestamp("2026-02-01 09:15", tz="Asia/Kolkata")
    assert out.loc[0, "volume"] == 30
    assert out.loc[1, "timestamp"] == pd.Timestamp("2026-02-01 09:30", tz="Asia/Kolkata")
    assert out.loc[1, "volume"] == 70
    assert out["volume"].sum() == df["volume"].sum()


def test_empty_input_returns_empty_with_schema():
    df = _candles("2026-01-05", [], [], [], [], [], [])
    out = resample_to_interval(df, target_minutes=15)
    assert out.empty
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# config.yaml timeframes: profile merging
# ---------------------------------------------------------------------------


def test_5min_profile_matches_weekend4_registered_defaults():
    """Regression guard: the base (no-timeframe) profile must still be
    exactly what Weekend 4's NO-GO verdict was computed against. If this
    fails, someone changed a pre-registered Weekend 4 default while adding
    the 15-min profile."""
    cfg = load_strategy_config(CONFIG_PATH)
    assert cfg.stop_floor_pct == pytest.approx(0.0035)
    assert cfg.atr_mult == pytest.approx(1.0)
    assert cfg.cost_viability_max_pct == pytest.approx(0.15)
    assert cfg.min_rr == pytest.approx(1.5)
    assert cfg.risk_pct == pytest.approx(0.005)
    assert cfg.capital == pytest.approx(100_000.0)
    assert cfg.max_trades_per_day == 2
    assert cfg.stop_cooldown_minutes == 30
    assert cfg.session_open_no_entry_minutes == 15
    assert cfg.no_entry_after.strftime("%H:%M") == "14:45"

    signals_cfg = load_signals_config(CONFIG_PATH)
    assert signals_cfg.acceptance_candles == 3


def test_15min_profile_overrides_only_the_specified_fields():
    cfg = load_strategy_config(CONFIG_PATH, timeframe="15min")
    # Overridden:
    assert cfg.session_open_no_entry_minutes == 30
    assert cfg.no_entry_after.strftime("%H:%M") == "14:30"
    # Inherited unchanged from the base profile:
    assert cfg.stop_floor_pct == pytest.approx(0.0035)
    assert cfg.atr_mult == pytest.approx(1.0)
    assert cfg.cost_viability_max_pct == pytest.approx(0.15)
    assert cfg.min_rr == pytest.approx(1.5)
    assert cfg.risk_pct == pytest.approx(0.005)
    assert cfg.capital == pytest.approx(100_000.0)
    assert cfg.max_trades_per_day == 2
    assert cfg.stop_cooldown_minutes == 30
    assert cfg.square_off_at.strftime("%H:%M") == "15:15"
    assert cfg.wide_band_guard_pct == pytest.approx(0.015)

    signals_cfg = load_signals_config(CONFIG_PATH, timeframe="15min")
    assert signals_cfg.acceptance_candles == 2
    assert signals_cfg.value_area_band == 1  # unchanged


def test_unknown_timeframe_falls_back_to_base_profile():
    cfg = load_strategy_config(CONFIG_PATH, timeframe="60min")  # not defined in config.yaml
    base = load_strategy_config(CONFIG_PATH)
    assert cfg == base


# ---------------------------------------------------------------------------
# 5-min zero-trade regression guard (mechanism-level, offline/synthetic)
# ---------------------------------------------------------------------------


def test_5min_default_config_still_rejects_the_weekend4_boundary_scenario():
    """Same scenario as tests/test_strategy.py::test_cost_viability_filter_boundary
    at the strict end -- reproduced here against the *actual loaded*
    config.yaml defaults (not a hand-built StrategyConfig) as a regression
    guard that adding the 15-min profile didn't accidentally change the
    5-min profile's real cost-viability outcome."""
    from backtest.costs import load_cost_config

    ts = pd.to_datetime(
        [f"2026-01-05 {t}" for t in ["09:15", "09:20", "09:25", "09:30"]]
    ).tz_localize("Asia/Kolkata")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 108, 108],
            "high": [102, 109, 111, 110],
            "low": [99, 107, 108, 104],
            "close": [101, 108, 110, 106],
            "vwap": [100, 100, 100, 100],
            "band_upper_1": [105, 105, 105, 105],
            "band_lower_1": [95, 95, 95, 95],
            "condition": [INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE],
            "acceptance_streak": [0, 3, 4, 5],
            "atr20": [0.1, 0.1, 0.1, 0.1],
        }
    )

    strategy_cfg = load_strategy_config(CONFIG_PATH)
    cost_cfg = load_cost_config(CONFIG_PATH)
    proposals = engine.generate_proposals(df, "TEST", strategy_cfg, cost_cfg)
    assert proposals == []  # rejected by cost-viability, exactly as Weekend 4 found
