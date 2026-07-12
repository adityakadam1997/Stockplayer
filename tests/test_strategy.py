"""Unit tests for strategy/ -- no network access, no cache/ dependency. Every
test hand-crafts the vwap/band/condition columns directly (rather than routing
through signals.vwap/signals.condition) so each setup's own entry/stop/target
logic can be tested in isolation with exact, known band and vwap levels."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.condition import ACCEPTED_ABOVE, ACCEPTED_BELOW, INSIDE_VALUE
from strategy import engine
from strategy.base import LONG, SHORT, StrategyConfig, compute_rr


def _frame(date, times, opens, highs, lows, closes, vwaps, band_upper, band_lower, conditions, streaks):
    ts = pd.to_datetime([f"{date} {t}" for t in times]).tz_localize("Asia/Kolkata")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "vwap": vwaps,
            "band_upper_1": band_upper,
            "band_lower_1": band_lower,
            "condition": conditions,
            "acceptance_streak": streaks,
        }
    )


# ---------------------------------------------------------------------------
# compute_rr / the R:R filter
# ---------------------------------------------------------------------------


def test_compute_rr_exact_values():
    assert compute_rr(100, 99, 101.4, LONG) == pytest.approx(1.4)
    assert compute_rr(100, 99, 101.6, LONG) == pytest.approx(1.6)


def test_engine_rejects_below_min_rr_accepts_above():
    # Same setup2 fade pattern, only vwap (the target) distance changes to push
    # r:r just under vs. just over the 1.5 threshold.
    cfg = StrategyConfig()

    def scenario(vwap_level: float) -> pd.DataFrame:
        return _frame(
            "2026-01-05",
            ["09:15", "09:20", "09:25", "09:30"],
            opens=[100.2, 99.8, 100.2, 100.6],
            highs=[100.3, 99.9, 100.3, 100.9],
            lows=[100.1, 99.7, 100.1, 100.5],
            closes=[100.2, 99.8, 100.2, 100.6],
            vwaps=[100.0, 100.0, 100.0, vwap_level],
            band_upper=[100.7, 100.7, 100.7, 100.7],
            band_lower=[99.3, 99.3, 99.3, 99.3],
            conditions=[INSIDE_VALUE, INSIDE_VALUE, INSIDE_VALUE, INSIDE_VALUE],
            streaks=[0, 0, 0, 0],
        )

    # risk = 100.9*1.0005 - 100.6 = 0.35045; reward = 100.6 - target.
    low_rr = scenario(100.4)  # reward=0.2 -> rr ~= 0.57 < 1.5
    high_rr = scenario(100.0)  # reward=0.6 -> rr ~= 1.71 >= 1.5

    assert engine.generate_proposals(low_rr, "TEST", cfg) == []
    proposals = engine.generate_proposals(high_rr, "TEST", cfg)
    assert len(proposals) == 1
    assert proposals[0].rr_ratio >= cfg.min_rr


# ---------------------------------------------------------------------------
# Setup 1 -- Price Discovery Continuation
# ---------------------------------------------------------------------------


def test_setup1_fires_on_retest_after_real_acceptance():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100, 101, 108, 108],
        highs=[102, 109, 111, 110],
        lows=[99, 107, 108, 104],
        closes=[101, 108, 110, 106],
        vwaps=[100, 100, 100, 100],
        band_upper=[105, 105, 105, 105],
        band_lower=[95, 95, 95, 95],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE],
        streaks=[0, 3, 4, 5],
    )
    cfg = StrategyConfig()
    proposals = engine.generate_proposals(df, "TEST", cfg)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.setup_id == "setup1_discovery"
    assert p.direction == LONG
    assert p.entry_price == pytest.approx(106.0)
    assert p.target_price == pytest.approx(111.0)  # session extreme high reached during acceptance
    assert p.timestamp == df.loc[3, "timestamp"]


def test_setup1_does_not_fire_without_real_acceptance():
    # A single candle pokes above the band and immediately reverts (matching
    # signals.condition's own definition of "not acceptance") -- condition
    # never reaches ACCEPTED_ABOVE, so even a later candle that geometrically
    # looks like a retest must not fire.
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100, 106, 100, 106],
        highs=[101, 110, 101, 106.5],
        lows=[99, 105, 99, 104],
        closes=[100, 100, 100, 106],
        vwaps=[100, 100, 100, 100],
        band_upper=[105, 105, 105, 105],
        band_lower=[95, 95, 95, 95],
        conditions=[INSIDE_VALUE, INSIDE_VALUE, INSIDE_VALUE, INSIDE_VALUE],
        streaks=[0, 1, 0, 0],
    )
    cfg = StrategyConfig()
    assert engine.generate_proposals(df, "TEST", cfg) == []


# ---------------------------------------------------------------------------
# Setup 2 -- Fade Value Area Extremes
# ---------------------------------------------------------------------------


def _setup2_frame(crossing_closes):
    """Candles that build up vwap crossings around a flat vwap=100 via
    ``crossing_closes``, then a final candle that tags and rejects the
    (deliberately tight, sub-guard-threshold) upper band."""
    closes = crossing_closes + [100.6]
    n = len(closes)
    times = ["09:15", "09:20", "09:25", "09:30", "09:35"][:n]
    highs = [c + 0.1 for c in closes[:-1]] + [100.9]
    lows = [c - 0.1 for c in closes[:-1]] + [100.5]
    opens = closes[:-1] + [100.6]
    return _frame(
        "2026-01-05",
        times,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        vwaps=[100.0] * n,
        band_upper=[100.7] * n,
        band_lower=[99.3] * n,
        conditions=[INSIDE_VALUE] * n,
        streaks=[0] * n,
    )


def test_setup2_fires_after_enough_vwap_crossings():
    # closes: 100.2 (above vwap), 99.8 (below, cross 1), 100.2 (above, cross 2),
    # then the tag-and-reject candle -- vwap_cross_count reaches 2 (the
    # default threshold) right before the fade candle.
    df = _setup2_frame([100.2, 99.8, 100.2])
    cfg = StrategyConfig()
    proposals = engine.generate_proposals(df, "TEST", cfg)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.setup_id == "setup2_fade"
    assert p.direction == SHORT
    assert p.entry_price == pytest.approx(100.6)
    assert p.target_price == pytest.approx(100.0)


def test_setup2_does_not_fire_without_rotation():
    # Only 1 crossing before the tag-and-reject candle -- below the default
    # threshold of 2, so the balance filter blocks it even though the tag and
    # reject itself is identical to the passing case above.
    df = _setup2_frame([100.2, 99.8])
    cfg = StrategyConfig()
    assert engine.generate_proposals(df, "TEST", cfg) == []


# ---------------------------------------------------------------------------
# Setup 3 -- Return to Value
# ---------------------------------------------------------------------------


def test_setup3_primary_fires_on_retest_and_hold():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100, 108, 103, 104],
        highs=[101, 109, 104, 105.2],
        lows=[99, 107, 102, 103],
        closes=[100, 108, 103, 103.5],
        vwaps=[100, 100, 100, 100],
        band_upper=[105, 105, 105, 105],
        band_lower=[95, 95, 95, 95],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, INSIDE_VALUE, INSIDE_VALUE],
        streaks=[0, 3, 0, 0],
    )
    cfg = StrategyConfig(setup3_allow_fallback=False)
    proposals = engine.generate_proposals(df, "TEST", cfg)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.setup_id == "setup3_return"
    assert p.direction == SHORT
    assert p.entry_price == pytest.approx(103.5)
    assert p.target_price == pytest.approx(100.0)
    assert "primary" in p.notes


def test_setup3_fallback_fires_only_when_enabled():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100, 108, 109, 104],
        highs=[101, 109, 110, 104.5],
        lows=[99, 107, 108, 102.5],
        closes=[100, 108, 109, 103],
        vwaps=[95, 95, 95, 95],
        band_upper=[104, 104, 104, 104],
        band_lower=[86, 86, 86, 86],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, INSIDE_VALUE],
        streaks=[0, 3, 4, 0],
    )
    cfg_off = StrategyConfig(setup3_allow_fallback=False)
    assert engine.generate_proposals(df, "TEST", cfg_off) == []

    cfg_on = StrategyConfig(setup3_allow_fallback=True)
    proposals = engine.generate_proposals(df, "TEST", cfg_on)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.setup_id == "setup3_return"
    assert p.direction == SHORT
    assert p.entry_price == pytest.approx(103.0)  # entered on the break candle's own close
    assert "fallback" in p.notes


# ---------------------------------------------------------------------------
# Setup 4 -- VWAP Bounce
# ---------------------------------------------------------------------------


def test_setup4_fires_after_touch_then_close_away():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "09:35"],
        opens=[100.0, 101.0, 100.3, 100.2, 100.2],
        highs=[100.2, 101.2, 100.6, 100.3, 100.35],
        lows=[99.8, 100.8, 100.1, 100.0, 100.15],
        closes=[100.0, 101.0, 100.3, 100.2, 100.3],
        vwaps=[100.0, 100.0, 100.0, 100.0, 100.0],
        band_upper=[100.7, 100.7, 100.7, 100.7, 100.7],
        band_lower=[99.3, 99.3, 99.3, 99.3, 99.3],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, INSIDE_VALUE, INSIDE_VALUE, INSIDE_VALUE],
        streaks=[0, 3, 0, 0, 0],
    )
    cfg = StrategyConfig()
    proposals = engine.generate_proposals(df, "TEST", cfg)

    bounce_proposals = [p for p in proposals if p.setup_id == "setup4_bounce"]
    assert len(bounce_proposals) == 1
    p = bounce_proposals[0]
    assert p.direction == LONG
    assert p.entry_price == pytest.approx(100.3)
    assert p.target_price == pytest.approx(100.7)
    assert p.timestamp == df.loc[4, "timestamp"]


def test_setup4_does_not_fire_without_a_prior_vwap_touch():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100.0, 101.0, 100.3, 100.35],
        highs=[100.2, 101.2, 100.6, 100.5],
        lows=[99.8, 100.8, 100.2, 100.3],
        closes=[100.0, 101.0, 100.3, 100.4],
        vwaps=[100.0, 100.0, 100.0, 100.0],
        band_upper=[100.7, 100.7, 100.7, 100.7],
        band_lower=[99.3, 99.3, 99.3, 99.3],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, INSIDE_VALUE, INSIDE_VALUE],
        streaks=[0, 3, 0, 0],
    )
    cfg = StrategyConfig()
    proposals = engine.generate_proposals(df, "TEST", cfg)
    assert all(p.setup_id != "setup4_bounce" for p in proposals)


# ---------------------------------------------------------------------------
# No-lookahead
# ---------------------------------------------------------------------------


def test_no_lookahead():
    day1 = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "09:35", "09:40"],
        opens=[100, 101, 108, 108, 106, 108],
        highs=[102, 109, 111, 110, 108, 112],
        lows=[99, 107, 108, 104, 105, 107],
        closes=[101, 108, 110, 106, 107, 111],
        vwaps=[100, 100, 100, 100, 100, 100],
        band_upper=[105, 105, 105, 105, 105, 105],
        band_lower=[95, 95, 95, 95, 95, 95],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE],
        streaks=[0, 3, 4, 5, 6, 7],
    )
    day2 = _setup2_frame([100.2, 99.8, 100.2])
    day2["timestamp"] = pd.to_datetime(
        [f"2026-01-06 {t.strftime('%H:%M')}" for t in day2["timestamp"].dt.tz_localize(None)]
    ).tz_localize("Asia/Kolkata")

    df = pd.concat([day1, day2], ignore_index=True)
    cfg = StrategyConfig()

    full = engine.generate_proposals(df, "TEST", cfg)

    for truncate_at in (3, 5, 7, len(df) - 1):
        truncated_df = df.iloc[: truncate_at + 1].reset_index(drop=True)
        truncated = engine.generate_proposals(truncated_df, "TEST", cfg)
        cutoff = df.loc[truncate_at, "timestamp"]
        expected_prefix = [p for p in full if p.timestamp <= cutoff]
        assert truncated == expected_prefix
