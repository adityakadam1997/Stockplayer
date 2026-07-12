"""Unit tests for strategy/ -- no network access, no cache/ dependency. Every
test hand-crafts the vwap/band/condition/atr20 columns directly (rather than
routing through signals.vwap/signals.condition/strategy.base.compute_atr) so
each setup's own entry/stop/target logic can be tested in isolation with
exact, known band, vwap, and volatility levels.

Most tests below exercise pre-Weekend-4 setup mechanics (unchanged this
weekend) and use ``_permissive_cfg()`` to switch off the stop floor and
cost-viability filter, which would otherwise dominate every stop/rr number in
these hand-picked, tight-stop scenarios. The Weekend 4 filters themselves
(stop floor, cost viability) get their own dedicated tests further down,
using realistic (default) config."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostConfig
from signals.condition import ACCEPTED_ABOVE, ACCEPTED_BELOW, INSIDE_VALUE
from strategy import engine
from strategy.base import LONG, SHORT, StrategyConfig, compute_atr, compute_rr

_DEFAULT_COST_CFG = CostConfig()


def _permissive_cfg(**overrides) -> StrategyConfig:
    """A StrategyConfig with the Weekend 4 filters neutralized, so tests
    written for Weekend 3's setup mechanics see the same structural
    stop/target/rr numbers they always did."""
    defaults = dict(stop_floor_pct=0.0, atr_mult=0.0, cost_viability_max_pct=float("inf"))
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def _frame(date, times, opens, highs, lows, closes, vwaps, band_upper, band_lower, conditions, streaks, atr=0.0):
    ts = pd.to_datetime([f"{date} {t}" for t in times]).tz_localize("Asia/Kolkata")
    n = len(times)
    atr_values = atr if isinstance(atr, list) else [atr] * n
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
            "atr20": atr_values,
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
    cfg = _permissive_cfg()

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

    assert engine.generate_proposals(low_rr, "TEST", cfg, _DEFAULT_COST_CFG) == []
    proposals = engine.generate_proposals(high_rr, "TEST", cfg, _DEFAULT_COST_CFG)
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
    cfg = _permissive_cfg()
    proposals = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)

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
    cfg = _permissive_cfg()
    assert engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG) == []


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
    cfg = _permissive_cfg()
    proposals = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)

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
    cfg = _permissive_cfg()
    assert engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG) == []


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
    cfg = _permissive_cfg(setup3_allow_fallback=False)
    proposals = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)

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
    cfg_off = _permissive_cfg(setup3_allow_fallback=False)
    assert engine.generate_proposals(df, "TEST", cfg_off, _DEFAULT_COST_CFG) == []

    cfg_on = _permissive_cfg(setup3_allow_fallback=True)
    proposals = engine.generate_proposals(df, "TEST", cfg_on, _DEFAULT_COST_CFG)
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
    cfg = _permissive_cfg()
    proposals = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)

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
    cfg = _permissive_cfg()
    proposals = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)
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
    cfg = _permissive_cfg()

    full = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)

    for truncate_at in (3, 5, 7, len(df) - 1):
        truncated_df = df.iloc[: truncate_at + 1].reset_index(drop=True)
        truncated = engine.generate_proposals(truncated_df, "TEST", cfg, _DEFAULT_COST_CFG)
        cutoff = df.loc[truncate_at, "timestamp"]
        expected_prefix = [p for p in full if p.timestamp <= cutoff]
        assert truncated == expected_prefix


# ---------------------------------------------------------------------------
# Weekend 4 -- ATR (session-aware, no-lookahead)
# ---------------------------------------------------------------------------


def test_atr_resets_at_session_boundary_and_first_candle_is_high_minus_low():
    day1 = pd.to_datetime(["2026-01-05 09:15", "2026-01-05 09:20"]).tz_localize("Asia/Kolkata")
    day2 = pd.to_datetime(["2026-01-06 09:15", "2026-01-06 09:20"]).tz_localize("Asia/Kolkata")
    df = pd.DataFrame(
        {
            "timestamp": list(day1) + list(day2),
            "open": [100, 200, 500, 500],
            "high": [102, 205, 503, 503],
            "low": [98, 195, 497, 497],
            "close": [100, 200, 500, 500],
        }
    )
    out = compute_atr(df, period=20)

    # First candle of each session: no same-session prior close -> TR = high-low.
    assert out.loc[0, "atr20"] == pytest.approx(4.0)
    assert out.loc[2, "atr20"] == pytest.approx(6.0)
    # Second candle of day1: TR = max(10, |205-100|=105, |195-100|=95) = 105;
    # atr20 = mean(4, 105).
    assert out.loc[1, "atr20"] == pytest.approx((4.0 + 105.0) / 2)
    # Day 2 must not be influenced by day 1's huge range.
    assert out.loc[3, "atr20"] == pytest.approx((6.0 + max(6.0, abs(503 - 500), abs(497 - 500))) / 2)


def test_atr_no_lookahead():
    ts = pd.to_datetime([f"2026-01-05 {9 + i // 12:02d}:{(15 + (i % 12) * 5) % 60:02d}" for i in range(30)]).tz_localize(
        "Asia/Kolkata"
    )
    import numpy as np

    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 1, size=30).cumsum()
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": closes + rng.uniform(0.1, 1.0, size=30),
            "low": closes - rng.uniform(0.1, 1.0, size=30),
            "close": closes,
        }
    )
    full = compute_atr(df, period=20)

    for truncate_at in (5, 15, 25, 29):
        truncated = compute_atr(df.iloc[: truncate_at + 1].reset_index(drop=True), period=20)
        pd.testing.assert_series_equal(
            truncated["atr20"], full["atr20"].iloc[: truncate_at + 1], check_names=False
        )


# ---------------------------------------------------------------------------
# Weekend 4 -- stop floor
# ---------------------------------------------------------------------------


def test_stop_floor_widens_a_structurally_tight_stop():
    # setup1 retest pattern identical to test_setup1_fires_on_retest_after_real_acceptance,
    # but this time with the REAL stop floor active: structural stop distance
    # (entry=106, low=104 -> stop~103.948, distance~2.052) is narrower than
    # atr_mult(1.0)*atr20(3.0)=3.0, so the ATR floor component must win.
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
        atr=3.0,
    )
    cfg = StrategyConfig(stop_floor_pct=0.0035, atr_mult=1.0, cost_viability_max_pct=float("inf"))
    proposals = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)

    assert len(proposals) == 1
    p = proposals[0]
    structural_distance = 106.0 - (104.0 * (1 - cfg.stop_buffer_pct))
    floor_distance = max(cfg.stop_floor_pct * 106.0, cfg.atr_mult * 3.0)
    assert floor_distance > structural_distance  # sanity: the floor is the binding constraint here
    expected_stop = 106.0 - floor_distance
    assert p.stop_price == pytest.approx(expected_stop)
    assert p.stop_price < 104.0 * (1 - cfg.stop_buffer_pct)  # floor only ever widens, never tightens
    assert "stop_floor_applied" in p.notes


def test_stop_floor_leaves_a_structurally_wide_stop_alone():
    # Same shape, but the retest candle's low is now further below the entry
    # (distance ~2.55) than both floor components (stop_floor_pct(0.35%)*106
    # = 0.371, atr_mult(1.0)*atr20(0.5) = 0.5) -- while still narrow enough
    # to clear the R:R filter against the fixed target=111 (reward=5, so
    # risk must stay under 5/1.5=3.33) -- the structural stop must win untouched.
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30"],
        opens=[100, 101, 108, 108],
        highs=[102, 109, 111, 110],
        lows=[99, 107, 108, 103.5],
        closes=[101, 108, 110, 106],
        vwaps=[100, 100, 100, 100],
        band_upper=[105, 105, 105, 105],
        band_lower=[95, 95, 95, 95],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE],
        streaks=[0, 3, 4, 5],
        atr=0.5,
    )
    cfg = StrategyConfig(stop_floor_pct=0.0035, atr_mult=1.0, cost_viability_max_pct=float("inf"))
    proposals = engine.generate_proposals(df, "TEST", cfg, _DEFAULT_COST_CFG)

    assert len(proposals) == 1
    p = proposals[0]
    expected_structural_stop = 103.5 * (1 - cfg.stop_buffer_pct)
    assert p.stop_price == pytest.approx(expected_structural_stop)
    assert "stop_floor_applied" not in p.notes


# ---------------------------------------------------------------------------
# Weekend 4 -- cost-viability filter
# ---------------------------------------------------------------------------


def test_cost_viability_filter_boundary():
    # Same setup1 retest pattern as test_setup1_fires_on_retest_after_real_acceptance
    # (structural stop distance ~2.05, comfortably past the floor with a tiny
    # atr20, rr~2.44 -- clears the R:R filter regardless of cost config).
    # Only cost_viability_max_pct decides whether this specific proposal survives.
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
        atr=0.1,
    )
    strict_cfg = StrategyConfig(stop_floor_pct=0.0035, atr_mult=1.0, cost_viability_max_pct=0.0001)
    lenient_cfg = StrategyConfig(stop_floor_pct=0.0035, atr_mult=1.0, cost_viability_max_pct=1.0)

    assert engine.generate_proposals(df, "TEST", strict_cfg, _DEFAULT_COST_CFG) == []
    proposals = engine.generate_proposals(df, "TEST", lenient_cfg, _DEFAULT_COST_CFG)
    assert len(proposals) == 1


# ---------------------------------------------------------------------------
# Weekend 4 -- daily trade cap and stop-out cooldown live in backtest.simulator;
# see tests/test_backtest.py.
# ---------------------------------------------------------------------------
