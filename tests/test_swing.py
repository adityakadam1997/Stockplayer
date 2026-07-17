"""Unit tests for Cycle 3 (swing, daily bars) -- no network access, no cache/
dependency, synthetic data only.

Covers: weekly/monthly anchor reset correctness, the periodic condition
classifier, the delivery cost model, and swing_simulator's next-open
execution, gap-through-stop fill, time-stop, portfolio caps, and trend
filter. Most simulator-level tests use a permissive StrategyConfig (stop
floor and cost-viability filter neutralized) to isolate the specific
mechanic under test, matching the pattern established in
tests/test_strategy.py and tests/test_backtest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs_delivery, swing_simulator
from backtest.swing_simulator import SwingPortfolioConfig
from signals import condition as condition_module
from signals import vwap as vwap_module
from signals.condition import ACCEPTED_ABOVE, ACCEPTED_BELOW, INSIDE_VALUE
from strategy.base import LONG, SHORT, StrategyConfig


def _permissive_cfg(**overrides) -> StrategyConfig:
    defaults = dict(stop_floor_pct=0.0, atr_mult=0.0, cost_viability_max_pct=float("inf"))
    defaults.update(overrides)
    return StrategyConfig(**defaults)


_ZERO_SLIPPAGE_COST_CFG = costs_delivery.DeliveryCostConfig(slippage_pct=0.0)


def _daily_frame(
    dates,
    opens,
    highs,
    lows,
    closes,
    vwaps,
    band_upper,
    band_lower,
    conditions,
    streaks,
    atr=0.0,
    monthly_vwap=None,
    band_upper_2=None,
    high_20d_prior=None,
):
    ts = pd.to_datetime(dates).tz_localize("Asia/Kolkata")
    n = len(dates)
    atr_values = atr if isinstance(atr, list) else [atr] * n
    monthly_values = monthly_vwap if monthly_vwap is not None else [0.0] * n  # 0.0 -> trend filter always fails; override per test
    # Defaults kept generously above band_upper so setup1's Cycle 3B target
    # recomputation (prior_20d_high else band_upper_2) has *something* valid
    # to fall back on in tests that don't care about target geometry.
    band_upper_2_values = band_upper_2 if band_upper_2 is not None else [b + 50.0 for b in band_upper]
    high_20d_prior_values = high_20d_prior if high_20d_prior is not None else [float("nan")] * n
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
            "band_upper_2": band_upper_2_values,
            "condition": conditions,
            "acceptance_streak": streaks,
            "atr14": atr_values,
            "monthly_vwap": monthly_values,
            "high_20d_prior": high_20d_prior_values,
        }
    )


# ---------------------------------------------------------------------------
# Weekly / monthly anchor reset correctness
# ---------------------------------------------------------------------------


def test_weekly_vwap_resets_on_monday():
    week1 = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]).tz_localize("Asia/Kolkata"),  # Mon-Wed
            "open": [100, 200, 300],
            "high": [100, 200, 300],
            "low": [100, 200, 300],
            "close": [100, 200, 300],
            "volume": [1000, 1000, 1000],
        }
    )
    week2 = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-12", "2026-01-13"]).tz_localize("Asia/Kolkata"),  # next Mon-Tue
            "open": [500, 500],
            "high": [500, 500],
            "low": [500, 500],
            "close": [500, 500],
            "volume": [1000, 1000],
        }
    )
    df = pd.concat([week1, week2], ignore_index=True)
    out = vwap_module.compute_weekly_vwap(df)

    assert pd.isna(out.loc[0, "vwap"])  # first candle of week 1
    assert not pd.isna(out.loc[1, "vwap"])
    assert not pd.isna(out.loc[2, "vwap"])
    # New week -> fresh reset, not a continuation of week 1's much lower prices.
    assert pd.isna(out.loc[3, "vwap"])
    assert out.loc[4, "vwap"] == pytest.approx(500.0)


def test_monthly_vwap_resets_on_first_trading_day_of_month():
    jan = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-28", "2026-01-29"]).tz_localize("Asia/Kolkata"),
            "open": [100, 200],
            "high": [100, 200],
            "low": [100, 200],
            "close": [100, 200],
            "volume": [1000, 1000],
        }
    )
    feb = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-02-02", "2026-02-03"]).tz_localize("Asia/Kolkata"),
            "open": [900, 900],
            "high": [900, 900],
            "low": [900, 900],
            "close": [900, 900],
            "volume": [1000, 1000],
        }
    )
    df = pd.concat([jan, feb], ignore_index=True)
    out = vwap_module.compute_monthly_vwap(df)

    assert pd.isna(out.loc[0, "vwap"])
    assert not pd.isna(out.loc[1, "vwap"])
    assert pd.isna(out.loc[2, "vwap"])  # new month -> fresh reset
    assert out.loc[3, "vwap"] == pytest.approx(900.0)  # unaffected by January's much lower prices


def test_condition_periodic_resets_on_custom_period_key():
    # Two "periods" of 2 candles each, using an arbitrary integer period key
    # (not a real calendar period) -- proves the reset boundary is driven by
    # whatever period_key is passed in, not a hardcoded day/week/month.
    ts = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]).tz_localize("Asia/Kolkata")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 106, 100, 106],
            "high": [100, 106, 100, 106],
            "low": [100, 106, 100, 106],
            "close": [100, 106, 100, 106],
            "vwap": [100.0] * 4,
            "band_upper_1": [105.0] * 4,
            "band_lower_1": [95.0] * 4,
            "band_upper_2": [110.0] * 4,
            "band_lower_2": [90.0] * 4,
        }
    )
    period_key = pd.Series([1, 1, 2, 2])  # two custom 2-candle "periods"
    out = condition_module.compute_condition_periodic(df, period_key, acceptance_candles=1)

    # Each period's second candle closes above the band -> accepted_above,
    # immediately (acceptance_candles=1) -- streak must NOT carry across the
    # period-2 boundary (period 2 starts fresh at streak=0 despite period 1
    # ending on streak=1).
    assert out.loc[1, "condition"] == condition_module.ACCEPTED_ABOVE
    assert out.loc[1, "acceptance_streak"] == 1
    assert out.loc[2, "condition"] == INSIDE_VALUE  # period 2's first candle, no bands-relative-history yet within period
    assert out.loc[3, "condition"] == condition_module.ACCEPTED_ABOVE
    assert out.loc[3, "acceptance_streak"] == 1


# ---------------------------------------------------------------------------
# Delivery cost model
# ---------------------------------------------------------------------------


def test_delivery_round_trip_costs_hand_computed_long():
    cfg = costs_delivery.DeliveryCostConfig()
    entry_fill, exit_fill, quantity = 1000.0, 1050.0, 100

    entry_value = 100_000.0
    exit_value = 105_000.0
    expected_stt = (entry_value + exit_value) * 0.001  # both legs, 0.1%
    expected_txn = (entry_value + exit_value) * 0.0000325
    expected_sebi = (entry_value + exit_value) * 0.000001
    expected_stamp_duty = entry_value * 0.00015  # buy leg = entry for a long
    expected_dp = 20.0
    expected_brokerage = 0.0
    expected_gst = (expected_brokerage + expected_txn + expected_sebi + expected_dp) * 0.18
    expected_total = (
        expected_brokerage + expected_stt + expected_txn + expected_sebi + expected_stamp_duty + expected_dp + expected_gst
    )

    result = costs_delivery.round_trip_costs(LONG, entry_fill, exit_fill, quantity, cfg)
    assert result.stt == pytest.approx(expected_stt)
    assert result.txn_charges == pytest.approx(expected_txn)
    assert result.sebi_fee == pytest.approx(expected_sebi)
    assert result.stamp_duty == pytest.approx(expected_stamp_duty)
    assert result.dp_charges == pytest.approx(expected_dp)
    assert result.gst == pytest.approx(expected_gst)
    assert result.total == pytest.approx(expected_total)

    expected_gross = (exit_fill - entry_fill) * quantity
    net = costs_delivery.net_pnl(LONG, entry_fill, exit_fill, quantity, result)
    assert net == pytest.approx(expected_gross - expected_total)


def test_delivery_stamp_duty_on_exit_leg_for_short():
    # For a short, the buy leg is the EXIT (covering), not the entry.
    cfg = costs_delivery.DeliveryCostConfig()
    result = costs_delivery.round_trip_costs(SHORT, 1050.0, 1000.0, 100, cfg)
    exit_value = 1000.0 * 100
    assert result.stamp_duty == pytest.approx(exit_value * cfg.stamp_duty_pct)


# ---------------------------------------------------------------------------
# Next-open execution / no same-close fills
# ---------------------------------------------------------------------------


def _setup1_signal_frame(dates, entry_day_low, entry_day_close, band=105.0, extreme_high=112.0, extra_days=None):
    """A minimal frame that fires a single setup1 LONG proposal on the last
    "acceptance+retest" day: day0 builds acceptance (accepted_above,
    high=extreme_high), day1 is the retest (low<=band, close>band)."""
    n = len(dates)
    opens = [100.0] + [entry_day_close] * (n - 1)
    highs = [extreme_high] + [entry_day_close + 1] * (n - 1)
    lows = [99.0] + [entry_day_low] + [entry_day_close - 1] * (n - 2)
    closes = [108.0] + [entry_day_close] * (n - 1)
    conditions = [ACCEPTED_ABOVE] + [INSIDE_VALUE] * (n - 1)
    streaks = [3] + [0] * (n - 1)
    return _daily_frame(
        dates,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        vwaps=[100.0] * n,
        band_upper=[band] * n,
        band_lower=[95.0] * n,
        conditions=conditions,
        streaks=streaks,
        atr=0.0,
        monthly_vwap=[50.0] * n,  # close > 50 always -> trend filter passes for longs
        # Cycle 3B: target is recomputed from current structure at signal
        # time, not from state.acceptance_extreme_high -- give every row a
        # prior-20d-high of extreme_high so recomputation stays valid
        # (target > entry) for tests that don't specifically probe geometry.
        high_20d_prior=[extreme_high] * n,
    )


def test_no_trade_when_signal_is_last_available_day():
    # Only 2 days: day0 builds acceptance, day1 IS the retest signal -- but
    # there's no day2 to fill at, so the signal must be dropped entirely.
    df = _setup1_signal_frame(["2026-01-05", "2026-01-06"], entry_day_low=104.0, entry_day_close=106.0)
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig()

    trades = swing_simulator.run_portfolio({"TEST": df}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    assert trades == []


def test_entry_fills_at_next_day_open_not_signal_close():
    # day0: acceptance. day1: retest signal (close=106). day2: fill day,
    # open=107 (deliberately different from day1's close=106, so a bug that
    # fills at the signal's own close would be caught).
    df = _setup1_signal_frame(["2026-01-05", "2026-01-06", "2026-01-07"], entry_day_low=104.0, entry_day_close=106.0)
    df.loc[2, "open"] = 107.0
    df.loc[2, "high"] = 108.0
    df.loc[2, "low"] = 106.5
    df.loc[2, "close"] = 107.5
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig()

    trades = swing_simulator.run_portfolio({"TEST": df}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.signal_timestamp == df.loc[1, "timestamp"]
    assert trade.entry_timestamp == df.loc[2, "timestamp"]
    assert trade.entry_signal_price == pytest.approx(106.0)  # the signal day's close (reference only)
    assert trade.entry_fill_price == pytest.approx(107.0)  # the NEXT day's open, not 106.0


# ---------------------------------------------------------------------------
# Gap-through-stop fill
# ---------------------------------------------------------------------------


def test_gap_through_stop_fills_at_worse_open_not_stop_price():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)
    # fill day (day2): open=107 (normal fill).
    df.loc[2, ["open", "high", "low", "close"]] = [107.0, 108.0, 106.5, 107.5]
    # day3: gaps down hard, opening at 95 -- well below whatever the stop
    # ends up being (structural stop = buffered_stop(104, 0.0005, LONG) ~103.95).
    df.loc[3, ["open", "high", "low", "close"]] = [95.0, 96.0, 94.0, 95.5]
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig()

    trades = swing_simulator.run_portfolio({"TEST": df}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "stop"
    assert trade.stop_price == pytest.approx(104.0 * (1 - cfg.stop_buffer_pct))
    assert trade.exit_fill_price == pytest.approx(95.0)  # the gapped-through open, worse than the nominal stop
    assert trade.exit_fill_price < trade.stop_price


def test_stop_touched_without_gap_fills_at_stop_price():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)
    df.loc[2, ["open", "high", "low", "close"]] = [107.0, 108.0, 106.5, 107.5]
    expected_stop = 104.0 * (1 - StrategyConfig().stop_buffer_pct)
    # day3: opens well above the stop, but the LOW touches it intraday -- no gap.
    df.loc[3, ["open", "high", "low", "close"]] = [107.0, 107.5, expected_stop - 0.1, 106.0]
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig()

    trades = swing_simulator.run_portfolio({"TEST": df}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_fill_price == pytest.approx(expected_stop)


# ---------------------------------------------------------------------------
# Time-stop
# ---------------------------------------------------------------------------


def test_time_stop_exits_at_close_after_configured_trading_days():
    # Signal + fill, then 12 more days that never touch stop or target
    # (target is intentionally unreachable) -- must force-exit at the close
    # of the (entry_index + time_stop_days)'th day.
    n_hold_days = 12
    dates = ["2026-01-05", "2026-01-06"] + [f"2026-01-{6 + i:02d}" for i in range(1, n_hold_days + 1)]
    # fix any month-rollover in the naive day-string construction by using date_range instead
    dates = pd.bdate_range("2026-01-05", periods=2 + n_hold_days).strftime("%Y-%m-%d").tolist()
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)
    for i in range(2, len(dates)):
        df.loc[i, ["open", "high", "low", "close"]] = [106.0, 106.5, 105.5, 106.0]  # flat, never hits stop/target

    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(time_stop_days=10)

    trades = swing_simulator.run_portfolio({"TEST": df}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "time_stop"
    assert trade.holding_days == 10
    entry_idx = 2  # fill day index
    assert trade.exit_timestamp == df.loc[entry_idx + 10, "timestamp"]
    assert trade.exit_signal_price == pytest.approx(106.0)  # that day's close


# ---------------------------------------------------------------------------
# Portfolio caps
# ---------------------------------------------------------------------------


def test_max_concurrent_positions_blocks_a_lower_priority_symbol():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    df_a = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)
    df_b = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)

    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(max_concurrent_positions=1)

    trades = swing_simulator.run_portfolio({"AAA": df_a, "ZZZ": df_b}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    symbols_traded = {t.symbol for t in trades}
    assert symbols_traded == {"AAA"}  # alphabetically first wins the single slot


def test_max_concurrent_positions_allows_both_when_raised():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    df_a = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)
    df_b = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)

    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(max_concurrent_positions=2)

    trades = swing_simulator.run_portfolio({"AAA": df_a, "ZZZ": df_b}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    symbols_traded = {t.symbol for t in trades}
    assert symbols_traded == {"AAA", "ZZZ"}


def test_max_one_position_per_symbol():
    # A symbol whose signal geometry could fire twice (once per acceptance
    # cycle) -- the second signal's fill day must find the symbol already
    # open and skip it.
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13"]
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0, extreme_high=200.0)
    # Keep the position open (flat, no stop/target hit) through days 2-6; a
    # second retest-shaped candle on day 4 must not open a 2nd position while
    # day1's is still open.
    for i in range(2, len(dates)):
        df.loc[i, ["open", "high", "low", "close"]] = [106.0, 106.5, 105.5, 106.0]
    df.loc[4, ["low", "close"]] = [104.0, 106.0]  # geometrically another valid retest, symbol already open though

    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(time_stop_days=100)

    trades = swing_simulator.run_portfolio({"TEST": df}, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg)
    assert len(trades) <= 1


# ---------------------------------------------------------------------------
# Trend filter
# ---------------------------------------------------------------------------


def test_trend_filter_blocks_long_below_monthly_vwap():
    dates = ["2026-01-05", "2026-01-06"]
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0)
    df["monthly_vwap"] = 200.0  # close (106) < monthly_vwap -> long blocked

    from strategy.swing_engine import generate_proposals

    proposals = generate_proposals(df, "TEST", _permissive_cfg(), _ZERO_SLIPPAGE_COST_CFG)
    assert proposals == []


def test_trend_filter_allows_long_above_monthly_vwap():
    dates = ["2026-01-05", "2026-01-06"]
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0)
    df["monthly_vwap"] = 50.0  # close (106) > monthly_vwap -> long allowed

    from strategy.swing_engine import generate_proposals

    proposals = generate_proposals(df, "TEST", _permissive_cfg(), _ZERO_SLIPPAGE_COST_CFG)
    assert len(proposals) == 1
    assert proposals[0].direction == LONG


# ---------------------------------------------------------------------------
# Cycle 3B: long-only suppression
# ---------------------------------------------------------------------------


def _setup1_short_signal_frame(dates, entry_day_high, entry_day_close, band=95.0, extreme_low=85.0):
    """Mirror of ``_setup1_signal_frame`` for the SHORT branch: day0 builds
    ``accepted_below`` (low=extreme_low), day1 is the retest (high>=band,
    close<band). ``monthly_vwap`` is set high so close < monthly_vwap ->
    the trend filter *passes* shorts, isolating the long-only filter as the
    thing that actually suppresses them."""
    n = len(dates)
    opens = [100.0] + [entry_day_close] * (n - 1)
    highs = [101.0] + [entry_day_high] + [entry_day_close + 1] * (n - 2)
    lows = [extreme_low] + [entry_day_close - 1] * (n - 1)
    closes = [92.0] + [entry_day_close] * (n - 1)
    conditions = [ACCEPTED_BELOW] + [INSIDE_VALUE] * (n - 1)
    streaks = [3] + [0] * (n - 1)
    return _daily_frame(
        dates,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        vwaps=[100.0] * n,
        band_upper=[105.0] * n,
        band_lower=[band] * n,
        conditions=conditions,
        streaks=streaks,
        atr=0.0,
        monthly_vwap=[200.0] * n,  # close always < 200 -> trend filter passes for shorts
    )


def test_long_only_filter_suppresses_shorts_and_counts_them_in_funnel():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    df = _setup1_short_signal_frame(dates, entry_day_high=96.0, entry_day_close=94.0)

    from strategy.swing_engine import generate_proposals_with_funnel

    proposals, funnel = generate_proposals_with_funnel(df, "TEST", _permissive_cfg(), _ZERO_SLIPPAGE_COST_CFG)
    assert proposals == []  # cash delivery cannot short -- must never reach execution
    assert all(p.direction != SHORT for p in proposals)
    assert funnel["after_trend_filter"] >= 1  # the short candidate genuinely reached the long-only stage
    assert funnel["suppressed_shorts"] >= 1
    assert funnel["after_long_only"] == 0


def test_long_only_filter_leaves_longs_untouched():
    dates = ["2026-01-05", "2026-01-06"]
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0)

    from strategy.swing_engine import generate_proposals_with_funnel

    proposals, funnel = generate_proposals_with_funnel(df, "TEST", _permissive_cfg(), _ZERO_SLIPPAGE_COST_CFG)
    assert len(proposals) == 1
    assert proposals[0].direction == LONG
    assert funnel["suppressed_shorts"] == 0


# ---------------------------------------------------------------------------
# Cycle 3B: target recomputation at signal time / invalid-geometry discard
# ---------------------------------------------------------------------------


def test_invalid_geometry_discards_proposal_with_nonpositive_recomputed_target():
    # Both of setup1's recomputation candidates (prior 20d high, band_upper_2)
    # sit BELOW entry -- the proposal must be discarded as invalid_geometry,
    # never reaching the R:R filter at all (which would otherwise happily
    # accept whatever stale target the detector originally proposed).
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    df = _setup1_signal_frame(dates, entry_day_low=104.0, entry_day_close=106.0)
    df["high_20d_prior"] = 50.0  # < entry (106)
    df["band_upper_2"] = 60.0  # < entry (106) -- fallback also invalid

    from strategy.swing_engine import generate_proposals_with_funnel

    proposals, funnel = generate_proposals_with_funnel(df, "TEST", _permissive_cfg(), _ZERO_SLIPPAGE_COST_CFG)
    assert proposals == []
    assert funnel["after_long_only"] >= 1  # candidate genuinely reached the geometry-recompute stage
    assert funnel["invalid_geometry"] >= 1
    assert funnel["after_valid_geometry"] == 0


def test_valid_geometry_but_failed_rr_is_a_different_funnel_bucket_than_invalid_geometry():
    # Reward is genuinely positive (target > entry) but small relative to a
    # wide stop -- this must fail the R:R bar, landing in a DIFFERENT funnel
    # bucket than a target that's on the wrong side of entry altogether.
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    df = _setup1_signal_frame(dates, entry_day_low=90.0, entry_day_close=106.0)  # wide stop: entry-low = 16
    df["high_20d_prior"] = 107.0  # target only Rs1 above entry -> reward=1, rr << min_rr(1.5)
    df["band_upper_2"] = 200.0

    from strategy.swing_engine import generate_proposals_with_funnel

    proposals, funnel = generate_proposals_with_funnel(df, "TEST", _permissive_cfg(), _ZERO_SLIPPAGE_COST_CFG)
    assert proposals == []
    assert funnel["invalid_geometry"] == 0  # geometry WAS valid (target > entry)
    assert funnel["after_valid_geometry"] >= 1
    assert funnel["after_rr"] == 0  # ... it just failed the R:R bar


# ---------------------------------------------------------------------------
# Cycle 3B: 20-day-high target correctness, no lookahead
# ---------------------------------------------------------------------------


def test_compute_prior_n_day_high_excludes_current_bar():
    from strategy.swing_engine import compute_prior_n_day_high

    highs = [10.0] * 25
    highs[24] = 9999.0  # today's own high must never count toward today's "prior" high
    df = pd.DataFrame({"high": highs})

    result = compute_prior_n_day_high(df, window=20)
    assert result.iloc[24] == pytest.approx(10.0)


def test_compute_prior_n_day_high_respects_window_size():
    from strategy.swing_engine import compute_prior_n_day_high

    highs = [10.0] * 50
    highs[5] = 500.0  # a spike far in the past
    df = pd.DataFrame({"high": highs})

    result = compute_prior_n_day_high(df, window=20)
    assert result.iloc[25] == pytest.approx(500.0)  # still inside the 20-bar lookback window
    assert result.iloc[26] == pytest.approx(10.0)  # window has now slid past the spike


def test_compute_prior_n_day_high_first_row_is_nan():
    from strategy.swing_engine import compute_prior_n_day_high

    df = pd.DataFrame({"high": [10.0, 20.0, 30.0]})
    result = compute_prior_n_day_high(df, window=20)
    assert pd.isna(result.iloc[0])  # no prior bars exist yet
    assert result.iloc[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Cycle 3B: current-band target read at signal close (setups 2/3/4)
# ---------------------------------------------------------------------------


def _single_row(**overrides) -> object:
    base = {
        "band_upper_1": float("nan"),
        "band_upper_2": float("nan"),
        "vwap": float("nan"),
        "high_20d_prior": float("nan"),
    }
    base.update(overrides)
    df = pd.DataFrame({k: [v] for k, v in base.items()})
    return next(df.itertuples())


def test_recompute_target_setup4_reads_current_row_band_not_stale_proposal_target():
    from strategy.base import TradeProposal
    from strategy.swing_engine import _recompute_target
    from strategy import setup4_bounce

    stale_proposal = TradeProposal(
        symbol="TEST",
        timestamp=pd.Timestamp("2026-01-06"),
        setup_id=setup4_bounce.SETUP_ID,
        direction=LONG,
        entry_price=100.0,
        stop_price=95.0,
        target_price=999.0,  # deliberately wrong/stale -- must be discarded, not trusted
        rr_ratio=10.0,
        condition_at_entry="inside_value",
        acceptance_streak_at_entry=0,
        notes="orig",
    )
    row = _single_row(band_upper_1=110.0)  # the CURRENT row's structure at signal close

    new_proposal = _recompute_target(stale_proposal, row)
    assert new_proposal is not None
    assert new_proposal.target_price == pytest.approx(110.0)  # from row, not the stale 999.0


def test_recompute_target_setup2_and_setup3_read_current_row_vwap():
    from strategy.base import TradeProposal
    from strategy.swing_engine import _recompute_target
    from strategy import setup2_fade, setup3_return

    for setup_id in (setup2_fade.SETUP_ID, setup3_return.SETUP_ID):
        stale_proposal = TradeProposal(
            symbol="TEST",
            timestamp=pd.Timestamp("2026-01-06"),
            setup_id=setup_id,
            direction=LONG,
            entry_price=100.0,
            stop_price=95.0,
            target_price=123.0,  # stale -- must be discarded in favor of row.vwap
            rr_ratio=5.0,
            condition_at_entry="inside_value",
            acceptance_streak_at_entry=0,
            notes="orig",
        )
        row = _single_row(vwap=108.0)

        new_proposal = _recompute_target(stale_proposal, row)
        assert new_proposal is not None
        assert new_proposal.target_price == pytest.approx(108.0)


def test_recompute_target_setup4_invalid_when_current_band_not_above_entry():
    from strategy.base import TradeProposal
    from strategy.swing_engine import _recompute_target
    from strategy import setup4_bounce

    proposal = TradeProposal(
        symbol="TEST",
        timestamp=pd.Timestamp("2026-01-06"),
        setup_id=setup4_bounce.SETUP_ID,
        direction=LONG,
        entry_price=100.0,
        stop_price=95.0,
        target_price=105.0,
        rr_ratio=1.0,
        condition_at_entry="inside_value",
        acceptance_streak_at_entry=0,
        notes="orig",
    )
    row = _single_row(band_upper_1=99.0)  # <= entry -> invalid geometry

    assert _recompute_target(proposal, row) is None
