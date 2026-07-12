"""Unit tests for backtest/ -- no network access, no cache/ dependency."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import costs, simulator
from signals.condition import ACCEPTED_ABOVE, INSIDE_VALUE
from strategy.base import LONG, SHORT, StrategyConfig


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
# Cost model
# ---------------------------------------------------------------------------


def test_round_trip_costs_hand_computed():
    cfg = costs.CostConfig(
        brokerage_per_order=20.0,
        stt_sell_pct=0.00025,
        txn_charge_pct=0.00005,
        gst_pct=0.18,
    )
    entry_fill, exit_fill, quantity = 100.0, 105.0, 10

    entry_value = 1000.0
    exit_value = 1050.0
    expected_stt = 1050.0 * 0.00025  # sell leg = exit for a long
    expected_txn = (1000.0 + 1050.0) * 0.00005
    expected_brokerage = 40.0
    expected_gst = (expected_brokerage + expected_txn) * 0.18
    expected_total = expected_brokerage + expected_stt + expected_txn + expected_gst

    result = costs.round_trip_costs(LONG, entry_fill, exit_fill, quantity, cfg)
    assert result.brokerage == pytest.approx(expected_brokerage)
    assert result.stt == pytest.approx(expected_stt)
    assert result.txn_charges == pytest.approx(expected_txn)
    assert result.gst == pytest.approx(expected_gst)
    assert result.total == pytest.approx(expected_total)

    expected_gross = (exit_fill - entry_fill) * quantity
    net = costs.net_pnl(LONG, entry_fill, exit_fill, quantity, result)
    assert net == pytest.approx(expected_gross - expected_total)


def test_round_trip_costs_stt_on_entry_for_short():
    # For a short, the sell leg is the entry, not the exit.
    cfg = costs.CostConfig()
    result_short = costs.round_trip_costs(SHORT, 105.0, 100.0, 10, cfg)
    expected_stt = (105.0 * 10) * cfg.stt_sell_pct
    assert result_short.stt == pytest.approx(expected_stt)


# ---------------------------------------------------------------------------
# Stop-first intra-candle fill rule
# ---------------------------------------------------------------------------


def test_stop_fills_first_when_candle_touches_both_stop_and_target():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "09:35"],
        opens=[100, 108, 109, 106, 106],
        highs=[102, 115, 116, 111, 113],
        lows=[99, 107, 108, 104, 103],
        closes=[100, 108, 110, 106, 108],
        vwaps=[100, 100, 100, 100, 100],
        band_upper=[105, 105, 105, 105, 105],
        band_lower=[95, 95, 95, 95, 95],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE],
        streaks=[0, 3, 4, 5, 5],
    )
    # row3 (09:30): low=104 touches band_upper_1=105, close=106 reclaims it ->
    # setup1 long entry=106, stop=104*(1-0.0005)=103.948, target=extreme high
    # reached during acceptance (116) -> comfortably clears the R:R filter.
    # row4 (09:35): range [103, 113] touches the stop (103.948) but not the
    # target (116 is NOT touched here on purpose).
    strategy_cfg = StrategyConfig()
    cost_cfg = costs.CostConfig()
    sim_cfg = simulator.SimulatorConfig()

    trades = simulator.run_symbol("TEST", df, strategy_cfg, cost_cfg, sim_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.direction == LONG
    assert trade.target_price == pytest.approx(116.0)
    assert trade.exit_reason == "stop"
    assert trade.stop_price == pytest.approx(103.948)


def test_stop_wins_over_target_when_both_touched_same_candle():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "09:35"],
        opens=[100, 108, 109, 106, 106],
        highs=[102, 112, 112.5, 111, 113],
        lows=[99, 107, 108, 104, 103],
        closes=[100, 108, 110, 106, 108],
        vwaps=[100, 100, 100, 100, 100],
        band_upper=[105, 105, 105, 105, 105],
        band_lower=[95, 95, 95, 95, 95],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE],
        streaks=[0, 3, 4, 5, 5],
    )
    # target = extreme high during acceptance = 112.5 (row2's high); row4's
    # range [103, 113] touches BOTH the stop (103.948) and the target (112.5).
    strategy_cfg = StrategyConfig()
    cost_cfg = costs.CostConfig()
    sim_cfg = simulator.SimulatorConfig()

    trades = simulator.run_symbol("TEST", df, strategy_cfg, cost_cfg, sim_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.target_price == pytest.approx(112.5)
    assert trade.exit_reason == "stop", "pessimistic assumption: stop fills first when a candle touches both"
    expected_fill = costs.apply_exit_slippage(trade.stop_price, LONG, cost_cfg, is_stop=True)
    assert trade.exit_fill_price == pytest.approx(expected_fill)


# ---------------------------------------------------------------------------
# Forced square-off at 15:15
# ---------------------------------------------------------------------------


def test_forced_square_off_at_1515():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "15:15"],
        opens=[100, 108, 109, 106, 106.5],
        highs=[102, 130, 131, 111, 108],
        lows=[99, 107, 108, 104, 106],
        closes=[100, 108, 110, 106, 107],
        vwaps=[100, 100, 100, 100, 100],
        band_upper=[105, 105, 105, 105, 105],
        band_lower=[95, 95, 95, 95, 95],
        conditions=[INSIDE_VALUE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE, ACCEPTED_ABOVE],
        streaks=[0, 3, 4, 5, 5],
    )
    # target = extreme high during acceptance = 131 (deliberately unreachable);
    # the 15:15 candle's range [106, 108] never touches the stop (103.948) or
    # the target either, so the position can only be closed by the forced
    # square-off rule.
    strategy_cfg = StrategyConfig()
    cost_cfg = costs.CostConfig()
    sim_cfg = simulator.SimulatorConfig()

    trades = simulator.run_symbol("TEST", df, strategy_cfg, cost_cfg, sim_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "square_off"
    assert trade.exit_signal_price == pytest.approx(107.0)  # the 15:15 candle's close
    assert trade.exit_timestamp == df.loc[4, "timestamp"]


# ---------------------------------------------------------------------------
# Setup 2's early "close beyond the band" exit
# ---------------------------------------------------------------------------


def test_setup2_exits_at_next_open_on_close_beyond_band():
    df = _frame(
        "2026-01-05",
        ["09:15", "09:20", "09:25", "09:30", "09:35", "09:40"],
        opens=[100.2, 99.8, 100.2, 100.6, 100.65, 100.85],
        highs=[100.3, 99.9, 100.3, 100.9, 100.8, 100.9],
        lows=[100.1, 99.7, 100.1, 100.5, 100.6, 100.7],
        closes=[100.2, 99.8, 100.2, 100.6, 100.75, 100.8],
        vwaps=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        band_upper=[100.7, 100.7, 100.7, 100.7, 100.7, 100.7],
        band_lower=[99.3, 99.3, 99.3, 99.3, 99.3, 99.3],
        conditions=[INSIDE_VALUE] * 6,
        streaks=[0] * 6,
    )
    # row3 (09:30): tag+reject of band_upper_1 -> setup2 short, entry=100.6,
    # stop=100.9*(1.0005)=100.9505, target=vwap=100.0.
    # row4 (09:35): closes at 100.75, beyond the faded band (100.7), without
    # touching the stop -- this should flag an exit at row5's open, not wait
    # for the stop.
    strategy_cfg = StrategyConfig()
    cost_cfg = costs.CostConfig()
    sim_cfg = simulator.SimulatorConfig()

    trades = simulator.run_symbol("TEST", df, strategy_cfg, cost_cfg, sim_cfg)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.setup_id == "setup2_fade"
    assert trade.exit_reason == "setup2_band_break"
    assert trade.exit_signal_price == pytest.approx(100.85)  # row5's open
    assert trade.exit_timestamp == df.loc[5, "timestamp"]
