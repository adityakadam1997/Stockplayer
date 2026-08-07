"""Unit tests for backtest/*.py -- no network access, no dependency on the
local cache/ directory. Synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostConfig, apply_entry_slippage, apply_exit_slippage, compute_trade_costs
from backtest.simulator import simulate_symbol
from strategy.base import LONG, SETUP1_DISCOVERY, SHORT, StrategyConfig


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_cost_model_hand_computed_round_trip() -> None:
    config = CostConfig(
        brokerage_per_order=20.0,
        stt_sell_side_pct=0.025,
        transaction_charges_pct=0.005,
        gst_pct=18.0,
    )
    # A long trade: buy 100 shares at 200, sell at 210.
    entry_fill = 200.0
    exit_fill = 210.0
    quantity = 100

    costs = compute_trade_costs(LONG, entry_fill, exit_fill, quantity, config)

    entry_turnover = 200.0 * 100  # 20,000
    exit_turnover = 210.0 * 100  # 21,000

    expected_brokerage = 20.0 * 2  # 40
    expected_stt = exit_turnover * (0.025 / 100.0)  # sell side = exit for a long
    expected_transaction_charges = (entry_turnover + exit_turnover) * (0.005 / 100.0)
    expected_gst = (expected_brokerage + expected_transaction_charges) * (18.0 / 100.0)

    assert costs.brokerage == pytest.approx(expected_brokerage)
    assert costs.stt == pytest.approx(expected_stt)
    assert costs.transaction_charges == pytest.approx(expected_transaction_charges)
    assert costs.gst == pytest.approx(expected_gst)
    assert costs.total == pytest.approx(
        expected_brokerage + expected_stt + expected_transaction_charges + expected_gst
    )


def test_cost_model_stt_applies_to_entry_leg_for_shorts() -> None:
    config = CostConfig()
    entry_fill = 200.0  # the sell (short) leg
    exit_fill = 190.0  # the buy-to-cover leg
    quantity = 50

    costs = compute_trade_costs(SHORT, entry_fill, exit_fill, quantity, config)
    expected_stt = (entry_fill * quantity) * (config.stt_sell_side_pct / 100.0)
    assert costs.stt == pytest.approx(expected_stt)


def test_entry_slippage_worse_for_both_directions() -> None:
    config = CostConfig(slippage_mode="ticks", slippage_ticks=1, tick_size=0.05)
    assert apply_entry_slippage(100.0, LONG, config) == pytest.approx(100.05)
    assert apply_entry_slippage(100.0, SHORT, config) == pytest.approx(99.95)


def test_exit_slippage_stop_uses_wider_rate() -> None:
    config = CostConfig(stop_slippage_pct=0.05, slippage_mode="ticks", slippage_ticks=1, tick_size=0.05)
    stop_price = 200.0
    filled = apply_exit_slippage(stop_price, LONG, config, is_stop=True)
    assert filled == pytest.approx(stop_price - stop_price * 0.0005)
    # A non-stop exit (target) uses the smaller general slippage instead.
    target_filled = apply_exit_slippage(stop_price, LONG, config, is_stop=False)
    assert target_filled == pytest.approx(stop_price - 0.05)


# ---------------------------------------------------------------------------
# Simulator: fills and time rules
# ---------------------------------------------------------------------------


def _candle(time: str, open_, high, low, close, date="2026-01-05", **extra) -> dict:
    return {
        "timestamp": pd.Timestamp(f"{date} {time}", tz="Asia/Kolkata"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vwap": extra.get("vwap", close),
        "band_upper_1": extra.get("band_upper_1", close + 10),
        "band_lower_1": extra.get("band_lower_1", close - 10),
        "band_upper_2": extra.get("band_upper_2", close + 20),
        "band_lower_2": extra.get("band_lower_2", close - 20),
        "condition": extra.get("condition", "inside_value"),
        "acceptance_streak": extra.get("acceptance_streak", 0),
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_stop_first_when_both_stop_and_target_touched_in_one_candle() -> None:
    # Long entry at 100.30 via setup1 acceptance -> pullback pattern, on the
    # first candle of the session. The next candle's range touches BOTH the
    # stop and the target -- the pessimistic rule says the stop fills.
    rows = [
        _candle("09:35", 108, 110, 108, 109, band_upper_1=100, condition="accepted_above"),
        _candle("09:40", 109, 112, 110, 111, band_upper_1=101, condition="accepted_above"),
        # Entry candle: pullback that tests the band and closes back above it.
        _candle("09:45", 103, 104, 101.8, 104, band_upper_1=102, condition="accepted_above"),
        # Next candle's range spans well below the stop AND above the target.
        _candle("09:50", 103, 200, 50, 103, band_upper_1=102, condition="accepted_above"),
    ]
    df = _make_df(rows)
    config = StrategyConfig()
    cost_config = CostConfig()

    records = simulate_symbol("TEST", df, config, cost_config, capital=100_000, risk_pct=0.5)

    assert len(records) == 1
    assert records[0].setup_id == SETUP1_DISCOVERY
    assert records[0].exit_reason == "stop"


def test_forced_square_off_at_configured_time() -> None:
    rows = [
        _candle("09:35", 108, 110, 108, 109, band_upper_1=100, condition="accepted_above"),
        _candle("09:40", 109, 112, 110, 111, band_upper_1=101, condition="accepted_above"),
        # Entry candle.
        _candle("09:45", 103, 104, 101.8, 104, band_upper_1=102, condition="accepted_above"),
        # A quiet candle -- neither stop nor target touched.
        _candle("09:50", 104, 105, 103.5, 104.5, band_upper_1=102, condition="accepted_above"),
        # The force-close candle: still doesn't touch stop or target, but
        # it's at/after the configured force_close_at time.
        _candle("15:15", 104.5, 105, 104, 104.8, band_upper_1=102, condition="accepted_above"),
        # A later candle that would otherwise hit the target -- must never
        # be reached because the position was already force-closed.
        _candle("15:20", 104.8, 200, 104, 150, band_upper_1=102, condition="accepted_above"),
    ]
    df = _make_df(rows)
    config = StrategyConfig()
    cost_config = CostConfig()

    records = simulate_symbol("TEST", df, config, cost_config, capital=100_000, risk_pct=0.5)

    assert len(records) == 1
    record = records[0]
    assert record.exit_reason == "time_exit"
    assert record.exit_timestamp == pd.Timestamp("2026-01-05 15:15", tz="Asia/Kolkata")
    assert record.exit_price == 104.8  # the 15:15 candle's close
