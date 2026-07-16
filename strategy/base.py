"""Shared building blocks for the VWAP Wave System setups.

``TradeProposal`` is the common output type every setup emits. ``SessionState``
is the per-(symbol, day) memory that setup detectors read from -- it always
reflects candles *strictly before* the one currently being evaluated, which is
what makes the no-lookahead property in ``strategy.engine`` structurally
guaranteed rather than merely convention: a detector call only ever sees (a)
the current candle's own OHLC/vwap/band/condition values and (b) state folded
from earlier candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path

import pandas as pd
import yaml

from signals.condition import ACCEPTED_ABOVE, ACCEPTED_BELOW, INSIDE_VALUE

LONG = "long"
SHORT = "short"

# SessionState transition events, computed by comparing the previous candle's
# condition to the current candle's.
BROKE_FROM_ABOVE = "broke_from_above"
BROKE_FROM_BELOW = "broke_from_below"
REACCEPTED_ABOVE = "reaccepted_above"
REACCEPTED_BELOW = "reaccepted_below"


@dataclass
class TradeProposal:
    symbol: str
    timestamp: pd.Timestamp
    setup_id: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    rr_ratio: float
    condition_at_entry: str
    acceptance_streak_at_entry: int
    notes: str = ""


ATR_PERIOD = 20
ATR_COLUMN = f"atr{ATR_PERIOD}"


@dataclass
class StrategyConfig:
    min_rr: float = 1.5
    stop_buffer_pct: float = 0.0005
    setup2_min_vwap_crossings: int = 2
    setup3_allow_fallback: bool = False
    setup4_vwap_touch_tolerance_pct: float = 0.0005
    wide_band_guard_pct: float = 0.015
    session_open_no_entry_minutes: int = 15
    no_entry_after: time = time(14, 45)
    square_off_at: time = time(15, 15)

    # Weekend 4: stop floor -- the stop can never be tighter than
    # max(stop_floor_pct * entry_price, atr_mult * ATR20). Structural stops
    # from the setups only ever get pushed further away, never tighter.
    stop_floor_pct: float = 0.0035
    atr_mult: float = 1.0

    # Weekend 4: skip a proposal if its estimated full round-trip cost (fees +
    # both-sides slippage, at the position size the floored stop implies)
    # exceeds this fraction of the nominal rupee risk on the trade.
    cost_viability_max_pct: float = 0.15
    # Mirrors backtest:capital/risk_pct -- needed here only to size the
    # cost-viability estimate at proposal time, before the simulator exists.
    capital: float = 100_000.0
    risk_pct: float = 0.005

    # Weekend 4: frequency discipline, enforced by backtest.simulator (it's
    # the only layer that knows which proposals actually got taken).
    max_trades_per_day: int = 2
    stop_cooldown_minutes: int = 30


def _parse_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    hour, minute = (int(part) for part in value.split(":"))
    return time(hour, minute)


def load_strategy_config(config_path: Path, timeframe: str | None = None) -> StrategyConfig:
    """``timeframe`` (e.g. ``"15min"``) merges ``timeframes.<timeframe>.strategy``
    on top of the base ``strategy:`` section -- see config.yaml's ``timeframes:``
    block. ``None`` (or a timeframe with no override section) uses the base
    5-min profile unchanged."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    strategy_raw = dict(raw.get("strategy", {}))
    backtest_raw = dict(raw.get("backtest", {}))
    if timeframe:
        overrides = raw.get("timeframes", {}).get(timeframe, {})
        strategy_raw.update(overrides.get("strategy", {}))
        backtest_raw.update(overrides.get("backtest", {}))
    defaults = StrategyConfig()
    return StrategyConfig(
        min_rr=strategy_raw.get("min_rr", defaults.min_rr),
        stop_buffer_pct=strategy_raw.get("stop_buffer_pct", defaults.stop_buffer_pct),
        setup2_min_vwap_crossings=strategy_raw.get(
            "setup2_min_vwap_crossings", defaults.setup2_min_vwap_crossings
        ),
        setup3_allow_fallback=strategy_raw.get("setup3_allow_fallback", defaults.setup3_allow_fallback),
        setup4_vwap_touch_tolerance_pct=strategy_raw.get(
            "setup4_vwap_touch_tolerance_pct", defaults.setup4_vwap_touch_tolerance_pct
        ),
        wide_band_guard_pct=strategy_raw.get("wide_band_guard_pct", defaults.wide_band_guard_pct),
        session_open_no_entry_minutes=strategy_raw.get(
            "session_open_no_entry_minutes", defaults.session_open_no_entry_minutes
        ),
        no_entry_after=_parse_time(strategy_raw.get("no_entry_after", defaults.no_entry_after)),
        square_off_at=_parse_time(strategy_raw.get("square_off_at", defaults.square_off_at)),
        stop_floor_pct=strategy_raw.get("stop_floor_pct", defaults.stop_floor_pct),
        atr_mult=strategy_raw.get("atr_mult", defaults.atr_mult),
        cost_viability_max_pct=strategy_raw.get("cost_viability_max_pct", defaults.cost_viability_max_pct),
        capital=backtest_raw.get("capital", defaults.capital),
        risk_pct=backtest_raw.get("risk_pct", defaults.risk_pct),
        max_trades_per_day=strategy_raw.get("max_trades_per_day", defaults.max_trades_per_day),
        stop_cooldown_minutes=strategy_raw.get("stop_cooldown_minutes", defaults.stop_cooldown_minutes),
    )


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Add an ``atr{period}`` column: a session-aware, no-lookahead average
    true range.

    True range resets at each session boundary -- the first candle of a
    session has no same-session prior close to reference, so its TR is just
    high-low (matching how ``signals.vwap`` treats a session's first candle).
    The rolling average is a backward-looking window over the *current and
    up to period-1 preceding* candles within the same session (never crosses
    a session boundary, never reaches into the future) -- so ``atr{period}``
    at row T is a pure function of rows <= T. See
    ``tests/test_strategy.py::test_atr_no_lookahead``.
    """
    df = df.copy()
    day = df["timestamp"].dt.date
    prev_close = df.groupby(day, sort=False)["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - prev_close).abs()
    low_prev_close = (df["low"] - prev_close).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    df[f"atr{period}"] = true_range.groupby(day, sort=False).transform(
        lambda s: s.rolling(window=period, min_periods=1).mean()
    )
    return df


def compute_rr(entry_price: float, stop_price: float, target_price: float, direction: str) -> float:
    """Reward:risk ratio at entry. Zero (never taken) if risk is zero or the
    target isn't actually on the profitable side of entry."""
    risk = abs(entry_price - stop_price)
    reward = (target_price - entry_price) if direction == LONG else (entry_price - target_price)
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk


def buffered_stop(reference_price: float, buffer_pct: float, direction: str) -> float:
    """A stop a configurable buffer beyond ``reference_price`` -- below it for a
    long (stop must sit under support), above it for a short (stop must sit
    over resistance)."""
    if direction == LONG:
        return reference_price * (1 - buffer_pct)
    return reference_price * (1 + buffer_pct)


@dataclass
class SessionState:
    """Per-(symbol, trading day) memory, folded forward one candle at a time by
    ``update_session_state``. Never contains information from the candle
    currently being evaluated by the setup detectors."""

    day: date | None = None

    had_acceptance_above: bool = False
    had_acceptance_below: bool = False
    # Running extreme reached *while* accepted this session -- setup 1's measured-move target.
    acceptance_extreme_high: float | None = None
    acceptance_extreme_low: float | None = None

    vwap_cross_count: int = 0
    prev_close_side: int = 0  # -1 below vwap, +1 above vwap, 0 unknown/unset

    prev_condition: str | None = None

    # Set when condition flips accepted_* -> inside_value ("broke back into value").
    # Cleared once price re-accepts on the same side, or a new session starts.
    awaiting_return_retest_direction: str | None = None  # trade direction for setup 3/4, or None
    awaiting_return_retest_band: float | None = None  # the value-area band level that was broken
    awaiting_bounce_direction: str | None = None  # trade direction for setup 4, or None
    touched_vwap_this_leg: bool = False


def classify_transition(prev_condition: str | None, curr_condition: str) -> str | None:
    if prev_condition == ACCEPTED_ABOVE and curr_condition == INSIDE_VALUE:
        return BROKE_FROM_ABOVE
    if prev_condition == ACCEPTED_BELOW and curr_condition == INSIDE_VALUE:
        return BROKE_FROM_BELOW
    if curr_condition == ACCEPTED_ABOVE and prev_condition != ACCEPTED_ABOVE:
        return REACCEPTED_ABOVE
    if curr_condition == ACCEPTED_BELOW and prev_condition != ACCEPTED_BELOW:
        return REACCEPTED_BELOW
    return None


def update_session_state(
    state: SessionState, row, transition: str | None, cfg: StrategyConfig, period: object | None = None
) -> SessionState:
    """Fold ``row`` (and the already-computed ``transition``) into ``state``,
    returning the state to use for the *next* candle. Starts a fresh state
    whenever ``period`` (default: the row's own calendar day) changes --
    Cycle 3's swing engine passes a weekly period key here to reset this
    exact same state machine on a WEEK boundary instead of a day boundary,
    reusing all of setup1-4's detectors unmodified."""
    period = period if period is not None else row.timestamp.date()
    if state.day != period:
        state = SessionState(day=period)

    if not pd.isna(row.vwap):
        if row.close > row.vwap:
            side = 1
        elif row.close < row.vwap:
            side = -1
        else:
            side = state.prev_close_side
        if state.prev_close_side != 0 and side != 0 and side != state.prev_close_side:
            state.vwap_cross_count += 1
        state.prev_close_side = side

    if row.condition == ACCEPTED_ABOVE:
        state.had_acceptance_above = True
        state.acceptance_extreme_high = (
            row.high if state.acceptance_extreme_high is None else max(state.acceptance_extreme_high, row.high)
        )
    if row.condition == ACCEPTED_BELOW:
        state.had_acceptance_below = True
        state.acceptance_extreme_low = (
            row.low if state.acceptance_extreme_low is None else min(state.acceptance_extreme_low, row.low)
        )

    if transition == BROKE_FROM_ABOVE:
        state.awaiting_return_retest_direction = SHORT
        state.awaiting_return_retest_band = row.band_upper_1
        state.awaiting_bounce_direction = LONG
        state.touched_vwap_this_leg = False
    elif transition == BROKE_FROM_BELOW:
        state.awaiting_return_retest_direction = LONG
        state.awaiting_return_retest_band = row.band_lower_1
        state.awaiting_bounce_direction = SHORT
        state.touched_vwap_this_leg = False
    elif transition in (REACCEPTED_ABOVE, REACCEPTED_BELOW):
        state.awaiting_return_retest_direction = None
        state.awaiting_return_retest_band = None
        state.awaiting_bounce_direction = None
        state.touched_vwap_this_leg = False

    if state.awaiting_bounce_direction is not None and not pd.isna(row.vwap):
        tol = cfg.setup4_vwap_touch_tolerance_pct
        if state.awaiting_bounce_direction == LONG and row.low <= row.vwap * (1 + tol):
            state.touched_vwap_this_leg = True
        if state.awaiting_bounce_direction == SHORT and row.high >= row.vwap * (1 - tol):
            state.touched_vwap_this_leg = True

    state.prev_condition = row.condition
    return state
