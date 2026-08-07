"""TradeProposal dataclass, StrategyConfig, and helpers shared by all four setups.

Every setup detector is a pure function of a *session-local* slice of the
signals DataFrame that ends at the candle being evaluated -- it must never be
given, and must never reach for, any row beyond "now". That structural
constraint (callers only ever pass ``session_so_far``, never the full future)
is what makes no-lookahead a property of the architecture rather than a
convention each setup has to remember.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from signals.condition import ACCEPTED_ABOVE, ACCEPTED_BELOW, INSIDE_VALUE

LONG = "long"
SHORT = "short"

SETUP1_DISCOVERY = "setup1_discovery"
SETUP2_FADE = "setup2_fade"
SETUP3_RETURN_PRIMARY = "setup3_return_primary"
SETUP3_RETURN_FALLBACK = "setup3_return_fallback"
SETUP4_BOUNCE = "setup4_bounce"


@dataclass
class TradeProposal:
    symbol: str
    timestamp: pd.Timestamp
    setup_id: str
    direction: str  # LONG | SHORT
    entry_price: float
    stop_price: float
    target_price: float
    rr_ratio: float
    condition_at_entry: str
    acceptance_streak_at_entry: int
    notes: str = ""


@dataclass
class StrategyConfig:
    min_rr: float = 1.5
    stop_buffer_pct: float = 0.05
    setup2_vwap_cross_count: int = 2
    wide_band_guard_pct: float = 1.5
    setup3_fallback_allowed: bool = False
    vwap_touch_tolerance_pct: float = 0.05
    no_entry_before: dt.time = dt.time(9, 30)
    no_entry_after: dt.time = dt.time(14, 45)
    force_close_at: dt.time = dt.time(15, 15)


def load_strategy_config(config_path: Path) -> StrategyConfig:
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    s = raw.get("strategy", {})
    return StrategyConfig(
        min_rr=s.get("min_rr", 1.5),
        stop_buffer_pct=s.get("stop_buffer_pct", 0.05),
        setup2_vwap_cross_count=s.get("setup2_vwap_cross_count", 2),
        wide_band_guard_pct=s.get("wide_band_guard_pct", 1.5),
        setup3_fallback_allowed=s.get("setup3_fallback_allowed", False),
        vwap_touch_tolerance_pct=s.get("vwap_touch_tolerance_pct", 0.05),
        no_entry_before=_parse_time(s.get("no_entry_before", "09:30")),
        no_entry_after=_parse_time(s.get("no_entry_after", "14:45")),
        force_close_at=_parse_time(s.get("force_close_at", "15:15")),
    )


def _parse_time(value: str) -> dt.time:
    hour, minute = value.split(":")
    return dt.time(int(hour), int(minute))


def buffer_amount(price: float, buffer_pct: float) -> float:
    """Convert a percentage config value (e.g. 0.05 meaning 0.05%) into a
    rupee amount for the given price."""
    return price * (buffer_pct / 100.0)


def compute_rr_ratio(direction: str, entry_price: float, stop_price: float, target_price: float) -> float:
    """Reward:risk ratio at entry. Returns a non-positive number if the setup
    is malformed (e.g. target on the wrong side of entry), which will always
    fail the R:R filter."""
    risk = entry_price - stop_price if direction == LONG else stop_price - entry_price
    reward = target_price - entry_price if direction == LONG else entry_price - target_price
    if risk <= 0:
        return 0.0
    return reward / risk


def passes_rr_filter(rr_ratio: float, min_rr: float) -> bool:
    return rr_ratio >= min_rr


def last_row(session_so_far: pd.DataFrame) -> pd.Series:
    """The candle being evaluated 'now' -- the last row of the session slice."""
    return session_so_far.iloc[-1]


def entries_allowed_now(now: pd.Series, config: StrategyConfig) -> bool:
    t = now["timestamp"].time()
    return config.no_entry_before <= t <= config.no_entry_after


def vwap_cross_count(session_so_far: pd.DataFrame) -> int:
    """How many times close has crossed from one side of VWAP to the other,
    so far this session (rows with no VWAP yet -- the first candle of the
    session -- are ignored)."""
    valid = session_so_far.dropna(subset=["vwap"])
    sides = (valid["close"] > valid["vwap"]).astype(int) - (valid["close"] < valid["vwap"]).astype(int)
    crossings = 0
    prev_side = 0
    for side in sides:
        if side == 0:
            continue
        if prev_side != 0 and side != prev_side:
            crossings += 1
        prev_side = side
    return crossings


def last_flip_to_inside(session_so_far: pd.DataFrame) -> tuple[int, str] | None:
    """Position (within the session) and prior side of the most recent
    accepted_above/below -> inside_value transition, or None if there hasn't
    been one yet this session. Used by setups 3 and 4, which both trade the
    market's return to value after a real acceptance phase failed to hold."""
    cond = session_so_far["condition"].reset_index(drop=True)
    found = None
    for i in range(1, len(cond)):
        if cond.iloc[i] == INSIDE_VALUE and cond.iloc[i - 1] in (ACCEPTED_ABOVE, ACCEPTED_BELOW):
            found = (i, cond.iloc[i - 1])
    return found


def is_wide_band(now: pd.Series, config: StrategyConfig) -> bool:
    """Post-shock wide-band guard: value area unusually wide relative to VWAP."""
    if pd.isna(now.get("vwap")) or now["vwap"] == 0:
        return False
    width_pct = (now["band_upper_1"] - now["band_lower_1"]) / now["vwap"] * 100.0
    return width_pct > config.wide_band_guard_pct
