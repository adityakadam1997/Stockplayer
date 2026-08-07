"""Setup #4 -- VWAP Bounce.

After a real acceptance phase has broken back into value (same precondition
as setup 3), price sometimes carries all the way through to test VWAP
itself. If it shows strength moving away from VWAP after that touch, trade
the bounce toward the 1st deviation band on that side.

This zone is trap-prone -- big players operate right at VWAP, and what looks
like a bounce can just as easily be absorption before a break-through. The
backtest is what tells us whether this survives on NSE stocks; don't assume
it does.
"""

from __future__ import annotations

import pandas as pd

from signals.condition import INSIDE_VALUE
from strategy.base import (
    LONG,
    SETUP4_BOUNCE,
    SHORT,
    StrategyConfig,
    TradeProposal,
    buffer_amount,
    compute_rr_ratio,
    last_flip_to_inside,
    last_row,
    passes_rr_filter,
)


def detect(session_so_far: pd.DataFrame, symbol: str, config: StrategyConfig) -> TradeProposal | None:
    now = last_row(session_so_far)
    if now["condition"] != INSIDE_VALUE:
        return None
    if pd.isna(now["vwap"]) or pd.isna(now["band_upper_1"]) or pd.isna(now["band_lower_1"]):
        return None

    # Precondition: a real acceptance phase broke back into value earlier
    # this session (same underlying context as setup 3), and price has now
    # carried through to actually test VWAP.
    if last_flip_to_inside(session_so_far) is None:
        return None

    tolerance = now["vwap"] * (config.vwap_touch_tolerance_pct / 100.0)
    touched_vwap = now["low"] <= now["vwap"] + tolerance and now["high"] >= now["vwap"] - tolerance
    if not touched_vwap:
        return None

    if now["close"] > now["vwap"] + tolerance:
        direction = LONG
        band_col = "band_upper_1"
    elif now["close"] < now["vwap"] - tolerance:
        direction = SHORT
        band_col = "band_lower_1"
    else:
        return None  # didn't clearly close away from VWAP

    entry_price = now["close"]
    buffer = buffer_amount(entry_price, config.stop_buffer_pct)
    stop_price = now["low"] - buffer if direction == LONG else now["high"] + buffer
    target_price = now[band_col]

    rr_ratio = compute_rr_ratio(direction, entry_price, stop_price, target_price)
    if not passes_rr_filter(rr_ratio, config.min_rr):
        return None

    return TradeProposal(
        symbol=symbol,
        timestamp=now["timestamp"],
        setup_id=SETUP4_BOUNCE,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        rr_ratio=rr_ratio,
        condition_at_entry=now["condition"],
        acceptance_streak_at_entry=int(now["acceptance_streak"]),
        notes="bounce off VWAP after breaking back into value",
    )
