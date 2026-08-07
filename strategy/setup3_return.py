"""Setup #3 -- Return to Value.

Once a real acceptance phase (accepted_above/below) has broken back inside
the value area, the market has rejected the extension and the higher-odds
trade is back toward VWAP. Primary entry waits for a retest of the band from
the inside that holds (doesn't re-accept); the aggressive fallback -- off by
default, flagged separately when used -- enters on the break-back-inside
candle's own close instead of waiting.
"""

from __future__ import annotations

import pandas as pd

from signals.condition import ACCEPTED_ABOVE, INSIDE_VALUE
from strategy.base import (
    LONG,
    SETUP3_RETURN_FALLBACK,
    SETUP3_RETURN_PRIMARY,
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

    flip = last_flip_to_inside(session_so_far)
    if flip is None:
        return None
    flip_idx, flip_side = flip
    is_flip_candle_now = flip_idx == len(session_so_far) - 1

    direction = SHORT if flip_side == ACCEPTED_ABOVE else LONG
    band_col = "band_upper_1" if flip_side == ACCEPTED_ABOVE else "band_lower_1"
    band_level = now[band_col]
    if pd.isna(band_level):
        return None

    entry_price = now["close"]
    buffer = buffer_amount(entry_price, config.stop_buffer_pct)

    if is_flip_candle_now:
        if not config.setup3_fallback_allowed:
            return None
        setup_id = SETUP3_RETURN_FALLBACK
        stop_price = band_level + buffer if direction == SHORT else band_level - buffer
        notes = "aggressive fallback: entered on the break-back-inside candle itself"
    else:
        if direction == SHORT:
            touched = now["high"] >= band_level
            rejected = now["close"] < band_level
            showed_strength = now["close"] < now["open"]
        else:
            touched = now["low"] <= band_level
            rejected = now["close"] > band_level
            showed_strength = now["close"] > now["open"]
        if not (touched and rejected and showed_strength):
            return None
        setup_id = SETUP3_RETURN_PRIMARY
        stop_price = now["high"] + buffer if direction == SHORT else now["low"] - buffer
        notes = "retested the band from inside and held, entering toward VWAP"

    target_price = now["vwap"]
    rr_ratio = compute_rr_ratio(direction, entry_price, stop_price, target_price)
    if not passes_rr_filter(rr_ratio, config.min_rr):
        return None

    return TradeProposal(
        symbol=symbol,
        timestamp=now["timestamp"],
        setup_id=setup_id,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        rr_ratio=rr_ratio,
        condition_at_entry=now["condition"],
        acceptance_streak_at_entry=int(now["acceptance_streak"]),
        notes=notes,
    )
