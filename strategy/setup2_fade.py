"""Setup #2 -- Fade Value Area Extremes.

Only in a *balanced* session (price has rotated across VWAP at least
``setup2_vwap_cross_count`` times so far) does a tag of the 1st deviation
band get faded: short the first rejection candle at the upper band, long the
first rejection candle at the lower band, targeting VWAP. If price instead
closes beyond the band being faded (a strong grind through it rather than a
rejection), the caller exits immediately at the next candle's open --
handled by the simulator, not here, since it's a post-entry monitoring rule.
"""

from __future__ import annotations

import pandas as pd

from signals.condition import INSIDE_VALUE
from strategy.base import (
    LONG,
    SETUP2_FADE,
    SHORT,
    StrategyConfig,
    TradeProposal,
    buffer_amount,
    compute_rr_ratio,
    last_row,
    passes_rr_filter,
    vwap_cross_count,
)


def detect(session_so_far: pd.DataFrame, symbol: str, config: StrategyConfig) -> TradeProposal | None:
    now = last_row(session_so_far)

    if now["condition"] != INSIDE_VALUE:
        return None
    if pd.isna(now["vwap"]) or pd.isna(now["band_upper_1"]) or pd.isna(now["band_lower_1"]):
        return None

    if vwap_cross_count(session_so_far) < config.setup2_vwap_cross_count:
        return None

    if now["high"] >= now["band_upper_1"] and now["close"] < now["band_upper_1"]:
        direction = SHORT
        buffer = buffer_amount(now["close"], config.stop_buffer_pct)
        stop_price = now["high"] + buffer
    elif now["low"] <= now["band_lower_1"] and now["close"] > now["band_lower_1"]:
        direction = LONG
        buffer = buffer_amount(now["close"], config.stop_buffer_pct)
        stop_price = now["low"] - buffer
    else:
        return None

    entry_price = now["close"]
    target_price = now["vwap"]

    rr_ratio = compute_rr_ratio(direction, entry_price, stop_price, target_price)
    if not passes_rr_filter(rr_ratio, config.min_rr):
        return None

    return TradeProposal(
        symbol=symbol,
        timestamp=now["timestamp"],
        setup_id=SETUP2_FADE,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        rr_ratio=rr_ratio,
        condition_at_entry=now["condition"],
        acceptance_streak_at_entry=int(now["acceptance_streak"]),
        notes="rejection at value-area band in a rotating session",
    )
