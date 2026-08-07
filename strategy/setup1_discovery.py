"""Setup #1 -- Price Discovery Continuation.

After the market has *accepted* a breakout above/below the 1st deviation band
(per signals.condition's persistence rule), wait for the first pullback that
tests the band and shows strength back in the breakout direction. Enter
there; stop just past that pullback candle's wick; target the extreme
reached during the acceptance phase (the measured move) -- skip if that
doesn't clear the R:R filter.
"""

from __future__ import annotations

import pandas as pd

from signals.condition import ACCEPTED_ABOVE, ACCEPTED_BELOW
from strategy.base import (
    LONG,
    SETUP1_DISCOVERY,
    SHORT,
    StrategyConfig,
    TradeProposal,
    buffer_amount,
    compute_rr_ratio,
    last_row,
    passes_rr_filter,
)


def detect(session_so_far: pd.DataFrame, symbol: str, config: StrategyConfig) -> TradeProposal | None:
    now = last_row(session_so_far)

    if now["condition"] == ACCEPTED_ABOVE:
        direction = LONG
    elif now["condition"] == ACCEPTED_BELOW:
        direction = SHORT
    else:
        return None

    band_col = "band_upper_1" if direction == LONG else "band_lower_1"
    band_level = now[band_col]
    if pd.isna(band_level):
        return None

    # The pullback candle must have touched the band and closed back beyond
    # it, in the breakout direction -- a single wick that reverts is not
    # "the first sign of strength".
    if direction == LONG:
        touched = now["low"] <= band_level
        closed_back = now["close"] > band_level
        showed_strength = now["close"] > now["open"]
    else:
        touched = now["high"] >= band_level
        closed_back = now["close"] < band_level
        showed_strength = now["close"] < now["open"]

    if not (touched and closed_back and showed_strength):
        return None

    # The measured-move target: the extreme reached during the acceptance
    # phase, strictly *before* this pullback candle.
    prior = session_so_far.iloc[:-1]
    phase = prior[prior["condition"] == now["condition"]]
    if phase.empty:
        return None
    target_price = phase["high"].max() if direction == LONG else phase["low"].min()

    entry_price = now["close"]
    if (direction == LONG and target_price <= entry_price) or (
        direction == SHORT and target_price >= entry_price
    ):
        return None

    buffer = buffer_amount(entry_price, config.stop_buffer_pct)
    stop_price = now["low"] - buffer if direction == LONG else now["high"] + buffer

    rr_ratio = compute_rr_ratio(direction, entry_price, stop_price, target_price)
    if not passes_rr_filter(rr_ratio, config.min_rr):
        return None

    return TradeProposal(
        symbol=symbol,
        timestamp=now["timestamp"],
        setup_id=SETUP1_DISCOVERY,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        rr_ratio=rr_ratio,
        condition_at_entry=now["condition"],
        acceptance_streak_at_entry=int(now["acceptance_streak"]),
        notes="pullback tested the band it broke, closed back in breakout direction",
    )
