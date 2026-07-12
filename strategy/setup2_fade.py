"""Setup #2 -- Fade Value Area Extremes.

A mean-reversion play, only taken when the session is genuinely balanced
(price has crossed VWAP at least ``setup2_min_vwap_crossings`` times so far --
a one-directional trend day is not a fade day) and the current candle is
still classified ``inside_value``. A tag of a value-area band followed by a
close back inside it is a rejection; fade it toward VWAP.

The extra "close beyond the band" stop-out rule (exit at the next candle's
open without waiting for the wick-based stop) lives in
``backtest.simulator`` -- it needs the *live* band level on each subsequent
candle while the position is held, which only the simulator's candle-by-candle
replay has.
"""

from __future__ import annotations

import pandas as pd

from signals.condition import INSIDE_VALUE
from strategy.base import LONG, SHORT, SessionState, StrategyConfig, TradeProposal, buffered_stop, compute_rr

SETUP_ID = "setup2_fade"


def detect(row, state: SessionState, transition: str | None, cfg: StrategyConfig, symbol: str) -> list[TradeProposal]:
    if row.condition != INSIDE_VALUE:
        return []
    if pd.isna(row.band_upper_1) or pd.isna(row.band_lower_1) or pd.isna(row.vwap):
        return []
    if state.vwap_cross_count < cfg.setup2_min_vwap_crossings:
        return []

    proposals: list[TradeProposal] = []

    if row.high >= row.band_upper_1 and row.close < row.band_upper_1:
        entry = row.close
        stop = buffered_stop(row.high, cfg.stop_buffer_pct, SHORT)
        target = row.vwap
        rr = compute_rr(entry, stop, target, SHORT)
        proposals.append(
            TradeProposal(
                symbol=symbol,
                timestamp=row.timestamp,
                setup_id=SETUP_ID,
                direction=SHORT,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                rr_ratio=rr,
                condition_at_entry=row.condition,
                acceptance_streak_at_entry=row.acceptance_streak,
                notes="fade rejection at band_upper_1, target=vwap",
            )
        )

    if row.low <= row.band_lower_1 and row.close > row.band_lower_1:
        entry = row.close
        stop = buffered_stop(row.low, cfg.stop_buffer_pct, LONG)
        target = row.vwap
        rr = compute_rr(entry, stop, target, LONG)
        proposals.append(
            TradeProposal(
                symbol=symbol,
                timestamp=row.timestamp,
                setup_id=SETUP_ID,
                direction=LONG,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                rr_ratio=rr,
                condition_at_entry=row.condition,
                acceptance_streak_at_entry=row.acceptance_streak,
                notes="fade rejection at band_lower_1, target=vwap",
            )
        )

    return proposals
