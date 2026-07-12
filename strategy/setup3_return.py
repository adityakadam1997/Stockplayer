"""Setup #3 -- Return to Value.

Once a session has genuinely accepted a level and then breaks back inside the
value area, the move tends to continue toward VWAP. Two entry modes:

- Primary: wait for a pullback that retests the broken band *from the inside*
  and holds (closes back away from it) -- enter toward VWAP on that strength.
- Aggressive fallback (only when ``setup3_allow_fallback`` is set): enter
  directly on the decisive break-back-inside candle's own close, no retest
  wait. These are flagged via ``notes`` so they can be broken out separately
  in the backtest report -- they're a materially different risk profile from
  the primary entries.

``state.awaiting_return_retest_direction``/``_band`` are set by
``strategy.base.update_session_state`` the candle *after* the break (folded
from the transition detected on the break candle), so the primary retest scan
can never fire on the break candle itself.
"""

from __future__ import annotations

import pandas as pd

from strategy.base import (
    BROKE_FROM_ABOVE,
    BROKE_FROM_BELOW,
    LONG,
    SHORT,
    SessionState,
    StrategyConfig,
    TradeProposal,
    buffered_stop,
    compute_rr,
)

SETUP_ID = "setup3_return"


def _fallback_proposal(row, transition: str, cfg: StrategyConfig, symbol: str) -> TradeProposal | None:
    if pd.isna(row.vwap):
        return None
    if transition == BROKE_FROM_ABOVE:
        direction = SHORT
        band_level = row.band_upper_1
    else:
        direction = LONG
        band_level = row.band_lower_1
    if pd.isna(band_level):
        return None

    entry = row.close
    stop = buffered_stop(band_level, cfg.stop_buffer_pct, direction)
    target = row.vwap
    rr = compute_rr(entry, stop, target, direction)
    return TradeProposal(
        symbol=symbol,
        timestamp=row.timestamp,
        setup_id=SETUP_ID,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        rr_ratio=rr,
        condition_at_entry=row.condition,
        acceptance_streak_at_entry=row.acceptance_streak,
        notes="fallback: entered on the break-back-inside candle itself",
    )


def _primary_proposal(row, state: SessionState, cfg: StrategyConfig, symbol: str) -> TradeProposal | None:
    direction = state.awaiting_return_retest_direction
    band_level = state.awaiting_return_retest_band
    if direction is None or band_level is None or pd.isna(row.vwap):
        return None

    if direction == SHORT:
        if not (row.high >= band_level and row.close < band_level):
            return None
        entry = row.close
        stop = buffered_stop(row.high, cfg.stop_buffer_pct, SHORT)
    else:
        if not (row.low <= band_level and row.close > band_level):
            return None
        entry = row.close
        stop = buffered_stop(row.low, cfg.stop_buffer_pct, LONG)

    target = row.vwap
    rr = compute_rr(entry, stop, target, direction)
    return TradeProposal(
        symbol=symbol,
        timestamp=row.timestamp,
        setup_id=SETUP_ID,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        rr_ratio=rr,
        condition_at_entry=row.condition,
        acceptance_streak_at_entry=row.acceptance_streak,
        notes="primary: retest of the broken band from inside, holds, toward vwap",
    )


def detect(row, state: SessionState, transition: str | None, cfg: StrategyConfig, symbol: str) -> list[TradeProposal]:
    proposals: list[TradeProposal] = []

    if cfg.setup3_allow_fallback and transition in (BROKE_FROM_ABOVE, BROKE_FROM_BELOW):
        fallback = _fallback_proposal(row, transition, cfg, symbol)
        if fallback is not None:
            proposals.append(fallback)

    primary = _primary_proposal(row, state, cfg, symbol)
    if primary is not None:
        proposals.append(primary)

    return proposals
