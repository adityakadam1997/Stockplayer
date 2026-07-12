"""Setup #1 -- Price Discovery Continuation.

After the market has genuinely accepted a new price level (persistent close
outside the value area, or an extreme-band touch -- see
``signals.condition``), price often pulls back to retest the value-area band
it just broke before continuing. This setup enters on the first candle that
shows strength off that retest: it touches the band (a wick back into the
value area) but closes back on the breakout side.

The retest can only be recognized on a candle *after* acceptance was already
established -- ``state.had_acceptance_above``/``had_acceptance_below`` are
folded from prior candles only, so the breakout candle itself (which is what
causes acceptance in the first place) can never be mistaken for its own
retest.
"""

from __future__ import annotations

import pandas as pd

from strategy.base import LONG, SHORT, SessionState, StrategyConfig, TradeProposal, buffered_stop, compute_rr

SETUP_ID = "setup1_discovery"


def detect(row, state: SessionState, transition: str | None, cfg: StrategyConfig, symbol: str) -> list[TradeProposal]:
    if pd.isna(row.band_upper_1) or pd.isna(row.band_lower_1):
        return []

    proposals: list[TradeProposal] = []

    if state.had_acceptance_above and state.acceptance_extreme_high is not None:
        if row.low <= row.band_upper_1 and row.close > row.band_upper_1:
            entry = row.close
            stop = buffered_stop(row.low, cfg.stop_buffer_pct, LONG)
            target = state.acceptance_extreme_high
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
                    notes="retest of band_upper_1 after accepted_above, target=session extreme high",
                )
            )

    if state.had_acceptance_below and state.acceptance_extreme_low is not None:
        if row.high >= row.band_lower_1 and row.close < row.band_lower_1:
            entry = row.close
            stop = buffered_stop(row.high, cfg.stop_buffer_pct, SHORT)
            target = state.acceptance_extreme_low
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
                    notes="retest of band_lower_1 after accepted_below, target=session extreme low",
                )
            )

    return proposals
