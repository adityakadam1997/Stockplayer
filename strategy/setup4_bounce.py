"""Setup #4 -- VWAP Bounce.

After a break back into value (setup 3's precondition), price sometimes runs
all the way to VWAP itself and bounces -- big players often defend VWAP with
size, but it is also the zone most prone to traps. This setup only fires
after price has *already* touched VWAP within tolerance on a prior candle
(``state.touched_vwap_this_leg``, latched by ``update_session_state``), and
then a later candle closes away from VWAP in the bounce direction, showing
the reversal actually has strength behind it.

Whether this setup survives real NSE data (as opposed to just looking good on
a chart) is exactly what this weekend's backtest report is for.
"""

from __future__ import annotations

import pandas as pd

from strategy.base import LONG, SHORT, SessionState, StrategyConfig, TradeProposal, buffered_stop, compute_rr

SETUP_ID = "setup4_bounce"


def detect(row, state: SessionState, transition: str | None, cfg: StrategyConfig, symbol: str) -> list[TradeProposal]:
    direction = state.awaiting_bounce_direction
    if direction is None or not state.touched_vwap_this_leg:
        return []
    if pd.isna(row.vwap) or pd.isna(row.band_upper_1) or pd.isna(row.band_lower_1):
        return []

    tol = cfg.setup4_vwap_touch_tolerance_pct

    if direction == LONG:
        if not (row.close > row.vwap * (1 + tol)):
            return []
        entry = row.close
        stop = buffered_stop(row.low, cfg.stop_buffer_pct, LONG)
        target = row.band_upper_1
    else:
        if not (row.close < row.vwap * (1 - tol)):
            return []
        entry = row.close
        stop = buffered_stop(row.high, cfg.stop_buffer_pct, SHORT)
        target = row.band_lower_1

    rr = compute_rr(entry, stop, target, direction)
    return [
        TradeProposal(
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
            notes="bounce off vwap after touch, target=1st deviation band",
        )
    ]
