"""Runs all four setup detectors per candle and applies the global rules that
apply to every setup regardless of which one fires:

- one open position per symbol at a time (no new proposals while one is open)
- no entries in the first 15 minutes of the session or after 14:45 IST
- the post-shock wide-band guard, which suppresses setups 2 and 4 (the
  mean-reversion fades) when the value area is unusually wide

Force-closing open positions at 15:15 IST is trade *management*, not signal
generation, so it lives in backtest/simulator.py, not here.
"""

from __future__ import annotations

import pandas as pd

from strategy import setup1_discovery, setup2_fade, setup3_return, setup4_bounce
from strategy.base import LONG, StrategyConfig, TradeProposal, entries_allowed_now, is_wide_band, last_row


def generate_proposals(
    session_so_far: pd.DataFrame,
    symbol: str,
    config: StrategyConfig,
    has_open_position: bool,
) -> list[TradeProposal]:
    """Candidate proposals for the candle at the end of ``session_so_far``.

    ``session_so_far`` must be a slice of one symbol's signals DataFrame
    containing only rows from the current trading day, up to and including
    the candle being evaluated -- callers must never pass anything from
    later in the session or from a future session.
    """
    if has_open_position or session_so_far.empty:
        return []

    now = last_row(session_so_far)
    if not entries_allowed_now(now, config):
        return []

    wide_band = is_wide_band(now, config)

    proposals = []
    for detect, suppressed_by_wide_band in (
        (setup1_discovery.detect, False),
        (setup2_fade.detect, True),
        (setup3_return.detect, False),
        (setup4_bounce.detect, True),
    ):
        if suppressed_by_wide_band and wide_band:
            continue
        proposal = detect(session_so_far, symbol, config)
        if proposal is not None:
            proposals.append(proposal)

    return proposals


def setup2_close_beyond_faded_band(direction: str, candle: pd.Series) -> bool:
    """Setup 2's early-exit rule: the fade failed and price closed beyond the
    band it was faded from (a grind-through rather than a rejection). The
    caller (the simulator) exits at the *next* candle's open when this fires,
    since it's detected off this candle's close."""
    if direction == LONG:
        return candle["close"] < candle["band_lower_1"]
    return candle["close"] > candle["band_upper_1"]
