"""Runs all four setups over a symbol's candles in strict chronological order
and applies the global rules that sit above any single setup.

No-lookahead is the single most important correctness property here: a
decision at candle T may only use data up to and including candle T. This
holds by construction --

- The loop below processes ``df`` with a single forward pass over
  ``itertuples()``, never indexing ahead.
- Each setup's ``detect(row, state, transition, cfg, symbol)`` receives the
  *current* row (fine -- a signal confirmed by candle T's own close is not
  lookahead) plus ``state``/``transition``, both derived only from candles
  strictly before T (see ``strategy.base.SessionState`` and
  ``update_session_state``, which folds row T into the state only *after*
  detection has already run for row T).
- Nothing here reads ``df`` out of order or precomputes anything from the
  tail of the frame.

See ``tests/test_strategy.py::test_no_lookahead`` -- it re-runs this function
on a truncated prefix of the same data and asserts the proposals up to the
truncation point are byte-for-byte identical to the full run.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from strategy import setup1_discovery, setup2_fade, setup3_return, setup4_bounce
from strategy.base import SessionState, StrategyConfig, TradeProposal, classify_transition, update_session_state

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "band_upper_1",
    "band_lower_1",
    "condition",
    "acceptance_streak",
]

_MEAN_REVERSION_SETUP_IDS = (setup2_fade.SETUP_ID, setup4_bounce.SETUP_ID)

_SESSION_OPEN = dt.time(9, 15)


def _entry_window(ts: pd.Timestamp, cfg: StrategyConfig) -> bool:
    t = ts.time()
    open_cutoff = (dt.datetime.combine(dt.date.today(), _SESSION_OPEN) + dt.timedelta(minutes=cfg.session_open_no_entry_minutes)).time()
    return open_cutoff <= t <= cfg.no_entry_after


def _wide_band_guard_active(row, cfg: StrategyConfig) -> bool:
    if pd.isna(row.band_upper_1) or pd.isna(row.band_lower_1) or row.close <= 0:
        return False
    width_pct = (row.band_upper_1 - row.band_lower_1) / row.close
    return width_pct > cfg.wide_band_guard_pct


def generate_proposals(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> list[TradeProposal]:
    """``df`` must already carry vwap/band/condition columns (the output of
    ``signals.vwap.compute_session_vwap`` + ``signals.condition.compute_condition``),
    sorted ascending by timestamp. Returns every proposal that clears the
    global entry-window, wide-band-guard, and R:R filters -- one-trade-per-symbol
    and stop-first fills are the simulator's job, not the engine's."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    proposals: list[TradeProposal] = []
    state = SessionState()

    for row in df.itertuples():
        day = row.timestamp.date()
        if state.day != day:
            state = SessionState(day=day)

        transition = classify_transition(state.prev_condition, row.condition)

        candidates: list[TradeProposal] = []
        if _entry_window(row.timestamp, cfg):
            candidates += setup1_discovery.detect(row, state, transition, cfg, symbol)
            candidates += setup2_fade.detect(row, state, transition, cfg, symbol)
            candidates += setup3_return.detect(row, state, transition, cfg, symbol)
            candidates += setup4_bounce.detect(row, state, transition, cfg, symbol)

        if _wide_band_guard_active(row, cfg):
            candidates = [p for p in candidates if p.setup_id not in _MEAN_REVERSION_SETUP_IDS]

        proposals.extend(p for p in candidates if p.rr_ratio >= cfg.min_rr)

        state = update_session_state(state, row, transition, cfg)

    return proposals
