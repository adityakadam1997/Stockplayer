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

import dataclasses
import datetime as dt
import math

import pandas as pd

from backtest import costs
from strategy import setup1_discovery, setup2_fade, setup3_return, setup4_bounce
from strategy.base import (
    ATR_COLUMN,
    LONG,
    SessionState,
    StrategyConfig,
    TradeProposal,
    classify_transition,
    compute_rr,
    update_session_state,
)

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
    ATR_COLUMN,
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


def _apply_stop_floor(proposal: TradeProposal, atr_value: float, cfg: StrategyConfig) -> TradeProposal:
    """Widen (never tighten) the stop to at least
    max(stop_floor_pct * entry_price, atr_mult * ATR20), and recompute
    rr_ratio against the (unchanged) target. The structural stop from the
    setup wins whenever it's already wider than the floor."""
    structural_distance = abs(proposal.entry_price - proposal.stop_price)
    atr_component = cfg.atr_mult * atr_value if not pd.isna(atr_value) else 0.0
    floor_distance = max(cfg.stop_floor_pct * proposal.entry_price, atr_component)
    final_distance = max(structural_distance, floor_distance)

    if final_distance <= structural_distance:
        return proposal

    new_stop = (
        proposal.entry_price - final_distance if proposal.direction == LONG else proposal.entry_price + final_distance
    )
    new_rr = compute_rr(proposal.entry_price, new_stop, proposal.target_price, proposal.direction)
    return dataclasses.replace(
        proposal, stop_price=new_stop, rr_ratio=new_rr, notes=f"{proposal.notes} | stop_floor_applied"
    )


def _position_size(risk_per_share: float, cfg: StrategyConfig) -> int:
    if risk_per_share <= 0:
        return 0
    return math.floor((cfg.capital * cfg.risk_pct) / risk_per_share)


def _cost_viable(proposal: TradeProposal, cfg: StrategyConfig, cost_cfg: costs.CostConfig) -> bool:
    """Estimate the full round-trip cost (fees + slippage on both legs) at the
    position size the (already-floored) stop implies, and reject the
    proposal if that cost eats more than ``cost_viability_max_pct`` of the
    nominal rupee risk. Uses raw entry/target prices for the estimate (not
    slippage-adjusted fills) -- it's a pre-trade viability check, not a
    prediction of the exact fill."""
    risk_per_share = abs(proposal.entry_price - proposal.stop_price)
    quantity = _position_size(risk_per_share, cfg)
    if quantity < 1:
        return False

    nominal_risk = risk_per_share * quantity
    entry_slippage = costs.slippage_amount(proposal.entry_price, cost_cfg) * quantity
    exit_slippage = costs.slippage_amount(proposal.target_price, cost_cfg) * quantity
    fees = costs.round_trip_costs(
        proposal.direction, proposal.entry_price, proposal.target_price, quantity, cost_cfg
    ).total
    estimated_cost = fees + entry_slippage + exit_slippage

    return estimated_cost <= cfg.cost_viability_max_pct * nominal_risk


def generate_proposals(
    df: pd.DataFrame, symbol: str, cfg: StrategyConfig, cost_cfg: costs.CostConfig
) -> list[TradeProposal]:
    """``df`` must already carry vwap/band/condition/atr20 columns (the output
    of ``signals.vwap.compute_session_vwap`` + ``signals.condition.compute_condition``
    + ``strategy.base.compute_atr``), sorted ascending by timestamp. Returns
    every proposal that clears the global entry-window, wide-band-guard, stop
    floor + R:R, and cost-viability filters -- one-trade-per-symbol and
    stop-first fills are the simulator's job, not the engine's."""
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

        atr_value = getattr(row, ATR_COLUMN)
        candidates = [_apply_stop_floor(p, atr_value, cfg) for p in candidates]
        candidates = [p for p in candidates if p.rr_ratio >= cfg.min_rr]
        candidates = [p for p in candidates if _cost_viable(p, cfg, cost_cfg)]

        proposals.extend(candidates)

        state = update_session_state(state, row, transition, cfg)

    return proposals
