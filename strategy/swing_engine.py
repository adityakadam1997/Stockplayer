"""Cycle 3 / 3B: the identical setup1-4 detectors, pointed at weekly-anchored
bands on daily bars instead of session-anchored bands on intraday bars.

Reuses:
- ``setup1_discovery.detect`` / ``setup2_fade.detect`` / ``setup3_return.detect``
  / ``setup4_bounce.detect`` verbatim -- they only read ``row.vwap`` /
  ``row.band_upper_1`` / ``row.condition`` / ``row.acceptance_streak`` /
  ``state.*``, all of which are anchor-agnostic.
- ``strategy.base.SessionState`` / ``update_session_state`` /
  ``classify_transition``, reset on a WEEKLY period key instead of a calendar
  day (``update_session_state`` gained an optional ``period`` parameter for
  exactly this).
- ``strategy.base.compute_atr`` (period=14 instead of 20).
- The same stop-floor and cost-viability machinery as ``strategy.engine``,
  adapted for the delivery cost model (``backtest.costs_delivery``) instead
  of the intraday one, and for swing's stop-floor formula (2.5% of price /
  1.5x ATR14).

New for swing:
- A trend filter: longs only when close > monthly VWAP, shorts only when
  close < monthly VWAP -- a slower anchor, computed separately, never fed
  into the weekly condition classifier (context only).
- No intraday entry-window concept -- daily bars don't have an "hour of
  day." The wide-band guard (value area > 1.5% of price suppresses setups 2
  & 4) still applies, unchanged formula.

No lookahead: a proposal's ``entry_price`` is the signal day's own close (a
completed daily candle) -- used only to size the stop/target/R:R, exactly
like the intraday engine. It is NOT the actual fill price: Cycle 3's core
adaptation is that fills happen at the *next* trading day's open (see
``backtest.swing_simulator``), never at the close a signal was read off of.

CYCLE 3B REPAIR (Cycle 3 returned zero trades and diagnosis showed why:
setup1's target was ``state.acceptance_extreme_high``, a value tracked
*across* candles -- on 5-min bars the extend->pullback->reclaim sequence
always spans several distinct candles, so by the time a retest fires, the
tracked extreme genuinely reflects an already-observed prior peak. On a
single **daily** candle that whole sequence can happen within one bar, so
the tracked extreme is frequently stale by the time the day's own close has
already moved past it -- 64-67% of Cycle 3's candidates had targets on the
wrong side of entry as a result. Two fixes, both applied at proposal time,
after detection, before the stop floor):
1. Long-only (``_long_only_filter``) -- cash delivery cannot short; the
   detectors are left completely intact, shorts are just discarded here,
   counted in the funnel.
2. Target recomputation (``_recompute_target``) -- replaces whatever target
   the detector computed with one read from CURRENT structure at the signal
   close, never from state accumulated earlier in the sequence. A proposal
   whose recomputed target isn't strictly above entry is INVALID and
   discarded (``invalid_geometry`` in the funnel) -- distinct from failing
   the R:R bar, which only applies to candidates with genuinely positive
   reward.
"""

from __future__ import annotations

import dataclasses
import math

import pandas as pd

from backtest import costs_delivery
from strategy import setup1_discovery, setup2_fade, setup3_return, setup4_bounce
from strategy.base import (
    LONG,
    SessionState,
    StrategyConfig,
    TradeProposal,
    classify_transition,
    compute_rr,
    update_session_state,
)

ATR_PERIOD = 14
ATR_COLUMN = f"atr{ATR_PERIOD}"
WEEK_FREQ = "W-SUN"  # Monday-Sunday weeks; matches signals.vwap.compute_weekly_vwap

ROLLING_HIGH_WINDOW = 20
ROLLING_HIGH_COLUMN = f"high_{ROLLING_HIGH_WINDOW}d_prior"

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "band_upper_1",
    "band_lower_1",
    "band_upper_2",
    "condition",
    "acceptance_streak",
    ATR_COLUMN,
    "monthly_vwap",
    ROLLING_HIGH_COLUMN,
]

_MEAN_REVERSION_SETUP_IDS = (setup2_fade.SETUP_ID, setup4_bounce.SETUP_ID)


def compute_prior_n_day_high(df: pd.DataFrame, window: int = ROLLING_HIGH_WINDOW) -> pd.Series:
    """Rolling max of ``high`` over the ``window`` bars strictly BEFORE the
    current row (``shift(1)`` excludes today) -- no lookahead by
    construction. Fewer than ``window`` prior bars available (start of data)
    just uses whatever exists (``min_periods=1``)."""
    return df["high"].shift(1).rolling(window=window, min_periods=1).max()


def _wide_band_guard_active(row, cfg: StrategyConfig) -> bool:
    if pd.isna(row.band_upper_1) or pd.isna(row.band_lower_1) or row.close <= 0:
        return False
    width_pct = (row.band_upper_1 - row.band_lower_1) / row.close
    return width_pct > cfg.wide_band_guard_pct


def _trend_filter_ok(proposal: TradeProposal, row) -> bool:
    """Longs only when close > monthly VWAP, shorts only when close < monthly
    VWAP -- swing trades fighting a multi-week flow otherwise. Rejects if the
    monthly anchor hasn't formed yet (first month of data)."""
    if pd.isna(row.monthly_vwap):
        return False
    if proposal.direction == LONG:
        return row.close > row.monthly_vwap
    return row.close < row.monthly_vwap


def _long_only_filter(candidates: list[TradeProposal]) -> list[TradeProposal]:
    """Cash delivery cannot short. Detector code stays intact; shorts are
    simply discarded here (counted separately in the funnel as
    ``suppressed_shorts``)."""
    return [p for p in candidates if p.direction == LONG]


def _recompute_target(proposal: TradeProposal, row) -> TradeProposal | None:
    """Cycle 3B: replace the detector's target with one read from CURRENT
    structure at the signal close, never from state tracked earlier in the
    sequence. Returns ``None`` (invalid geometry) if the recomputed target
    isn't strictly above entry -- long-only, so reward must be positive."""
    entry = proposal.entry_price

    if proposal.setup_id == setup1_discovery.SETUP_ID:
        prior_high = row.__getattribute__(ROLLING_HIGH_COLUMN)
        if not pd.isna(prior_high) and prior_high > entry:
            new_target = prior_high
        elif not pd.isna(row.band_upper_2) and row.band_upper_2 > entry:
            new_target = row.band_upper_2
        else:
            return None
    elif proposal.setup_id in (setup2_fade.SETUP_ID, setup3_return.SETUP_ID):
        if pd.isna(row.vwap) or row.vwap <= entry:
            return None
        new_target = row.vwap
    elif proposal.setup_id == setup4_bounce.SETUP_ID:
        if pd.isna(row.band_upper_1) or row.band_upper_1 <= entry:
            return None
        new_target = row.band_upper_1
    else:
        return None

    new_rr = compute_rr(entry, proposal.stop_price, new_target, proposal.direction)
    return dataclasses.replace(
        proposal, target_price=new_target, rr_ratio=new_rr, notes=f"{proposal.notes} | target_recomputed_at_signal"
    )


def _apply_stop_floor(proposal: TradeProposal, atr_value: float, cfg: StrategyConfig) -> TradeProposal:
    """Same mechanism as ``strategy.engine._apply_stop_floor``: widen (never
    tighten) the stop to at least max(stop_floor_pct * entry_price, atr_mult
    * ATR14), recompute rr_ratio against the unchanged target."""
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


def _cost_viable(proposal: TradeProposal, cfg: StrategyConfig, cost_cfg: costs_delivery.DeliveryCostConfig) -> bool:
    """Same idea as ``strategy.engine._cost_viable``, priced with the
    delivery cost model instead of the intraday one."""
    risk_per_share = abs(proposal.entry_price - proposal.stop_price)
    quantity = _position_size(risk_per_share, cfg)
    if quantity < 1:
        return False

    nominal_risk = risk_per_share * quantity
    entry_slippage = proposal.entry_price * cost_cfg.slippage_pct * quantity
    exit_slippage = proposal.target_price * cost_cfg.slippage_pct * quantity
    fees = costs_delivery.round_trip_costs(
        proposal.direction, proposal.entry_price, proposal.target_price, quantity, cost_cfg
    ).total
    estimated_cost = fees + entry_slippage + exit_slippage

    return estimated_cost <= cfg.cost_viability_max_pct * nominal_risk


def _trace_row(proposal: TradeProposal, stage: str) -> dict:
    """One diagnostic record for ``generate_proposal_trace`` (Phase 1's paper
    journal) -- the funnel stage a single raw candidate reached, whatever its
    fields looked like at that point (pre- or post-recomputation depending on
    how far it got)."""
    return {
        "symbol": proposal.symbol,
        "timestamp": proposal.timestamp,
        "setup_id": proposal.setup_id,
        "direction": proposal.direction,
        "entry_price": proposal.entry_price,
        "stop_price": proposal.stop_price,
        "target_price": proposal.target_price,
        "rr_ratio": proposal.rr_ratio,
        "funnel_stage": stage,
        "notes": proposal.notes,
    }


def _run_pipeline(
    df: pd.DataFrame, symbol: str, cfg: StrategyConfig, cost_cfg: costs_delivery.DeliveryCostConfig
) -> tuple[list[TradeProposal], dict[str, int], list[float], list[dict]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    proposals: list[TradeProposal] = []
    valid_geometry_rr: list[float] = []
    trace: list[dict] = []
    funnel = {
        "raw": 0,
        "after_trend_filter": 0,
        "suppressed_shorts": 0,
        "after_long_only": 0,
        "invalid_geometry": 0,
        "after_valid_geometry": 0,
        "after_rr": 0,
        "after_cost_viability": 0,
    }
    state = SessionState()

    week_periods = df["timestamp"].dt.tz_localize(None).dt.to_period(WEEK_FREQ)

    for row, week_period in zip(df.itertuples(), week_periods):
        if state.day != week_period:
            state = SessionState(day=week_period)

        transition = classify_transition(state.prev_condition, row.condition)

        candidates: list[TradeProposal] = []
        candidates += setup1_discovery.detect(row, state, transition, cfg, symbol)
        candidates += setup2_fade.detect(row, state, transition, cfg, symbol)
        candidates += setup3_return.detect(row, state, transition, cfg, symbol)
        candidates += setup4_bounce.detect(row, state, transition, cfg, symbol)

        if _wide_band_guard_active(row, cfg):
            for p in candidates:
                if p.setup_id in _MEAN_REVERSION_SETUP_IDS:
                    trace.append(_trace_row(p, "wide_band_guard"))
            candidates = [p for p in candidates if p.setup_id not in _MEAN_REVERSION_SETUP_IDS]
        funnel["raw"] += len(candidates)

        survivors: list[TradeProposal] = []
        for p in candidates:
            if _trend_filter_ok(p, row):
                survivors.append(p)
            else:
                trace.append(_trace_row(p, "failed_trend_filter"))
        candidates = survivors
        funnel["after_trend_filter"] += len(candidates)

        survivors = []
        for p in candidates:
            if p.direction == LONG:
                survivors.append(p)
            else:
                trace.append(_trace_row(p, "suppressed_short"))
        funnel["suppressed_shorts"] += len(candidates) - len(survivors)
        candidates = survivors
        funnel["after_long_only"] += len(candidates)

        recomputed: list[TradeProposal] = []
        for p in candidates:
            new_p = _recompute_target(p, row)
            if new_p is None:
                funnel["invalid_geometry"] += 1
                trace.append(_trace_row(p, "invalid_geometry"))
            else:
                recomputed.append(new_p)
        candidates = recomputed
        funnel["after_valid_geometry"] += len(candidates)

        atr_value = getattr(row, ATR_COLUMN)
        candidates = [_apply_stop_floor(p, atr_value, cfg) for p in candidates]
        valid_geometry_rr.extend(p.rr_ratio for p in candidates)

        survivors = []
        for p in candidates:
            if p.rr_ratio >= cfg.min_rr:
                survivors.append(p)
            else:
                trace.append(_trace_row(p, "failed_rr"))
        candidates = survivors
        funnel["after_rr"] += len(candidates)

        survivors = []
        for p in candidates:
            if _cost_viable(p, cfg, cost_cfg):
                survivors.append(p)
            else:
                trace.append(_trace_row(p, "failed_cost_viability"))
        candidates = survivors
        funnel["after_cost_viability"] += len(candidates)

        for p in candidates:
            trace.append(_trace_row(p, "executed_pending"))

        proposals.extend(candidates)

        state = update_session_state(state, row, transition, cfg, period=week_period)

    return proposals, funnel, valid_geometry_rr, trace


def generate_proposals(
    df: pd.DataFrame, symbol: str, cfg: StrategyConfig, cost_cfg: costs_delivery.DeliveryCostConfig
) -> list[TradeProposal]:
    """``df`` must carry weekly-anchored vwap/band/condition columns (from
    ``signals.vwap.compute_weekly_vwap`` + ``signals.condition.compute_condition_periodic``),
    an ``atr14`` column (``strategy.base.compute_atr(df, period=14)``), and a
    ``monthly_vwap`` column (``signals.vwap.compute_monthly_vwap``), sorted
    ascending by timestamp."""
    proposals, _, _, _ = _run_pipeline(df, symbol, cfg, cost_cfg)
    return proposals


def generate_proposals_with_funnel(
    df: pd.DataFrame, symbol: str, cfg: StrategyConfig, cost_cfg: costs_delivery.DeliveryCostConfig
) -> tuple[list[TradeProposal], dict[str, int]]:
    """Same as ``generate_proposals``, plus a funnel dict counting candidates
    surviving each filter stage -- diagnostic only. Funnel stage order: raw ->
    after_trend_filter -> (suppressed_shorts /) after_long_only ->
    (invalid_geometry /) after_valid_geometry -> after_rr ->
    after_cost_viability. Proposals returned are what survived all stages
    (pre next-day-fill / portfolio-capacity trimming, which happens in
    ``backtest.swing_simulator``)."""
    proposals, funnel, _, _ = _run_pipeline(df, symbol, cfg, cost_cfg)
    return proposals, funnel


def generate_proposals_with_diagnostics(
    df: pd.DataFrame, symbol: str, cfg: StrategyConfig, cost_cfg: costs_delivery.DeliveryCostConfig
) -> tuple[list[TradeProposal], dict[str, int], list[float]]:
    """Same as ``generate_proposals_with_funnel``, plus the list of rr_ratio
    values for every candidate that reached the valid-geometry stage
    (post stop-floor, pre R:R-filter) -- feeds the R:R distribution
    diagnostic."""
    proposals, funnel, valid_geometry_rr, _ = _run_pipeline(df, symbol, cfg, cost_cfg)
    return proposals, funnel, valid_geometry_rr


def generate_proposal_trace(
    df: pd.DataFrame, symbol: str, cfg: StrategyConfig, cost_cfg: costs_delivery.DeliveryCostConfig
) -> list[dict]:
    """Phase 1 (paper trading): one diagnostic record per RAW candidate (from
    any of the 4 setups, before the wide-band guard) showing exactly which
    funnel stage it reached -- ``wide_band_guard`` / ``failed_trend_filter``
    / ``suppressed_short`` / ``invalid_geometry`` / ``failed_rr`` /
    ``failed_cost_viability`` / ``executed_pending``. This is strictly a
    read-out of ``_run_pipeline``'s existing decisions (same function,
    same call, zero behavior change) -- it does not alter what
    ``generate_proposals`` returns."""
    _, _, _, trace = _run_pipeline(df, symbol, cfg, cost_cfg)
    return trace
