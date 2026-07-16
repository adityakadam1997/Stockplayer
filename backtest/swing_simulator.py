"""Portfolio-level replay for Cycle 3's daily-bar swing trades.

Materially different from ``backtest.simulator`` (intraday) in three ways
that all trace back to the same fact -- a swing position spans multiple
days and competes for a shared, portfolio-wide capacity:

1. **Next-open execution.** A signal forms on a daily close (a completed
   candle); it is filled at the *next* trading day's open, never at the
   close it was read off of -- that close hasn't "happened" yet from a
   live-trading perspective when the signal fires, so filling at it would be
   lookahead. If a signal is the last row of a symbol's data (no next day
   available), it is simply dropped -- there is nothing to fill against.
2. **Gap-through-stop fills.** Stops are only checked once per day (daily
   OHLC). If a day's open already gapped past the stop, the fill is at that
   open (worse than the stop price) -- the same pessimism-on-fills principle
   as the intraday simulator's stop-first rule, adapted for daily granularity:
   fill = min(open, stop) for a long stop, max(open, stop) for a short stop.
   Targets are never gap-adjusted upward/favorably (a limit-style fill caps
   at the target price even on a favorable gap) -- consistent with the
   intraday simulator never rewarding a favorable gap either.
3. **Portfolio-wide capacity**, not per-symbol independence. Every symbol
   competes for ``max_concurrent_positions`` open slots at once (plus
   ``max_positions_per_symbol``, effectively 1 here) -- this requires walking
   all symbols in one global, date-synchronized loop instead of processing
   each symbol's full history independently the way the intraday simulator
   does (there, no shared constraint existed).

Overnight/weekend gap risk is inherent to any position held across a close --
the stop is only ever checked once per day at daily OHLC resolution, so a
loss can exceed the nominal stop distance if price gaps hard between one
day's close and the next day's open. That risk is real and not eliminated by
the gap-fill rule above (which only prices the fill correctly once a gap is
observed -- it doesn't prevent the gap itself).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from backtest import costs_delivery
from strategy import swing_engine
from strategy.base import LONG, StrategyConfig, TradeProposal


@dataclass
class SwingPortfolioConfig:
    time_stop_days: int = 10
    max_concurrent_positions: int = 5
    max_positions_per_symbol: int = 1


def load_swing_portfolio_config(config_path: Path) -> SwingPortfolioConfig:
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    portfolio_raw = raw.get("swing_portfolio", {})
    defaults = SwingPortfolioConfig()
    return SwingPortfolioConfig(
        time_stop_days=portfolio_raw.get("time_stop_days", defaults.time_stop_days),
        max_concurrent_positions=portfolio_raw.get("max_concurrent_positions", defaults.max_concurrent_positions),
        max_positions_per_symbol=portfolio_raw.get("max_positions_per_symbol", defaults.max_positions_per_symbol),
    )


@dataclass
class SwingTradeRecord:
    symbol: str
    setup_id: str
    direction: str
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp  # next trading day after the signal
    entry_signal_price: float  # signal day's close (reference used to size stop/target)
    entry_fill_price: float
    stop_price: float
    target_price: float
    rr_ratio: float
    condition_at_entry: str
    acceptance_streak_at_entry: int
    quantity: int
    exit_timestamp: pd.Timestamp
    exit_signal_price: float
    exit_fill_price: float
    exit_reason: str  # "stop" | "target" | "time_stop" | "end_of_data"
    holding_days: int
    total_costs: float
    gross_pnl: float
    net_pnl: float
    r_multiple: float
    notes: str


def _position_size(risk_per_share: float, cfg: StrategyConfig) -> int:
    if risk_per_share <= 0:
        return 0
    return math.floor((cfg.capital * cfg.risk_pct) / risk_per_share)


def _gap_adjusted_stop_fill(direction: str, open_price: float, stop_price: float) -> float:
    if direction == LONG:
        return min(open_price, stop_price)
    return max(open_price, stop_price)


def _close_position(
    position: dict,
    exit_index: int,
    exit_timestamp: pd.Timestamp,
    exit_signal_price: float,
    exit_fill_price: float,
    reason: str,
    cost_cfg: costs_delivery.DeliveryCostConfig,
) -> SwingTradeRecord:
    proposal: TradeProposal = position["proposal"]
    quantity = position["quantity"]

    round_trip = costs_delivery.round_trip_costs(
        proposal.direction, position["entry_fill"], exit_fill_price, quantity, cost_cfg
    )
    net = costs_delivery.net_pnl(proposal.direction, position["entry_fill"], exit_fill_price, quantity, round_trip)
    gross = (
        (exit_fill_price - position["entry_fill"]) * quantity
        if proposal.direction == LONG
        else (position["entry_fill"] - exit_fill_price) * quantity
    )

    risk_per_share = abs(proposal.entry_price - proposal.stop_price)
    r_multiple = (net / quantity) / risk_per_share if risk_per_share > 0 else 0.0
    holding_days = exit_index - position["entry_index"]

    return SwingTradeRecord(
        symbol=proposal.symbol,
        setup_id=proposal.setup_id,
        direction=proposal.direction,
        signal_timestamp=proposal.timestamp,
        entry_timestamp=position["entry_timestamp"],
        entry_signal_price=proposal.entry_price,
        entry_fill_price=position["entry_fill"],
        stop_price=proposal.stop_price,
        target_price=proposal.target_price,
        rr_ratio=proposal.rr_ratio,
        condition_at_entry=proposal.condition_at_entry,
        acceptance_streak_at_entry=proposal.acceptance_streak_at_entry,
        quantity=quantity,
        exit_timestamp=exit_timestamp,
        exit_signal_price=exit_signal_price,
        exit_fill_price=exit_fill_price,
        exit_reason=reason,
        holding_days=holding_days,
        total_costs=round_trip.total,
        gross_pnl=gross,
        net_pnl=net,
        r_multiple=r_multiple,
        notes=proposal.notes,
    )


def run_portfolio(
    symbol_data: dict[str, pd.DataFrame],
    strategy_cfg: StrategyConfig,
    cost_cfg: costs_delivery.DeliveryCostConfig,
    portfolio_cfg: SwingPortfolioConfig,
) -> list[SwingTradeRecord]:
    """Replays every symbol's daily bars in one global, date-synchronized
    walk so ``max_concurrent_positions``/``max_positions_per_symbol`` can be
    enforced portfolio-wide. ``symbol_data`` values must already carry the
    columns ``strategy.swing_engine.generate_proposals`` requires."""
    rows_by_symbol: dict[str, list] = {}
    index_by_date: dict[str, dict] = {}
    entries_by_fill_date: dict[str, dict] = {}

    for symbol, df in symbol_data.items():
        df = df.sort_values("timestamp").reset_index(drop=True)
        rows = list(df.itertuples())
        rows_by_symbol[symbol] = rows
        index_by_date[symbol] = {row.timestamp.date(): i for i, row in enumerate(rows)}

        proposals = swing_engine.generate_proposals(df, symbol, strategy_cfg, cost_cfg)
        proposals_by_signal_index: dict[int, TradeProposal] = {}
        for p in proposals:
            idx = index_by_date[symbol].get(p.timestamp.date())
            if idx is not None and idx not in proposals_by_signal_index:
                proposals_by_signal_index[idx] = p  # first setup wins if more than one fires the same day

        fill_map: dict = {}
        for signal_idx, proposal in proposals_by_signal_index.items():
            fill_idx = signal_idx + 1
            if fill_idx < len(rows):  # drop signals with no next trading day to fill against
                fill_map[rows[fill_idx].timestamp.date()] = (proposal, fill_idx)
        entries_by_fill_date[symbol] = fill_map

    all_dates = sorted({row.timestamp.date() for rows in rows_by_symbol.values() for row in rows})

    trades: list[SwingTradeRecord] = []
    open_positions: dict[str, dict] = {}

    for date in all_dates:
        # Exits first.
        for symbol in list(open_positions.keys()):
            idx = index_by_date[symbol].get(date)
            if idx is None:
                continue
            row = rows_by_symbol[symbol][idx]
            position = open_positions[symbol]
            proposal: TradeProposal = position["proposal"]

            if proposal.direction == LONG:
                hit_stop = row.low <= proposal.stop_price
                hit_target = row.high >= proposal.target_price
            else:
                hit_stop = row.high >= proposal.stop_price
                hit_target = row.low <= proposal.target_price

            holding_days_so_far = idx - position["entry_index"]

            if hit_stop:
                gap_fill = _gap_adjusted_stop_fill(proposal.direction, row.open, proposal.stop_price)
                fill = costs_delivery.apply_exit_slippage(gap_fill, proposal.direction, cost_cfg)
                trades.append(_close_position(position, idx, row.timestamp, proposal.stop_price, fill, "stop", cost_cfg))
                del open_positions[symbol]
            elif hit_target:
                fill = costs_delivery.apply_exit_slippage(proposal.target_price, proposal.direction, cost_cfg)
                trades.append(
                    _close_position(position, idx, row.timestamp, proposal.target_price, fill, "target", cost_cfg)
                )
                del open_positions[symbol]
            elif holding_days_so_far >= portfolio_cfg.time_stop_days:
                fill = costs_delivery.apply_exit_slippage(row.close, proposal.direction, cost_cfg)
                trades.append(
                    _close_position(position, idx, row.timestamp, row.close, fill, "time_stop", cost_cfg)
                )
                del open_positions[symbol]

        # Entries, alphabetical by symbol for determinism when capacity is scarce.
        # open_positions is keyed one-per-symbol by construction, so
        # `symbol in open_positions` already enforces exactly 1 concurrent
        # position per symbol -- max_positions_per_symbol is only meaningful
        # (and only ever set) at its spec value of 1; anything else isn't
        # supported by this data structure.
        for symbol in sorted(entries_by_fill_date.keys()):
            if symbol in open_positions and portfolio_cfg.max_positions_per_symbol <= 1:
                continue
            if len(open_positions) >= portfolio_cfg.max_concurrent_positions:
                break
            candidate = entries_by_fill_date[symbol].get(date)
            if candidate is None:
                continue
            proposal, fill_idx = candidate
            row = rows_by_symbol[symbol][fill_idx]

            risk_per_share = abs(proposal.entry_price - proposal.stop_price)
            quantity = _position_size(risk_per_share, strategy_cfg)
            if quantity < 1:
                continue

            entry_fill = costs_delivery.apply_entry_slippage(row.open, proposal.direction, cost_cfg)
            open_positions[symbol] = {
                "proposal": proposal,
                "entry_fill": entry_fill,
                "quantity": quantity,
                "entry_index": fill_idx,
                "entry_timestamp": row.timestamp,
            }

    # End of data: force-close anything still open at the last available price.
    for symbol, position in open_positions.items():
        rows = rows_by_symbol[symbol]
        last = rows[-1]
        idx = len(rows) - 1
        trades.append(_close_position(position, idx, last.timestamp, last.close, last.close, "end_of_data", cost_cfg))

    return trades
