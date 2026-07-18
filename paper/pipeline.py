"""The daily paper-trading step: fill pending orders, manage open
positions, generate new proposals -- for exactly ONE new trading day, given
the full (indicator-augmented) history up to and including that day.

This is a faithful, day-granular re-expression of
``backtest.swing_simulator.run_portfolio``'s single-date loop body (exits
first, then entries, same stop-priority-over-target rule, same gap-through-
stop fill, same alphabetical-tie-break-with-early-cutoff capacity rule) --
not a second implementation of the underlying economics. Position sizing,
gap-adjusted stop fills, and trade-closing (cost/R-multiple) all reuse
``swing_simulator``'s own private helpers verbatim, so a paper trade's
numbers are computed by the exact same code as a backtest trade's.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from backtest import costs_delivery
from backtest.swing_simulator import (
    SwingPortfolioConfig,
    SwingTradeRecord,
    _close_position,
    _gap_adjusted_stop_fill,
    _position_size,
)
from strategy import swing_engine
from strategy.base import LONG, StrategyConfig
from paper.state import PaperState, position_proposal, position_to_dict, proposal_from_dict, proposal_to_dict


def _row_index(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    matches = df.index[df["timestamp"] == ts]
    if len(matches) == 0:
        raise KeyError(f"timestamp {ts} not found in df")
    return int(matches[0])


@dataclass
class DailyStepResult:
    state: PaperState
    trades: list[SwingTradeRecord] = field(default_factory=list)  # newly closed today
    fills: list[dict] = field(default_factory=list)  # newly opened today (for the Telegram message)
    new_pending: list[dict] = field(default_factory=list)  # new signals stashed for tomorrow
    journal_trace: list[dict] = field(default_factory=list)  # today's raw candidates, every funnel stage


def run_daily_step(
    symbol_data: dict[str, pd.DataFrame],
    state: PaperState,
    strategy_cfg: StrategyConfig,
    cost_cfg: costs_delivery.DeliveryCostConfig,
    portfolio_cfg: SwingPortfolioConfig,
    today: dt.date,
    watchlist: list[str],
) -> DailyStepResult:
    symbol_data = {symbol: df.sort_values("timestamp").reset_index(drop=True) for symbol, df in symbol_data.items()}

    open_positions = dict(state.open_positions)
    pending_orders = dict(state.pending_orders)
    trades: list[SwingTradeRecord] = []
    fills: list[dict] = []

    today_rows: dict[str, object] = {}
    for symbol, df in symbol_data.items():
        matches = df.index[df["timestamp"].dt.date == today]
        if len(matches) == 0:
            continue  # this symbol has no candle today (e.g. a symbol-specific trading halt) -- can't act on it
        today_rows[symbol] = df.loc[matches[0]]

    # --- 1. Manage existing open positions against today's completed bar (exits first, matching
    #     swing_simulator's per-date loop -- so a same-day exit can free portfolio capacity for a
    #     same-day new fill, exactly like the batch replay). ---
    for symbol in list(open_positions.keys()):
        if symbol not in today_rows:
            continue
        df = symbol_data[symbol]
        row = today_rows[symbol]
        pos = open_positions[symbol]
        proposal = position_proposal(pos)

        if proposal.direction == LONG:
            hit_stop = row["low"] <= proposal.stop_price
            hit_target = row["high"] >= proposal.target_price
        else:
            hit_stop = row["high"] >= proposal.stop_price
            hit_target = row["low"] <= proposal.target_price

        entry_ts = pd.Timestamp(pos["entry_timestamp"])
        entry_idx = _row_index(df, entry_ts)
        today_idx = _row_index(df, row["timestamp"])
        holding_days_so_far = today_idx - entry_idx

        position_for_close = {
            "proposal": proposal,
            "entry_fill": pos["entry_fill"],
            "quantity": pos["quantity"],
            "entry_index": entry_idx,
            "entry_timestamp": entry_ts,
        }

        trade = None
        if hit_stop:
            gap_fill = _gap_adjusted_stop_fill(proposal.direction, row["open"], proposal.stop_price)
            fill = costs_delivery.apply_exit_slippage(gap_fill, proposal.direction, cost_cfg)
            trade = _close_position(
                position_for_close, today_idx, row["timestamp"], proposal.stop_price, fill, "stop", cost_cfg
            )
        elif hit_target:
            fill = costs_delivery.apply_exit_slippage(proposal.target_price, proposal.direction, cost_cfg)
            trade = _close_position(
                position_for_close, today_idx, row["timestamp"], proposal.target_price, fill, "target", cost_cfg
            )
        elif holding_days_so_far >= portfolio_cfg.time_stop_days:
            fill = costs_delivery.apply_exit_slippage(row["close"], proposal.direction, cost_cfg)
            trade = _close_position(
                position_for_close, today_idx, row["timestamp"], row["close"], fill, "time_stop", cost_cfg
            )

        if trade is not None:
            trades.append(trade)
            state.realized_pnl += trade.net_pnl
            state.trade_count += 1
            del open_positions[symbol]

    # --- 2. Fill pending orders at today's open. Mirrors swing_simulator's entries loop exactly,
    #     including its capacity-cutoff semantic: once max_concurrent_positions is hit, EVERY
    #     remaining alphabetically-later symbol is cut off for today (its pending order, if any,
    #     is dropped -- a fill_map entry is only ever valid for one specific calendar date, there
    #     is no "retry tomorrow"). ---
    for symbol in sorted(watchlist):
        if symbol in open_positions and portfolio_cfg.max_positions_per_symbol <= 1:
            pending_orders.pop(symbol, None)
            continue
        if len(open_positions) >= portfolio_cfg.max_concurrent_positions:
            for later_symbol in sorted(watchlist):
                if later_symbol >= symbol:
                    pending_orders.pop(later_symbol, None)
            break

        candidate_dict = pending_orders.pop(symbol, None)
        if candidate_dict is None:
            continue
        if symbol not in today_rows:
            continue  # no candle today for this symbol -- can't fill; matches "no next trading day" drop

        proposal = proposal_from_dict(candidate_dict)
        row = today_rows[symbol]

        risk_per_share = abs(proposal.entry_price - proposal.stop_price)
        quantity = _position_size(risk_per_share, strategy_cfg)
        if quantity < 1:
            continue

        entry_fill = costs_delivery.apply_entry_slippage(row["open"], proposal.direction, cost_cfg)
        open_positions[symbol] = position_to_dict(proposal, entry_fill, quantity, row["timestamp"])
        fills.append(
            {
                "symbol": symbol,
                "direction": proposal.direction,
                "quantity": quantity,
                "entry_fill_price": entry_fill,
                "stop_price": proposal.stop_price,
                "target_price": proposal.target_price,
                "rr_ratio": proposal.rr_ratio,
            }
        )

    # --- 3. Generate new proposals from today's close -> pending orders for tomorrow. ---
    new_pending: list[dict] = []
    journal_trace: list[dict] = []
    for symbol in watchlist:
        if symbol not in symbol_data:
            continue
        trace = swing_engine.generate_proposal_trace(symbol_data[symbol], symbol, strategy_cfg, cost_cfg)
        today_trace = [t for t in trace if t["timestamp"].date() == today]
        journal_trace.extend(today_trace)

        executed_today = [t for t in today_trace if t["funnel_stage"] == "executed_pending"]
        if executed_today:
            winner = executed_today[0]  # first setup wins, matching swing_simulator's tie-break
            proposal = swing_engine.TradeProposal(
                symbol=winner["symbol"],
                timestamp=winner["timestamp"],
                setup_id=winner["setup_id"],
                direction=winner["direction"],
                entry_price=winner["entry_price"],
                stop_price=winner["stop_price"],
                target_price=winner["target_price"],
                rr_ratio=winner["rr_ratio"],
                condition_at_entry=symbol_data[symbol].loc[
                    symbol_data[symbol]["timestamp"].dt.date == today, "condition"
                ].iloc[0],
                acceptance_streak_at_entry=int(
                    symbol_data[symbol]
                    .loc[symbol_data[symbol]["timestamp"].dt.date == today, "acceptance_streak"]
                    .iloc[0]
                ),
                notes=winner["notes"],
            )
            pending_orders[symbol] = proposal_to_dict(proposal)
            new_pending.append(proposal_to_dict(proposal))

    new_state = PaperState(
        capital=state.capital,
        realized_pnl=state.realized_pnl,
        trade_count=state.trade_count,
        paper_start_date=state.paper_start_date,
        last_processed_date=today,
        open_positions=open_positions,
        pending_orders=pending_orders,
    )

    return DailyStepResult(
        state=new_state, trades=trades, fills=fills, new_pending=new_pending, journal_trace=journal_trace
    )
