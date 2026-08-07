"""Replays candles in strict chronological order, generates and fills trade
proposals via strategy.engine, and tracks P&L including realistic costs.

Correctness property: a decision at candle T only ever sees
``session_so_far = day_df.iloc[:i+1]`` for the current trading day -- built
fresh from a slice that structurally cannot contain anything past "now".
That's what makes no-lookahead a property of the loop below, not a
convention the strategy code has to remember.

Position sizing uses fixed *starting* capital for every trade (non-
compounding) rather than a running/compounding equity figure. Real trading
capital is shared across all symbols, and several can have open positions
concurrently (the "one trade per symbol" rule is per-symbol, not portfolio-
wide), so compounding sizing would require processing every symbol in
lock-step global chronological order to share one running equity number.
Fixed-capital sizing decouples that: every symbol can be simulated
independently, sizing is deterministic and reproducible, and the equity
curve used for drawdown reporting is still a real running total of net P&L
on top of the starting capital -- it just isn't fed back into position size.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.costs import CostConfig, apply_entry_slippage, apply_exit_slippage, compute_trade_costs
from data import store
from signals.condition import compute_condition
from signals.config import SignalsConfig
from signals.vwap import compute_session_vwap
from strategy import engine
from strategy.base import LONG, SETUP2_FADE, StrategyConfig, TradeProposal


@dataclass
class TradeRecord:
    symbol: str
    timestamp: pd.Timestamp
    setup_id: str
    direction: str
    entry_price: float  # theoretical trigger price (pre-slippage)
    stop_price: float
    target_price: float
    rr_ratio: float
    condition_at_entry: str
    acceptance_streak_at_entry: int
    notes: str
    entry_fill_price: float
    quantity: int
    exit_timestamp: pd.Timestamp
    exit_price: float  # theoretical trigger price at exit (pre-slippage)
    exit_fill_price: float
    exit_reason: str  # "stop" | "target" | "setup2_early_exit" | "time_exit"
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    duration_minutes: float


def prepare_symbol_df(
    symbol: str, cache_dir: Path, interval_minutes: int, signals_config: SignalsConfig
) -> pd.DataFrame | None:
    df = store.read_symbol(symbol, interval_minutes, cache_dir)
    if df is None or df.empty:
        return None
    df = compute_session_vwap(df, deviation_bands=signals_config.deviation_bands)
    df = compute_condition(
        df,
        acceptance_candles=signals_config.acceptance_candles,
        value_area_band=signals_config.value_area_band,
    )
    return df


def simulate_symbol(
    symbol: str,
    df: pd.DataFrame,
    strategy_config: StrategyConfig,
    cost_config: CostConfig,
    capital: float,
    risk_pct: float,
) -> list[TradeRecord]:
    """Walk one symbol's candles session by session, in order, generating and
    managing trades."""
    records: list[TradeRecord] = []
    day = df["timestamp"].dt.date

    for _, day_df in df.groupby(day, sort=False):
        day_df = day_df.reset_index(drop=True)
        open_trade: dict | None = None

        for i in range(len(day_df)):
            candle = day_df.iloc[i]

            if open_trade is not None:
                record = _manage_open_trade(open_trade, candle, strategy_config, cost_config)
                if record is not None:
                    records.append(record)
                    open_trade = None

            if open_trade is None:
                session_so_far = day_df.iloc[: i + 1]
                proposals = engine.generate_proposals(
                    session_so_far, symbol, strategy_config, has_open_position=False
                )
                if proposals:
                    open_trade = _open_trade(proposals[0], cost_config, capital, risk_pct)

        # Defensive: anything still open at day end (shouldn't happen given
        # the 15:15 force-close rule) is closed at the session's last close.
        if open_trade is not None:
            last_candle = day_df.iloc[-1]
            records.append(_finalize_trade(open_trade, last_candle, "time_exit", cost_config))

    return records


def _open_trade(proposal: TradeProposal, cost_config: CostConfig, capital: float, risk_pct: float) -> dict | None:
    risk_amount = capital * (risk_pct / 100.0)
    risk_per_share = abs(proposal.entry_price - proposal.stop_price)
    if risk_per_share <= 0:
        return None
    quantity = int(risk_amount // risk_per_share)
    if quantity < 1:
        return None
    entry_fill_price = apply_entry_slippage(proposal.entry_price, proposal.direction, cost_config)
    return {
        "proposal": proposal,
        "quantity": quantity,
        "entry_fill_price": entry_fill_price,
        "pending_exit_at_open": False,
    }


def _manage_open_trade(
    open_trade: dict, candle: pd.Series, strategy_config: StrategyConfig, cost_config: CostConfig
) -> TradeRecord | None:
    proposal: TradeProposal = open_trade["proposal"]
    direction = proposal.direction

    if open_trade["pending_exit_at_open"]:
        return _finalize_trade(
            open_trade, candle, "setup2_early_exit", cost_config, exit_price_override=candle["open"]
        )

    if direction == LONG:
        stop_hit = candle["low"] <= proposal.stop_price
        target_hit = candle["high"] >= proposal.target_price
    else:
        stop_hit = candle["high"] >= proposal.stop_price
        target_hit = candle["low"] <= proposal.target_price

    # Pessimistic intra-candle fill rule: if both stop and target were
    # touched in the same candle's range, assume the stop filled first.
    if stop_hit:
        return _finalize_trade(open_trade, candle, "stop", cost_config, exit_price_override=proposal.stop_price)
    if target_hit:
        return _finalize_trade(open_trade, candle, "target", cost_config, exit_price_override=proposal.target_price)

    if proposal.setup_id == SETUP2_FADE and engine.setup2_close_beyond_faded_band(direction, candle):
        # Detected off this candle's close; exit at the *next* candle's open.
        open_trade["pending_exit_at_open"] = True
        return None

    if candle["timestamp"].time() >= strategy_config.force_close_at:
        return _finalize_trade(open_trade, candle, "time_exit", cost_config, exit_price_override=candle["close"])

    return None


def _finalize_trade(
    open_trade: dict,
    exit_candle: pd.Series,
    exit_reason: str,
    cost_config: CostConfig,
    exit_price_override: float | None = None,
) -> TradeRecord:
    proposal: TradeProposal = open_trade["proposal"]
    quantity = open_trade["quantity"]
    direction = proposal.direction
    entry_fill_price = open_trade["entry_fill_price"]

    exit_price = exit_price_override if exit_price_override is not None else exit_candle["close"]
    is_stop = exit_reason == "stop"
    exit_fill_price = apply_exit_slippage(exit_price, direction, cost_config, is_stop=is_stop)

    sign = 1 if direction == LONG else -1
    gross_pnl = (exit_price - proposal.entry_price) * quantity * sign
    costs = compute_trade_costs(direction, entry_fill_price, exit_fill_price, quantity, cost_config).total
    net_pnl = (exit_fill_price - entry_fill_price) * quantity * sign - costs

    dollar_risk = quantity * abs(proposal.entry_price - proposal.stop_price)
    r_multiple = net_pnl / dollar_risk if dollar_risk > 0 else 0.0

    duration_minutes = (exit_candle["timestamp"] - proposal.timestamp).total_seconds() / 60.0

    return TradeRecord(
        symbol=proposal.symbol,
        timestamp=proposal.timestamp,
        setup_id=proposal.setup_id,
        direction=direction,
        entry_price=proposal.entry_price,
        stop_price=proposal.stop_price,
        target_price=proposal.target_price,
        rr_ratio=proposal.rr_ratio,
        condition_at_entry=proposal.condition_at_entry,
        acceptance_streak_at_entry=proposal.acceptance_streak_at_entry,
        notes=proposal.notes,
        entry_fill_price=entry_fill_price,
        quantity=quantity,
        exit_timestamp=exit_candle["timestamp"],
        exit_price=exit_price,
        exit_fill_price=exit_fill_price,
        exit_reason=exit_reason,
        gross_pnl=gross_pnl,
        costs=costs,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        duration_minutes=duration_minutes,
    )


def run_backtest(
    symbols: list[str],
    cache_dir: Path,
    interval_minutes: int,
    signals_config: SignalsConfig,
    strategy_config: StrategyConfig,
    cost_config: CostConfig,
    capital: float,
    risk_pct: float,
) -> list[TradeRecord]:
    """Simulate every symbol independently and return the combined,
    chronologically-sorted trade records."""
    all_records: list[TradeRecord] = []
    for symbol in symbols:
        df = prepare_symbol_df(symbol, cache_dir, interval_minutes, signals_config)
        if df is None:
            continue
        all_records.extend(
            simulate_symbol(symbol, df, strategy_config, cost_config, capital, risk_pct)
        )
    all_records.sort(key=lambda r: r.exit_timestamp)
    return all_records
