"""Replays a symbol's candles against its ``strategy.engine`` proposals,
applying position sizing, fill/cost mechanics, and the position-management
rules that live above any single setup (one open trade per symbol, forced
square-off, setup 2's early band-break exit, Weekend 4's daily trade cap and
post-stop-out cooldown).

Entries are only considered on a candle where the symbol started the candle
flat -- a stop/target/square-off exit resolved *during* a candle never opens
a fresh position on that same candle. The one exception is setup 2's
band-break rule, which resolves at the *next* candle's open (see below); a
fresh entry signal from that same candle's close is fine since the exit has
already happened before the close is evaluated.

The daily trade cap and stop-out cooldown live here (not in strategy.engine)
because they depend on which proposals actually got *taken* and how they
were *resolved* -- the engine only knows about candidate signals, not fills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from backtest import costs
from strategy import engine, setup2_fade
from strategy.base import LONG, SHORT, StrategyConfig, TradeProposal


@dataclass
class SimulatorConfig:
    capital: float = 100_000.0
    risk_pct: float = 0.005


def load_simulator_config(config_path: Path) -> SimulatorConfig:
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    backtest_raw = raw.get("backtest", {})
    defaults = SimulatorConfig()
    return SimulatorConfig(
        capital=backtest_raw.get("capital", defaults.capital),
        risk_pct=backtest_raw.get("risk_pct", defaults.risk_pct),
    )


@dataclass
class TradeRecord:
    symbol: str
    setup_id: str
    direction: str
    entry_timestamp: pd.Timestamp
    entry_signal_price: float
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
    exit_reason: str
    total_costs: float
    gross_pnl: float
    net_pnl: float
    r_multiple: float
    duration_minutes: float
    notes: str


def _position_size(proposal: TradeProposal, sim_cfg: SimulatorConfig) -> int:
    risk_per_share = abs(proposal.entry_price - proposal.stop_price)
    if risk_per_share <= 0:
        return 0
    return math.floor((sim_cfg.capital * sim_cfg.risk_pct) / risk_per_share)


def _open_position(proposal: TradeProposal, row, sim_cfg: SimulatorConfig, cost_cfg: costs.CostConfig) -> dict | None:
    quantity = _position_size(proposal, sim_cfg)
    if quantity < 1:
        return None

    entry_fill = costs.apply_entry_slippage(proposal.entry_price, proposal.direction, cost_cfg)

    fade_band_level = None
    if proposal.setup_id == setup2_fade.SETUP_ID:
        fade_band_level = row.band_upper_1 if proposal.direction == SHORT else row.band_lower_1

    return {
        "proposal": proposal,
        "entry_fill": entry_fill,
        "quantity": quantity,
        "fade_band_level": fade_band_level,
    }


def _close_position(position: dict, exit_timestamp, exit_signal_price: float, reason: str, cost_cfg: costs.CostConfig) -> TradeRecord:
    proposal: TradeProposal = position["proposal"]
    is_stop = reason == "stop"
    exit_fill = costs.apply_exit_slippage(exit_signal_price, proposal.direction, cost_cfg, is_stop=is_stop)

    quantity = position["quantity"]
    round_trip = costs.round_trip_costs(proposal.direction, position["entry_fill"], exit_fill, quantity, cost_cfg)
    net = costs.net_pnl(proposal.direction, position["entry_fill"], exit_fill, quantity, round_trip)
    gross = (
        (exit_fill - position["entry_fill"]) * quantity
        if proposal.direction == LONG
        else (position["entry_fill"] - exit_fill) * quantity
    )

    risk_per_share = abs(proposal.entry_price - proposal.stop_price)
    r_multiple = (net / quantity) / risk_per_share if risk_per_share > 0 else 0.0

    duration_minutes = (exit_timestamp - proposal.timestamp).total_seconds() / 60.0

    return TradeRecord(
        symbol=proposal.symbol,
        setup_id=proposal.setup_id,
        direction=proposal.direction,
        entry_timestamp=proposal.timestamp,
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
        exit_fill_price=exit_fill,
        exit_reason=reason,
        total_costs=round_trip.total,
        gross_pnl=gross,
        net_pnl=net,
        r_multiple=r_multiple,
        duration_minutes=duration_minutes,
        notes=proposal.notes,
    )


def run_symbol(
    symbol: str,
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    cost_cfg: costs.CostConfig,
    sim_cfg: SimulatorConfig,
) -> list[TradeRecord]:
    """``df`` must already carry the signals columns required by
    ``strategy.engine.generate_proposals``, sorted ascending by timestamp."""
    proposals = engine.generate_proposals(df, symbol, strategy_cfg, cost_cfg)
    proposals_by_ts: dict[pd.Timestamp, list[TradeProposal]] = {}
    for p in proposals:
        proposals_by_ts.setdefault(p.timestamp, []).append(p)

    trades: list[TradeRecord] = []
    position: dict | None = None
    pending_exit_reason: str | None = None
    current_day = None
    trades_today = 0
    cooldown_until: pd.Timestamp | None = None

    rows = list(df.itertuples())
    for row in rows:
        ts = row.timestamp
        flat_for_entry = position is None

        day = ts.date()
        if day != current_day:
            current_day = day
            trades_today = 0
            cooldown_until = None

        if pending_exit_reason is not None and position is not None:
            trade = _close_position(position, ts, row.open, pending_exit_reason, cost_cfg)
            trades.append(trade)
            if trade.exit_reason == "stop":
                cooldown_until = ts + pd.Timedelta(minutes=strategy_cfg.stop_cooldown_minutes)
            position = None
            pending_exit_reason = None
            flat_for_entry = True

        if position is not None:
            flat_for_entry = False

            if ts.time() >= strategy_cfg.square_off_at:
                trades.append(_close_position(position, ts, row.close, "square_off", cost_cfg))
                position = None
            else:
                proposal: TradeProposal = position["proposal"]
                if proposal.setup_id == setup2_fade.SETUP_ID:
                    band_level = position["fade_band_level"]
                    if proposal.direction == SHORT and not pd.isna(row.band_upper_1) and row.close > band_level:
                        pending_exit_reason = "setup2_band_break"
                    elif proposal.direction == LONG and not pd.isna(row.band_lower_1) and row.close < band_level:
                        pending_exit_reason = "setup2_band_break"

                if position is not None and pending_exit_reason is None:
                    if proposal.direction == LONG:
                        hit_stop = row.low <= proposal.stop_price
                        hit_target = row.high >= proposal.target_price
                    else:
                        hit_stop = row.high >= proposal.stop_price
                        hit_target = row.low <= proposal.target_price

                    if hit_stop:
                        trade = _close_position(position, ts, proposal.stop_price, "stop", cost_cfg)
                        trades.append(trade)
                        cooldown_until = ts + pd.Timedelta(minutes=strategy_cfg.stop_cooldown_minutes)
                        position = None
                    elif hit_target:
                        trades.append(_close_position(position, ts, proposal.target_price, "target", cost_cfg))
                        position = None

        can_enter = (
            flat_for_entry
            and position is None
            and trades_today < strategy_cfg.max_trades_per_day
            and (cooldown_until is None or ts >= cooldown_until)
        )
        if can_enter:
            for candidate in proposals_by_ts.get(ts, []):
                position = _open_position(candidate, row, sim_cfg, cost_cfg)
                if position is not None:
                    trades_today += 1
                    break

    if position is not None:
        last = rows[-1]
        trades.append(_close_position(position, last.timestamp, last.close, "end_of_data", cost_cfg))

    return trades


def run_backtest(
    symbol_data: dict[str, pd.DataFrame],
    strategy_cfg: StrategyConfig,
    cost_cfg: costs.CostConfig,
    sim_cfg: SimulatorConfig,
) -> list[TradeRecord]:
    """Runs every symbol independently (each only depends on its own history --
    there's no shared capital pool to contend for) and returns every trade
    across all symbols."""
    trades: list[TradeRecord] = []
    for symbol, df in symbol_data.items():
        trades.extend(run_symbol(symbol, df, strategy_cfg, cost_cfg, sim_cfg))
    return trades
