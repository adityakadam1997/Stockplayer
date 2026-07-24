"""``paper/state.json`` -- the single source of truth for what the paper
account currently holds: open positions, pending (unfilled) orders, and
running equity. Loaded fresh at the start of every daily run, saved at the
end. All timestamps are stored as ISO-8601 strings (round-tripped through
``pd.Timestamp``); dates as ``YYYY-MM-DD``.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from strategy.base import TradeProposal


@dataclass
class PaperState:
    capital: float = 300_000.0
    realized_pnl: float = 0.0
    trade_count: int = 0
    paper_start_date: dt.date | None = None
    last_processed_date: dt.date | None = None
    # The last calendar date (Asia/Kolkata) a weekly summary was actually
    # sent -- independent of last_processed_date, since the summary decision
    # runs every invocation regardless of whether that day had a new trading
    # candle to process. Drives both the "already sent today" idempotency
    # guard and the >7-day catch-up check in scripts/paper_daily.py.
    last_weekly_summary_date: dt.date | None = None
    # symbol -> open-position dict (see `position_to_dict`/`position_from_dict`)
    open_positions: dict = field(default_factory=dict)
    # symbol -> pending-order dict (a TradeProposal, serialized)
    pending_orders: dict = field(default_factory=dict)

    @property
    def equity(self) -> float:
        """Realized-only equity (no mark-to-market on open positions --
        matches the backtest, which only ever books P&L at exit)."""
        return self.capital + self.realized_pnl


def proposal_to_dict(proposal: TradeProposal) -> dict:
    return {
        "symbol": proposal.symbol,
        "timestamp": proposal.timestamp.isoformat(),
        "setup_id": proposal.setup_id,
        "direction": proposal.direction,
        "entry_price": proposal.entry_price,
        "stop_price": proposal.stop_price,
        "target_price": proposal.target_price,
        "rr_ratio": proposal.rr_ratio,
        "condition_at_entry": proposal.condition_at_entry,
        "acceptance_streak_at_entry": proposal.acceptance_streak_at_entry,
        "notes": proposal.notes,
    }


def proposal_from_dict(d: dict) -> TradeProposal:
    return TradeProposal(
        symbol=d["symbol"],
        timestamp=pd.Timestamp(d["timestamp"]),
        setup_id=d["setup_id"],
        direction=d["direction"],
        entry_price=d["entry_price"],
        stop_price=d["stop_price"],
        target_price=d["target_price"],
        rr_ratio=d["rr_ratio"],
        condition_at_entry=d["condition_at_entry"],
        acceptance_streak_at_entry=d["acceptance_streak_at_entry"],
        notes=d.get("notes", ""),
    )


def position_to_dict(proposal: TradeProposal, entry_fill: float, quantity: int, entry_timestamp: pd.Timestamp) -> dict:
    """An open position, as stored in ``state.open_positions[symbol]``."""
    d = proposal_to_dict(proposal)
    d["entry_fill"] = entry_fill
    d["quantity"] = quantity
    d["entry_timestamp"] = entry_timestamp.isoformat()
    return d


def position_proposal(d: dict) -> TradeProposal:
    """Recover the original ``TradeProposal`` embedded in an open-position dict."""
    return proposal_from_dict(d)


def _date_to_str(value: dt.date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _str_to_date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value is not None else None


def state_to_dict(state: PaperState) -> dict:
    return {
        "capital": state.capital,
        "realized_pnl": state.realized_pnl,
        "trade_count": state.trade_count,
        "paper_start_date": _date_to_str(state.paper_start_date),
        "last_processed_date": _date_to_str(state.last_processed_date),
        "last_weekly_summary_date": _date_to_str(state.last_weekly_summary_date),
        "open_positions": state.open_positions,
        "pending_orders": state.pending_orders,
    }


def state_from_dict(d: dict) -> PaperState:
    return PaperState(
        capital=d.get("capital", 300_000.0),
        realized_pnl=d.get("realized_pnl", 0.0),
        trade_count=d.get("trade_count", 0),
        paper_start_date=_str_to_date(d.get("paper_start_date")),
        last_processed_date=_str_to_date(d.get("last_processed_date")),
        last_weekly_summary_date=_str_to_date(d.get("last_weekly_summary_date")),
        open_positions=d.get("open_positions", {}),
        pending_orders=d.get("pending_orders", {}),
    )


def load_state(path: Path, capital: float = 300_000.0) -> PaperState:
    """Returns a fresh ``PaperState`` (with the given starting capital) if
    ``path`` doesn't exist yet -- the very first run of the paper job."""
    if not path.exists():
        return PaperState(capital=capital)
    with path.open() as f:
        return state_from_dict(json.load(f))


def save_state(state: PaperState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(state_to_dict(state), f, indent=2, sort_keys=True)
        f.write("\n")
