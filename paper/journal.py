"""Append-only CSV journals: ``paper/journal.csv`` (every candidate proposal,
whatever funnel stage it reached -- diagnostic) and ``paper/trades.csv``
(fills/exits, full economics -- the thing ``scripts/verify_fidelity.py``
diffs against the backtest engine). Both are plain CSVs so the git history
of each file is a literal, human-readable audit trail.
"""

from __future__ import annotations

import csv
from pathlib import Path

from backtest.swing_simulator import SwingTradeRecord

JOURNAL_COLUMNS = [
    "run_date",
    "symbol",
    "signal_timestamp",
    "setup_id",
    "direction",
    "entry_price",
    "stop_price",
    "target_price",
    "rr_ratio",
    "funnel_stage",
    "notes",
]

TRADE_COLUMNS = [
    "symbol",
    "setup_id",
    "direction",
    "signal_timestamp",
    "entry_timestamp",
    "entry_signal_price",
    "entry_fill_price",
    "stop_price",
    "target_price",
    "rr_ratio",
    "condition_at_entry",
    "acceptance_streak_at_entry",
    "quantity",
    "exit_timestamp",
    "exit_signal_price",
    "exit_fill_price",
    "exit_reason",
    "holding_days",
    "total_costs",
    "gross_pnl",
    "net_pnl",
    "r_multiple",
    "notes",
]


def _append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ensure_header_only(path: Path, columns: list[str]) -> None:
    """Create ``path`` with just a header row if it doesn't exist yet --
    idempotent, does nothing if the file (with any content) already exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()


def ensure_files(paper_dir: Path) -> None:
    """Create ``journal.csv``, ``trades.csv``, and ``run_log.csv`` (header
    row only) if they don't exist yet. A zero-activity day -- which is most
    days, and is guaranteed on the very first run before any trade has ever
    filled or exited -- would otherwise leave ``trades.csv`` (and possibly
    ``journal.csv``, if literally no setup fired anywhere that day) absent
    from disk entirely, since ``_append_rows`` is a no-op on an empty list.
    That's exactly what broke the paper workflow's first real run: its
    ``git add paper/trades.csv`` failed with "pathspec did not match any
    files" because the file had genuinely never been created. Call this
    unconditionally at the start of every ``paper_daily.py`` invocation
    (before the holiday/no-op check) so the files always exist from the
    first run onward, activity or not."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    _ensure_header_only(paper_dir / "journal.csv", JOURNAL_COLUMNS)
    _ensure_header_only(paper_dir / "trades.csv", TRADE_COLUMNS)
    _ensure_header_only(paper_dir / "run_log.csv", RUN_LOG_COLUMNS)


def append_journal_rows(path: Path, run_date: str, trace: list[dict]) -> None:
    """``trace`` is ``strategy.swing_engine.generate_proposal_trace``'s
    output for one symbol/day -- one row per raw candidate."""
    rows = []
    for t in trace:
        rows.append(
            {
                "run_date": run_date,
                "symbol": t["symbol"],
                "signal_timestamp": t["timestamp"].isoformat(),
                "setup_id": t["setup_id"],
                "direction": t["direction"],
                "entry_price": t["entry_price"],
                "stop_price": t["stop_price"],
                "target_price": t["target_price"],
                "rr_ratio": t["rr_ratio"],
                "funnel_stage": t["funnel_stage"],
                "notes": t["notes"],
            }
        )
    _append_rows(path, JOURNAL_COLUMNS, rows)


def trade_record_to_row(trade: SwingTradeRecord) -> dict:
    return {
        "symbol": trade.symbol,
        "setup_id": trade.setup_id,
        "direction": trade.direction,
        "signal_timestamp": trade.signal_timestamp.isoformat(),
        "entry_timestamp": trade.entry_timestamp.isoformat(),
        "entry_signal_price": trade.entry_signal_price,
        "entry_fill_price": trade.entry_fill_price,
        "stop_price": trade.stop_price,
        "target_price": trade.target_price,
        "rr_ratio": trade.rr_ratio,
        "condition_at_entry": trade.condition_at_entry,
        "acceptance_streak_at_entry": trade.acceptance_streak_at_entry,
        "quantity": trade.quantity,
        "exit_timestamp": trade.exit_timestamp.isoformat(),
        "exit_signal_price": trade.exit_signal_price,
        "exit_fill_price": trade.exit_fill_price,
        "exit_reason": trade.exit_reason,
        "holding_days": trade.holding_days,
        "total_costs": trade.total_costs,
        "gross_pnl": trade.gross_pnl,
        "net_pnl": trade.net_pnl,
        "r_multiple": trade.r_multiple,
        "notes": trade.notes,
    }


def append_trade_rows(path: Path, trades: list[SwingTradeRecord]) -> None:
    _append_rows(path, TRADE_COLUMNS, [trade_record_to_row(t) for t in trades])


def read_trades_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


RUN_LOG_COLUMNS = ["run_date"]


def append_run_log(path: Path, run_date: str) -> None:
    """One row per trading day the job actually processed (regardless of
    whether anything happened that day) -- the source of truth for the
    weekly summary's reliability metric, since a zero-activity day never
    appears in ``journal.csv``."""
    _append_rows(path, RUN_LOG_COLUMNS, [{"run_date": run_date}])


def read_run_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_journal_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# Every journal row's funnel_stage is TERMINAL -- the one stage that
# candidate's journey stopped at. Cumulative pass-counts for a given stage
# are reconstructed by walking this ordered list of (stage, drop-count)
# pairs and subtracting each drop count from the running total, matching
# strategy.swing_engine's own funnel stage order exactly.
_FUNNEL_STAGE_ORDER = [
    ("wide_band_guard", "raw"),
    ("failed_trend_filter", "after_trend_filter"),
    ("suppressed_short", "after_long_only"),
    ("invalid_geometry", "after_valid_geometry"),
    ("failed_rr", "after_rr"),
    ("failed_cost_viability", "after_cost_viability"),
]


def compute_funnel_totals(journal_rows: list[dict]) -> dict:
    """Reconstructs cumulative funnel-stage pass-counts from the journal's
    per-candidate terminal-stage log -- for the weekly summary."""
    running = len(journal_rows)
    totals = {}
    for drop_stage, survivor_label in _FUNNEL_STAGE_ORDER:
        running -= sum(1 for r in journal_rows if r["funnel_stage"] == drop_stage)
        totals[survivor_label] = running
    return totals
