#!/usr/bin/env python3
"""Fidelity stress test: walks ``scripts/paper_daily.py``'s exact production
day-by-day logic (``_process_one_day`` / ``_pending_trading_dates``, the
same functions the real cron uses, imported and reused verbatim -- not
reimplemented) across YEARS of cached history in one sitting, and diffs the
result against a fresh batch backtest (``backtest.swing_simulator.run_portfolio``)
over the identical range and config, using ``scripts/verify_fidelity.py``'s
own ``diff_trades`` (the same diff PR #10 established as Phase 1's pass/fail
instrument). If the day-by-day pipeline and the batch engine ever disagree
on a single trade, this proves it isn't just correct "recently" -- it's
correct across the entire cached history, not merely on the handful of
weeks Phase 1 has actually run in production so far.

    python scripts/fidelity_stress_test.py
    python scripts/fidelity_stress_test.py --start-date 2022-01-01 --end-date 2024-12-31

Defaults to the full range of cached history common to every watchlist
symbol (earliest common date to latest common date) if --start-date/
--end-date aren't given.

## Safety

This tool must NEVER touch production state -- it never reads, writes, or
modifies paper/journal.csv, paper/trades.csv, paper/state.json, or
paper/run_log.csv, and it never calls Telegram (``_process_one_day`` is
always invoked with ``no_telegram=True``, unconditionally -- there is no
flag to turn this off). All simulated journaling/state lives in a fresh
throwaway temp directory (or an explicit --work-dir, still never paper/),
cleaned up automatically. ``run_stress_test`` snapshots every file's mtime
under the real ``paper/`` directory before and after the run and raises
``ProductionStateTouchedError`` if a single byte of that changed --
see ``tests/test_fidelity_stress_test.py`` for the same guarantee as an
explicit regression test, safe to re-run any time (e.g. after any future
paper/ code change) as a standing diagnostic.

Reads ONLY already-cached local parquet data (``data.store.read_symbol``)
-- no downloader, no instrument-key resolution, no network calls at all.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ itself -- for `import paper_daily`

from backtest import costs_delivery, swing_simulator
from data import store
from paper import journal as journal_module
from paper.state import PaperState
from signals.config import load_signals_config
from strategy.base import load_strategy_config

import paper_daily
import verify_fidelity

TIMEFRAME = "swing"
INTERVAL_LABEL = "1d"


class ProductionStateTouchedError(RuntimeError):
    """Raised if any file under the real paper/ directory changed during a run."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk paper_daily.py's day-by-day logic across cached history and diff it against a batch backtest."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--start-date", default=None, help="ISO date (YYYY-MM-DD). Default: earliest common cached date.")
    parser.add_argument("--end-date", default=None, help="ISO date (YYYY-MM-DD). Default: latest common cached date.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Temp directory for the isolated day-by-day walk's journal/state files. "
        "Default: a fresh auto-cleaned tempdir. NEVER pass the real paper/ directory here.",
    )
    return parser.parse_args()


def _snapshot_mtimes(paper_dir: Path) -> dict[str, float]:
    if not paper_dir.exists():
        return {}
    return {str(p): p.stat().st_mtime for p in sorted(paper_dir.rglob("*")) if p.is_file()}


def _load_watchlist_symbol_data(watchlist: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Reads whatever daily candles are already cached -- no downloader, no
    network. Symbols with no cache at all are silently skipped (mirrors
    ``paper_daily.py``'s own "no data at all, skipping" behavior)."""
    raw: dict = {}
    for symbol in watchlist:
        df = store.read_symbol(symbol, 1440, cache_dir, interval_label=INTERVAL_LABEL)
        if df is not None and not df.empty:
            raw[symbol] = df
    return raw


@dataclass
class StressTestReport:
    start_date: dt.date
    end_date: dt.date
    symbols: list[str]
    trading_days_walked: int
    day_by_day_trades: int
    batch_trades: int
    common_trades: int
    exact_matches: int
    mismatches: list[str] = field(default_factory=list)
    open_positions_at_end: int = 0
    runtime_seconds: float = 0.0
    paper_dir_untouched: bool = True

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.paper_dir_untouched


def _mismatched_trade_keys(ground_truth_rows: list[dict], paper_rows: list[dict]) -> set[tuple[str, str]]:
    """Same key/field definitions ``verify_fidelity.diff_trades`` uses
    (imported, not re-derived independently) -- used only to turn its
    human-readable mismatch strings into a count of distinct affected
    trades for the summary report. ``diff_trades`` itself remains the
    single source of truth for what counts as a mismatch."""
    gt_by_key = {
        verify_fidelity._trade_key(r["symbol"], r["entry_timestamp"]): r
        for r in ground_truth_rows
        if r["exit_reason"] != "end_of_data"
    }
    paper_by_key = {verify_fidelity._trade_key(r["symbol"], r["entry_timestamp"]): r for r in paper_rows}

    bad: set[tuple[str, str]] = set()
    for key, paper_row in paper_by_key.items():
        gt_row = gt_by_key.get(key)
        if gt_row is None:
            bad.add(key)
            continue
        for f in verify_fidelity._COMPARE_FIELDS_EXACT + verify_fidelity._COMPARE_FIELDS_TIMESTAMP:
            if str(paper_row[f]) != str(gt_row[f]):
                bad.add(key)
        for f in verify_fidelity._COMPARE_FIELDS_FLOAT:
            paper_val, gt_val = float(paper_row[f]), float(gt_row[f])
            if abs(paper_val - gt_val) > max(1e-6, abs(gt_val) * 1e-6):
                bad.add(key)
    for key in gt_by_key:
        if key not in paper_by_key:
            bad.add(key)
    return bad


def run_stress_test(
    config_path: Path,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    work_dir: Path | None = None,
) -> StressTestReport:
    t0 = time.time()

    real_paper_dir = REPO_ROOT / "paper"
    before_mtimes = _snapshot_mtimes(real_paper_dir)

    with config_path.open() as f:
        config = yaml.safe_load(f)
    cache_dir = REPO_ROOT / config["data"]["cache_dir"]
    watchlist = sorted(config["watchlist"])

    signals_cfg = load_signals_config(config_path, timeframe=TIMEFRAME)
    strategy_cfg = load_strategy_config(config_path, timeframe=TIMEFRAME)
    cost_cfg = costs_delivery.load_delivery_cost_config(config_path)
    portfolio_cfg = swing_simulator.load_swing_portfolio_config(config_path)

    raw_symbol_data = _load_watchlist_symbol_data(watchlist, cache_dir)
    if not raw_symbol_data:
        raise RuntimeError(
            "No cached daily data found for any watchlist symbol. Run scripts/paper_daily.py or "
            "scripts/download_data.py at least once to populate cache/ before running this tool."
        )

    common_dates_all = set.intersection(*(set(df["timestamp"].dt.date) for df in raw_symbol_data.values()))
    if not common_dates_all:
        raise RuntimeError("No trading date is present in every watchlist symbol's cache -- nothing to walk.")

    resolved_start = start_date or min(common_dates_all)
    resolved_end = end_date or max(common_dates_all)
    if resolved_start > resolved_end:
        raise ValueError(f"--start-date {resolved_start} is after --end-date {resolved_end}.")

    # Truncate to end_date FIRST, then compute indicators -- same slice-first
    # discipline used throughout this repo (e.g. scripts/run_swing_backtest.py),
    # so indicators warm up identically for both the day-by-day walk and the
    # batch backtest below, which share this exact symbol_data object.
    symbol_data = {}
    for symbol, df in raw_symbol_data.items():
        truncated = df[df["timestamp"].dt.date <= resolved_end].reset_index(drop=True)
        if truncated.empty:
            continue
        symbol_data[symbol] = paper_daily._compute_indicators(truncated, signals_cfg)

    # ---- Day-by-day walk via the SAME production _process_one_day / _pending_trading_dates. ----
    own_tempdir = work_dir is None
    work_dir = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="stockplayer_fidelity_stress_"))
    if work_dir.resolve() == real_paper_dir.resolve():
        raise ValueError("--work-dir must never be the real paper/ directory.")
    isolated_paper_dir = work_dir / "paper"

    try:
        journal_module.ensure_files(isolated_paper_dir)

        # _pending_trading_dates needs a non-None last_processed_date to walk a
        # bounded range instead of triggering its intentional first-run
        # short-circuit (which would return only the single latest date) --
        # seeding it one day before resolved_start makes it return every
        # common trading date from resolved_start onward, exactly the range
        # this tool needs (symbol_data is already truncated to <= resolved_end).
        seed_state = PaperState(capital=strategy_cfg.capital, last_processed_date=resolved_start - dt.timedelta(days=1))
        trading_dates = paper_daily._pending_trading_dates(symbol_data, seed_state)

        state = PaperState(capital=strategy_cfg.capital)
        for today in trading_dates:
            state = paper_daily._process_one_day(
                symbol_data, state, strategy_cfg, cost_cfg, portfolio_cfg, watchlist, isolated_paper_dir, today,
                no_telegram=True,
            )

        day_by_day_rows = journal_module.read_trades_csv(isolated_paper_dir / "trades.csv")
        open_positions_at_end = len(state.open_positions)
    finally:
        if own_tempdir:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ---- Fresh batch backtest over the identical range/config. ----
    batch_trades = swing_simulator.run_portfolio(
        symbol_data, strategy_cfg, cost_cfg, portfolio_cfg, walk_start_date=resolved_start
    )
    batch_rows = [journal_module.trade_record_to_row(t) for t in batch_trades]
    batch_closed_rows = [r for r in batch_rows if r["exit_reason"] != "end_of_data"]

    mismatches = verify_fidelity.diff_trades(batch_rows, day_by_day_rows)

    gt_keys = {verify_fidelity._trade_key(r["symbol"], r["entry_timestamp"]) for r in batch_closed_rows}
    paper_keys = {verify_fidelity._trade_key(r["symbol"], r["entry_timestamp"]) for r in day_by_day_rows}
    common_keys = gt_keys & paper_keys
    common_trades = len(common_keys)
    if mismatches:
        bad_common_keys = _mismatched_trade_keys(batch_rows, day_by_day_rows) & common_keys
        exact_matches = common_trades - len(bad_common_keys)
    else:
        exact_matches = common_trades

    after_mtimes = _snapshot_mtimes(real_paper_dir)
    paper_dir_untouched = before_mtimes == after_mtimes
    if not paper_dir_untouched:
        changed = sorted(set(before_mtimes) ^ set(after_mtimes)) or [
            p for p in before_mtimes if before_mtimes.get(p) != after_mtimes.get(p)
        ]
        raise ProductionStateTouchedError(
            f"paper/ was modified during the stress test run -- this must never happen. Changed: {changed}"
        )

    return StressTestReport(
        start_date=resolved_start,
        end_date=resolved_end,
        symbols=watchlist,
        trading_days_walked=len(trading_dates),
        day_by_day_trades=len(day_by_day_rows),
        batch_trades=len(batch_closed_rows),
        common_trades=common_trades,
        exact_matches=exact_matches,
        mismatches=mismatches,
        open_positions_at_end=open_positions_at_end,
        runtime_seconds=time.time() - t0,
        paper_dir_untouched=paper_dir_untouched,
    )


def format_report(report: StressTestReport) -> str:
    lines = [
        "=== Fidelity stress test ===",
        f"Range: {report.start_date} to {report.end_date} ({report.trading_days_walked} trading days walked)",
        f"Symbols: {len(report.symbols)} ({', '.join(report.symbols)})",
        f"Day-by-day trades (isolated walk): {report.day_by_day_trades}",
        f"Batch backtest trades (excl. still-open end_of_data): {report.batch_trades}",
        f"Open positions still held at end_date in the day-by-day walk: {report.open_positions_at_end} "
        "(expected -- excluded from comparison on both sides, same as verify_fidelity.py)",
        f"Trades comparable on both sides: {report.common_trades}",
        f"Exact matches: {report.exact_matches}/{report.common_trades}",
        f"Mismatches: {len(report.mismatches)}",
    ]
    if report.mismatches:
        lines.append("")
        lines.append("MISMATCH DETAIL (every mismatch reported in full -- nothing averaged away):")
        for m in report.mismatches:
            lines.append(f"  {m}")
    lines.append(f"Runtime: {report.runtime_seconds:.1f}s")
    lines.append(f"Real paper/ directory untouched: {report.paper_dir_untouched} (verified via mtime snapshot before/after)")
    lines.append(f"RESULT: {'PASS -- 100% fidelity' if report.passed else 'FAIL -- see mismatches above'}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    start_date = dt.date.fromisoformat(args.start_date) if args.start_date else None
    end_date = dt.date.fromisoformat(args.end_date) if args.end_date else None
    work_dir = Path(args.work_dir) if args.work_dir else None

    report = run_stress_test(config_path, start_date=start_date, end_date=end_date, work_dir=work_dir)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
