#!/usr/bin/env python3
"""CLI entry point for the watchlist-candidate vetting pipeline.

Vets every symbol in ``data/candidate_watchlist.yaml`` against Cycle 3B's
EXACT strategy/cost/portfolio logic as it exists on main -- zero parameter
changes, single-symbol backtests only (no portfolio-cap competition; see
the Step 4 caveat baked into every generated SUMMARY.md). Entirely
additive: touches only ``cache/candidates/`` (gitignored) and
``backtest/results/candidates/`` (committed as research output). Never
imports, reads, or writes anything under ``paper/``, and never touches
``config.yaml``'s ``watchlist:`` key -- see ``tests/test_candidate_vetting.py``'s
guard test.

Usage:
    python scripts/vet_candidates.py
    python scripts/vet_candidates.py --force   # reprocess symbols already reported on
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import candidate_vetting, costs_delivery, swing_simulator
from data import candidate_instrument_keys, downloader, store
from signals.config import load_signals_config
from strategy.base import load_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEFRAME = "swing"
CANDIDATE_CACHE_DIR = REPO_ROOT / "cache" / "candidates"
RESULTS_DIR = REPO_ROOT / "backtest" / "results" / "candidates"
WATCHLIST_PATH = REPO_ROOT / "data" / "candidate_watchlist.yaml"
UNRESOLVED_PATH = RESULTS_DIR / "_unresolved.json"
SUMMARY_PATH = RESULTS_DIR / "SUMMARY.md"
LOOKBACK_YEARS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vet candidate watchlist symbols against Cycle 3B, unmodified.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--force", action="store_true", help="Reprocess every candidate, including ones already reported on."
    )
    return parser.parse_args()


def load_candidates() -> list[str]:
    with WATCHLIST_PATH.open() as f:
        raw = yaml.safe_load(f)
    return raw["candidates"]


def load_unresolved() -> list[str]:
    if not UNRESOLVED_PATH.exists():
        return []
    return json.loads(UNRESOLVED_PATH.read_text())


def save_unresolved(symbols: list[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UNRESOLVED_PATH.write_text(json.dumps(sorted(symbols), indent=2) + "\n")


def load_summary_row(symbol: str) -> dict | None:
    sidecar = RESULTS_DIR / f"{symbol}.json"
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text())


def save_summary_row(symbol: str, row: dict) -> None:
    sidecar = RESULTS_DIR / f"{symbol}.json"
    sidecar.write_text(json.dumps(row, indent=2) + "\n")


def already_reported(symbol: str, previously_unresolved: set[str]) -> bool:
    return (RESULTS_DIR / f"{symbol}.md").exists() or symbol in previously_unresolved


def vet_one_symbol(
    symbol: str,
    instrument_key: str,
    dl_cfg: downloader.DownloaderConfig,
    signals_cfg,
    strategy_cfg,
    cost_cfg,
    portfolio_cfg,
) -> dict | None:
    """Downloads, vets, and writes the per-symbol report for one resolved
    symbol. Returns its SUMMARY.md row dict, or ``None`` if no usable data
    could be downloaded (treated as unresolved by the caller)."""
    print(f"[vet_candidates] {symbol}: downloading up to {LOOKBACK_YEARS} years of daily candles...")
    try:
        df = downloader.download_symbol_daily_history(instrument_key, lookback_years=LOOKBACK_YEARS, config=dl_cfg)
    except Exception as exc:
        print(f"[vet_candidates] {symbol}: download failed ({exc}).")
        return None

    merged = store.write_symbol(symbol, df, 1440, CANDIDATE_CACHE_DIR, interval_label="1d")
    if merged is None or merged.empty:
        print(f"[vet_candidates] {symbol}: no candle data available.")
        return None

    coverage = candidate_vetting.compute_coverage(symbol, merged)
    split = candidate_vetting.compute_split_dates(coverage.first_date, coverage.last_date)

    indicators = candidate_vetting.compute_indicators(merged, signals_cfg)
    is_window = candidate_vetting.slice_by_date(indicators, split.is_start, split.is_end)
    oos_window = candidate_vetting.slice_by_date(indicators, split.oos_start, split.oos_end)

    is_result = candidate_vetting.run_single_symbol_window(symbol, is_window, strategy_cfg, cost_cfg, portfolio_cfg)
    oos_result = candidate_vetting.run_single_symbol_window(symbol, oos_window, strategy_cfg, cost_cfg, portfolio_cfg)

    tier = candidate_vetting.classify_tier(oos_result.report.overall, oos_result.cost_risk_distribution)

    report_md = candidate_vetting.render_symbol_report_md(symbol, coverage, split, is_result, oos_result, tier)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{symbol}.md").write_text(report_md)

    coverage_label = f"{coverage.first_date}..{coverage.last_date}" + (" (LIMITED)" if coverage.limited_history else "")
    row = {
        "symbol": symbol,
        "oos_trades": oos_result.report.overall["trades"],
        "oos_expectancy_r": oos_result.report.overall["expectancy_r"],
        "oos_profit_factor": oos_result.report.overall["profit_factor"]
        if oos_result.report.overall["profit_factor"] != float("inf")
        else "inf",
        "oos_cost_risk_avg_pct": oos_result.cost_risk_distribution["mean_pct"],
        "coverage_label": coverage_label,
        "tier": tier,
    }
    save_summary_row(symbol, row)
    print(
        f"[vet_candidates] {symbol}: {tier} "
        f"(OOS {row['oos_trades']} trades, expectancy {oos_result.report.overall['expectancy_r']:+.3f}R)"
    )
    return row


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    candidates = load_candidates()
    previously_unresolved = set(load_unresolved())

    if args.force:
        pending = list(candidates)
    else:
        pending = [s for s in candidates if not already_reported(s, previously_unresolved)]
    skipped = [s for s in candidates if s not in pending]
    if skipped:
        print(
            f"[vet_candidates] skipping {len(skipped)} already-reported symbol(s) "
            f"(use --force to reprocess): {', '.join(skipped)}"
        )
    if not pending:
        print("[vet_candidates] nothing new to do.")
        _write_summary(candidates)
        return

    print(f"[vet_candidates] resolving instrument keys for {len(pending)} candidate(s)...")
    resolved, unresolved = candidate_instrument_keys.resolve_candidates(pending, CANDIDATE_CACHE_DIR)
    if unresolved:
        print(f"[vet_candidates] UNRESOLVED ({len(unresolved)}): {', '.join(unresolved)}")

    signals_cfg = load_signals_config(config_path, timeframe=TIMEFRAME)
    strategy_cfg = load_strategy_config(config_path, timeframe=TIMEFRAME)
    cost_cfg = costs_delivery.load_delivery_cost_config(config_path)
    portfolio_cfg = swing_simulator.load_swing_portfolio_config(config_path)

    dl_raw = config.get("data", {})
    dl_cfg = downloader.DownloaderConfig(
        interval_minutes=1440,
        request_chunk_days=dl_raw.get("request_chunk_days", 28),
        request_sleep_seconds=dl_raw.get("request_sleep_seconds", 0.35),
        max_retries=dl_raw.get("max_retries", 5),
        access_token=os.environ.get("UPSTOX_ACCESS_TOKEN"),
    )

    still_unresolved = set(unresolved)
    for symbol, instrument_key in resolved.items():
        row = vet_one_symbol(symbol, instrument_key, dl_cfg, signals_cfg, strategy_cfg, cost_cfg, portfolio_cfg)
        if row is None:
            still_unresolved.add(symbol)

    # Union with symbols confirmed unresolved on a prior run, minus anything
    # resolved just now (so a symbol that becomes resolvable later -- e.g.
    # the fallback map grows -- drops out of UNRESOLVED on the next --force run).
    all_unresolved = (previously_unresolved | still_unresolved) - set(resolved.keys())
    save_unresolved(sorted(all_unresolved))

    _write_summary(candidates)


def _write_summary(candidates: list[str]) -> None:
    unresolved = load_unresolved()
    rows = []
    for symbol in candidates:
        if symbol in unresolved:
            continue
        row = load_summary_row(symbol)
        if row is not None:
            rows.append(row)

    summary_md = candidate_vetting.render_summary_md(rows, sorted(unresolved))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary_md)
    print(f"[vet_candidates] wrote {SUMMARY_PATH} ({len(rows)} resolved, {len(unresolved)} unresolved).")


if __name__ == "__main__":
    main()
