"""Unit tests for the watchlist-candidate vetting pipeline
(data/candidate_instrument_keys.py, backtest/candidate_vetting.py,
scripts/vet_candidates.py) -- no network access required.

Covers: the instrument-resolution fallback chain, history-depth flagging,
IS/OOS split math, report rendering on a synthetic fixture, and -- the
non-negotiable one -- a guard that this entirely-additive research pipeline
never touches paper/ or config.yaml's live watchlist.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import candidate_vetting, costs_delivery, swing_report, swing_simulator
from backtest.swing_simulator import SwingPortfolioConfig, SwingTradeRecord
from data import candidate_instrument_keys, instruments
from strategy.base import StrategyConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def _candidates() -> list[str]:
    with (REPO_ROOT / "data" / "candidate_watchlist.yaml").open() as f:
        return yaml.safe_load(f)["candidates"]


# ---------------------------------------------------------------------------
# Instrument resolution fallback chain
# ---------------------------------------------------------------------------


def test_watchlist_has_51_deduplicated_candidates():
    candidates = _candidates()
    assert len(candidates) == 51
    assert len(set(candidates)) == 51


def test_resolve_candidates_uses_hardcoded_map_when_bulk_source_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        instruments, "load_instrument_map", lambda cache_dir, force_refresh=False: (_ for _ in ()).throw(RuntimeError("network down"))
    )
    resolved, unresolved = candidate_instrument_keys.resolve_candidates(["TANLA", "SUNPHARMA"], tmp_path)
    assert resolved == {
        "TANLA": candidate_instrument_keys.CANDIDATE_VERIFIED_INSTRUMENT_KEYS["TANLA"],
        "SUNPHARMA": candidate_instrument_keys.CANDIDATE_VERIFIED_INSTRUMENT_KEYS["SUNPHARMA"],
    }
    assert unresolved == []


def test_resolve_candidates_prefers_bulk_source_over_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        instruments, "load_instrument_map", lambda cache_dir, force_refresh=False: {"TANLA": "NSE_EQ|SOME_PRIMARY_KEY"}
    )
    resolved, unresolved = candidate_instrument_keys.resolve_candidates(["TANLA"], tmp_path)
    assert resolved["TANLA"] == "NSE_EQ|SOME_PRIMARY_KEY"
    assert unresolved == []


def test_resolve_candidates_reports_unresolved_by_name_never_guesses(tmp_path, monkeypatch):
    monkeypatch.setattr(instruments, "load_instrument_map", lambda cache_dir, force_refresh=False: {})
    resolved, unresolved = candidate_instrument_keys.resolve_candidates(["FINELABS", "TANLA"], tmp_path)
    assert unresolved == ["FINELABS"]
    assert "TANLA" in resolved
    assert "FINELABS" not in resolved


def test_50_of_51_candidates_are_in_the_verified_fallback_map():
    # FINELABS is the one documented UNRESOLVED symbol -- see the module
    # docstring. Every other candidate must have a verified key.
    candidates = _candidates()
    missing = [s for s in candidates if s not in candidate_instrument_keys.CANDIDATE_VERIFIED_INSTRUMENT_KEYS]
    assert missing == ["FINELABS"], f"unexpected unresolved set: {missing}"


def test_verified_keys_are_well_formed_instrument_keys():
    # <SEGMENT>|<12-char ISIN> -- almost always NSE_EQ, but NSDL is
    # deliberately BSE_EQ (see module docstring: not registered under
    # NSE_EQ in Upstox's V3 API despite being NSE_EQ|INE301O01023 by ISIN).
    pattern = re.compile(r"^(NSE_EQ|BSE_EQ)\|[A-Z0-9]{12}$")
    for symbol, key in candidate_instrument_keys.CANDIDATE_VERIFIED_INSTRUMENT_KEYS.items():
        assert pattern.match(key), f"{symbol}: {key!r} doesn't look like a well-formed instrument key"


def test_candidate_verified_keys_are_disjoint_from_live_fallback_map():
    # This module must never silently override the live watchlist's own
    # fallback map -- they're deliberately kept separate (module docstring).
    from data import instrument_fallback

    overlap = set(candidate_instrument_keys.CANDIDATE_VERIFIED_INSTRUMENT_KEYS) & set(
        instrument_fallback.VERIFIED_INSTRUMENT_KEYS
    )
    assert overlap == set(), f"candidate keys unexpectedly overlap the live watchlist's symbols: {overlap}"


# ---------------------------------------------------------------------------
# History-depth flagging / IS-OOS split
# ---------------------------------------------------------------------------


def test_compute_coverage_flags_limited_history():
    dates = pd.date_range("2025-01-01", periods=200, freq="D", tz="Asia/Kolkata")
    df = pd.DataFrame({"timestamp": dates})
    coverage = candidate_vetting.compute_coverage("XYZ", df)
    assert coverage.limited_history is True
    assert coverage.trading_days == 200
    assert coverage.first_date == dates.min().date()
    assert coverage.last_date == dates.max().date()


def test_compute_coverage_does_not_flag_full_five_year_history():
    dates = pd.date_range("2020-01-01", periods=365 * 5 + 10, freq="D", tz="Asia/Kolkata")
    df = pd.DataFrame({"timestamp": dates})
    coverage = candidate_vetting.compute_coverage("XYZ", df)
    assert coverage.limited_history is False


def test_split_dates_uses_18_months_oos_for_full_five_year_history():
    first_date = dt.date(2021, 7, 9)
    last_date = dt.date(2026, 7, 10)
    split = candidate_vetting.compute_split_dates(first_date, last_date)
    assert split.is_start == first_date
    assert split.oos_end == last_date
    # ~18 months back from last_date
    expected_oos_start = (pd.Timestamp(last_date) - pd.DateOffset(months=18)).date()
    assert split.oos_start == expected_oos_start
    assert split.is_end == expected_oos_start - dt.timedelta(days=1)


def test_split_dates_uses_30_percent_oos_for_limited_history():
    first_date = dt.date(2024, 1, 1)
    last_date = dt.date(2025, 1, 1)  # 366 days total -- well under 5 years
    split = candidate_vetting.compute_split_dates(first_date, last_date)
    span_days = (last_date - first_date).days
    expected_oos_start = first_date + dt.timedelta(days=round(span_days * 0.7))
    assert split.oos_start == expected_oos_start
    assert split.is_start == first_date
    assert split.oos_end == last_date


def test_split_dates_never_produces_an_empty_side_for_very_short_history():
    first_date = dt.date(2026, 1, 1)
    last_date = dt.date(2026, 1, 3)  # 2 days total
    split = candidate_vetting.compute_split_dates(first_date, last_date)
    assert split.is_start <= split.is_end
    assert split.oos_start <= split.oos_end
    assert split.oos_start > first_date or split.oos_start == last_date


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------


def _stats(trades=0, expectancy_r=0.0, profit_factor=1.0):
    return {"trades": trades, "expectancy_r": expectancy_r, "profit_factor": profit_factor}


def _cost_risk(mean_pct=0.0, n=1):
    return {"mean_pct": mean_pct, "n": n}


def test_classify_tier_is_insufficient_data_with_zero_trades():
    # Zero OOS trades is "no evidence either way", not "tested and failed" --
    # must be its own tier, distinct from TIER_FAILS.
    tier = candidate_vetting.classify_tier(_stats(trades=0), _cost_risk(n=0))
    assert tier == candidate_vetting.TIER_INSUFFICIENT_DATA
    assert tier != candidate_vetting.TIER_FAILS


def test_classify_tier_fails_when_cost_risk_too_high():
    stats = _stats(trades=10, expectancy_r=0.5, profit_factor=2.0)
    assert candidate_vetting.classify_tier(stats, _cost_risk(mean_pct=20.0)) == candidate_vetting.TIER_FAILS


def test_classify_tier_comparable_when_all_bars_cleared():
    stats = _stats(trades=10, expectancy_r=0.3, profit_factor=1.5)
    assert candidate_vetting.classify_tier(stats, _cost_risk(mean_pct=10.0)) == candidate_vetting.TIER_COMPARABLE


def test_classify_tier_marginal_when_trade_count_too_thin():
    stats = _stats(trades=2, expectancy_r=0.3, profit_factor=1.5)
    assert candidate_vetting.classify_tier(stats, _cost_risk(mean_pct=10.0)) == candidate_vetting.TIER_MARGINAL


# ---------------------------------------------------------------------------
# Report rendering on a synthetic fixture
# ---------------------------------------------------------------------------


def _synthetic_trade(symbol: str, net_pnl: float, r_multiple: float, day: int) -> SwingTradeRecord:
    ts = pd.Timestamp("2026-01-01", tz="Asia/Kolkata") + pd.Timedelta(days=day)
    return SwingTradeRecord(
        symbol=symbol,
        setup_id="setup1_discovery",
        direction="long",
        signal_timestamp=ts,
        entry_timestamp=ts + pd.Timedelta(days=1),
        entry_signal_price=100.0,
        entry_fill_price=100.5,
        stop_price=95.0,
        target_price=110.0,
        rr_ratio=1.8,
        condition_at_entry="accepted_above",
        acceptance_streak_at_entry=2,
        quantity=10,
        exit_timestamp=ts + pd.Timedelta(days=3),
        exit_signal_price=105.0,
        exit_fill_price=105.0,
        exit_reason="target",
        holding_days=2,
        total_costs=15.0,
        gross_pnl=net_pnl + 15.0,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        notes="",
    )


def _window_result(symbol: str, trades: list[SwingTradeRecord]) -> candidate_vetting.WindowResult:
    report = swing_report.build_report(trades)
    cost_risk = swing_report.compute_cost_risk_ratio_distribution(report.trades_df)
    funnel_table = pd.DataFrame([{"symbol": symbol, "raw": len(trades), "executed": len(trades)}])
    return candidate_vetting.WindowResult(
        report=report, funnel_table=funnel_table, cost_risk_distribution=cost_risk, max_dd_pct_of_capital=0.0
    )


def test_render_symbol_report_md_synthetic_two_symbol_fixture():
    coverage = candidate_vetting.DataCoverage(
        symbol="ALPHA",
        first_date=dt.date(2021, 1, 1),
        last_date=dt.date(2026, 1, 1),
        trading_days=1200,
        limited_history=False,
    )
    split = candidate_vetting.compute_split_dates(coverage.first_date, coverage.last_date)

    is_trades = [_synthetic_trade("ALPHA", 500.0, 1.0, day=0), _synthetic_trade("ALPHA", -200.0, -0.5, day=10)]
    oos_trades = [_synthetic_trade("ALPHA", 300.0, 0.8, day=20)]

    is_result = _window_result("ALPHA", is_trades)
    oos_result = _window_result("ALPHA", oos_trades)
    tier = candidate_vetting.classify_tier(oos_result.report.overall, oos_result.cost_risk_distribution)

    md = candidate_vetting.render_symbol_report_md("ALPHA", coverage, split, is_result, oos_result, tier)

    assert "# ALPHA -- candidate vetting report" in md
    assert "## Data coverage" in md
    assert "## IS/OOS split" in md
    assert "## In-sample results" in md
    assert "## Out-of-sample results" in md
    assert "- Trades: 2" in md  # IS trade count
    assert "- Trades: 1" in md  # OOS trade count
    assert f"## Tier: {tier}" in md


def test_render_symbol_report_md_flags_limited_history():
    coverage = candidate_vetting.DataCoverage(
        symbol="BETA", first_date=dt.date(2025, 1, 1), last_date=dt.date(2025, 6, 1), trading_days=100,
        limited_history=True,
    )
    split = candidate_vetting.compute_split_dates(coverage.first_date, coverage.last_date)
    empty_result = _window_result("BETA", [])
    tier = candidate_vetting.classify_tier(empty_result.report.overall, empty_result.cost_risk_distribution)
    md = candidate_vetting.render_symbol_report_md("BETA", coverage, split, empty_result, empty_result, tier)
    assert "LIMITED HISTORY" in md


def test_render_summary_md_ranks_by_oos_expectancy_and_lists_unresolved():
    rows = [
        {
            "symbol": "WEAK",
            "oos_trades": 6,
            "oos_expectancy_r": 0.05,
            "oos_profit_factor": 1.1,
            "oos_cost_risk_avg_pct": 9.0,
            "coverage_label": "2021-01-01..2026-01-01",
            "tier": candidate_vetting.TIER_MARGINAL,
        },
        {
            "symbol": "STRONG",
            "oos_trades": 8,
            "oos_expectancy_r": 0.40,
            "oos_profit_factor": 2.0,
            "oos_cost_risk_avg_pct": 7.0,
            "coverage_label": "2021-01-01..2026-01-01",
            "tier": candidate_vetting.TIER_COMPARABLE,
        },
        {
            "symbol": "UNTESTED",
            "oos_trades": 0,
            "oos_expectancy_r": 0.0,
            "oos_profit_factor": "n/a",
            "oos_cost_risk_avg_pct": 0.0,
            "coverage_label": "2025-01-01..2026-01-01 (LIMITED)",
            "tier": candidate_vetting.TIER_INSUFFICIENT_DATA,
        },
    ]
    md = candidate_vetting.render_summary_md(rows, ["FINELABS"])

    assert "# Candidate watchlist vetting -- SUMMARY" in md
    assert "FIRST FILTER only" in md
    assert "combined portfolio backtest" in md.lower() or "COMBINED portfolio backtest" in md
    strong_idx = md.index("STRONG")
    weak_idx = md.index("WEAK")
    untested_idx = md.index("UNTESTED")
    assert strong_idx < weak_idx  # ranked by OOS expectancy_R descending
    assert weak_idx < untested_idx  # zero-trade (no evidence) symbols sort last, distinct from tested-and-failed
    assert candidate_vetting.TIER_INSUFFICIENT_DATA in md
    assert "FINELABS" in md
    assert "## Unresolved symbols" in md
    assert "## Tier definitions" in md


def test_render_summary_md_distinguishes_insufficient_data_from_fails():
    # The core regression this fix targets: a zero-trade symbol's tier text
    # must never equal a tested-and-failed symbol's tier text.
    assert candidate_vetting.TIER_INSUFFICIENT_DATA != candidate_vetting.TIER_FAILS
    md = candidate_vetting.render_summary_md([], [])
    assert candidate_vetting.TIER_FAILS in md
    assert candidate_vetting.TIER_INSUFFICIENT_DATA in md


def test_render_summary_md_with_no_unresolved_symbols_says_so():
    md = candidate_vetting.render_summary_md([], [])
    assert "None" in md


# ---------------------------------------------------------------------------
# Live-system safety guard -- the non-negotiable one
# ---------------------------------------------------------------------------

PIPELINE_SOURCE_FILES = [
    REPO_ROOT / "scripts" / "vet_candidates.py",
    REPO_ROOT / "backtest" / "candidate_vetting.py",
    REPO_ROOT / "data" / "candidate_instrument_keys.py",
]


def test_pipeline_never_imports_the_paper_trading_package():
    import_pattern = re.compile(r"^\s*(import\s+paper\b|from\s+paper\b)", re.MULTILINE)
    for path in PIPELINE_SOURCE_FILES:
        source = path.read_text()
        assert not import_pattern.search(source), f"{path} must never import paper/ (live cron code)"


def test_pipeline_never_references_config_watchlist_key_for_writing():
    # The pipeline is allowed to READ config.yaml (strategy/cost/portfolio
    # settings, exactly like run_swing_backtest.py does), but must never
    # write to it or mutate the `watchlist:` key.
    for path in PIPELINE_SOURCE_FILES:
        source = path.read_text()
        assert 'config["watchlist"]' not in source
        assert "config['watchlist']" not in source
        assert ".open(\"w\")" not in source or "config.yaml" not in source


def test_pipeline_writes_only_to_its_own_cache_and_results_directories():
    vet_candidates_source = (REPO_ROOT / "scripts" / "vet_candidates.py").read_text()
    assert 'REPO_ROOT / "cache" / "candidates"' in vet_candidates_source
    assert 'REPO_ROOT / "backtest" / "results" / "candidates"' in vet_candidates_source
    # Must never point at the live paper/ directory or cache/ root (the live
    # watchlist's own cache, shared with the daily cron).
    assert '"paper"' not in vet_candidates_source
    assert "'paper'" not in vet_candidates_source


def test_candidate_watchlist_yaml_is_disjoint_from_live_watchlist():
    with (REPO_ROOT / "config.yaml").open() as f:
        live_watchlist = set(yaml.safe_load(f)["watchlist"])
    candidates = set(_candidates())
    assert live_watchlist & candidates == set(), "candidate list must not duplicate the live watchlist"


def test_full_pipeline_smoke_does_not_touch_paper_dir_or_config(tmp_path, monkeypatch):
    """End-to-end guard: actually run resolution + a single-symbol backtest
    window against a synthetic price series, and confirm neither paper/'s
    on-disk files nor config.yaml change as a byte."""
    paper_dir = REPO_ROOT / "paper"
    config_path = REPO_ROOT / "config.yaml"
    paper_snapshot = {p: p.read_bytes() for p in paper_dir.glob("*") if p.is_file()}
    config_snapshot = config_path.read_bytes()

    n = 400
    dates = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")
    close = pd.Series(100.0 + pd.Series(range(n)) * 0.05)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10_000,
        }
    )

    from signals.config import load_signals_config
    from strategy.base import load_strategy_config

    signals_cfg = load_signals_config(config_path, timeframe="swing")
    strategy_cfg = load_strategy_config(config_path, timeframe="swing")
    cost_cfg = costs_delivery.load_delivery_cost_config(config_path)
    portfolio_cfg = swing_simulator.load_swing_portfolio_config(config_path)

    coverage = candidate_vetting.compute_coverage("SYNTH", df)
    split = candidate_vetting.compute_split_dates(coverage.first_date, coverage.last_date)
    indicators = candidate_vetting.compute_indicators(df, signals_cfg)
    oos_window = candidate_vetting.slice_by_date(indicators, split.oos_start, split.oos_end)
    result = candidate_vetting.run_single_symbol_window("SYNTH", oos_window, strategy_cfg, cost_cfg, portfolio_cfg)

    assert result.report is not None  # pipeline actually ran

    for p, content in paper_snapshot.items():
        assert p.read_bytes() == content, f"{p} was modified by the candidate-vetting pipeline"
    assert config_path.read_bytes() == config_snapshot, "config.yaml was modified by the candidate-vetting pipeline"
