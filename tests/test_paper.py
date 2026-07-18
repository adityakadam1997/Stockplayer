"""Unit tests for Phase 1 (paper trading) -- no network access, synthetic
data only.

The central test (``test_day_by_day_matches_batch_replay``) is the actual
point of this phase: it proves ``paper.pipeline.run_daily_step``, called
once per trading day, produces EXACTLY the same trades as
``backtest.swing_simulator.run_portfolio`` called once in batch over the
same data with the same ``walk_start_date`` -- i.e. that the day-granular
incremental job and the batch backtest engine agree, which is what
``scripts/verify_fidelity.py`` checks in production against the real,
git-committed paper journal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backtest import costs_delivery, swing_simulator
from backtest.swing_simulator import SwingPortfolioConfig
from paper import journal as journal_module
from paper import state as state_module
from paper.pipeline import run_daily_step
from paper.state import PaperState
from signals.condition import ACCEPTED_ABOVE, INSIDE_VALUE
from strategy.base import LONG, StrategyConfig

_ZERO_SLIPPAGE_COST_CFG = costs_delivery.DeliveryCostConfig(slippage_pct=0.0)


def _permissive_cfg(**overrides) -> StrategyConfig:
    defaults = dict(stop_floor_pct=0.0, atr_mult=0.0, cost_viability_max_pct=float("inf"))
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def _daily_frame(dates, opens, highs, lows, closes, band_upper=105.0, band_lower=95.0, monthly_vwap=50.0):
    """A long, otherwise-flat synthetic daily series with one setup1 LONG
    signal baked in at index 1 (day0 builds acceptance, day1 is the retest),
    followed by many flat days so time-stops / multi-signal scenarios can be
    exercised. Every row gets a generous prior-20d-high / band_upper_2 so
    Cycle 3B's target recomputation stays valid throughout."""
    n = len(dates)
    ts = pd.to_datetime(dates).tz_localize("Asia/Kolkata")
    conditions = [ACCEPTED_ABOVE] + [INSIDE_VALUE] * (n - 1)
    streaks = [3] + [0] * (n - 1)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "vwap": [100.0] * n,
            "band_upper_1": [band_upper] * n,
            "band_lower_1": [95.0] * n,
            "band_upper_2": [band_upper + 100.0] * n,
            "condition": conditions,
            "acceptance_streak": streaks,
            "atr14": [0.0] * n,
            "monthly_vwap": [monthly_vwap] * n,
            "high_20d_prior": [band_upper + 100.0] * n,
        }
    )


def _two_symbol_history():
    """~14 trading days for 2 symbols. AAA fires a setup1 signal on day1
    (retest), filled day2; then flat for the rest (long enough to exercise
    a time-stop). BBB never fires (stays purely inside_value)."""
    dates = pd.bdate_range("2026-01-05", periods=14).strftime("%Y-%m-%d").tolist()
    n = len(dates)

    aaa_opens = [100.0, 106.0] + [106.0] * (n - 2)
    aaa_highs = [112.0, 106.5] + [106.5] * (n - 2)
    aaa_lows = [99.0, 104.0] + [105.5] * (n - 2)
    aaa_closes = [108.0, 106.0] + [106.0] * (n - 2)
    df_aaa = _daily_frame(dates, aaa_opens, aaa_highs, aaa_lows, aaa_closes)

    bbb_flat = [100.0] * n
    df_bbb = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates).tz_localize("Asia/Kolkata"),
            "open": bbb_flat,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": bbb_flat,
            "vwap": [100.0] * n,
            "band_upper_1": [105.0] * n,
            "band_lower_1": [95.0] * n,
            "band_upper_2": [110.0] * n,
            "condition": [INSIDE_VALUE] * n,
            "acceptance_streak": [0] * n,
            "atr14": [0.0] * n,
            "monthly_vwap": [50.0] * n,
            "high_20d_prior": [110.0] * n,
        }
    )
    return {"AAA": df_aaa, "BBB": df_bbb}, dates


def test_day_by_day_matches_batch_replay():
    symbol_data, dates = _two_symbol_history()
    watchlist = sorted(symbol_data.keys())
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(time_stop_days=10, max_concurrent_positions=5, max_positions_per_symbol=1)
    paper_start = pd.Timestamp(dates[0]).date()

    # Batch ground truth: swing_simulator over the whole history, walk scoped to paper_start.
    batch_trades = swing_simulator.run_portfolio(
        symbol_data, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, walk_start_date=paper_start
    )

    # Day-by-day: run_daily_step once per trading day, starting from an empty state.
    state = PaperState(capital=300_000.0, paper_start_date=paper_start)
    all_trades = []
    for date_str in dates:
        today = pd.Timestamp(date_str).date()
        result = run_daily_step(symbol_data, state, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, today, watchlist)
        all_trades.extend(result.trades)
        state = result.state

    assert len(all_trades) == len(batch_trades) > 0

    def _key(t):
        return (t.symbol, t.entry_timestamp, t.exit_timestamp)

    day_by_day_sorted = sorted(all_trades, key=_key)
    batch_sorted = sorted(batch_trades, key=_key)

    for a, b in zip(day_by_day_sorted, batch_sorted):
        assert a.symbol == b.symbol
        assert a.setup_id == b.setup_id
        assert a.direction == b.direction
        assert a.signal_timestamp == b.signal_timestamp
        assert a.entry_timestamp == b.entry_timestamp
        assert a.entry_fill_price == pytest.approx(b.entry_fill_price)
        assert a.stop_price == pytest.approx(b.stop_price)
        assert a.target_price == pytest.approx(b.target_price)
        assert a.exit_timestamp == b.exit_timestamp
        assert a.exit_fill_price == pytest.approx(b.exit_fill_price)
        assert a.exit_reason == b.exit_reason
        assert a.r_multiple == pytest.approx(b.r_multiple)
        assert a.net_pnl == pytest.approx(b.net_pnl)


def test_pending_order_fills_at_next_open_matching_simulator():
    symbol_data, dates = _two_symbol_history()
    watchlist = sorted(symbol_data.keys())
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(time_stop_days=100)
    paper_start = pd.Timestamp(dates[0]).date()

    state = PaperState(capital=300_000.0, paper_start_date=paper_start)
    day0 = pd.Timestamp(dates[0]).date()
    result0 = run_daily_step(symbol_data, state, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, day0, watchlist)
    assert result0.new_pending == []  # day0 is acceptance only, no retest signal yet
    assert result0.state.pending_orders == {}

    day1 = pd.Timestamp(dates[1]).date()
    result1 = run_daily_step(
        symbol_data, result0.state, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, day1, watchlist
    )
    # day1 IS the retest signal day -- it also fires a new proposal (dropped by
    # the trend filter or geometry doesn't apply here since state carries fresh);
    # more importantly, nothing should be *filled* yet (day0's signal doesn't
    # exist -- the retest only fires on day1 itself, so day1's fill queue was empty).
    assert result1.fills == []

    day2 = pd.Timestamp(dates[2]).date()
    result2 = run_daily_step(
        symbol_data, result1.state, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, day2, watchlist
    )
    assert len(result2.fills) == 1
    fill = result2.fills[0]
    assert fill["symbol"] == "AAA"
    assert fill["entry_fill_price"] == pytest.approx(106.0)  # day2's open (see _two_symbol_history)
    assert "AAA" in result2.state.open_positions
    assert "AAA" not in result2.state.pending_orders


def test_idempotent_same_day_rerun_changes_nothing():
    symbol_data, dates = _two_symbol_history()
    watchlist = sorted(symbol_data.keys())
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(time_stop_days=100)
    paper_start = pd.Timestamp(dates[0]).date()

    state = PaperState(capital=300_000.0, paper_start_date=paper_start)
    for date_str in dates[:3]:
        today = pd.Timestamp(date_str).date()
        result = run_daily_step(symbol_data, state, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, today, watchlist)
        state = result.state

    # Re-running the SAME last day again (simulating a re-triggered job) must
    # be driven at the CLI layer by last_processed_date == today -- but at the
    # pipeline layer itself, calling run_daily_step twice for the same day on
    # the same state must be side-effect-free/deterministic (no duplicate fills).
    today = pd.Timestamp(dates[2]).date()
    result_again = run_daily_step(symbol_data, state, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, today, watchlist)
    assert result_again.trades == []
    assert result_again.fills == []
    assert state_module.state_to_dict(result_again.state) == state_module.state_to_dict(state)


def test_capacity_cutoff_matches_batch_replay_with_three_symbols():
    # Three symbols all fire the identical setup1 signal on the same day, but
    # max_concurrent_positions=2 -- alphabetically AAA and BBB should fill,
    # CCC should be permanently dropped (not retried later), matching
    # swing_simulator's early-break tie-break rule exactly.
    dates = pd.bdate_range("2026-01-05", periods=14).strftime("%Y-%m-%d").tolist()
    symbol_data = {}
    for symbol in ["AAA", "BBB", "CCC"]:
        n = len(dates)
        opens = [100.0, 106.0] + [106.0] * (n - 2)
        highs = [112.0, 106.5] + [106.5] * (n - 2)
        lows = [99.0, 104.0] + [105.5] * (n - 2)
        closes = [108.0, 106.0] + [106.0] * (n - 2)
        symbol_data[symbol] = _daily_frame(dates, opens, highs, lows, closes)

    watchlist = sorted(symbol_data.keys())
    cfg = _permissive_cfg()
    portfolio_cfg = SwingPortfolioConfig(time_stop_days=100, max_concurrent_positions=2)
    paper_start = pd.Timestamp(dates[0]).date()

    batch_trades = swing_simulator.run_portfolio(
        symbol_data, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, walk_start_date=paper_start
    )
    batch_symbols_entered = {t.symbol for t in batch_trades}

    state = PaperState(capital=300_000.0, paper_start_date=paper_start)
    day_by_day_symbols_entered = set()
    for date_str in dates:
        today = pd.Timestamp(date_str).date()
        result = run_daily_step(symbol_data, state, cfg, _ZERO_SLIPPAGE_COST_CFG, portfolio_cfg, today, watchlist)
        day_by_day_symbols_entered.update(f["symbol"] for f in result.fills)
        state = result.state

    assert day_by_day_symbols_entered == batch_symbols_entered == {"AAA", "BBB"}
    assert "CCC" not in state.open_positions
    assert "CCC" not in state.pending_orders  # dropped permanently, not sitting around for a future retry


def test_state_json_round_trip(tmp_path):
    from strategy.base import TradeProposal

    proposal = TradeProposal(
        symbol="AAA",
        timestamp=pd.Timestamp("2026-01-05", tz="Asia/Kolkata"),
        setup_id="setup1_discovery",
        direction=LONG,
        entry_price=106.0,
        stop_price=104.0,
        target_price=112.0,
        rr_ratio=3.0,
        condition_at_entry="accepted_above",
        acceptance_streak_at_entry=3,
        notes="test",
    )
    state = PaperState(
        capital=300_000.0,
        realized_pnl=1234.5,
        trade_count=2,
        paper_start_date=pd.Timestamp("2026-01-05").date(),
        last_processed_date=pd.Timestamp("2026-01-07").date(),
        open_positions={
            "AAA": state_module.position_to_dict(proposal, 106.5, 85, pd.Timestamp("2026-01-06", tz="Asia/Kolkata"))
        },
        pending_orders={"BBB": state_module.proposal_to_dict(proposal)},
    )

    path = tmp_path / "state.json"
    state_module.save_state(state, path)
    reloaded = state_module.load_state(path)

    assert reloaded.capital == state.capital
    assert reloaded.realized_pnl == state.realized_pnl
    assert reloaded.trade_count == state.trade_count
    assert reloaded.paper_start_date == state.paper_start_date
    assert reloaded.last_processed_date == state.last_processed_date
    assert reloaded.open_positions == state.open_positions
    assert reloaded.pending_orders == state.pending_orders

    # And the embedded proposal really does round-trip back to an equal TradeProposal.
    recovered = state_module.position_proposal(reloaded.open_positions["AAA"])
    assert recovered == proposal


def test_holiday_no_op_when_no_new_candle():
    # Exercises the actual guard paper_daily.main() calls: an NSE holiday
    # (no symbol's cache advanced past the prior run) and an accidental
    # same-day re-run both look identical -- candidate_today isn't strictly
    # newer than state.last_processed_date -- and both must be a no-op.
    import paper_daily

    symbol_data, dates = _two_symbol_history()
    latest_dates = [df["timestamp"].max().date() for df in symbol_data.values()]
    candidate_today = min(latest_dates)

    already_processed_state = PaperState(capital=300_000.0, last_processed_date=candidate_today)
    assert paper_daily._should_skip(already_processed_state, candidate_today) is True

    holiday_state = PaperState(capital=300_000.0, last_processed_date=candidate_today)  # no fresher candle appeared
    assert paper_daily._should_skip(holiday_state, candidate_today) is True

    fresh_state = PaperState(capital=300_000.0, last_processed_date=None)
    assert paper_daily._should_skip(fresh_state, candidate_today) is False

    earlier_state = PaperState(capital=300_000.0, last_processed_date=pd.Timestamp(dates[0]).date())
    later_today = pd.Timestamp(dates[-1]).date()
    assert paper_daily._should_skip(earlier_state, later_today) is False


def test_fidelity_harness_catches_corrupted_journal_row():
    import verify_fidelity

    good_row = {
        "symbol": "AAA",
        "setup_id": "setup1_discovery",
        "direction": "long",
        "signal_timestamp": "2026-01-06T00:00:00+05:30",
        "entry_timestamp": "2026-01-07T00:00:00+05:30",
        "entry_fill_price": "106.0",
        "stop_price": "104.0",
        "target_price": "112.0",
        "exit_timestamp": "2026-01-20T00:00:00+05:30",
        "exit_fill_price": "112.0",
        "exit_reason": "target",
        "r_multiple": "3.0",
        "net_pnl": "500.0",
    }
    ground_truth = [good_row]

    # Sanity: identical rows -> no mismatches.
    assert verify_fidelity.diff_trades(ground_truth, [dict(good_row)]) == []

    # A deliberately corrupted paper row (wrong exit_fill_price, as if a bug
    # mis-recorded the fill) must be caught.
    corrupted = dict(good_row)
    corrupted["exit_fill_price"] = "999.0"
    mismatches = verify_fidelity.diff_trades(ground_truth, [corrupted])
    assert len(mismatches) == 1
    assert "exit_fill_price" in mismatches[0]

    # A paper row for a trade the backtest replay never produced.
    phantom = dict(good_row)
    phantom["entry_timestamp"] = "2026-02-01T00:00:00+05:30"
    mismatches = verify_fidelity.diff_trades(ground_truth, [phantom])
    assert len(mismatches) == 2  # phantom trade missing from backtest AND the real one missing from paper


def test_ensure_files_creates_header_only_csvs(tmp_path):
    paper_dir = tmp_path / "paper"
    journal_module.ensure_files(paper_dir)

    for name, columns in [
        ("journal.csv", journal_module.JOURNAL_COLUMNS),
        ("trades.csv", journal_module.TRADE_COLUMNS),
        ("run_log.csv", journal_module.RUN_LOG_COLUMNS),
    ]:
        path = paper_dir / name
        assert path.exists()
        with path.open() as f:
            header = f.readline().strip().split(",")
        assert header == columns

    assert journal_module.read_journal_csv(paper_dir / "journal.csv") == []
    assert journal_module.read_trades_csv(paper_dir / "trades.csv") == []
    assert journal_module.read_run_log(paper_dir / "run_log.csv") == []


def test_ensure_files_does_not_clobber_existing_data(tmp_path):
    paper_dir = tmp_path / "paper"
    journal_module.ensure_files(paper_dir)
    journal_module.append_run_log(paper_dir / "run_log.csv", "2026-01-05")

    # Calling ensure_files again (as every single invocation of
    # paper_daily.py does) must be a pure no-op on files that already have
    # real content.
    journal_module.ensure_files(paper_dir)
    assert journal_module.read_run_log(paper_dir / "run_log.csv") == [{"run_date": "2026-01-05"}]


def test_zero_activity_first_run_leaves_git_addable_files(tmp_path):
    # Reproduces exactly what broke the paper workflow's first real run:
    # a day is processed, nothing fires (0 trades, possibly 0 candidates),
    # and the workflow then does `git add paper/trades.csv` unconditionally.
    # After the fix, that file must exist -- with a valid header and zero
    # data rows -- even though append_trade_rows/append_journal_rows were
    # both called with empty lists.
    paper_dir = tmp_path / "paper"
    journal_module.ensure_files(paper_dir)  # paper_daily.py calls this before anything else

    journal_module.append_trade_rows(paper_dir / "trades.csv", [])
    journal_module.append_journal_rows(paper_dir / "journal.csv", "2026-01-05", [])
    journal_module.append_run_log(paper_dir / "run_log.csv", "2026-01-05")

    assert (paper_dir / "trades.csv").exists()
    assert (paper_dir / "journal.csv").exists()
    assert (paper_dir / "run_log.csv").exists()
    assert journal_module.read_trades_csv(paper_dir / "trades.csv") == []
    assert journal_module.read_journal_csv(paper_dir / "journal.csv") == []
    assert journal_module.read_run_log(paper_dir / "run_log.csv") == [{"run_date": "2026-01-05"}]

    # The actual regression: `git add <path>` fails with "pathspec did not
    # match any files" if and only if the path doesn't exist on disk.
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    result = subprocess.run(
        ["git", "add", "paper/trades.csv", "paper/journal.csv", "paper/run_log.csv"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_load_state_missing_file_returns_fresh_default(tmp_path):
    state = state_module.load_state(tmp_path / "does_not_exist.json", capital=300_000.0)
    assert state.capital == 300_000.0
    assert state.open_positions == {}
    assert state.pending_orders == {}
    assert state.last_processed_date is None
