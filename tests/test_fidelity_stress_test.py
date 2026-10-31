"""Unit tests for scripts/fidelity_stress_test.py -- no network access,
synthetic data only (this repo's cache/ is gitignored and empty on a fresh
checkout, so these tests build their own isolated config.yaml + cache/
rather than depending on the real ones being pre-populated).

The two things that matter most here, matching the tool's own stated
purpose: (1) the day-by-day walk (via paper_daily._process_one_day /
_pending_trading_dates, imported and reused, not reimplemented) actually
reproduces the batch backtest exactly on realistic multi-year-shaped data,
and (2) it is IMPOSSIBLE for a run of this tool to touch the real paper/
directory, config.yaml, or Telegram, regardless of what data it's pointed at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data import store

import fidelity_stress_test as fst
import paper_daily

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Synthetic fixture: realistic-shaped raw daily OHLCV that actually clears
# Cycle 3B's R:R / cost-viability / trend-filter bars (a flat or tiny-move
# series produces zero trades on both sides, which would make the diff
# vacuously "pass" without proving anything -- verified empirically that this
# shape reliably produces real trades under the real swing strategy config).
# ---------------------------------------------------------------------------


def _synthetic_symbol_frame(seed: int, n: int = 500, start: str = "2021-01-04", price0: float = 1000.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, periods=n, tz="Asia/Kolkata")
    rets = rng.normal(0.0006, 0.018, n)
    cyc = 0.01 * np.sin(np.arange(n) / 12.0)
    close = price0 * np.cumprod(1 + rets + cyc * 0.3)
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.006, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.006, 0.004, n)))
    volume = rng.randint(100_000, 500_000, n)
    return pd.DataFrame(
        {"timestamp": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _write_isolated_config(tmp_path: Path, symbols: list[str]) -> Path:
    """A full copy of the real config.yaml's strategy/signals/cost/portfolio
    sections (so the test exercises the REAL swing parameters, not a
    permissive stand-in), with only ``data.cache_dir`` and ``watchlist``
    redirected to an isolated, temp location -- never the real cache/."""
    with (REPO_ROOT / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    config["data"]["cache_dir"] = str(tmp_path / "cache") + "/"
    config["watchlist"] = symbols

    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)
    return config_path


def _populate_cache(cache_dir: Path, symbols: list[str], **frame_kwargs) -> None:
    for i, symbol in enumerate(symbols):
        df = _synthetic_symbol_frame(seed=i + 1, **frame_kwargs)
        store.write_symbol(symbol, df, 1440, cache_dir, interval_label="1d")


@pytest.fixture
def isolated_env(tmp_path):
    symbols = ["AAA", "BBB"]
    config_path = _write_isolated_config(tmp_path, symbols)
    with config_path.open() as f:
        cache_dir = Path(yaml.safe_load(f)["data"]["cache_dir"])
    _populate_cache(cache_dir, symbols)
    return config_path, symbols


# ---------------------------------------------------------------------------
# Core behavior: day-by-day reproduces the batch backtest exactly.
# ---------------------------------------------------------------------------


def test_run_stress_test_reports_zero_mismatches_on_realistic_synthetic_data(isolated_env, tmp_path):
    config_path, symbols = isolated_env
    report = fst.run_stress_test(config_path, work_dir=tmp_path / "work")

    assert report.mismatches == []
    assert report.passed is True
    assert report.day_by_day_trades == report.batch_trades == report.common_trades
    assert report.exact_matches == report.common_trades
    # The fixture must actually generate real trades -- otherwise this test
    # would pass vacuously without exercising the diff at all.
    assert report.day_by_day_trades > 0
    assert report.trading_days_walked > 400
    assert set(report.symbols) == set(symbols)


def test_run_stress_test_respects_start_and_end_date(isolated_env, tmp_path):
    config_path, _ = isolated_env
    import datetime as dt

    start = dt.date(2021, 6, 1)
    end = dt.date(2021, 12, 31)
    report = fst.run_stress_test(config_path, start_date=start, end_date=end, work_dir=tmp_path / "work")

    assert report.start_date == start
    assert report.end_date == end
    assert report.mismatches == []


def test_run_stress_test_defaults_to_full_common_cached_range(isolated_env, tmp_path):
    config_path, _ = isolated_env
    report = fst.run_stress_test(config_path, work_dir=tmp_path / "work")

    with config_path.open() as f:
        cache_dir = Path(yaml.safe_load(f)["data"]["cache_dir"])
    raw = fst._load_watchlist_symbol_data(["AAA", "BBB"], cache_dir)
    common_dates = set.intersection(*(set(df["timestamp"].dt.date) for df in raw.values()))

    assert report.start_date == min(common_dates)
    assert report.end_date == max(common_dates)


# ---------------------------------------------------------------------------
# Safety: the real paper/ directory, config.yaml, and Telegram must never be
# touched, regardless of what this tool is pointed at.
# ---------------------------------------------------------------------------


def _snapshot(paths_root: Path) -> dict[str, float]:
    if not paths_root.exists():
        return {}
    return {str(p): p.stat().st_mtime for p in sorted(paths_root.rglob("*")) if p.is_file()}


def test_real_paper_directory_mtimes_unchanged_after_a_full_run(isolated_env, tmp_path):
    """The explicit guard the task calls for: snapshot the REAL repo paper/
    directory's file mtimes before and after a full stress-test run, and
    assert byte-for-byte (well, mtime-for-mtime) that nothing changed."""
    real_paper_dir = REPO_ROOT / "paper"
    before = _snapshot(real_paper_dir)

    config_path, _ = isolated_env
    report = fst.run_stress_test(config_path, work_dir=tmp_path / "work")

    after = _snapshot(real_paper_dir)
    assert before == after, "the real paper/ directory was modified by the stress test tool"
    assert report.paper_dir_untouched is True


def test_real_config_yaml_is_never_written_to(isolated_env, tmp_path):
    real_config_path = REPO_ROOT / "config.yaml"
    before = real_config_path.read_bytes()

    config_path, _ = isolated_env
    fst.run_stress_test(config_path, work_dir=tmp_path / "work")

    assert real_config_path.read_bytes() == before


def test_real_cache_directory_is_never_written_to(isolated_env, tmp_path):
    real_cache_dir = REPO_ROOT / "cache"
    before = _snapshot(real_cache_dir)

    config_path, _ = isolated_env
    fst.run_stress_test(config_path, work_dir=tmp_path / "work")

    after = _snapshot(real_cache_dir)
    assert before == after, "the real cache/ directory was modified by the stress test tool"


def test_telegram_is_never_called(isolated_env, tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("Telegram must never be called by the fidelity stress test tool")

    monkeypatch.setattr(paper_daily.telegram_module, "send_message", _boom)

    config_path, _ = isolated_env
    report = fst.run_stress_test(config_path, work_dir=tmp_path / "work")
    assert report.passed is True  # ran to completion without the monkeypatched send_message firing


def test_work_dir_may_never_be_the_real_paper_directory(isolated_env):
    config_path, _ = isolated_env
    with pytest.raises(ValueError):
        fst.run_stress_test(config_path, work_dir=REPO_ROOT / "paper")


def test_work_dir_temp_directory_is_cleaned_up_after_a_run(isolated_env, tmp_path):
    config_path, _ = isolated_env
    import tempfile

    before_tmp_entries = set(Path(tempfile.gettempdir()).glob("stockplayer_fidelity_stress_*"))
    fst.run_stress_test(config_path)  # no work_dir given -> auto tempdir, must self-clean
    after_tmp_entries = set(Path(tempfile.gettempdir()).glob("stockplayer_fidelity_stress_*"))
    assert after_tmp_entries == before_tmp_entries


# ---------------------------------------------------------------------------
# Smaller units
# ---------------------------------------------------------------------------


def test_load_watchlist_symbol_data_skips_symbols_with_no_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    _populate_cache(cache_dir, ["AAA"], n=50)
    result = fst._load_watchlist_symbol_data(["AAA", "NOPE"], cache_dir)
    assert set(result.keys()) == {"AAA"}


def test_mismatched_trade_keys_detects_a_corrupted_field():
    ground_truth = [
        {
            "symbol": "AAA",
            "signal_timestamp": "2021-01-04T00:00:00+05:30",
            "entry_timestamp": "2021-01-05T00:00:00+05:30",
            "exit_timestamp": "2021-01-10T00:00:00+05:30",
            "setup_id": "setup1_discovery",
            "direction": "long",
            "exit_reason": "target",
            "entry_fill_price": 100.0,
            "stop_price": 95.0,
            "target_price": 110.0,
            "exit_fill_price": 110.0,
            "r_multiple": 2.0,
            "net_pnl": 500.0,
        }
    ]
    corrupted = [dict(ground_truth[0], net_pnl=999.0)]  # a real bug would look exactly like this
    bad = fst._mismatched_trade_keys(ground_truth, corrupted)
    assert bad == {("AAA", "2021-01-05T00:00:00+05:30")}


def test_mismatched_trade_keys_empty_when_rows_match():
    ground_truth = [
        {
            "symbol": "AAA",
            "signal_timestamp": "2021-01-04T00:00:00+05:30",
            "entry_timestamp": "2021-01-05T00:00:00+05:30",
            "exit_timestamp": "2021-01-10T00:00:00+05:30",
            "setup_id": "setup1_discovery",
            "direction": "long",
            "exit_reason": "target",
            "entry_fill_price": 100.0,
            "stop_price": 95.0,
            "target_price": 110.0,
            "exit_fill_price": 110.0,
            "r_multiple": 2.0,
            "net_pnl": 500.0,
        }
    ]
    assert fst._mismatched_trade_keys(ground_truth, [dict(ground_truth[0])]) == set()


def test_format_report_includes_mismatch_detail_when_present():
    report = fst.StressTestReport(
        start_date=pd.Timestamp("2021-01-01").date(),
        end_date=pd.Timestamp("2021-12-31").date(),
        symbols=["AAA"],
        trading_days_walked=250,
        day_by_day_trades=3,
        batch_trades=2,
        common_trades=2,
        exact_matches=1,
        mismatches=["('AAA', '2021-03-01T00:00:00+05:30'): field 'net_pnl' differs -- paper=1.0 vs backtest=2.0"],
    )
    text = fst.format_report(report)
    assert "FAIL" in text
    assert "field 'net_pnl' differs" in text
    assert report.passed is False


def test_format_report_pass_when_no_mismatches():
    report = fst.StressTestReport(
        start_date=pd.Timestamp("2021-01-01").date(),
        end_date=pd.Timestamp("2021-12-31").date(),
        symbols=["AAA"],
        trading_days_walked=250,
        day_by_day_trades=2,
        batch_trades=2,
        common_trades=2,
        exact_matches=2,
        mismatches=[],
    )
    text = fst.format_report(report)
    assert "PASS" in text
    assert report.passed is True
