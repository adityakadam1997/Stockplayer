"""Unit tests for strategy/*.py -- no network access, no dependency on the
local cache/ directory. Synthetic data only, matching this repo's existing
test philosophy (see tests/test_signals.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.condition import ACCEPTED_ABOVE, ACCEPTED_BELOW, INSIDE_VALUE, compute_condition
from signals.vwap import compute_session_vwap
from strategy import engine, setup1_discovery, setup2_fade, setup3_return, setup4_bounce
from strategy.base import LONG, SETUP3_RETURN_FALLBACK, SETUP3_RETURN_PRIMARY, SHORT, StrategyConfig, passes_rr_filter


def _session(date: str, rows: list[dict]) -> pd.DataFrame:
    """Build a session DataFrame with explicit, hand-picked signal columns
    (vwap/bands/condition/acceptance_streak set directly) so a setup's
    pattern-recognition logic can be tested in isolation from signals/*."""
    defaults = {"open": None, "high": None, "low": None, "condition": INSIDE_VALUE, "acceptance_streak": 0}
    out = []
    for i, row in enumerate(rows):
        r = {**defaults, **row}
        if r["open"] is None:
            r["open"] = r["close"]
        if r["high"] is None:
            r["high"] = max(r["open"], r["close"])
        if r["low"] is None:
            r["low"] = min(r["open"], r["close"])
        r["timestamp"] = pd.Timestamp(f"{date} 09:{15 + i * 5:02d}", tz="Asia/Kolkata")
        out.append(r)
    return pd.DataFrame(out)


DEFAULT_CONFIG = StrategyConfig()


# ---------------------------------------------------------------------------
# Setup 1 -- Price Discovery Continuation
# ---------------------------------------------------------------------------


def test_setup1_triggers_on_pullback_and_close_back_above_band() -> None:
    df = _session(
        "2026-01-05",
        [
            {"close": 109, "open": 108, "high": 110, "low": 108, "band_upper_1": 100, "condition": ACCEPTED_ABOVE},
            {"close": 111, "open": 110, "high": 112, "low": 110, "band_upper_1": 101, "condition": ACCEPTED_ABOVE},
            {"close": 104, "open": 103, "high": 104, "low": 101.8, "band_upper_1": 102, "condition": ACCEPTED_ABOVE},
        ],
    )
    proposal = setup1_discovery.detect(df, "TEST", DEFAULT_CONFIG)
    assert proposal is not None
    assert proposal.direction == LONG
    assert proposal.entry_price == 104
    assert proposal.target_price == 112  # session high reached during the acceptance phase, before this candle
    assert proposal.rr_ratio >= DEFAULT_CONFIG.min_rr


def test_setup1_does_not_trigger_on_single_wick_without_acceptance() -> None:
    # A single candle that pokes above where a band would be but reverts is
    # NOT acceptance -- the Weekend 2 classifier would keep this inside_value,
    # so setup 1 (which requires condition == accepted_above/below) must not
    # fire here.
    df = _session(
        "2026-01-05",
        [
            {"close": 100, "band_upper_1": 105, "condition": INSIDE_VALUE},
            {"close": 100.5, "open": 100, "high": 106, "low": 100, "band_upper_1": 105, "condition": INSIDE_VALUE},
        ],
    )
    assert setup1_discovery.detect(df, "TEST", DEFAULT_CONFIG) is None


def test_setup1_skips_when_rr_filter_fails() -> None:
    # Same shape as the triggering case, but the acceptance-phase extreme is
    # only barely above entry -- not enough reward for the risk taken.
    df = _session(
        "2026-01-05",
        [
            {"close": 105, "high": 106, "low": 104, "band_upper_1": 100, "condition": ACCEPTED_ABOVE},
            {"close": 104, "open": 103, "high": 104, "low": 101.8, "band_upper_1": 102, "condition": ACCEPTED_ABOVE},
        ],
    )
    assert setup1_discovery.detect(df, "TEST", DEFAULT_CONFIG) is None


# ---------------------------------------------------------------------------
# Setup 2 -- Fade Value Area Extremes
# ---------------------------------------------------------------------------


def test_setup2_triggers_on_rejection_at_upper_band_with_rotation() -> None:
    df = _session(
        "2026-01-05",
        [
            {"close": 101, "vwap": 100},
            {"close": 99, "vwap": 100},  # crossing 1
            {"close": 101, "vwap": 100},  # crossing 2
            {"close": 100.5, "vwap": 100},
            {
                "close": 102,
                "open": 103.5,
                "high": 104,
                "low": 99,
                "vwap": 95,
                "band_upper_1": 103.5,
                "band_lower_1": 85,
                "condition": INSIDE_VALUE,
            },
        ],
    )
    proposal = setup2_fade.detect(df, "TEST", DEFAULT_CONFIG)
    assert proposal is not None
    assert proposal.direction == SHORT
    assert proposal.target_price == 95
    assert proposal.rr_ratio >= DEFAULT_CONFIG.min_rr


def test_setup2_does_not_trigger_without_enough_rotation() -> None:
    df = _session(
        "2026-01-05",
        [
            {"close": 90, "vwap": 100},
            {"close": 91, "vwap": 100},
            {
                "close": 102,
                "open": 103.5,
                "high": 104,
                "low": 99,
                "vwap": 95,
                "band_upper_1": 103.5,
                "band_lower_1": 85,
                "condition": INSIDE_VALUE,
            },
        ],
    )
    assert setup2_fade.detect(df, "TEST", DEFAULT_CONFIG) is None


def test_setup2_requires_inside_value_condition() -> None:
    df = _session(
        "2026-01-05",
        [
            {"close": 101, "vwap": 100},
            {"close": 99, "vwap": 100},
            {"close": 101, "vwap": 100},
            {
                "close": 102,
                "open": 103.5,
                "high": 104,
                "low": 99,
                "vwap": 95,
                "band_upper_1": 103.5,
                "band_lower_1": 85,
                "condition": ACCEPTED_ABOVE,
            },
        ],
    )
    assert setup2_fade.detect(df, "TEST", DEFAULT_CONFIG) is None


# ---------------------------------------------------------------------------
# Setup 3 -- Return to Value
# ---------------------------------------------------------------------------


def _setup3_base_rows() -> list[dict]:
    return [
        {"close": 109, "high": 110, "low": 108, "condition": ACCEPTED_ABOVE},
        {  # the break-back-inside ("flip") candle
            "close": 98,
            "open": 101,
            "high": 101,
            "low": 97,
            "band_upper_1": 100,
            "vwap": 90,
            "condition": INSIDE_VALUE,
        },
    ]


def test_setup3_fallback_triggers_on_flip_candle_when_allowed() -> None:
    df = _session("2026-01-05", _setup3_base_rows())
    config = StrategyConfig(setup3_fallback_allowed=True)
    proposal = setup3_return.detect(df, "TEST", config)
    assert proposal is not None
    assert proposal.setup_id == SETUP3_RETURN_FALLBACK
    assert proposal.direction == SHORT
    assert proposal.entry_price == 98


def test_setup3_fallback_disabled_by_default_does_not_trigger() -> None:
    df = _session("2026-01-05", _setup3_base_rows())
    assert setup3_return.detect(df, "TEST", DEFAULT_CONFIG) is None


def test_setup3_primary_triggers_on_later_retest_and_hold() -> None:
    rows = _setup3_base_rows() + [
        {  # retest of the band from inside, holds, closes back down (red)
            "close": 97,
            "open": 98.5,
            "high": 99.5,
            "low": 96.5,
            "band_upper_1": 99,
            "vwap": 90,
            "condition": INSIDE_VALUE,
        }
    ]
    df = _session("2026-01-05", rows)
    proposal = setup3_return.detect(df, "TEST", DEFAULT_CONFIG)
    assert proposal is not None
    assert proposal.setup_id == SETUP3_RETURN_PRIMARY
    assert proposal.direction == SHORT
    assert proposal.entry_price == 97


def test_setup3_no_trigger_without_a_prior_acceptance_phase() -> None:
    df = _session(
        "2026-01-05",
        [
            {"close": 100, "vwap": 99, "band_upper_1": 103, "condition": INSIDE_VALUE},
            {"close": 100.5, "vwap": 99, "band_upper_1": 103, "condition": INSIDE_VALUE},
        ],
    )
    assert setup3_return.detect(df, "TEST", DEFAULT_CONFIG) is None


# ---------------------------------------------------------------------------
# Setup 4 -- VWAP Bounce
# ---------------------------------------------------------------------------


def test_setup4_triggers_on_bounce_away_from_vwap() -> None:
    rows = _setup3_base_rows() + [
        {
            "close": 100.5,
            "open": 99.95,
            "high": 100.6,
            "low": 99.9,
            "vwap": 100,
            "band_upper_1": 103,
            "band_lower_1": 97,
            "condition": INSIDE_VALUE,
        }
    ]
    df = _session("2026-01-05", rows)
    proposal = setup4_bounce.detect(df, "TEST", DEFAULT_CONFIG)
    assert proposal is not None
    assert proposal.direction == LONG
    assert proposal.target_price == 103


def test_setup4_no_trigger_without_prior_acceptance_phase() -> None:
    df = _session(
        "2026-01-05",
        [
            {
                "close": 100.5,
                "open": 99.95,
                "high": 100.6,
                "low": 99.9,
                "vwap": 100,
                "band_upper_1": 103,
                "band_lower_1": 97,
                "condition": INSIDE_VALUE,
            }
        ],
    )
    assert setup4_bounce.detect(df, "TEST", DEFAULT_CONFIG) is None


# ---------------------------------------------------------------------------
# R:R filter
# ---------------------------------------------------------------------------


def test_rr_filter_rejects_below_threshold() -> None:
    assert passes_rr_filter(1.4, min_rr=1.5) is False


def test_rr_filter_accepts_above_threshold() -> None:
    assert passes_rr_filter(1.6, min_rr=1.5) is True


# ---------------------------------------------------------------------------
# No-lookahead
# ---------------------------------------------------------------------------


def _synthetic_multiday_candles(seed: int = 7, days: int = 3, candles_per_day: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for d in range(days):
        date = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d * 2)  # skip weekends loosely
        for i in range(candles_per_day):
            drift = 0.15 if d % 2 == 0 else -0.1  # a trend leg most days, to get acceptance phases
            price += drift + rng.normal(0, 0.3)
            open_ = price
            close = price + rng.normal(0, 0.25)
            high = max(open_, close) + abs(rng.normal(0, 0.2))
            low = min(open_, close) - abs(rng.normal(0, 0.2))
            price = close
            ts = date + pd.Timedelta(hours=9, minutes=15 + i * 5)
            rows.append(
                {
                    "timestamp": ts.tz_localize("Asia/Kolkata"),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": int(rng.integers(1000, 5000)),
                }
            )
    return pd.DataFrame(rows)


def _all_proposals(signals_df: pd.DataFrame, symbol: str, config: StrategyConfig) -> list:
    """Signal-generation only (ignores open-position state) across every
    candle, day by day -- exactly what the no-lookahead property is about."""
    results = []
    day = signals_df["timestamp"].dt.date
    for _, day_df in signals_df.groupby(day, sort=False):
        day_df = day_df.reset_index(drop=True)
        for i in range(len(day_df)):
            session_so_far = day_df.iloc[: i + 1]
            proposals = engine.generate_proposals(session_so_far, symbol, config, has_open_position=False)
            results.append(tuple((p.setup_id, p.direction, p.entry_price, p.stop_price, p.target_price) for p in proposals))
    return results


def test_no_lookahead_truncated_dataset_matches_full_dataset() -> None:
    raw = _synthetic_multiday_candles()
    config = StrategyConfig()

    def signals(raw_df: pd.DataFrame) -> pd.DataFrame:
        return compute_condition(compute_session_vwap(raw_df))

    full_proposals = _all_proposals(signals(raw), "TEST", config)

    truncate_at = 55  # partway into day 2
    truncated_raw = raw.iloc[: truncate_at + 1]
    truncated_proposals = _all_proposals(signals(truncated_raw), "TEST", config)

    assert len(truncated_proposals) == truncate_at + 1
    assert truncated_proposals == full_proposals[: truncate_at + 1]
    # Sanity: the synthetic data actually exercises the engine (otherwise
    # this test would trivially pass with nothing ever generated).
    assert any(p for p in full_proposals)
