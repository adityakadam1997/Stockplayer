"""Reusable logic for the watchlist-candidate vetting pipeline
(``scripts/vet_candidates.py``) -- factored out of the CLI script so the
IS/OOS split math and report rendering are independently testable on a
synthetic fixture.

Everything here calls straight into the existing, unmodified
``signals``/``strategy``/``backtest`` modules exactly the way
``scripts/run_swing_backtest.py`` does (truncate the date range FIRST,
THEN compute weekly/monthly VWAP, condition, ATR, rolling-high on the
truncated slice) -- this module adds zero new strategy parameters and
changes no existing ones. Its only job is (a) picking the IS/OOS split
dates per symbol, (b) running that exact pipeline per window, and
(c) turning ``backtest.swing_report``'s existing stats/funnel/cost-risk
outputs into markdown.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from backtest import costs_delivery, swing_report, swing_simulator
from backtest.swing_report import SwingReport
from signals import condition, vwap
from signals.config import SignalsConfig
from strategy import swing_engine
from strategy.base import StrategyConfig, compute_atr

FIVE_YEAR_DAYS = 365 * 5
OOS_MONTHS = 18


@dataclass
class DataCoverage:
    symbol: str
    first_date: dt.date
    last_date: dt.date
    trading_days: int
    limited_history: bool  # True if total span < FIVE_YEAR_DAYS


@dataclass
class SplitDates:
    is_start: dt.date
    is_end: dt.date
    oos_start: dt.date
    oos_end: dt.date


@dataclass
class WindowResult:
    report: SwingReport
    funnel_table: pd.DataFrame
    cost_risk_distribution: dict
    max_dd_pct_of_capital: float


def compute_coverage(symbol: str, df: pd.DataFrame) -> DataCoverage:
    dates = df["timestamp"].dt.date
    first_date, last_date = dates.min(), dates.max()
    span_days = (last_date - first_date).days
    return DataCoverage(
        symbol=symbol,
        first_date=first_date,
        last_date=last_date,
        trading_days=len(df),
        limited_history=span_days < FIVE_YEAR_DAYS,
    )


def compute_split_dates(first_date: dt.date, last_date: dt.date) -> SplitDates:
    """Mirrors Cycle 3B's actual split (in-sample 2021-07-09 -> 2025-01-07,
    out-of-sample 2025-01-08 -> 2026-07-10 -- an ~18-month OOS window out of
    a ~5-year total). For a symbol with less than 5 years of history, a
    fixed 18-month OOS window could swallow most or all of it, so the same
    ~30% OOS proportion is used instead, applied to whatever history
    actually exists."""
    span_days = (last_date - first_date).days
    if span_days >= FIVE_YEAR_DAYS:
        oos_start = (pd.Timestamp(last_date) - pd.DateOffset(months=OOS_MONTHS)).date()
    else:
        oos_start = first_date + dt.timedelta(days=round(span_days * 0.7))

    # Clamp: always leave at least one day on each side.
    if oos_start <= first_date:
        oos_start = first_date + dt.timedelta(days=1)
    if oos_start > last_date:
        oos_start = last_date

    return SplitDates(
        is_start=first_date,
        is_end=oos_start - dt.timedelta(days=1),
        oos_start=oos_start,
        oos_end=last_date,
    )


def compute_indicators(df: pd.DataFrame, signals_cfg: SignalsConfig) -> pd.DataFrame:
    """Identical call sequence to ``scripts/run_swing_backtest.py``'s
    per-symbol block: weekly VWAP + deviation bands, monthly VWAP (trend
    filter context), the periodic condition classifier reset on week
    boundaries, ATR14, and the prior-N-trading-day high Cycle 3B's setup1
    target recomputation needs. Called AFTER truncating ``df`` to a date
    window, never before, so indicators warm up fresh at the window's own
    start -- exactly how Cycle 3B's own IS/OOS evaluation was run."""
    df = vwap.compute_weekly_vwap(df, deviation_bands=signals_cfg.deviation_bands)
    monthly = vwap.compute_monthly_vwap(
        df[["timestamp", "open", "high", "low", "close", "volume"]], deviation_bands=[1]
    )
    df["monthly_vwap"] = monthly["vwap"]
    week_period = df["timestamp"].dt.tz_localize(None).dt.to_period("W-SUN")
    df = condition.compute_condition_periodic(
        df,
        period_key=week_period,
        acceptance_candles=signals_cfg.acceptance_candles,
        value_area_band=signals_cfg.value_area_band,
    )
    df = compute_atr(df, period=14)
    df[swing_engine.ROLLING_HIGH_COLUMN] = swing_engine.compute_prior_n_day_high(df)
    return df


def slice_by_date(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    day = df["timestamp"].dt.date
    mask = (day >= start) & (day <= end)
    return df.loc[mask].reset_index(drop=True)


def run_single_symbol_window(
    symbol: str,
    df_window: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    cost_cfg: costs_delivery.DeliveryCostConfig,
    portfolio_cfg: swing_simulator.SwingPortfolioConfig,
) -> WindowResult:
    """Runs the identical Cycle 3B pipeline for ONE symbol in isolation --
    no portfolio-cap competition from other symbols (that's Step 4's
    combined backtest, a separate later exercise). ``max_positions_per_symbol``
    still applies (unchanged from config.yaml), but with only one symbol in
    play ``max_concurrent_positions`` can never bind."""
    symbol_data = {symbol: df_window}
    trades = swing_simulator.run_portfolio(symbol_data, strategy_cfg, cost_cfg, portfolio_cfg)
    report = swing_report.build_report(trades)
    funnel_table = swing_report.compute_funnel_table(symbol_data, strategy_cfg, cost_cfg, {symbol: len(trades)})
    cost_risk_distribution = swing_report.compute_cost_risk_ratio_distribution(report.trades_df)
    max_dd_pct = swing_report.compute_max_drawdown_pct_of_capital(report, strategy_cfg.capital)
    return WindowResult(
        report=report,
        funnel_table=funnel_table,
        cost_risk_distribution=cost_risk_distribution,
        max_dd_pct_of_capital=max_dd_pct,
    )


# ---------------------------------------------------------------------------
# Tiering (documented thresholds, applied consistently -- not tuned per
# symbol). Only ever looks at OUT-OF-SAMPLE numbers, matching the project's
# standing rule that in-sample is context, not the verdict.
# ---------------------------------------------------------------------------

TIER_COMPARABLE = "comparable to current watchlist"
TIER_MARGINAL = "marginal -- thin sample or borderline cost/risk"
TIER_FAILS = "fails cost-viability or insufficient valid trades"
TIER_UNRESOLVED = "unresolved"

MIN_TRADES_FOR_COMPARABLE = 5
MAX_COST_RISK_PCT_COMPARABLE = 15.0
MAX_COST_RISK_PCT_BEFORE_FAIL = 18.0
MIN_EXPECTANCY_BEFORE_FAIL = -0.10


def classify_tier(oos_stats: dict, oos_cost_risk: dict) -> str:
    """Deterministic, documented thresholds -- see SUMMARY.md's own
    rendering of this same text. The existing 15-symbol watchlist's known
    OOS cost/risk range (Cycle 3B) is roughly 6-18%; the two
    MAX_COST_RISK_PCT_* constants are set around that observed range, not
    tuned per candidate."""
    trades = oos_stats["trades"]
    if trades == 0:
        return TIER_FAILS

    cost_risk_mean = oos_cost_risk["mean_pct"] if oos_cost_risk["n"] > 0 else 0.0
    if cost_risk_mean >= MAX_COST_RISK_PCT_BEFORE_FAIL:
        return TIER_FAILS
    if oos_stats["expectancy_r"] <= MIN_EXPECTANCY_BEFORE_FAIL:
        return TIER_FAILS

    profit_factor = oos_stats["profit_factor"]
    pf_ok = profit_factor == float("inf") or profit_factor >= 1.0
    if (
        trades >= MIN_TRADES_FOR_COMPARABLE
        and oos_stats["expectancy_r"] > 0
        and pf_ok
        and cost_risk_mean < MAX_COST_RISK_PCT_COMPARABLE
    ):
        return TIER_COMPARABLE

    return TIER_MARGINAL


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

STEP4_CAVEAT = (
    "**This is a FIRST FILTER only.** Every result above comes from a "
    "single-symbol backtest run in isolation -- no competition for the "
    "portfolio's `max_concurrent_positions` cap (5) or `max_positions_per_symbol` "
    "cap, and no correlation effects between symbols. A symbol that looks strong "
    "here could still add little (or even reduce net expectancy) once it has to "
    "compete for the same 5 concurrent slots as the other candidates and the "
    "existing 15, or if its signals cluster in time with symbols already held. "
    "A COMBINED portfolio backtest -- every promoted candidate plus the existing "
    "15, together, respecting the real caps, run once -- is required before any "
    "adoption decision. That combined run is deliberately NOT part of this "
    "pipeline; it should be a separate, later, pre-registered step, timed near "
    "the Phase 1 review."
)


def _fmt_pf(pf) -> str:
    if isinstance(pf, str):
        return pf
    if pf == float("inf"):
        return "inf"
    if pf != pf:  # NaN -- no trades, profit factor is undefined
        return "n/a"
    return f"{pf:.2f}"


def _fmt_stats_block(stats: dict) -> list[str]:
    return [
        f"- Trades: {stats['trades']}",
        f"- Win rate: {stats['win_rate']:.1%}",
        f"- Expectancy: {stats['expectancy_r']:+.3f}R",
        f"- Profit factor: {_fmt_pf(stats['profit_factor'])}",
        f"- Max drawdown: Rs{stats['max_drawdown']:,.0f}",
        f"- Avg holding days: {stats['avg_holding_days']:.1f}",
    ]


def render_symbol_report_md(
    symbol: str,
    coverage: DataCoverage,
    split: SplitDates,
    is_result: WindowResult,
    oos_result: WindowResult,
    tier: str,
) -> str:
    lines = [f"# {symbol} -- candidate vetting report", ""]

    lines.append("## Data coverage")
    if coverage.limited_history:
        lines.append(
            f"**LIMITED HISTORY** -- less than 5 years available: "
            f"{coverage.first_date} to {coverage.last_date} ({coverage.trading_days} trading days)."
        )
    else:
        lines.append(
            f"{coverage.first_date} to {coverage.last_date} ({coverage.trading_days} trading days)."
        )
    lines.append("")

    lines.append("## IS/OOS split")
    lines.append(f"- In-sample: {split.is_start} to {split.is_end}")
    lines.append(f"- Out-of-sample: {split.oos_start} to {split.oos_end}")
    lines.append("")

    lines.append("## In-sample results")
    lines.extend(_fmt_stats_block(is_result.report.overall))
    lines.append("")

    lines.append("## Out-of-sample results")
    lines.extend(_fmt_stats_block(oos_result.report.overall))
    lines.append("")

    lines.append("## Cost-viability funnel (out-of-sample)")
    lines.append("```")
    lines.append(oos_result.funnel_table.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Cost/risk ratio of executed trades (out-of-sample)")
    cr = oos_result.cost_risk_distribution
    if cr["n"] == 0:
        lines.append("(no executed trades)")
    else:
        lines.append(
            f"n={cr['n']}  mean={cr['mean_pct']:.1f}%  median={cr['median_pct']:.1f}%  "
            f"min={cr['min_pct']:.1f}%  max={cr['max_pct']:.1f}%  "
            f"(existing 15-symbol watchlist's known Cycle 3B range: ~6-18%)"
        )
    lines.append("")

    lines.append(f"## Tier: {tier}")
    lines.append("")

    return "\n".join(lines) + "\n"


def render_summary_md(rows: list[dict], unresolved: list[str]) -> str:
    """``rows`` -- one dict per resolved symbol, each with keys: symbol,
    oos_trades, oos_expectancy_r, oos_profit_factor, oos_cost_risk_avg_pct,
    coverage_label, tier. Ranked by ``oos_expectancy_r`` descending (symbols
    with zero OOS trades, i.e. TIER_FAILS with no trades, sort last)."""
    lines = ["# Candidate watchlist vetting -- SUMMARY", ""]

    lines.append("## Caveat")
    lines.append(STEP4_CAVEAT)
    lines.append("")

    lines.append("## Resolved symbols, ranked by out-of-sample expectancy_R")
    lines.append("")
    lines.append("| Symbol | OOS trades | OOS expectancy_R | OOS PF | OOS cost/risk avg % | Coverage | Tier |")
    lines.append("|---|---|---|---|---|---|---|")

    ranked = sorted(rows, key=lambda r: (r["oos_trades"] == 0, -r["oos_expectancy_r"]))
    for r in ranked:
        lines.append(
            f"| {r['symbol']} | {r['oos_trades']} | {r['oos_expectancy_r']:+.3f} | "
            f"{_fmt_pf(r['oos_profit_factor'])} | {r['oos_cost_risk_avg_pct']:.1f}% | "
            f"{r['coverage_label']} | {r['tier']} |"
        )
    lines.append("")

    lines.append("## Unresolved symbols")
    if unresolved:
        lines.append(
            "Could not resolve an instrument key via assets.upstox.com, the "
            "api.upstox.com symbol-search path, or the web-search-plus-live-"
            "verification fallback. Not guessed -- not included anywhere above."
        )
        lines.append("")
        for symbol in unresolved:
            lines.append(f"- {symbol}")
    else:
        lines.append("None -- all 51 candidates resolved to a verified instrument key.")
    lines.append("")

    return "\n".join(lines) + "\n"
