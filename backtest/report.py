"""Turns a flat list of ``backtest.simulator.TradeRecord`` into the numbers
that decide whether the VWAP Wave System has positive expectancy: overall,
and broken down per-setup and per-symbol.

Drawdown and losing-streak are computed on the chronological (by exit time)
net-P&L sequence *within whatever subset is being reported* -- for the
overall table that's the true combined equity curve; for a per-setup or
per-symbol breakdown it's that subset's own sub-sequence, which is a
reasonable approximation for comparison purposes but is not a curve any
single account actually traded (documented here since it's a real, if
minor, source of ambiguity).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.costs import CostConfig
from backtest.simulator import TradeRecord
from strategy.base import StrategyConfig
from strategy.engine import generate_proposals_with_funnel


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([vars(t) for t in trades])
    return df.sort_values("exit_timestamp").reset_index(drop=True)


def _max_drawdown(net_pnl_chronological: pd.Series) -> float:
    if net_pnl_chronological.empty:
        return 0.0
    equity = net_pnl_chronological.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    return float(drawdown.min())


def _longest_losing_streak(net_pnl_chronological: pd.Series) -> int:
    longest = streak = 0
    for pnl in net_pnl_chronological:
        if pnl < 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return longest


def compute_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_r_winners": 0.0,
            "avg_r_losers": 0.0,
            "expectancy_r": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown": 0.0,
            "longest_losing_streak": 0,
            "avg_duration_minutes": 0.0,
        }

    df = trades_df.sort_values("exit_timestamp")
    winners = df[df["net_pnl"] > 0]
    losers = df[df["net_pnl"] <= 0]

    gross_profit = winners["net_pnl"].sum()
    gross_loss = losers["net_pnl"].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")

    return {
        "trades": len(df),
        "win_rate": len(winners) / len(df),
        "avg_r_winners": winners["r_multiple"].mean() if not winners.empty else 0.0,
        "avg_r_losers": losers["r_multiple"].mean() if not losers.empty else 0.0,
        "expectancy_r": df["r_multiple"].mean(),
        "gross_pnl": df["gross_pnl"].sum(),
        "net_pnl": df["net_pnl"].sum(),
        "profit_factor": profit_factor,
        "max_drawdown": _max_drawdown(df["net_pnl"]),
        "longest_losing_streak": _longest_losing_streak(df["net_pnl"]),
        "avg_duration_minutes": df["duration_minutes"].mean(),
    }


@dataclass
class Report:
    overall: dict
    per_setup: pd.DataFrame
    per_symbol: pd.DataFrame
    trades_df: pd.DataFrame


_STAT_COLUMNS = [
    "trades",
    "win_rate",
    "avg_r_winners",
    "avg_r_losers",
    "expectancy_r",
    "gross_pnl",
    "net_pnl",
    "profit_factor",
    "max_drawdown",
    "longest_losing_streak",
    "avg_duration_minutes",
]


def _breakdown_table(trades_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=[group_col] + _STAT_COLUMNS)
    rows = []
    for key, group in trades_df.groupby(group_col, sort=True):
        stats = compute_stats(group)
        rows.append({group_col: key, **stats})
    return pd.DataFrame(rows, columns=[group_col] + _STAT_COLUMNS)


def build_report(trades: list[TradeRecord]) -> Report:
    trades_df = trades_to_dataframe(trades)
    overall = compute_stats(trades_df)
    per_setup = _breakdown_table(trades_df, "setup_id")
    per_symbol = _breakdown_table(trades_df, "symbol")
    return Report(overall=overall, per_setup=per_setup, per_symbol=per_symbol, trades_df=trades_df)


def _format_stats_row(stats: dict) -> dict:
    return {
        "trades": stats["trades"],
        "win_rate": f"{stats['win_rate']:.1%}",
        "avg_R_win": f"{stats['avg_r_winners']:.2f}",
        "avg_R_loss": f"{stats['avg_r_losers']:.2f}",
        "expectancy_R": f"{stats['expectancy_r']:.3f}",
        "gross_pnl": f"{stats['gross_pnl']:,.0f}",
        "net_pnl": f"{stats['net_pnl']:,.0f}",
        "profit_factor": f"{stats['profit_factor']:.2f}" if stats["profit_factor"] not in (float("inf"),) else "inf",
        "max_dd": f"{stats['max_drawdown']:,.0f}",
        "losing_streak": stats["longest_losing_streak"],
        "avg_duration_min": f"{stats['avg_duration_minutes']:.0f}",
    }


def print_report(report: Report) -> None:
    print("\n=== OVERALL ===")
    overall_df = pd.DataFrame([_format_stats_row(report.overall)])
    print(overall_df.to_string(index=False))

    print("\n=== PER SETUP ===")
    if report.per_setup.empty:
        print("(no trades)")
    else:
        rows = [{"setup_id": row["setup_id"], **_format_stats_row(row)} for _, row in report.per_setup.iterrows()]
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== PER SYMBOL ===")
    if report.per_symbol.empty:
        print("(no trades)")
    else:
        rows = [{"symbol": row["symbol"], **_format_stats_row(row)} for _, row in report.per_symbol.iterrows()]
        print(pd.DataFrame(rows).to_string(index=False))


def compute_funnel_table(
    symbol_data: dict[str, pd.DataFrame],
    strategy_cfg: StrategyConfig,
    cost_cfg: CostConfig,
    executed_by_symbol: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Diagnostic only. How many candidate signals survive each filter stage,
    per symbol and overall: ``raw`` (setup-detector candidates that clear the
    entry window + wide-band guard), ``after_stop_floor_rr``, and
    ``after_cost_viability``. If ``executed_by_symbol`` is given (trade counts
    from the simulator, keyed by symbol), an ``executed`` column is added --
    further trimmed by one-trade-at-a-time, the daily cap, and the stop-out
    cooldown, none of which the engine itself knows about."""
    rows = []
    for symbol, df in symbol_data.items():
        _, funnel = generate_proposals_with_funnel(df, symbol, strategy_cfg, cost_cfg)
        row = {"symbol": symbol, **funnel}
        if executed_by_symbol is not None:
            row["executed"] = executed_by_symbol.get(symbol, 0)
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    totals = {"symbol": "TOTAL"}
    for col in table.columns:
        if col != "symbol":
            totals[col] = int(table[col].sum())
    return pd.concat([table, pd.DataFrame([totals])], ignore_index=True)


def print_funnel_table(funnel_table: pd.DataFrame) -> None:
    print("\n=== FUNNEL (candidate signals surviving each filter stage) ===")
    if funnel_table.empty:
        print("(no data)")
        return
    print(funnel_table.to_string(index=False))


def compute_cost_risk_ratio_distribution(trades_df: pd.DataFrame) -> dict:
    """Diagnostic only. For *executed* trades: estimated round-trip cost as a
    fraction of the trade's nominal rupee risk (``total_costs / (risk_per_share
    * quantity)``), summarized. Compare against ``cost_viability_max_pct`` --
    every executed trade should sit at or under it by construction, but the
    distribution shows how close to the bar they are."""
    empty = {"mean_pct": 0.0, "median_pct": 0.0, "min_pct": 0.0, "max_pct": 0.0, "n": 0}
    if trades_df.empty:
        return empty

    risk_per_share = (trades_df["entry_signal_price"] - trades_df["stop_price"]).abs()
    nominal_risk = risk_per_share * trades_df["quantity"]
    ratio = (trades_df["total_costs"] / nominal_risk.replace(0, pd.NA)).dropna()
    if ratio.empty:
        return empty
    return {
        "mean_pct": float(ratio.mean() * 100),
        "median_pct": float(ratio.median() * 100),
        "min_pct": float(ratio.min() * 100),
        "max_pct": float(ratio.max() * 100),
        "n": len(ratio),
    }


def print_cost_risk_ratio_distribution(distribution: dict) -> None:
    print("\n=== COST/RISK RATIO OF EXECUTED TRADES (total_costs / nominal_risk) ===")
    if distribution["n"] == 0:
        print("(no trades)")
        return
    print(
        f"n={distribution['n']}  mean={distribution['mean_pct']:.1f}%  "
        f"median={distribution['median_pct']:.1f}%  min={distribution['min_pct']:.1f}%  "
        f"max={distribution['max_pct']:.1f}%"
    )


_STOP_DISTANCE_COLUMNS = ["setup_id", "avg_stop_rs", "median_stop_rs", "avg_stop_pct", "median_stop_pct"]


def compute_stop_distance_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only. Average/median stop distance (rupees and % of entry
    price) of *executed* trades, overall and per setup -- compare against
    ``compute_candle_range_diagnostics`` to see stop sizes against noise."""
    if trades_df.empty:
        return pd.DataFrame(columns=_STOP_DISTANCE_COLUMNS)

    df = trades_df.copy()
    df["stop_distance_rs"] = (df["entry_signal_price"] - df["stop_price"]).abs()
    df["stop_distance_pct"] = df["stop_distance_rs"] / df["entry_signal_price"] * 100

    def _row(label: str, group: pd.DataFrame) -> dict:
        return {
            "setup_id": label,
            "avg_stop_rs": group["stop_distance_rs"].mean(),
            "median_stop_rs": group["stop_distance_rs"].median(),
            "avg_stop_pct": group["stop_distance_pct"].mean(),
            "median_stop_pct": group["stop_distance_pct"].median(),
        }

    rows = [_row("OVERALL", df)]
    rows += [_row(setup_id, group) for setup_id, group in df.groupby("setup_id", sort=True)]
    return pd.DataFrame(rows, columns=_STOP_DISTANCE_COLUMNS)


def print_stop_distance_table(table: pd.DataFrame) -> None:
    print("\n=== STOP DISTANCE (executed trades, at entry) ===")
    if table.empty:
        print("(no trades)")
        return
    formatted = table.copy()
    formatted["avg_stop_rs"] = formatted["avg_stop_rs"].map("{:.2f}".format)
    formatted["median_stop_rs"] = formatted["median_stop_rs"].map("{:.2f}".format)
    formatted["avg_stop_pct"] = formatted["avg_stop_pct"].map("{:.3f}%".format)
    formatted["median_stop_pct"] = formatted["median_stop_pct"].map("{:.3f}%".format)
    print(formatted.to_string(index=False))


_TRADES_PER_DAY_COLUMNS = ["setup_id", "total_trades", "trades_per_day_all_symbols", "trades_per_day_per_symbol"]


def compute_trades_per_day_table(trades_df: pd.DataFrame, n_days: int, n_symbols: int) -> pd.DataFrame:
    """Diagnostic only. Trade frequency, overall and per setup, aggregated
    across all symbols in the run and averaged per symbol."""
    if trades_df.empty or n_days <= 0:
        return pd.DataFrame(columns=_TRADES_PER_DAY_COLUMNS)

    def _row(label: str, count: int) -> dict:
        return {
            "setup_id": label,
            "total_trades": count,
            "trades_per_day_all_symbols": count / n_days,
            "trades_per_day_per_symbol": (count / n_days / n_symbols) if n_symbols else 0.0,
        }

    rows = [_row("OVERALL", len(trades_df))]
    rows += [_row(setup_id, len(group)) for setup_id, group in trades_df.groupby("setup_id", sort=True)]
    return pd.DataFrame(rows, columns=_TRADES_PER_DAY_COLUMNS)


def print_trades_per_day_table(table: pd.DataFrame) -> None:
    print("\n=== TRADES PER DAY ===")
    if table.empty:
        print("(no trades)")
        return
    formatted = table.copy()
    formatted["trades_per_day_all_symbols"] = formatted["trades_per_day_all_symbols"].map("{:.3f}".format)
    formatted["trades_per_day_per_symbol"] = formatted["trades_per_day_per_symbol"].map("{:.3f}".format)
    print(formatted.to_string(index=False))


def compute_candle_range_diagnostics(symbol_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Diagnostic only -- no effect on any trading decision. Average/median
    candle range as a percentage of price, per symbol, so stop sizes can
    always be compared against the market's own intra-candle noise floor at
    whatever timeframe ``symbol_data`` is on (this is what motivated Weekend
    4's stop floor: Weekend 3's 5-min stops averaged ~0.11-0.13% of price,
    inside single-candle noise)."""
    rows = []
    for symbol, df in symbol_data.items():
        range_pct = (df["high"] - df["low"]) / df["close"] * 100
        rows.append(
            {
                "symbol": symbol,
                "avg_candle_range_pct": range_pct.mean(),
                "median_candle_range_pct": range_pct.median(),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "avg_candle_range_pct", "median_candle_range_pct"]).sort_values(
        "symbol"
    ).reset_index(drop=True)


def print_candle_range_diagnostics(diagnostics: pd.DataFrame, interval_label: str = "") -> None:
    label = f" ({interval_label})" if interval_label else ""
    print(f"\n=== CANDLE RANGE DIAGNOSTIC{label} (% of price) ===")
    if diagnostics.empty:
        print("(no data)")
        return
    formatted = diagnostics.copy()
    formatted["avg_candle_range_pct"] = formatted["avg_candle_range_pct"].map("{:.4f}%".format)
    formatted["median_candle_range_pct"] = formatted["median_candle_range_pct"].map("{:.4f}%".format)
    print(formatted.to_string(index=False))


def save_report(report: Report, out_dir: Path, candle_range_diagnostics: pd.DataFrame | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report.trades_df.to_csv(out_dir / "trades.csv", index=False)
    if candle_range_diagnostics is not None:
        candle_range_diagnostics.to_csv(out_dir / "candle_range_diagnostics.csv", index=False)

    summary_rows = [{"breakdown": "overall", "key": "overall", **report.overall}]
    summary_rows += [
        {"breakdown": "setup", "key": row["setup_id"], **{k: row[k] for k in _STAT_COLUMNS}}
        for _, row in report.per_setup.iterrows()
    ]
    summary_rows += [
        {"breakdown": "symbol", "key": row["symbol"], **{k: row[k] for k in _STAT_COLUMNS}}
        for _, row in report.per_symbol.iterrows()
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
