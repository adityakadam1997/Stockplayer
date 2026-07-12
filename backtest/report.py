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

from backtest.simulator import TradeRecord


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


def compute_candle_range_diagnostics(symbol_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Diagnostic only -- no effect on any trading decision. Average/median
    5-min candle range as a percentage of price, per symbol, so stop sizes
    can always be compared against the market's own intra-candle noise floor
    (this is what motivated Weekend 4's stop floor: Weekend 3's stops
    averaged ~0.11-0.13% of price, inside single-candle noise)."""
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


def print_candle_range_diagnostics(diagnostics: pd.DataFrame) -> None:
    print("\n=== CANDLE RANGE DIAGNOSTIC (5-min, % of price) ===")
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
