"""Swing-trade equivalent of ``backtest.report`` -- same win-rate/expectancy/
profit-factor/drawdown math (that logic doesn't care about time units), but
keyed on ``holding_days`` instead of ``duration_minutes``, plus swing-specific
diagnostics (trades/month, funnel with the trend-filter stage, cost/risk
distribution against the delivery cost model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.costs_delivery import DeliveryCostConfig
from backtest.swing_simulator import SwingTradeRecord
from strategy.base import StrategyConfig
from strategy.swing_engine import generate_proposals_with_funnel


def trades_to_dataframe(trades: list[SwingTradeRecord]) -> pd.DataFrame:
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
            "avg_holding_days": 0.0,
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
        "avg_holding_days": df["holding_days"].mean(),
    }


@dataclass
class SwingReport:
    overall: dict
    per_setup: pd.DataFrame
    per_symbol: pd.DataFrame
    per_direction: pd.DataFrame
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
    "avg_holding_days",
]


def _breakdown_table(trades_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=[group_col] + _STAT_COLUMNS)
    rows = []
    for key, group in trades_df.groupby(group_col, sort=True):
        stats = compute_stats(group)
        rows.append({group_col: key, **stats})
    return pd.DataFrame(rows, columns=[group_col] + _STAT_COLUMNS)


def build_report(trades: list[SwingTradeRecord]) -> SwingReport:
    trades_df = trades_to_dataframe(trades)
    overall = compute_stats(trades_df)
    per_setup = _breakdown_table(trades_df, "setup_id")
    per_symbol = _breakdown_table(trades_df, "symbol")
    per_direction = _breakdown_table(trades_df, "direction")
    return SwingReport(
        overall=overall, per_setup=per_setup, per_symbol=per_symbol, per_direction=per_direction, trades_df=trades_df
    )


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
        "avg_holding_days": f"{stats['avg_holding_days']:.1f}",
    }


def print_report(report: SwingReport) -> None:
    print("\n=== OVERALL ===")
    print(pd.DataFrame([_format_stats_row(report.overall)]).to_string(index=False))

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

    print("\n=== PER DIRECTION (long vs short -- see costs_delivery.py's cash-delivery short-selling caveat) ===")
    if report.per_direction.empty:
        print("(no trades)")
    else:
        rows = [{"direction": row["direction"], **_format_stats_row(row)} for _, row in report.per_direction.iterrows()]
        print(pd.DataFrame(rows).to_string(index=False))


def compute_max_drawdown_pct_of_capital(report: SwingReport, capital: float) -> float:
    """The overall max drawdown as a percentage of starting capital -- what
    the GO/NO-GO verdict's ``max drawdown <= 20% of capital`` bar checks."""
    if capital <= 0:
        return 0.0
    return abs(report.overall["max_drawdown"]) / capital * 100


def compute_funnel_table(
    symbol_data: dict[str, pd.DataFrame],
    strategy_cfg: StrategyConfig,
    cost_cfg: DeliveryCostConfig,
    executed_by_symbol: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Diagnostic only. Candidate signals surviving each filter stage, per
    symbol and overall -- ``raw`` (post wide-band-guard), ``after_trend_filter``,
    ``after_stop_floor_rr``, ``after_cost_viability``, and (if given) ``executed``
    (further trimmed by next-day-fill availability and portfolio capacity,
    neither of which the engine itself knows about)."""
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


_HOLDING_PERIOD_COLUMNS = ["setup_id", "avg_holding_days", "median_holding_days"]


def compute_holding_period_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=_HOLDING_PERIOD_COLUMNS)

    def _row(label: str, group: pd.DataFrame) -> dict:
        return {
            "setup_id": label,
            "avg_holding_days": group["holding_days"].mean(),
            "median_holding_days": group["holding_days"].median(),
        }

    rows = [_row("OVERALL", trades_df)]
    rows += [_row(setup_id, group) for setup_id, group in trades_df.groupby("setup_id", sort=True)]
    return pd.DataFrame(rows, columns=_HOLDING_PERIOD_COLUMNS)


def print_holding_period_table(table: pd.DataFrame) -> None:
    print("\n=== HOLDING PERIOD (trading days) ===")
    if table.empty:
        print("(no trades)")
        return
    formatted = table.copy()
    formatted["avg_holding_days"] = formatted["avg_holding_days"].map("{:.2f}".format)
    formatted["median_holding_days"] = formatted["median_holding_days"].map("{:.2f}".format)
    print(formatted.to_string(index=False))


_TRADES_PER_MONTH_COLUMNS = ["setup_id", "total_trades", "trades_per_month_all_symbols", "trades_per_month_per_symbol"]


def compute_trades_per_month_table(trades_df: pd.DataFrame, n_months: float, n_symbols: int) -> pd.DataFrame:
    if trades_df.empty or n_months <= 0:
        return pd.DataFrame(columns=_TRADES_PER_MONTH_COLUMNS)

    def _row(label: str, count: int) -> dict:
        return {
            "setup_id": label,
            "total_trades": count,
            "trades_per_month_all_symbols": count / n_months,
            "trades_per_month_per_symbol": (count / n_months / n_symbols) if n_symbols else 0.0,
        }

    rows = [_row("OVERALL", len(trades_df))]
    rows += [_row(setup_id, len(group)) for setup_id, group in trades_df.groupby("setup_id", sort=True)]
    return pd.DataFrame(rows, columns=_TRADES_PER_MONTH_COLUMNS)


def print_trades_per_month_table(table: pd.DataFrame) -> None:
    print("\n=== TRADES PER MONTH ===")
    if table.empty:
        print("(no trades)")
        return
    formatted = table.copy()
    formatted["trades_per_month_all_symbols"] = formatted["trades_per_month_all_symbols"].map("{:.2f}".format)
    formatted["trades_per_month_per_symbol"] = formatted["trades_per_month_per_symbol"].map("{:.3f}".format)
    print(formatted.to_string(index=False))


def save_report(report: SwingReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report.trades_df.to_csv(out_dir / "trades.csv", index=False)

    summary_rows = [{"breakdown": "overall", "key": "overall", **report.overall}]
    summary_rows += [
        {"breakdown": "setup", "key": row["setup_id"], **{k: row[k] for k in _STAT_COLUMNS}}
        for _, row in report.per_setup.iterrows()
    ]
    summary_rows += [
        {"breakdown": "symbol", "key": row["symbol"], **{k: row[k] for k in _STAT_COLUMNS}}
        for _, row in report.per_symbol.iterrows()
    ]
    summary_rows += [
        {"breakdown": "direction", "key": row["direction"], **{k: row[k] for k in _STAT_COLUMNS}}
        for _, row in report.per_direction.iterrows()
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
