"""The numbers that decide go/no-go: expectancy, win rate, profit factor,
drawdown, and cost drag, overall and broken down per-setup and per-symbol.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.simulator import TradeRecord

STAT_COLUMNS = [
    "trades",
    "win_rate_pct",
    "avg_r_winners",
    "avg_r_losers",
    "expectancy_r",
    "gross_pnl",
    "net_pnl",
    "profit_factor",
    "max_drawdown_pct",
    "longest_losing_streak",
    "avg_duration_min",
]


def trades_to_dataframe(records: list[TradeRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "symbol", "timestamp", "setup_id", "direction", "entry_price", "stop_price",
                "target_price", "rr_ratio", "condition_at_entry", "acceptance_streak_at_entry",
                "notes", "entry_fill_price", "quantity", "exit_timestamp", "exit_price",
                "exit_fill_price", "exit_reason", "gross_pnl", "costs", "net_pnl", "r_multiple",
                "duration_minutes",
            ]
        )
    return pd.DataFrame([r.__dict__ for r in records])


def compute_stats(trades: pd.DataFrame, starting_capital: float) -> dict:
    """Stats for one group of trades (overall, or a per-setup/per-symbol
    slice). The equity curve used for drawdown is that group's own trades in
    their own chronological order -- a standard breakdown convention, not
    the literal shared-portfolio sequence (see simulator.py's module
    docstring on why sizing/equity isn't compounded across symbols)."""
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": float("nan"),
            "avg_r_winners": float("nan"),
            "avg_r_losers": float("nan"),
            "expectancy_r": float("nan"),
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown_pct": float("nan"),
            "longest_losing_streak": 0,
            "avg_duration_min": float("nan"),
        }

    trades = trades.sort_values("exit_timestamp").reset_index(drop=True)
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    equity = starting_capital + trades["net_pnl"].cumsum()
    running_peak = equity.cummax()
    drawdown_pct = (running_peak - equity) / running_peak * 100.0

    longest_losing_streak = 0
    current_streak = 0
    for is_loss in (trades["net_pnl"] <= 0):
        if is_loss:
            current_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_streak)
        else:
            current_streak = 0

    return {
        "trades": len(trades),
        "win_rate_pct": 100.0 * len(wins) / len(trades),
        "avg_r_winners": wins["r_multiple"].mean() if not wins.empty else float("nan"),
        "avg_r_losers": losses["r_multiple"].mean() if not losses.empty else float("nan"),
        "expectancy_r": trades["r_multiple"].mean(),
        "gross_pnl": trades["gross_pnl"].sum(),
        "net_pnl": trades["net_pnl"].sum(),
        "profit_factor": profit_factor,
        "max_drawdown_pct": drawdown_pct.max(),
        "longest_losing_streak": longest_losing_streak,
        "avg_duration_min": trades["duration_minutes"].mean(),
    }


def build_summary(trades: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    """One combined table: an 'overall' row, one row per setup, one row per
    symbol -- distinguished by the ``breakdown``/``group`` columns."""
    rows = [{"breakdown": "overall", "group": "overall", **compute_stats(trades, starting_capital)}]
    if not trades.empty:
        for setup_id, group in trades.groupby("setup_id"):
            rows.append({"breakdown": "setup", "group": setup_id, **compute_stats(group, starting_capital)})
        for symbol, group in trades.groupby("symbol"):
            rows.append({"breakdown": "symbol", "group": symbol, **compute_stats(group, starting_capital)})
    return pd.DataFrame(rows, columns=["breakdown", "group"] + STAT_COLUMNS)


def _format_for_print(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    round_map = {
        "win_rate_pct": 1,
        "avg_r_winners": 2,
        "avg_r_losers": 2,
        "expectancy_r": 3,
        "gross_pnl": 0,
        "net_pnl": 0,
        "profit_factor": 2,
        "max_drawdown_pct": 1,
        "avg_duration_min": 0,
    }
    for col, decimals in round_map.items():
        if col in df.columns:
            df[col] = df[col].round(decimals)
    return df


def print_report(summary: pd.DataFrame, starting_capital: float) -> None:
    overall = summary[summary["breakdown"] == "overall"].drop(columns=["breakdown"]).rename(columns={"group": ""})
    per_setup = summary[summary["breakdown"] == "setup"].drop(columns=["breakdown"]).rename(columns={"group": "setup_id"})
    per_symbol = summary[summary["breakdown"] == "symbol"].drop(columns=["breakdown"]).rename(columns={"group": "symbol"})

    print(f"Starting capital: Rs {starting_capital:,.0f}")
    print()
    print("=== Overall ===")
    print(_format_for_print(overall).to_string(index=False))
    print()
    print("=== Per setup ===")
    print(_format_for_print(per_setup).to_string(index=False))
    print()
    print("=== Per symbol ===")
    print(_format_for_print(per_symbol).to_string(index=False))


def save_results(trades: pd.DataFrame, summary: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(results_dir / "trades.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
