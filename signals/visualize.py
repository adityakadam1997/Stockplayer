"""Debugging aid: plot price + VWAP + deviation bands + condition for one
symbol/day, so we can eyeball whether the signals look right. Not used in
production -- just for visual sanity-checking during development.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from data import store
from signals import condition as condition_mod
from signals import vwap as vwap_mod
from signals.config import load_signals_config

REPO_ROOT = Path(__file__).resolve().parent.parent

CONDITION_COLORS = {
    condition_mod.INSIDE_VALUE: "#d6f0d6",  # light green
    condition_mod.ACCEPTED_ABOVE: "#f7d6d6",  # light red
    condition_mod.ACCEPTED_BELOW: "#f7d6d6",  # light red
}


def plot_day(
    symbol: str,
    date: str | dt.date,
    cache_dir: Path | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot price, VWAP, deviation bands, and condition shading for one session.

    Reads the symbol's cached candles, computes VWAP/bands/condition using the
    thresholds in config.yaml, and returns the resulting Figure. If
    ``output_path`` is given, also saves it there as a PNG.
    """
    with (REPO_ROOT / "config.yaml").open() as f:
        raw_config = yaml.safe_load(f)

    cache_dir = cache_dir or (REPO_ROOT / raw_config["data"]["cache_dir"])
    interval_minutes = raw_config["data"]["interval_minutes"]
    signals_config = load_signals_config(REPO_ROOT / "config.yaml")

    target_date = pd.Timestamp(date).date() if not isinstance(date, dt.date) else date

    df = store.read_symbol(symbol, interval_minutes, cache_dir)
    if df is None:
        raise ValueError(f"No cached data for {symbol} in {cache_dir}")

    day_df = df[df["timestamp"].dt.date == target_date].reset_index(drop=True)
    if day_df.empty:
        raise ValueError(f"No cached candles for {symbol} on {target_date}")

    day_df = vwap_mod.compute_session_vwap(day_df, deviation_bands=signals_config.deviation_bands)
    day_df = condition_mod.compute_condition(
        day_df,
        acceptance_candles=signals_config.acceptance_candles,
        value_area_band=signals_config.value_area_band,
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    x = day_df["timestamp"]

    _shade_condition(ax, day_df)

    ax.fill_between(
        x, day_df["band_lower_2"], day_df["band_upper_2"], color="#4c72b0", alpha=0.08, label="±2 std"
    )
    ax.fill_between(
        x, day_df["band_lower_1"], day_df["band_upper_1"], color="#4c72b0", alpha=0.15, label="±1 std (value area)"
    )
    ax.plot(x, day_df["close"], color="#333333", linewidth=1.3, label="close")
    ax.plot(x, day_df["vwap"], color="#c44e52", linewidth=1.5, label="VWAP")

    ax.set_title(f"{symbol} -- {target_date.isoformat()}")
    ax.set_xlabel("time (IST)")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=120)

    return fig


def _shade_condition(ax: plt.Axes, day_df: pd.DataFrame) -> None:
    """Tint the plot background by condition, one axvspan per contiguous run."""
    conditions = day_df["condition"]
    x = day_df["timestamp"]
    start_idx = 0
    for i in range(1, len(conditions) + 1):
        if i == len(conditions) or conditions.iloc[i] != conditions.iloc[start_idx]:
            color = CONDITION_COLORS.get(conditions.iloc[start_idx])
            if color is not None:
                span_end = x.iloc[i] if i < len(conditions) else x.iloc[-1] + pd.Timedelta(minutes=5)
                ax.axvspan(x.iloc[start_idx], span_end, color=color, zorder=0)
            start_idx = i
