"""Session VWAP and deviation bands.

Both are cumulative *within a trading day* -- they reset at the first candle of
every session (09:15 IST) and never look across day boundaries. Callers pass in
the full multi-day candle DataFrame for a symbol (as returned by
``data.store.read_symbol``); grouping by day happens internally.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def compute_session_vwap(df: pd.DataFrame, deviation_bands: list[int] | None = None) -> pd.DataFrame:
    """Add session VWAP and deviation-band columns to ``df``.

    Adds ``vwap`` and, for each ``k`` in ``deviation_bands``, ``band_upper_{k}``
    and ``band_lower_{k}`` (``vwap + k*std`` / ``vwap - k*std``). ``std`` is the
    cumulative volume-weighted standard deviation of typical price
    ``(high+low+close)/3`` from VWAP, within the session so far.

    The first candle of each session has no prior data to average, so its
    ``vwap`` and band columns are ``NaN``; values are populated from the second
    candle of the session onward.
    """
    deviation_bands = deviation_bands if deviation_bands is not None else [1, 2]
    df = df.copy()
    day = df["timestamp"].dt.date

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]
    pv2 = typical_price.pow(2) * df["volume"]

    tmp = pd.DataFrame({"day": day, "volume": df["volume"], "pv": pv, "pv2": pv2})
    grouped = tmp.groupby("day", sort=False)
    cum = grouped[["volume", "pv", "pv2"]].cumsum()

    cum_vol = cum["volume"]
    vwap = cum["pv"] / cum_vol
    mean_sq = cum["pv2"] / cum_vol
    variance = (mean_sq - vwap.pow(2)).clip(lower=0)
    std = variance.pow(0.5)

    is_first_of_session = grouped.cumcount() == 0
    vwap = vwap.mask(is_first_of_session)

    df["vwap"] = vwap
    for k in deviation_bands:
        df[f"band_upper_{k}"] = df["vwap"] + k * std
        df[f"band_lower_{k}"] = df["vwap"] - k * std

    return df
