"""Three-state market condition classifier: inside_value / accepted_above / accepted_below.

Operates on the output of ``signals.vwap.compute_session_vwap``. "Acceptance" is
about persistence: price outside the value-area band only becomes
accepted_above/accepted_below once it has closed there for
``acceptance_candles`` consecutive candles in the same session, or once price
has touched the "extreme" deviation band at any point in the session -- a
single candle poking outside the value area and immediately reverting is not
acceptance, so it stays classified as inside_value until one of those two
triggers fires.
"""

from __future__ import annotations

import pandas as pd

INSIDE_VALUE = "inside_value"
ACCEPTED_ABOVE = "accepted_above"
ACCEPTED_BELOW = "accepted_below"


def compute_condition(
    df: pd.DataFrame,
    acceptance_candles: int = 3,
    value_area_band: int = 1,
    extreme_band: int = 2,
) -> pd.DataFrame:
    """Add ``condition`` and ``acceptance_streak`` columns to ``df``.

    ``df`` must already have ``band_upper_{value_area_band}``,
    ``band_lower_{value_area_band}``, ``band_upper_{extreme_band}``, and
    ``band_lower_{extreme_band}`` columns (from ``compute_session_vwap``).

    ``acceptance_streak`` is the signed count of consecutive candles closed
    outside the value-area band in the current session (positive above,
    negative below), reset to 0 whenever price closes back inside the value
    area or a new session starts.
    """
    df = df.copy()
    upper_va = f"band_upper_{value_area_band}"
    lower_va = f"band_lower_{value_area_band}"
    upper_ext = f"band_upper_{extreme_band}"
    lower_ext = f"band_lower_{extreme_band}"

    day = df["timestamp"].dt.date
    conditions = pd.Series(INSIDE_VALUE, index=df.index, dtype=object)
    streaks = pd.Series(0, index=df.index, dtype=int)

    for _, group in df.groupby(day, sort=False):
        streak = 0
        touched_extreme_above = False
        touched_extreme_below = False
        group_conditions = []
        group_streaks = []

        for row in group.itertuples():
            upper = getattr(row, upper_va)
            lower = getattr(row, lower_va)
            ext_upper = getattr(row, upper_ext)
            ext_lower = getattr(row, lower_ext)

            if pd.isna(upper) or pd.isna(lower):
                # First candle of the session: no bands yet, nothing to classify.
                streak = 0
                group_conditions.append(INSIDE_VALUE)
                group_streaks.append(streak)
                continue

            if not pd.isna(ext_upper) and row.high >= ext_upper:
                touched_extreme_above = True
            if not pd.isna(ext_lower) and row.low <= ext_lower:
                touched_extreme_below = True

            if row.close > upper:
                streak = streak + 1 if streak > 0 else 1
                cond = ACCEPTED_ABOVE if (streak >= acceptance_candles or touched_extreme_above) else INSIDE_VALUE
            elif row.close < lower:
                streak = streak - 1 if streak < 0 else -1
                cond = ACCEPTED_BELOW if (-streak >= acceptance_candles or touched_extreme_below) else INSIDE_VALUE
            else:
                streak = 0
                cond = INSIDE_VALUE

            group_conditions.append(cond)
            group_streaks.append(streak)

        conditions.loc[group.index] = group_conditions
        streaks.loc[group.index] = group_streaks

    df["condition"] = conditions
    df["acceptance_streak"] = streaks
    return df


def compute_condition_periodic(
    df: pd.DataFrame,
    period_key: pd.Series,
    acceptance_candles: int = 3,
    value_area_band: int = 1,
    extreme_band: int = 2,
) -> pd.DataFrame:
    """Cycle 3 (swing, daily bars): same classification logic as
    ``compute_condition``, generalized to reset on an arbitrary ``period_key``
    (e.g. ``signals.vwap.compute_weekly_vwap``'s week-period column) instead
    of hardcoding calendar day -- added alongside ``compute_condition``, not
    replacing it, so Weekend 2-4's intraday behavior is untouched.

    ``df`` must already have ``band_upper_{value_area_band}``,
    ``band_lower_{value_area_band}``, ``band_upper_{extreme_band}``, and
    ``band_lower_{extreme_band}`` columns matching that same anchor.
    """
    df = df.copy()
    upper_va = f"band_upper_{value_area_band}"
    lower_va = f"band_lower_{value_area_band}"
    upper_ext = f"band_upper_{extreme_band}"
    lower_ext = f"band_lower_{extreme_band}"

    conditions = pd.Series(INSIDE_VALUE, index=df.index, dtype=object)
    streaks = pd.Series(0, index=df.index, dtype=int)

    for _, group in df.groupby(period_key, sort=False):
        streak = 0
        touched_extreme_above = False
        touched_extreme_below = False
        group_conditions = []
        group_streaks = []

        for row in group.itertuples():
            upper = getattr(row, upper_va)
            lower = getattr(row, lower_va)
            ext_upper = getattr(row, upper_ext)
            ext_lower = getattr(row, lower_ext)

            if pd.isna(upper) or pd.isna(lower):
                # First candle of the period: no bands yet, nothing to classify.
                streak = 0
                group_conditions.append(INSIDE_VALUE)
                group_streaks.append(streak)
                continue

            if not pd.isna(ext_upper) and row.high >= ext_upper:
                touched_extreme_above = True
            if not pd.isna(ext_lower) and row.low <= ext_lower:
                touched_extreme_below = True

            if row.close > upper:
                streak = streak + 1 if streak > 0 else 1
                cond = ACCEPTED_ABOVE if (streak >= acceptance_candles or touched_extreme_above) else INSIDE_VALUE
            elif row.close < lower:
                streak = streak - 1 if streak < 0 else -1
                cond = ACCEPTED_BELOW if (-streak >= acceptance_candles or touched_extreme_below) else INSIDE_VALUE
            else:
                streak = 0
                cond = INSIDE_VALUE

            group_conditions.append(cond)
            group_streaks.append(streak)

        conditions.loc[group.index] = group_conditions
        streaks.loc[group.index] = group_streaks

    df["condition"] = conditions
    df["acceptance_streak"] = streaks
    return df
