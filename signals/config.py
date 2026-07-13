"""Signal-generation thresholds, pulled from config.yaml's ``signals:`` section."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SignalsConfig:
    deviation_bands: list[int] = field(default_factory=lambda: [1, 2])
    acceptance_candles: int = 3
    value_area_band: int = 1


def load_signals_config(config_path: Path, timeframe: str | None = None) -> SignalsConfig:
    """``timeframe`` (e.g. ``"15min"``) merges ``timeframes.<timeframe>.signals``
    on top of the base ``signals:`` section -- see config.yaml's ``timeframes:``
    block. ``None`` (or a timeframe with no override section) uses the base
    5-min profile unchanged."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    signals_raw = dict(raw.get("signals", {}))
    if timeframe:
        signals_raw.update(raw.get("timeframes", {}).get(timeframe, {}).get("signals", {}))
    return SignalsConfig(
        deviation_bands=signals_raw.get("deviation_bands", [1, 2]),
        acceptance_candles=signals_raw.get("acceptance_candles", 3),
        value_area_band=signals_raw.get("value_area_band", 1),
    )
