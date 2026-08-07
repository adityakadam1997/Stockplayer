"""Indian intraday-equity cost model: brokerage, STT, exchange/SEBI/stamp
charges, GST, and slippage on entry/exit fills. All rates are configurable
via config.yaml's ``costs:`` section so later work can stress-test with
doubled slippage etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from strategy.base import LONG


@dataclass
class CostConfig:
    brokerage_per_order: float = 20.0
    stt_sell_side_pct: float = 0.025
    transaction_charges_pct: float = 0.005
    gst_pct: float = 18.0
    slippage_mode: str = "ticks"  # "ticks" | "percentage"
    slippage_ticks: float = 1.0
    tick_size: float = 0.05
    slippage_pct: float = 0.02
    stop_slippage_pct: float = 0.05


def load_cost_config(config_path: Path) -> CostConfig:
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    c = raw.get("costs", {})
    return CostConfig(
        brokerage_per_order=c.get("brokerage_per_order", 20.0),
        stt_sell_side_pct=c.get("stt_sell_side_pct", 0.025),
        transaction_charges_pct=c.get("transaction_charges_pct", 0.005),
        gst_pct=c.get("gst_pct", 18.0),
        slippage_mode=c.get("slippage_mode", "ticks"),
        slippage_ticks=c.get("slippage_ticks", 1.0),
        tick_size=c.get("tick_size", 0.05),
        slippage_pct=c.get("slippage_pct", 0.02),
        stop_slippage_pct=c.get("stop_slippage_pct", 0.05),
    )


def _general_slippage_amount(price: float, config: CostConfig) -> float:
    if config.slippage_mode == "percentage":
        return price * (config.slippage_pct / 100.0)
    return config.slippage_ticks * config.tick_size


def _stop_slippage_amount(price: float, config: CostConfig) -> float:
    return price * (config.stop_slippage_pct / 100.0)


def apply_entry_slippage(trigger_price: float, direction: str, config: CostConfig) -> float:
    """Entry fills are always worse than the trigger price: higher for longs
    (buying), lower for shorts (selling short)."""
    amount = _general_slippage_amount(trigger_price, config)
    return trigger_price + amount if direction == LONG else trigger_price - amount


def apply_exit_slippage(trigger_price: float, direction: str, config: CostConfig, is_stop: bool) -> float:
    """Exit fills are always worse than the trigger price: lower for longs
    (selling), higher for shorts (buying to cover). Stop-loss fills use the
    wider, dedicated stop-slippage rate -- stops slip worse in fast markets."""
    amount = _stop_slippage_amount(trigger_price, config) if is_stop else _general_slippage_amount(trigger_price, config)
    return trigger_price - amount if direction == LONG else trigger_price + amount


@dataclass
class TradeCosts:
    brokerage: float
    stt: float
    transaction_charges: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.transaction_charges + self.gst


def compute_trade_costs(
    direction: str, entry_fill: float, exit_fill: float, quantity: int, config: CostConfig
) -> TradeCosts:
    """All-in round-trip costs for one trade, given the actual (slippage-
    adjusted) entry and exit fill prices."""
    entry_turnover = entry_fill * quantity
    exit_turnover = exit_fill * quantity

    brokerage = config.brokerage_per_order * 2

    # STT applies only to the sell-side leg: exit for longs, entry for shorts
    # (a short sells first, at entry).
    sell_turnover = exit_turnover if direction == LONG else entry_turnover
    stt = sell_turnover * (config.stt_sell_side_pct / 100.0)

    transaction_charges = (entry_turnover + exit_turnover) * (config.transaction_charges_pct / 100.0)

    gst = (brokerage + transaction_charges) * (config.gst_pct / 100.0)

    return TradeCosts(brokerage=brokerage, stt=stt, transaction_charges=transaction_charges, gst=gst)
