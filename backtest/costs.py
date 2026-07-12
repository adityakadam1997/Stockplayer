"""Indian intraday-equity cost model.

Per round trip (one entry order + one exit order):

- Brokerage: a flat rupee amount per *executed order* (so 2x per round trip).
- STT (securities transaction tax): charged on the sell leg only -- for a
  long that's the exit, for a short that's the entry.
- Exchange transaction charges + SEBI fees + stamp duty: bundled into one
  configurable percentage, charged on both legs.
- GST: 18% on (brokerage + the bundled transaction charges).
- Slippage: fills are worse than the trigger price on both entry and exit.
  The effective slippage is the *larger* of the tick-based and
  percentage-based components (``max(ticks * tick_size, price * pct)``) --
  configuring either alone still gives a sane floor. Stop-loss exits use a
  separate, larger percentage override (stops slip more in fast markets).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from strategy.base import LONG


@dataclass
class CostConfig:
    brokerage_per_order: float = 20.0
    stt_sell_pct: float = 0.00025
    txn_charge_pct: float = 0.00005
    gst_pct: float = 0.18
    slippage_ticks: float = 1.0
    tick_size: float = 0.05
    slippage_pct: float = 0.0002
    stop_slippage_pct: float = 0.0005


def load_cost_config(config_path: Path) -> CostConfig:
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    costs_raw = raw.get("costs", {})
    defaults = CostConfig()
    return CostConfig(
        brokerage_per_order=costs_raw.get("brokerage_per_order", defaults.brokerage_per_order),
        stt_sell_pct=costs_raw.get("stt_sell_pct", defaults.stt_sell_pct),
        txn_charge_pct=costs_raw.get("txn_charge_pct", defaults.txn_charge_pct),
        gst_pct=costs_raw.get("gst_pct", defaults.gst_pct),
        slippage_ticks=costs_raw.get("slippage_ticks", defaults.slippage_ticks),
        tick_size=costs_raw.get("tick_size", defaults.tick_size),
        slippage_pct=costs_raw.get("slippage_pct", defaults.slippage_pct),
        stop_slippage_pct=costs_raw.get("stop_slippage_pct", defaults.stop_slippage_pct),
    )


@dataclass
class RoundTripCosts:
    brokerage: float
    stt: float
    txn_charges: float
    gst: float
    total: float


def slippage_amount(price: float, cfg: CostConfig) -> float:
    """Rupee slippage per share for a non-stop entry/exit fill."""
    return max(cfg.slippage_ticks * cfg.tick_size, price * cfg.slippage_pct)


def apply_entry_slippage(trigger_price: float, direction: str, cfg: CostConfig) -> float:
    slip = slippage_amount(trigger_price, cfg)
    # Buying (long entry) costs more; selling (short entry) receives less.
    return trigger_price + slip if direction == LONG else trigger_price - slip


def apply_exit_slippage(trigger_price: float, direction: str, cfg: CostConfig, is_stop: bool = False) -> float:
    slip = trigger_price * cfg.stop_slippage_pct if is_stop else slippage_amount(trigger_price, cfg)
    # Exiting a long = selling (receive less); exiting a short = buying back (pay more).
    return trigger_price - slip if direction == LONG else trigger_price + slip


def round_trip_costs(direction: str, entry_fill: float, exit_fill: float, quantity: int, cfg: CostConfig) -> RoundTripCosts:
    entry_value = entry_fill * quantity
    exit_value = exit_fill * quantity

    sell_value = exit_value if direction == LONG else entry_value
    stt = sell_value * cfg.stt_sell_pct

    txn_charges = (entry_value + exit_value) * cfg.txn_charge_pct
    brokerage = cfg.brokerage_per_order * 2
    gst = (brokerage + txn_charges) * cfg.gst_pct

    total = brokerage + stt + txn_charges + gst
    return RoundTripCosts(brokerage=brokerage, stt=stt, txn_charges=txn_charges, gst=gst, total=total)


def net_pnl(direction: str, entry_fill: float, exit_fill: float, quantity: int, costs: RoundTripCosts) -> float:
    gross = (exit_fill - entry_fill) * quantity if direction == LONG else (entry_fill - exit_fill) * quantity
    return gross - costs.total
