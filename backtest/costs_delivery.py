"""Indian cash-DELIVERY equity cost model (swing/positional trades held
overnight, T+1 settlement) -- as opposed to ``backtest.costs``, which models
INTRADAY square-off trades and has materially different STT/DP-charge rules.

SOURCING NOTE: upstox.com is blocked by this environment's egress policy --
the same root cause tracked in issue #3 (``assets.upstox.com`` was blocked
there; here ``curl`` to the whole domain returns ``CONNECT tunnel failed,
response 403``, confirmed directly in this session). Every figure below was
therefore verified via >=2 independent secondary sources (aggregators that
mirror/cache Upstox's published numbers) rather than fetched directly from
upstox.com. **Re-verify against https://upstox.com/brokerage-charges/ once
that host is reachable from wherever this runs next.**

- Brokerage (delivery): Rs 0 per order. Upstox charges no brokerage on
  delivery-based equity trades (only intraday/F&O are charged per-order).
  Sources: stockcalc.in ("UPSTOX Brokerage Charges 2026: Rs0 Delivery"),
  Upstox community forum ("brokerage for equity delivery ... not imposed").
- STT (Securities Transaction Tax): 0.1% on BOTH the buy and the sell leg
  for delivery equity -- higher, and charged both-sided, vs. intraday's
  0.025% sell-only. SEBI-mandated, uniform across every Indian broker by
  law (not Upstox-specific).
- Exchange transaction charges (NSE): 0.00325% of turnover, both legs.
  Exchange-mandated, broker-independent.
- SEBI turnover fee: Rs 10 per crore = 0.0001% of turnover, both legs.
  SEBI-mandated, broker-independent.
- Stamp duty: 0.015% of turnover (or Rs 1500/crore), BUY side only.
  State-government-mandated under the Indian Stamp Act (uniform nationally
  since July 2020), broker-independent.
- DP (Depository Participant) charges: Rs 20 + GST, flat, per scrip per
  SELL day -- charged once per exit here, independent of quantity.
  CDSL/Upstox-levied; consistently reported at this figure across sources.
- GST: 18% on (brokerage + exchange transaction charges + SEBI fee + DP
  charges) -- i.e. on the *service* fees, not on STT or stamp duty (those
  are government taxes, not GST-able services).

Every regulatory figure (STT, exchange charges, SEBI fee, stamp duty) is
broker-independent and SEBI/exchange/government-mandated, so those carry
high confidence regardless of Upstox-specific sourcing limitations. The
broker-specific figures (Rs0 brokerage, Rs20 DP charge) were each
corroborated by 2+ independent secondary sources; if wrong, the most likely
error direction is understating costs (a nonzero brokerage would make this
weekend's cost-viability bar *harder* to clear, not easier).

CAVEAT (flagged prominently, per the brief): retail cash-delivery (CNC) in
India cannot actually initiate a short position -- short-selling requires
margin/F&O or intraday square-off, not delivery settlement. This backtest's
setups still generate short signals per the pre-registered spec (the trend
filter explicitly describes both directions), and this cost model is applied
symmetrically to both. If the report's expectancy is materially driven by
short trades, that edge is not realizable through literal cash-delivery
execution -- it would need a margin/SLB/F&O account, a different product
than what this cost model prices. The report breaks out long vs. short so
this can be checked directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from strategy.base import LONG


@dataclass
class DeliveryCostConfig:
    brokerage_per_order: float = 0.0
    stt_pct: float = 0.001  # 0.1%, both legs
    txn_charge_pct: float = 0.0000325  # 0.00325%, both legs
    sebi_fee_pct: float = 0.000001  # 0.0001% (Rs 10/crore), both legs
    stamp_duty_pct: float = 0.00015  # 0.015%, buy leg only
    dp_charge_rs: float = 20.0  # flat, sell leg only, once per exit
    gst_pct: float = 0.18
    slippage_pct: float = 0.0003  # 0.03% per side, "limit-ish patience" entries/exits


def load_delivery_cost_config(config_path: Path) -> DeliveryCostConfig:
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    costs_raw = raw.get("costs_delivery", {})
    defaults = DeliveryCostConfig()
    return DeliveryCostConfig(
        brokerage_per_order=costs_raw.get("brokerage_per_order", defaults.brokerage_per_order),
        stt_pct=costs_raw.get("stt_pct", defaults.stt_pct),
        txn_charge_pct=costs_raw.get("txn_charge_pct", defaults.txn_charge_pct),
        sebi_fee_pct=costs_raw.get("sebi_fee_pct", defaults.sebi_fee_pct),
        stamp_duty_pct=costs_raw.get("stamp_duty_pct", defaults.stamp_duty_pct),
        dp_charge_rs=costs_raw.get("dp_charge_rs", defaults.dp_charge_rs),
        gst_pct=costs_raw.get("gst_pct", defaults.gst_pct),
        slippage_pct=costs_raw.get("slippage_pct", defaults.slippage_pct),
    )


@dataclass
class DeliveryRoundTripCosts:
    brokerage: float
    stt: float
    txn_charges: float
    sebi_fee: float
    stamp_duty: float
    dp_charges: float
    gst: float
    total: float


def apply_entry_slippage(trigger_price: float, direction: str, cfg: DeliveryCostConfig) -> float:
    slip = trigger_price * cfg.slippage_pct
    return trigger_price + slip if direction == LONG else trigger_price - slip


def apply_exit_slippage(trigger_price: float, direction: str, cfg: DeliveryCostConfig) -> float:
    slip = trigger_price * cfg.slippage_pct
    return trigger_price - slip if direction == LONG else trigger_price + slip


def round_trip_costs(
    direction: str, entry_fill: float, exit_fill: float, quantity: int, cfg: DeliveryCostConfig
) -> DeliveryRoundTripCosts:
    """STT/exchange/SEBI charges are the same rate on both legs regardless of
    direction, but stamp duty (buy leg only) and DP charges (sell leg only)
    depend on which leg is actually the buy vs. the sell -- for a long that's
    entry=buy/exit=sell; for a short (see the module note on cash-delivery
    short-selling) it's the reverse: entry=sell/exit=buy."""
    entry_value = entry_fill * quantity
    exit_value = exit_fill * quantity
    buy_value = entry_value if direction == LONG else exit_value

    brokerage = cfg.brokerage_per_order * 2
    stt = (entry_value + exit_value) * cfg.stt_pct
    txn_charges = (entry_value + exit_value) * cfg.txn_charge_pct
    sebi_fee = (entry_value + exit_value) * cfg.sebi_fee_pct
    stamp_duty = buy_value * cfg.stamp_duty_pct
    dp_charges = cfg.dp_charge_rs

    gst = (brokerage + txn_charges + sebi_fee + dp_charges) * cfg.gst_pct
    total = brokerage + stt + txn_charges + sebi_fee + stamp_duty + dp_charges + gst

    return DeliveryRoundTripCosts(
        brokerage=brokerage,
        stt=stt,
        txn_charges=txn_charges,
        sebi_fee=sebi_fee,
        stamp_duty=stamp_duty,
        dp_charges=dp_charges,
        gst=gst,
        total=total,
    )


def net_pnl(direction: str, entry_fill: float, exit_fill: float, quantity: int, costs: DeliveryRoundTripCosts) -> float:
    gross = (exit_fill - entry_fill) * quantity if direction == LONG else (entry_fill - exit_fill) * quantity
    return gross - costs.total
