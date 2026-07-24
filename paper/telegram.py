"""Telegram notifications for the paper-trading daily job. Message
formatting is pure (no network) and separately testable; ``send_message``
is the only function that actually talks to the network, and never raises:
it returns ``False`` if no token/chat id is configured (e.g. a local manual
run) AND if the actual HTTP call fails (network error, bad token, wrong
chat id, Telegram-side error, timeout) -- printing the specific reason
either way. A transient Telegram outage must never crash the paper-trading
job or block whatever else that run still needs to do (journal writes,
the day's git commit) -- it should just be a clearly logged, non-fatal
"send failed" rather than an uncaught exception that takes the whole
process down with it.
"""

from __future__ import annotations

import requests

TELEGRAM_API_BASE = "https://api.telegram.org"


def send_message(text: str, bot_token: str | None, chat_id: str | None, timeout: float = 15.0) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[telegram] send_message failed: {exc}")
        return False
    return True


def format_daily_message(
    run_date: str,
    fills: list[dict],
    exits: list[dict],
    new_pending: list[dict],
    equity: float,
    market_closed: bool = False,
) -> str | None:
    """Returns ``None`` on a silent no-activity day (market closed, or no
    fills/exits/new pending orders) -- the caller should send nothing."""
    if market_closed:
        return None
    if not fills and not exits and not new_pending:
        return None

    lines = [f"Paper trading -- {run_date}"]

    if fills:
        lines.append("")
        lines.append(f"FILLED ({len(fills)}):")
        for f in fills:
            lines.append(
                f"  {f['symbol']} {f['direction']} x{f['quantity']} @ Rs{f['entry_fill_price']:.2f} "
                f"(stop Rs{f['stop_price']:.2f}, target Rs{f['target_price']:.2f}, R:R {f['rr_ratio']:.2f})"
            )

    if exits:
        lines.append("")
        lines.append(f"EXITED ({len(exits)}):")
        for e in exits:
            lines.append(
                f"  {e['symbol']} {e['exit_reason']} @ Rs{e['exit_fill_price']:.2f} "
                f"-> {e['r_multiple']:+.2f}R (Rs{e['net_pnl']:+,.0f} net)"
            )

    if new_pending:
        lines.append("")
        lines.append(f"NEW SIGNAL, pending next-open fill ({len(new_pending)}):")
        for p in new_pending:
            lines.append(f"  {p['symbol']} {p['direction']} {p['setup_id']}, R:R {p['rr_ratio']:.2f}")

    lines.append("")
    lines.append(f"Equity: Rs{equity:,.0f}")
    return "\n".join(lines)


def format_weekly_summary(
    week_label: str,
    trades_this_week: list[dict],
    cumulative_expectancy_r: float,
    cumulative_trade_count: int,
    equity: float,
    funnel_totals: dict,
    days_run: int,
    days_expected: int,
) -> str:
    lines = [f"Weekly summary -- {week_label}", ""]

    if trades_this_week:
        lines.append(f"Trades this week ({len(trades_this_week)}):")
        for t in trades_this_week:
            lines.append(f"  {t['symbol']} {t['exit_reason']} {float(t['r_multiple']):+.2f}R")
    else:
        lines.append("Trades this week: none")

    lines.append("")
    lines.append(f"Cumulative: {cumulative_trade_count} trades, expectancy {cumulative_expectancy_r:+.3f}R")
    lines.append(f"Equity: Rs{equity:,.0f}")
    lines.append("")
    lines.append(
        "Funnel (cumulative): raw={raw} after_trend_filter={after_trend_filter} "
        "after_long_only={after_long_only} after_valid_geometry={after_valid_geometry} "
        "after_rr={after_rr} after_cost_viability={after_cost_viability}".format(**funnel_totals)
    )
    lines.append("")
    reliability_pct = (days_run / days_expected * 100) if days_expected else 0.0
    lines.append(f"Reliability: job ran on {days_run}/{days_expected} expected trading days ({reliability_pct:.1f}%)")
    return "\n".join(lines)


def format_fidelity_alert(mismatches: list[str], n_compared: int) -> str:
    lines = [
        "FIDELITY MISMATCH -- paper journal vs backtest engine replay disagree",
        "",
        f"{len(mismatches)} of {n_compared} compared trades mismatched:",
    ]
    for m in mismatches[:20]:
        lines.append(f"  {m}")
    if len(mismatches) > 20:
        lines.append(f"  ... and {len(mismatches) - 20} more")
    return "\n".join(lines)
