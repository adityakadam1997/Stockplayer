"""Phase 1: fully automated PAPER trading for the Cycle 3B GO system.

No broker orders, no order API, no live capital -- ``scripts/paper_daily.py``
runs once per trading day and journals everything to ``paper/journal.csv``
(every candidate proposal, whatever funnel stage it reached) and
``paper/trades.csv`` (fills/exits with full economics), with
``paper/state.json`` as the single source of truth for what's currently open
or pending. These files are committed by the job -- the git history is the
audit trail.

This package intentionally does not reimplement any strategy/signal/cost
logic: ``paper.pipeline`` calls straight into ``strategy.swing_engine`` and
reuses ``backtest.swing_simulator``'s private position-sizing/gap-fill/
trade-closing helpers, so the paper system's economics are the existing,
already-tested backtest code, not a second copy of it.
"""
