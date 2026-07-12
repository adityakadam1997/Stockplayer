# trading-agent

Personal NSE intraday trading system: instrument resolution + historical
candle download (`data/`), session VWAP/deviation-band/condition signals
(`signals/`), the VWAP Wave System's four trade setups and backtest engine
(`strategy/`, `backtest/`). Risk, execution, and journaling are stubbed for
future weekends -- this is a research/backtesting system only, it does not
place orders.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11+.

## Configure

Edit `config.yaml`:

- `data.interval_minutes` — candle size (default 5).
- `data.lookback_months` — how far back to fetch (default 12).
- `watchlist` — NSE trading symbols to track.

### Upstox access token (optional)

The Upstox V3 Historical Candle API appears to work without authentication, but
if you have an access token, set it as an environment variable — it is never
read from a file or hardcoded:

```bash
export UPSTOX_ACCESS_TOKEN="your-token-here"
```

> Note: the official Upstox docs site (upstox.com) was not reachable from the
> environment this repo was built in, so the V3 endpoint shape, auth
> requirement, and response format were cross-checked via Upstox's public
> GitHub SDK docs and developer community threads instead of the live docs
> page. Re-verify against
> https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/
> before relying on this in production, and watch for `UpstoxAPIError`s that
> may indicate the endpoint contract has changed.

## Run

```bash
# Full download for every symbol in the watchlist
python scripts/download_data.py

# Incremental update (only fetches candles newer than what's cached)
python scripts/download_data.py --update

# Quality/coverage report only, no network calls
python scripts/download_data.py --report

# Override the watchlist for a quick smoke test
python scripts/download_data.py --symbols RELIANCE,SBIN
```

Cached candles land in `cache/{SYMBOL}_{interval}min.parquet` (gitignored).

## Data quality checks

After every download, a per-symbol summary table is printed covering:

- strictly increasing, duplicate-free timestamps
- all candles within NSE market hours (09:15-15:30 IST) on weekdays
- OHLC sanity (`high >= max(open, close)`, `low <= min(open, close)`)
- gap report: weekdays with missing candles (a day with zero candles is
  assumed to be an exchange holiday; a day with *some but not all* candles is
  flagged as a real gap)
- coverage: first date, last date, total candle count

Run `python scripts/download_data.py --report` any time to regenerate this
without hitting the network.

## Strategy: the VWAP Wave System

Four setups (`strategy/setup1_discovery.py` ... `setup4_bounce.py`), all
requiring reward:risk >= 1.5 at entry (`strategy.min_rr` in `config.yaml`) and
gated by the market condition classifier from `signals/`:

1. **Price discovery continuation** -- after a genuine acceptance move, enter
   on the retest of the band that broke, targeting the session extreme.
2. **Fade value area extremes** -- mean-reversion off the value-area band,
   only in a rotating (non-trending) session, targeting VWAP.
3. **Return to value** -- after price breaks back inside from an acceptance
   phase, ride the continuation toward VWAP (primary: wait for a retest;
   optional aggressive fallback: enter on the break candle itself).
4. **VWAP bounce** -- a reversal off VWAP itself after price runs all the way
   back to it; the setup most prone to being a trap, per its own code
   comments.

`strategy/engine.py` runs all four per candle with no lookahead (see
`tests/test_strategy.py::test_no_lookahead`), applies the wide-band guard
(suppresses the mean-reversion setups 2 & 4 after a violent move blows the
bands out), the entry time windows, and the R:R filter.

## Backtest

```bash
# Full watchlist, full lookback, default config.yaml
python scripts/run_backtest.py

# Override the watchlist
python scripts/run_backtest.py --symbols RELIANCE,SBIN
```

`backtest/simulator.py` replays cached candles one-trade-per-symbol, with a
realistic Indian-equity cost model (`backtest/costs.py`: brokerage, STT,
exchange/SEBI/stamp charges, GST, slippage -- all configurable under `costs:`
in `config.yaml`), a pessimistic stop-first fill rule when a candle touches
both stop and target, and forced square-off at 15:15 IST.
`backtest/report.py` prints overall / per-setup / per-symbol expectancy
tables and writes `backtest/results/summary.csv` + `trades.csv` (gitignored).

**This is a backtesting/research tool only** -- it never places an order;
`execution/` stays stubbed until that's explicitly in scope.

## Tests

```bash
python -m pytest tests/ -v
```

Tests are fully offline — they exercise `data/`, `signals/`, `strategy/`, and
`backtest/` against synthetic data, not the live Upstox API or the local
`cache/`.

## Repo layout

```
data/           instrument resolution, downloader, parquet storage
signals/        session VWAP, deviation bands, market condition classifier
strategy/       the four VWAP Wave System setups + the no-lookahead engine
backtest/       cost model, candle-replay simulator, expectancy report
risk/           stub
execution/      stub -- no order placement anywhere in this repo
journal/        stub
scripts/        CLI entry points
tests/          offline unit tests
cache/          gitignored parquet cache
```
