# trading-agent

Personal NSE intraday trading system. **Weekend 1 scope: data foundation only** —
resolving instruments, downloading historical candles, and caching them as
parquet. Signals, strategy, risk, execution, and journaling are stubbed for
future weekends.

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

## Tests

```bash
python -m pytest tests/ -v
```

Tests are fully offline — they exercise `data/store.py` and `data/quality.py`
against synthetic data, not the live Upstox API.

## Repo layout

```
data/           instrument resolution, downloader, parquet storage
signals/        stub -- Weekend 2
strategy/       stub -- Weekend 3
risk/           stub
execution/      stub
journal/        stub
scripts/        CLI entry points
tests/          offline unit tests
cache/          gitignored parquet cache
```
