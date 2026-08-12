# Candidate watchlist vetting -- SUMMARY

## Caveat
**This is a FIRST FILTER only.** Every result above comes from a single-symbol backtest run in isolation -- no competition for the portfolio's `max_concurrent_positions` cap (5) or `max_positions_per_symbol` cap, and no correlation effects between symbols. A symbol that looks strong here could still add little (or even reduce net expectancy) once it has to compete for the same 5 concurrent slots as the other candidates and the existing 15, or if its signals cluster in time with symbols already held. A COMBINED portfolio backtest -- every promoted candidate plus the existing 15, together, respecting the real caps, run once -- is required before any adoption decision. That combined run is deliberately NOT part of this pipeline; it should be a separate, later, pre-registered step, timed near the Phase 1 review.

## Tier definitions
- **comparable to current watchlist**
- **marginal -- thin sample or borderline cost/risk**
- **fails cost-viability or negative expectancy** -- generated real OOS trades and either lost money, or the cost/risk ratio blew the viability bar.
- **insufficient data -- zero OOS trades, not tested** -- the strategy never fired in the OOS window for this symbol. This is NOT the same as failing: an untested symbol has no evidence either way, whereas a 'fails cost-viability or negative expectancy' symbol was tested and the evidence was negative.

## Resolved symbols, ranked by out-of-sample expectancy_R

| Symbol | OOS trades | OOS expectancy_R | OOS PF | OOS cost/risk avg % | Coverage | Tier |
|---|---|---|---|---|---|---|
| COCHINSHIP | 3 | +1.837 | 6.75 | 7.0% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| SHAILY | 2 | +1.462 | 3.36 | 6.1% | 2022-04-04..2026-08-11 (LIMITED) | marginal -- thin sample or borderline cost/risk |
| EIMCOELECO | 9 | +1.317 | 3.76 | 5.5% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| VOLTAS | 7 | +1.042 | 4.90 | 7.5% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| MGL | 8 | +0.732 | 2.63 | 8.5% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| BEML | 5 | +0.657 | 3.27 | 5.4% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| WEBELSOLAR | 3 | +0.573 | 1.82 | 3.7% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| HCLTECH | 2 | +0.509 | 2.73 | 5.9% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| DRREDDY | 6 | +0.318 | 1.70 | 8.8% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| GODFRYPHLP | 5 | +0.232 | 1.33 | 4.6% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| IRCTC | 8 | +0.155 | 1.37 | 8.9% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| HIRECT | 5 | +0.024 | 1.05 | 5.1% | 2021-08-09..2026-08-11 | comparable to current watchlist |
| IDEA | 4 | -0.016 | 0.97 | 6.1% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| JKTYRE | 3 | -0.044 | 0.88 | 7.3% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| SUNPHARMA | 6 | -0.074 | 0.90 | 9.7% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| IIFL | 4 | -0.086 | 0.79 | 5.0% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| RAILTEL | 5 | -0.086 | 0.87 | 5.4% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| HINDALCO | 5 | -0.094 | 0.74 | 7.5% | 2021-08-09..2026-08-11 | marginal -- thin sample or borderline cost/risk |
| ERIS | 9 | -0.129 | 0.77 | 7.5% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| PPLPHARMA | 2 | -0.230 | 0.59 | 6.3% | 2022-10-19..2026-08-11 (LIMITED) | fails cost-viability or negative expectancy |
| TATAINVEST | 6 | -0.267 | 0.73 | 8.2% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| KPIGREEN | 4 | -0.269 | 0.51 | 4.4% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| INDSWFTLAB | 4 | -0.306 | 0.59 | 4.5% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| MPSLTD | 4 | -0.339 | 0.46 | 4.6% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| PARAGMILK | 3 | -0.351 | 0.49 | 4.5% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| TANLA | 8 | -0.545 | 0.40 | 5.8% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| IDFCFIRSTB | 3 | -0.691 | 0.12 | 8.0% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| REDINGTON | 4 | -0.711 | 0.00 | 7.0% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| NPST | 5 | -0.721 | 0.03 | 4.7% | 2021-08-10..2026-08-11 | fails cost-viability or negative expectancy |
| THERMAX | 6 | -0.727 | 0.00 | 7.4% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| DIXON | 8 | -0.728 | 0.30 | 7.1% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| SCODATUBES | 1 | -0.763 | 0.00 | 5.3% | 2025-06-04..2026-08-11 (LIMITED) | fails cost-viability or negative expectancy |
| SOLEX | 8 | -0.823 | 0.21 | 5.1% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| LLOYDSENGG | 4 | -0.975 | 0.00 | 4.4% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| CDSL | 6 | -1.029 | 0.00 | 6.4% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| RATNAVEER | 3 | -1.068 | 0.00 | 5.3% | 2023-09-11..2026-08-11 (LIMITED) | fails cost-viability or negative expectancy |
| IONEXCHANG | 1 | -1.078 | 0.00 | 5.3% | 2022-02-22..2026-08-11 (LIMITED) | fails cost-viability or negative expectancy |
| SAFARI | 3 | -1.115 | 0.00 | 7.0% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| NMDC | 3 | -1.134 | 0.00 | 8.5% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| ZOTA | 4 | -1.144 | 0.00 | 6.4% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| BALAMINES | 5 | -1.192 | 0.00 | 6.3% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| CREDITACC | 2 | -1.215 | 0.00 | 7.9% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| DIAMONDYD | 2 | -1.269 | 0.00 | 7.6% | 2021-08-09..2026-08-11 | fails cost-viability or negative expectancy |
| NSDL | 0 | +0.000 | n/a | 0.0% | 2025-08-06..2026-08-11 (LIMITED) | insufficient data -- zero OOS trades, not tested |
| KPL | 0 | +0.000 | n/a | 0.0% | 2026-04-20..2026-08-11 (LIMITED) | insufficient data -- zero OOS trades, not tested |
| BIRLANU | 0 | +0.000 | n/a | 0.0% | 2021-08-09..2026-08-11 | insufficient data -- zero OOS trades, not tested |
| PIRAMALFIN | 0 | +0.000 | n/a | 0.0% | 2025-11-07..2026-08-11 (LIMITED) | insufficient data -- zero OOS trades, not tested |
| LLOYDSENT | 0 | +0.000 | n/a | 0.0% | 2024-10-17..2026-08-11 (LIMITED) | insufficient data -- zero OOS trades, not tested |
| TMCV | 0 | +0.000 | n/a | 0.0% | 2025-11-12..2026-08-11 (LIMITED) | insufficient data -- zero OOS trades, not tested |
| OLAELEC | 0 | +0.000 | n/a | 0.0% | 2024-08-09..2026-08-11 (LIMITED) | insufficient data -- zero OOS trades, not tested |

## Unresolved symbols
Could not resolve an instrument key via assets.upstox.com, the api.upstox.com symbol-search path, or the web-search-plus-live-verification fallback. Not guessed -- not included anywhere above.

- FINELABS

