# POWERGRID -- candidate vetting report

## Data coverage
2021-08-09 to 2026-08-11 (1242 trading days).

## IS/OOS split
- In-sample: 2021-08-09 to 2025-02-10
- Out-of-sample: 2025-02-11 to 2026-08-11

## In-sample results
- Trades: 6
- Win rate: 50.0%
- Expectancy: +0.075R
- Profit factor: 1.13
- Max drawdown: Rs-10,251
- Avg holding days: 4.5

## Out-of-sample results
- Trades: 5
- Win rate: 20.0%
- Expectancy: -0.853R
- Profit factor: 0.19
- Max drawdown: Rs-15,742
- Avg holding days: 5.6

## Cost-viability funnel (out-of-sample)
```
   symbol  raw  after_trend_filter  suppressed_shorts  after_long_only  invalid_geometry  after_valid_geometry  after_rr  after_cost_viability  executed
POWERGRID  114                  88                 47               41                 7                    34         7                     7         5
    TOTAL  114                  88                 47               41                 7                    34         7                     7         5
```

## Cost/risk ratio of executed trades (out-of-sample)
n=5  mean=8.4%  median=9.6%  min=4.3%  max=9.6%  (existing 15-symbol watchlist's known Cycle 3B range: ~6-18%)

## Tier: fails cost-viability or negative expectancy

