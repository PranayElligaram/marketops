# Contribution-matched benchmark (SPY total return, in account currency)

Status: implemented. Code: `valuation/benchmark.py` (pure simulation), `valuation/benchmark_data.py`
(cache, provenance, currency conversion), `valuation/comparison.py` (like-for-like comparison),
wired into `HistoricalValuation.benchmark`. Tests: `test_valuation_benchmark.py`,
`test_valuation_comparison.py`, `test_benchmark_integration.py`.

Question answered: *what would the same external deposits and withdrawals be worth if every pound had
been put into the S&P 500 instead?*

## Method
1. **Series.** SPY dividend-adjusted close (`adj_close`, USD) per trading day is the total-return
   proxy: dividends are reinvested inside the price and are never added again. This is the only place
   adjusted prices are used; the portfolio valuation deliberately uses unadjusted closes plus dividend cash.
2. **Currency.** `price = adj_close_usd x GBP-per-USD`, using the ECB reference rate for the same date
   (last rate on or before the day, at most 5 days old, same rule as `valuation/fx.py`). A day with a
   missing/invalid price or FX rate is dropped and counted, never filled or interpolated.
3. **Flows.** The portfolio's external deposits and withdrawals (`DayNav.flow`). Units bought or sold
   at the close of the execution day: the flow's own date, or the next priced day.
4. **Simulation** (`simulate_benchmark`): `units_t = units_{t-1} + flow_t / price_t`,
   `value_t = units_t x price_t`. Decimal arithmetic throughout; floats are rejected. No negative
   holdings: a withdrawal larger than the holdings raises `ValueError`.
5. **Comparison window.** Only dates on which the portfolio is valued *and* the benchmark is priced,
   starting at the first such date on/after the first flow. Both sides are read at the same close,
   in the same currency, with the same flows; the benchmark's cumulative contributions are checked
   against the portfolio's net invested capital on every compared day (tolerance 0.01).

## Metrics (`benchmark.metrics`, each `available` / `partial` / `unavailable`)
Portfolio and benchmark value, net invested capital, value difference; time-weighted return for both
and the percentage-point difference; annualised TWR (window of at least one year); maximum drawdown
for both (at least 20 returns) and the difference; **closing-price XIRR** for both and the difference.

*Closing-price XIRR* uses the same dated flows for both sides and a terminal value at the last common
close. It is deliberately different from the live-snapshot XIRR in `/performance/historical`, whose
terminal value is a live broker balance that has no benchmark counterpart. The dashboard says so.
It is withheld when the period is under 30 days, the flows are one-sided, or the root is ambiguous.

## Timing
Same as `twr.end_of_day`: a flow is part of the value on its execution day's close and earns nothing
earlier. The benchmark never uses an earlier price for a flow. ECB fixes at ~16:00 CET and the US close
is ~21:00 UTC; same-date pairing is accepted and documented.

## No leakage / no double counting
- The output at day t depends only on prices and flows up to t (tested by prefix equality and by
  corrupting only later prices).
- `adj_close` is retroactively rescaled when dividends go ex; the simulation is invariant to a uniform
  rescaling (tested), so historical returns do not change when the provider re-adjusts.
- Dividends enter only through `adj_close`; the dividend column is never added.

## When the comparison is `unavailable` (exact reason shown, nothing substituted)
No cached benchmark data / fingerprint mismatch / quote currency not USD / no FX; benchmark ends more
than 5 days before the portfolio; a flow with no benchmark price within 5 days after it; unvalued
portfolio weekdays inside the window (the portfolio index would be broken); a benchmark price gap over
5 days; a one-day benchmark move over 25% (treated as corrupt data); fewer than two common dates;
withdrawals the benchmark cannot fund; contribution mismatch. If the portfolio itself is not reconciled to
the live account the comparison is `partial` with that blocker.

## Provenance
`POST /performance/valuation/refresh` fetches SPY through the existing provider (symbol and dates
only) into `data/market/benchmarks/SPY.csv` with a metadata file recording provider, quote currency,
download time and a SHA-256 of the CSV. It is git-ignored and never redistributed. `GET
/performance/valuation` reads the cache only and reports the source, row counts, drops, age and fingerprint.

## Limitations (shown beside every result)
- SPY is an ETF proxy for the S&P 500: ~0.09% expense ratio, tracking error and the provider's dividend
  adjustment method; it is not the index total-return series.
- Fractional units, zero costs, spreads and taxes (no UK withholding / stamp duty modelling).
- A GBP investor in SPY holds USD exposure; the result mixes equity and USD/GBP moves.
- Same-date ECB / US-close pairing; no intraday timing.
- The portfolio holds a different risk mix than the index; a gap is not evidence of skill.
- Free personal-use price data with no second source.
