# Historical valuation and time-weighted performance

Code: `src/marketops/valuation/` (`symbols`, `fx`, `cache`, `nav`, `twr`, `report`, `service`, plus the benchmark modules described in
[benchmark-design.md](benchmark-design.md)),
routes in `api/routes/performance.py`, dashboard section "Historical valuation and time-weighted
performance". The money-weighted XIRR in `/performance/historical` is a separate measure and is
unchanged.

## Endpoints (both need `X-MarketOps-Token`)
- `GET /performance/valuation`: daily NAV, net invested capital, time-weighted return, volatility,
  drawdown, monthly/annual returns and a reconciliation to the live snapshot. Reads PostgreSQL and
  the local cache only; it never downloads market data.
- `POST /performance/valuation/refresh`: downloads daily bars for every instrument ever traded
  (only public symbols and dates are sent), the SPY benchmark series and daily ECB reference FX
  (currency codes and dates).
- `GET /performance/demo` (public, no token): the same response built from an invented account, prices,
  FX and a synthetic SPY stand-in (`marketops/demo.py`). No credentials, database or network are used.

The response includes `benchmark`: the contribution-matched S&P 500 comparison (status, exact reasons
when unavailable, compared series, metrics, source provenance, conventions and limitations).

## Data sources (free, personal research use)
- Prices: the Yahoo chart endpoint already used for research (unofficial; personal use; not
  redistributed; cache is git-ignored under `data/market/`).
- FX: ECB euro reference rates via the open-source Frankfurter API. A download that is stale,
  truncated or missing a currency is rejected. (The ECB's own bulk CSV was found to be stale in
  testing, which is why freshness is enforced.)

## Instrument mapping
Trading 212 tickers map to provider symbols by rule (`AAPL_US_EQ` -> `AAPL`, `...l_EQ` -> `.L`,
etc.) or by a local override file `data/market/symbol_overrides.json` (`{"T212_TICKER": "SYMBOL"}`).
A mapping is only `verified` if at least 95 % of the instrument's own executed fills (buys and
sells, so fully sold instruments are covered) fall inside the provider's daily low-high range
(+/-2 %, +/-4 days, as-traded price) and the provider currency matches the instrument currency.
A wrong or renamed symbol (a real example during development: a renamed ticker that the provider
now assigns to a different company) fails verification and leaves the instrument unvalued until an
override is supplied. GBX/GBp prices are divided by 100.

## Conventions
- Event date is the UTC calendar date; valuation days are Monday to Friday; NAV is end-of-day.
- Price: provider split-adjusted `close`, restated to the as-traded price for the day (x product
  of later splits). `adj_close` is never used for the portfolio, so dividends are counted once, as
  cash received. (The only exception is the SPY benchmark, which needs a total-return series.)
- A close up to 5 calendar days old may be used (flagged stale), never across a split. FX uses the
  last published rate up to 5 days old. Null provider bars are dropped, not treated as zero.
- Splits from the provider multiply the shares held on the ex-date. If Trading 212 itself reports
  corporate-action fills for an instrument, valuation of that instrument is unavailable from that
  date (the same event could be described twice). Corporate-action cash effects are not modelled.
- Cash ledger: deposits - withdrawals - fees + interest + dividends/distributions - purchases +
  sales, identical to the ledger already reconciled to the broker balance. Negative cash beyond
  GBP 1.00 marks the day unvalued (a sign of missing history).
- Time-weighted return: external flows occur after the close of their day (daily return is
  `(NAV_t - flow_t) / NAV_{t-1} - 1`); the start-of-day variant is reported as a sensitivity.
  A gap in valuation breaks the chain into segments; nothing is bridged. Volatility needs 60
  returns, drawdown 20, annualisation at least a year. Periods are `complete` only if every
  weekday in them is covered by one unbroken segment.

## Status gates
`verified` requires: external flows verified (complete history, clean transactions, cash ledger
matching the broker), share counts (after provider splits) equal to the broker's, last-close NAV
within 1.5 % of the live total (`drift` up to 4 %), and every weekday valued in one segment.
Otherwise metrics are `partial` with blockers, or `unavailable` with a reason.

## Limitations
- Free price data can contain errors or gaps and can change its terms; no second source is used to
  cross-check closes (fills are used to verify the mapping only).
- Exchange holidays reuse the prior close (flagged); weekends are not valued.
- Daily closes cannot reflect intraday flow timing; see the sensitivity metric.
- Corporate actions other than provider-reported splits (mergers, spin-offs, ticker changes) are
  not interpreted.
- The reconciliation compares a last close with a live price, so small differences are expected.

## Dashboard presentation (Historical performance tab)

Code: `app/perf_view.py` (formatting, labels, display-period filtering, Altair charts) and the tab in
`app/dashboard.py`. Presentation only: every figure is taken from the API response; no calculation changed.

- Eight headline KPIs in two rows: portfolio value, net invested capital, net investment gain, total and annualised
  time-weighted return, money-weighted return (XIRR), annualised volatility, maximum drawdown. Each card names its
  basis: **Historical closes** (daily closing prices and ECB FX from `/performance/valuation`) or **Live snapshot**
  (XIRR, from `/performance/historical`, which ends at the broker-reported live value).
- **TWR vs XIRR:** TWR removes the timing of your deposits and shows how the investments performed; XIRR is the
  yearly rate on your actual dated cash flows. They answer different questions and need not match.
- Period selector (1M, 3M, 6M, 1Y, ALL) only chooses which days are plotted. Cumulative TWR and drawdown are derived
  from the API's time-weighted index over the full history before cropping, so KPIs and chart values never change
  with the selection. Leading days before the first valuation are not plotted; gaps stay gaps.
- Status wording: API `verified` is shown as **Reconciled**, `partial` as **Partially reconciled**, `unverified` as
  **Not reconciled**. The API values and the validation rules are unchanged. Significant warnings (unvalued days,
  unconfirmed symbols, reconciliation reasons, failed audit checks) stay visible; coverage, share-count
  reconciliation, provider/mapping diagnostics and conventions are in expanders.

## Benchmark comparison in the dashboard
The tab section "Portfolio vs S&P 500 benchmark" (below the portfolio charts) shows: twelve comparative KPI
cards (time-weighted return, annualised return, maximum drawdown, closing-price XIRR for both sides and the
differences, value versus the benchmark), a value chart (portfolio, benchmark, net invested capital), a
cumulative-return chart and a drawdown chart, a 1M / 3M / 6M / 1Y / ALL display window, a plain-language
explanation of XIRR versus time-weighted return, and an expander with data quality (source, rows used and
dropped, download time, file fingerprint, window) and methodology/limitations. Headline KPIs always cover the
full comparison window; the display window only crops the charts, and cumulative-return lines are rebased to
the window start (an exact operation for chain-linked returns). If the comparison is unavailable, the exact
reason is shown instead. **Load synthetic demo** fills the tab from `/performance/demo`.
