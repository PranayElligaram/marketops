# Historical valuation and time-weighted performance

Code: `src/marketops/valuation/` (`symbols`, `fx`, `cache`, `nav`, `twr`, `report`, `service`),
routes in `api/routes/performance.py`, dashboard section "Historical valuation and time-weighted
performance". The money-weighted XIRR in `/performance/historical` is a separate measure and is
unchanged.

## Endpoints (both need `X-MarketOps-Token`)
- `GET /performance/valuation`: daily NAV, net invested capital, time-weighted return, volatility,
  drawdown, monthly/annual returns and a reconciliation to the live snapshot. Reads PostgreSQL and
  the local cache only; it never downloads market data.
- `POST /performance/valuation/refresh`: downloads daily bars for every instrument ever traded
  (only public symbols and dates are sent) and daily ECB reference FX (currency codes and dates).

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
  of later splits). `adj_close` is never used, so dividends are counted once, as cash received.
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
