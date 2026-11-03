# Trading 212 account history and performance data contracts

Source of truth: the Trading 212 Public API OpenAPI description (`docs.trading212.com/_bundle/api.yaml`, checked 2026-10-29,
API v0 beta). Everything is read-only; the only POST in MarketOps is `POST /history/sync`, which starts a *local import*.

## 1. What is imported

| Endpoint (GET) | Limit | Scope | Stored in | Stable id |
|---|---|---|---|---|
| `/equity/history/orders` | 20 req/min | `history:orders` | `broker_order_fill` | `<order.id>:<fill.id or ->` |
| `/equity/history/transactions` | 20 req/min | `history:transactions` | `broker_cash_transaction` | `reference` |
| `/equity/history/dividends` | 20 req/min | `history:dividends` | `broker_dividend` | `reference` |

Pagination: `limit` max 50; the response carries `nextPagePath` (null on the last page). The client accepts only a relative
path equal to the *same* endpoint (`/api/v0/equity/history/<kind>`), query keys from `{limit, cursor, ticker, time}`, a numeric
`limit` of 1-50 and a cursor of `[A-Za-z0-9._:-]{1,128}`. Absolute URLs, other hosts/paths, userinfo, fragments, traversal,
repeated keys and control characters are refused. The request is rebuilt from the configured origin, so no broker-supplied string
is requested verbatim. Requests are spaced 3.1 s apart; 429 waits for `x-ratelimit-reset` (up to 70 s) and retries twice;
408/5xx/network errors retry twice; 401/403 never retry.

## 2. Classification (no guessing)

* **Cash transactions** (`type`): `DEPOSIT`=deposit, `WITHDRAW`=withdrawal, `FEE`=fee, `INTEREST_ON_FREE_CASH` and
  `LENDING_INTEREST`=interest, `TRANSFER`=transfer, anything else=unclassified. **`TRANSFER` is never counted as a deposit or
  withdrawal**: the API documents neither direction nor counterparty. Direction of deposits/withdrawals comes from the type; the
  amount is used as a magnitude (the docs do not state the sign convention). A non-positive `DEPOSIT` is flagged inconsistent
  and excluded from totals.
* **Orders**: a history item is `{order, fill}`. A `TRADE` fill with side BUY/SELL is a `buy`/`sell`; other documented fill types
  (splits, spin-offs, FOP, ...) are `corporate_action`; unknown fill types `unclassified_fill`; an order with no fill is
  `unfilled`. Buys/sells are *internal* to the account and are never external cash flows.
* **Dividends**: `ORDINARY`, `BONUS`, `DIVIDEND`, `DIVIDENDS_PAID_BY_*` (and their `_MANUFACTURED_PAYMENT` forms) are
  `dividend`; types starting `INTEREST` are `interest`; every other documented or unknown type is `other_distribution`
  (capital gains, return of capital, ...). Only `dividend` feeds "dividends received".
* Fees: `walletImpact.taxes[]` entries are stored verbatim; `fees_total` is set only when every entry is in the wallet currency.

## 3. Integrity rules

* A record missing its identifier, timestamp or a core amount is **rejected** with a value-free reason code (`missing:reference`,
  `invalid:amount`, ...) and counted; nothing is repaired. Naive timestamps are rejected rather than given an assumed zone.
* Optional values stay `NULL` when absent. Nothing is converted between currencies; totals are per currency.
* Identical re-delivery = duplicate (skipped). Same identifier with different content = **conflict** (first copy kept, counted,
  shown in the dashboard). Inconsistent records (`anomaly` column: non-positive deposit, negative dividend, wallet or dividend
  currency different from the account currency) are stored but excluded from totals.
* Every row carries `environment` (demo/live) and `account_key = sha256("<env>:<account id>")[:16]`; the account number and raw
  payloads are never stored. Uniqueness: `(environment, account_key, source_id)`.

## 4. Sync lifecycle and completeness

`broker_sync_run` has one row per dataset per run (`mode` full/incremental; `status` running/complete/interrupted/failed;
`reached_end`; page and record counters; `error_code`; reject reasons). Progress is committed per page.

* A run is **complete** when `nextPagePath` became null (full) or when a page of already-stored valid records was reached
  (incremental). Incremental is used only after a *finished* full pass; before that every sync is full. Re-running a full pass is
  always safe (idempotent).
* Anything else is `interrupted`/`failed` with a code (`rate_limit`, `unavailable`, `network`, `scope`, `auth`, `pagination`,
  `pagination_loop`, `empty_page_with_cursor`, `page_limit`, `storage`, `schema`, `process_terminated`).
* Dataset state: `never_synced`, `running`, `partial` (no finished full pass: may be incomplete), `stale` (backfill finished but
  the latest update failed), `complete`.
* Known limit: an incremental sync cannot see a record back-dated *older* than data already stored. A forced full pass can.
* Known limit: the API cannot prove history starts at account opening. The earliest stored record date is shown; nothing before it
  or between records is inferred.

## 5. Performance data contracts (what each metric needs)

Status as of this milestone is reported live by `GET /history/readiness`.

| Metric | Required inputs | Status now |
|---|---|---|
| Net deposits/withdrawals | Complete `transactions` pass; every flow in the account currency; zero `TRANSFER`/unclassified; no inconsistent rows | **Computed** (as recorded by the API) when gates pass, else `partial` with blockers |
| Dividend income | Complete `dividends` pass; account-currency amounts | **Computed**; sums `dividend` category only |
| Cash fees and interest | Complete `transactions` pass | **Computed** |
| Realised P/L (reported) | Complete `orders` pass; broker per-fill `realisedProfitLoss`; reconciliation to the account summary's all-time realised P/L captured at the same sync | Shown as `available` only if the two agree (tolerance GBP 0.005 per fill + 0.01), else `partial` |
| Unrealised P/L | Current positions (`walletImpact`) | Available in the Portfolio tab as a *snapshot*, not a return |
| Money-weighted return (XIRR) | Dated external flows from inception (verified complete, single currency) **and** a verified start/terminal valuation; flows must exclude internal buys/sells/dividends | **Computed** by `GET /performance/historical` only when holdings and the cash ledger reconcile to the live account; see `docs/status.md` |
| Time-weighted return | Portfolio value immediately before every external flow (or daily valuations) | **Unavailable**: no valuation history |
| Portfolio equity curve | Daily positions x historical prices x FX, from verified holdings history | **Unavailable**; not approximated from current holdings |
| Total return % | The above, with deposits separated from profit | **Unavailable** |
| Benchmark comparison | A verified portfolio return series on the same dates and currency | **Unavailable** |

Rules for the future calculations: deposits are never profit; buy/sell proceeds are never external flows; unrealised P/L is never a
total return; no return percentage is shown unless cash flows, valuations and currency conversions are all verified.
Candidate next input (not built): reconstructing holdings from fills and valuing them with the existing price store needs split
and corporate-action handling and FX; it must be validated against the live positions snapshot before any curve is shown.

## 6. Security

All `/history/*` routes use the existing token dependency (403 when no token is configured, 401 when missing/wrong). Only
`POST /history/sync` contacts the broker (one account-summary request, then the history pages in a background task); the
dashboard requires an explicit confirmation tick. Live needs `MARKETOPS_T212_ENVIRONMENT=live` and `MARKETOPS_T212_ALLOW_LIVE=true`.
No history is passed to the agent or Ollama (the agent tools are unchanged). Logs carry counts and codes only.
