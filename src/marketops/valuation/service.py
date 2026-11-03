"""Glue between stored account history, the cached market data and the pure valuation code."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from marketops.data.provider import Provider
from marketops.history.read import Coverage
from marketops.performance.report import _load, _utc, build_report
from marketops.persistence.models import (
    broker_cash_transaction,
    broker_dividend,
    broker_order_fill,
)
from marketops.portfolio.analytics import PortfolioOverview
from marketops.valuation import fx as fxmod
from marketops.valuation.cache import fetch_symbol, load_series
from marketops.valuation.report import HistoricalValuation, build_valuation
from marketops.valuation.symbols import candidate_symbol, load_overrides, normalise_currency

OVERRIDES_FILE = "symbol_overrides.json"


def compute_valuation(
    engine: Engine,
    environment: str,
    account_key: str,
    cov: Coverage,
    snapshot: PortfolioOverview | None,
    cache_dir: Path,
    now: datetime | None = None,
) -> HistoricalValuation:
    ccy = cov.account_currency
    as_of = _utc(snapshot.snapshot_utc) if snapshot else _utc(now or datetime.now(UTC))
    tx, divs, fills = (
        [r for r in _load(engine, t, environment, account_key) if r["occurred_at"] <= as_of]
        for t in (broker_cash_transaction, broker_dividend, broker_order_fill)
    )
    # The existing report decides whether stored history is reconciled to the broker.
    perf = build_report(engine, environment, account_key, cov, snapshot, now)
    # "Available" external flows already require complete history, a clean transaction set and a
    # cash ledger that reproduces the broker balance; share counts are checked by the valuation.
    flows = next((m for m in perf.metrics if m.name == "total_external_deposits"), None)
    history_verified = flows is not None and flows.status == "available"
    return build_valuation(
        fills=fills,
        tx=tx,
        divs=divs,
        ccy=ccy,
        as_of=as_of,
        price_root=cache_dir / "prices",
        fx=fxmod.load_cached(cache_dir),
        overrides=load_overrides(cache_dir / OVERRIDES_FILE),
        history_verified=history_verified,
        ledger_cash=perf.cash_ledger.reconstructed_cash,
        broker_positions=({h.ticker: h.quantity for h in snapshot.holdings} if snapshot else None),
        broker_values={h.ticker: h.current_value for h in snapshot.holdings} if snapshot else None,
        broker_total=(
            snapshot.broker_reported_total_value or snapshot.portfolio_value_incl_cash
            if snapshot
            else None
        ),
    )


def refresh_market_data(
    engine: Engine,
    environment: str,
    account_key: str,
    cache_dir: Path,
    provider: Provider,
    fx_client: Any = None,
    today: date | None = None,
) -> dict[str, str]:
    """Download prices for every instrument ever traded plus ECB FX. Returns per-item status."""
    fills = _load(engine, broker_order_fill, environment, account_key)
    trades = [f for f in fills if f["category"] in ("buy", "sell")]
    overrides = load_overrides(cache_dir / OVERRIDES_FILE)
    first: dict[str, date] = {}
    for f in trades:
        sym = candidate_symbol(f["ticker"], overrides).symbol
        if sym:
            d = f["occurred_at"].date()
            first[sym] = min(first.get(sym, d), d)
    result: dict[str, str] = {}
    for sym, d in sorted(first.items()):
        result[sym] = fetch_symbol(cache_dir / "prices", provider, sym, d, today)
    for f in trades:
        if candidate_symbol(f["ticker"], overrides).symbol is None:
            result[f["ticker"]] = "unmapped: add an override"
    ccys = {c for c in (_major(f.get("instrument_currency")) for f in trades) if c}
    ccys |= {c for c in (_major(f.get("wallet_currency")) for f in trades) if c}
    for sym in first:
        s = load_series(cache_dir / "prices", sym)
        if s and (c := _major(s.currency)):
            ccys.add(c)
    start = min((f["occurred_at"].date() for f in trades), default=today or date.today())
    try:
        table = fxmod.refresh_cache(cache_dir, ccys, start - timedelta(days=14), fx_client, today)
        result["FX"] = f"ok: {', '.join(table.currencies)}"
    except fxmod.FxError as exc:
        result["FX"] = f"failed: {exc}"
    return result


def _major(code: str | None) -> str | None:
    return normalise_currency(code)[0]
