"""Yahoo Finance chart-endpoint provider (unofficial; personal research use only).

The endpoint is undocumented and its terms restrict redistribution, so downloaded data is cached
locally, git-ignored, and never shipped. Swap `Provider` for another source if needed.

Price convention: `close` is split-adjusted only (Yahoo's raw `close`); `adj_close` is split- and
dividend-adjusted (total-return). Open/high/low are split-adjusted. Dividends/splits are kept.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, Protocol

import httpx
import pandas as pd

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


class ProviderError(Exception):
    """Provider failed in a way retrying will not fix, or retries are exhausted."""


class Provider(Protocol):
    name: str

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...


def _ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def parse_chart(payload: Any, symbol: str) -> pd.DataFrame:
    """Parse a chart payload into date/OHLC/adj_close/volume/dividend/split columns.

    Raises ProviderError on any structural problem; never fabricates values.
    """
    try:
        chart = payload["chart"]
        if chart.get("error"):
            raise ProviderError(f"{symbol}: provider error {chart['error'].get('code')}")
        result = chart["result"][0]
        stamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"]["adjclose"][0]["adjclose"]
        tz = result["meta"].get("exchangeTimezoneName", "America/New_York")
        currency = result["meta"].get("currency")
        cols = {k: quote[k] for k in ("open", "high", "low", "close", "volume")}
    except ProviderError:
        raise
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError(f"{symbol}: malformed response ({type(exc).__name__})") from None
    n = len(stamps)
    if any(len(v) != n for v in cols.values()) or len(adj) != n:
        raise ProviderError(f"{symbol}: inconsistent array lengths in provider response")
    idx = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    df = pd.DataFrame({**cols, "adj_close": adj}, index=idx)
    df.index.name = "date"
    df["dividend"] = 0.0
    df["split"] = 1.0
    events = result.get("events") or {}
    try:
        for e in (events.get("dividends") or {}).values():
            d = pd.Timestamp(datetime.fromtimestamp(e["date"], UTC).date())
            if d in df.index:
                df.loc[d, "dividend"] = float(e["amount"])
        for e in (events.get("splits") or {}).values():
            d = pd.Timestamp(datetime.fromtimestamp(e["date"], UTC).date())
            if d in df.index:
                df.loc[d, "split"] = float(e["numerator"]) / float(e["denominator"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProviderError(f"{symbol}: malformed corporate-action event") from exc
    out = df.reset_index()
    out.attrs["currency"] = currency
    return out


class YahooProvider:
    name = "yahoo_chart"

    def __init__(
        self,
        client: httpx.Client | None = None,
        retries: int = 3,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=15.0, headers={"User-Agent": "Mozilla/5.0 (MarketOps research)"}
        )
        self.retries, self.backoff_s, self._sleep = retries, backoff_s, sleep

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        params: dict[str, str | int] = {
            "period1": _ts(start),
            "period2": _ts(end),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        last = ""
        for attempt in range(self.retries):
            try:
                r = self._client.get(f"{BASE_URL}/{symbol}", params=params)
            except httpx.RequestError as exc:
                last = type(exc).__name__
                self._sleep(self.backoff_s * 2**attempt)
                continue
            if r.status_code == 404:
                raise ProviderError(f"{symbol}: unknown symbol")
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After", "")
                last = f"HTTP {r.status_code}"
                self._sleep(float(ra) if ra.isdigit() else self.backoff_s * 2**attempt)
                continue
            if r.status_code >= 400:
                raise ProviderError(f"{symbol}: HTTP {r.status_code}")
            try:
                payload = r.json()
            except ValueError:
                raise ProviderError(f"{symbol}: response was not valid JSON") from None
            return parse_chart(payload, symbol)
        raise ProviderError(f"{symbol}: giving up after {self.retries} attempts ({last})")
