"""Trading 212 instrument ticker -> market-data symbol, with the facts needed to verify it.

A mapping is a hypothesis. `verify_mapping` tests it against the account's own executed fills
(price inside the provider's daily range, currency agreement), so an unverified or wrong symbol
makes the instrument's valuation unavailable rather than silently wrong. Fully sold instruments
are mapped and verified from their historical fills in exactly the same way.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# Trading 212 single-letter exchange markers on non-US tickers -> Yahoo suffix.
_EXCHANGE_SUFFIX = {
    "l": ".L",  # London
    "d": ".DE",  # Xetra
    "a": ".AS",  # Amsterdam
    "p": ".PA",  # Paris
    "m": ".MI",  # Milan
    "e": ".MC",  # Madrid
    "z": ".SW",  # Switzerland
}
_US = re.compile(r"^(?P<root>[A-Z0-9]+(?:_[A-Z])?)_US_EQ$")
_OTHER = re.compile(r"^(?P<root>[A-Z0-9]+)(?P<ex>[a-z])_EQ$")
MINOR_UNITS = {"GBX": ("GBP", Decimal(100)), "GBp": ("GBP", Decimal(100))}
PRICE_BAND_TOLERANCE = 0.02  # extended-hours fills and quote timing
MIN_VERIFIED_FRACTION = 0.95

MappingStatus = Literal["verified", "unverified", "unmapped"]


@dataclass(frozen=True)
class Candidate:
    t212_ticker: str
    symbol: str | None
    source: Literal["rule", "override", "none"]


def normalise_currency(code: str | None) -> tuple[str | None, Decimal]:
    """Return (major currency, divisor). `GBX`/`GBp` are pence: prices divide by 100."""
    if code is None:
        return None, Decimal(1)
    if code in MINOR_UNITS:
        return MINOR_UNITS[code]
    return code.upper(), Decimal(1)


def candidate_symbol(t212_ticker: str, overrides: Mapping[str, str] | None = None) -> Candidate:
    if overrides and t212_ticker in overrides:
        return Candidate(t212_ticker, overrides[t212_ticker], "override")
    m = _US.match(t212_ticker)
    if m:
        return Candidate(t212_ticker, m["root"].replace("_", "-"), "rule")
    m = _OTHER.match(t212_ticker)
    if m and m["ex"] in _EXCHANGE_SUFFIX:
        return Candidate(t212_ticker, m["root"] + _EXCHANGE_SUFFIX[m["ex"]], "rule")
    return Candidate(t212_ticker, None, "none")


def load_overrides(path: Path) -> dict[str, str]:
    """Optional local JSON {"T212_TICKER": "PROVIDER_SYMBOL"}; malformed content is rejected."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in raw.items()
    ):
        raise ValueError("symbol override file must map strings to non-empty strings")
    return dict(raw)


class FillCheck:
    __slots__ = ("detail", "ok")

    def __init__(self, ok: bool, detail: str) -> None:
        self.ok, self.detail = ok, detail


class MappingVerdict:
    def __init__(
        self,
        ticker: str,
        symbol: str | None,
        source: str,
        status: MappingStatus,
        fills_checked: int,
        fills_passed: int,
        provider_currency: str | None,
        instrument_currency: str | None,
        reasons: list[str],
    ) -> None:
        self.ticker, self.symbol, self.source, self.status = ticker, symbol, source, status
        self.fills_checked, self.fills_passed = fills_checked, fills_passed
        self.provider_currency, self.instrument_currency = provider_currency, instrument_currency
        self.reasons = reasons


def as_traded_factor(splits: pd.Series) -> pd.Series:
    """Per-date factor turning the provider's split-adjusted price into the price as traded then.

    Yahoo's `close`/`low`/`high` are split-adjusted to today's share count. A fill executed
    before a split is in old shares, so as-traded = adjusted x product of later split ratios.
    `splits` is indexed by date with the ratio on its ex-date (1.0 otherwise).
    """
    ratios = splits.fillna(1.0).astype(float)
    later = ratios[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return later


def verify_mapping(
    ticker: str,
    candidate: Candidate,
    frame: pd.DataFrame | None,
    provider_currency: str | None,
    fills: Sequence[Mapping[str, Any]],
) -> MappingVerdict:
    """Verify against executed trade fills of this instrument (buys and sells)."""
    trades = [
        f
        for f in fills
        if f["ticker"] == ticker
        and f["category"] in ("buy", "sell")
        and not f.get("anomaly")
        and f.get("price") is not None
    ]
    inst_ccy = next(
        (f.get("instrument_currency") for f in trades if f.get("instrument_currency")), None
    )
    reasons: list[str] = []

    def verdict(status: MappingStatus, checked: int = 0, passed: int = 0) -> MappingVerdict:
        return MappingVerdict(
            ticker,
            candidate.symbol,
            candidate.source,
            status,
            checked,
            passed,
            provider_currency,
            inst_ccy,
            reasons,
        )

    if candidate.symbol is None:
        reasons.append("No symbol rule for this Trading 212 ticker; add a local override.")
        return verdict("unmapped")
    if frame is None or frame.empty:
        reasons.append("No cached provider history for the candidate symbol.")
        return verdict("unverified")
    if not trades:
        reasons.append("No executed fills to verify the symbol against.")
        return verdict("unverified")
    p_major, p_div = normalise_currency(provider_currency)
    i_major, i_div = normalise_currency(inst_ccy)
    if p_major is None or i_major is None or p_major != i_major:
        reasons.append("Provider currency and instrument currency are not the same currency.")
        return verdict("unverified")
    f = frame.set_index(pd.to_datetime(frame["date"])).sort_index()
    factor = as_traded_factor(f["split"])
    low = f["low"] * factor
    high = f["high"] * factor
    passed = 0
    for fill in trades:
        d = pd.Timestamp(fill["occurred_at"].date())
        window = slice(d - timedelta(days=4), d + timedelta(days=1))
        lo, hi = low.loc[window].min(), high.loc[window].max()
        if pd.isna(lo) or pd.isna(hi):
            continue
        # Fill price is in the instrument's quoted unit; provider quote unit may differ (GBX/GBP).
        price = float(Decimal(str(fill["price"])) / i_div * p_div)
        if lo * (1 - PRICE_BAND_TOLERANCE) <= price <= hi * (1 + PRICE_BAND_TOLERANCE):
            passed += 1
    frac = passed / len(trades)
    if frac < MIN_VERIFIED_FRACTION:
        reasons.append(
            f"Only {passed} of {len(trades)} fills fall inside the provider's daily price range."
        )
        return verdict("unverified", len(trades), passed)
    return verdict("verified", len(trades), passed)


def coverage_range(frame: pd.DataFrame | None) -> tuple[date, date] | None:
    if frame is None or frame.empty:
        return None
    d = pd.to_datetime(frame["date"])
    return d.min().date(), d.max().date()
