"""Historical FX from the ECB euro foreign-exchange reference rates (free, reuse permitted with
attribution), fetched through the open-source Frankfurter API. Rates are EUR-based:
1 EUR = r units, published each TARGET business day.

A rate is never invented: a date with no published rate within `MAX_STALE_DAYS` returns None and
the caller must mark the valuation unavailable. Cross rates triangulate through EUR. A download
that is stale, truncated or missing a requested currency is rejected rather than cached.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

FRANKFURTER_URL = "https://api.frankfurter.dev/v1"
FRESH_DAYS = 10
MAX_STALE_DAYS = 5
CACHE_NAME = "ecb_eurofxref_hist.csv"


class FxError(Exception):
    pass


def parse_ecb_csv(text: str) -> dict[str, dict[date, Decimal]]:
    """Wide ECB-style CSV (Date, USD, JPY, ...) -> {currency: {date: units per EUR}}."""
    reader = csv.reader(io.StringIO(text))
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        raise FxError("empty FX file") from None
    if not header or header[0] != "Date":
        raise FxError("unexpected FX file header")
    out: dict[str, dict[date, Decimal]] = {c: {} for c in header[1:] if c}
    for row in reader:
        if not row or not row[0]:
            continue
        try:
            d = date.fromisoformat(row[0].strip())
        except ValueError:
            raise FxError("malformed FX date") from None
        for ccy, cell in zip(header[1:], row[1:], strict=False):
            cell = cell.strip()
            if not ccy or cell in ("", "N/A"):
                continue
            try:
                v = Decimal(cell)
            except ArithmeticError:
                raise FxError("malformed FX rate") from None
            if v < 0:
                raise FxError("negative FX rate")
            if v == 0:
                continue  # the ECB file uses 0 as a placeholder for "not quoted": missing
            out[ccy][d] = v
    return out


def parse_frankfurter(payload: Any) -> dict[str, dict[date, Decimal]]:
    """Frankfurter JSON (ECB reference rates, EUR base) -> {currency: {date: units per EUR}}."""
    try:
        if payload["base"] != "EUR":
            raise FxError("FX base currency is not EUR")
        out: dict[str, dict[date, Decimal]] = {}
        for day, per in payload["rates"].items():
            d = date.fromisoformat(day)
            for ccy, v in per.items():
                rate = Decimal(str(v))
                if rate <= 0:
                    raise FxError("non-positive FX rate")
                out.setdefault(ccy, {})[d] = rate
    except (KeyError, TypeError, ValueError, AttributeError, ArithmeticError):
        raise FxError("malformed FX response") from None
    return out


class FxTable:
    def __init__(
        self, rates: dict[str, dict[date, Decimal]], max_stale_days: int = MAX_STALE_DAYS
    ) -> None:
        self._rates = rates
        self.max_stale_days = max_stale_days

    @property
    def currencies(self) -> list[str]:
        return sorted(self._rates)

    def coverage(self, ccy: str) -> tuple[date, date] | None:
        t = self._rates.get(ccy)
        return (min(t), max(t)) if t else None

    def _per_eur(self, ccy: str, on: date) -> Decimal | None:
        """Units of `ccy` per EUR on `on`, using the latest rate not after it (bounded)."""
        if ccy == "EUR":
            return Decimal(1)
        table = self._rates.get(ccy)
        if not table:
            return None
        for back in range(self.max_stale_days + 1):
            v = table.get(on - timedelta(days=back))
            if v is not None:
                return v
        return None

    def rate(self, src: str, dst: str, on: date) -> Decimal | None:
        """Units of `dst` per 1 `src`. None if either leg is unavailable (never fabricated)."""
        if src == dst:
            return Decimal(1)
        s, d = self._per_eur(src, on), self._per_eur(dst, on)
        if s is None or d is None:
            return None
        return d / s


def load_cached(root: Path) -> FxTable | None:
    path = root / CACHE_NAME
    if not path.exists():
        return None
    return FxTable(parse_ecb_csv(path.read_text(encoding="utf-8")))


def write_cache(root: Path, rates: dict[str, dict[date, Decimal]]) -> None:
    ccys = sorted(rates)
    days = sorted({d for r in rates.values() for d in r}, reverse=True)
    lines = ["Date," + ",".join(ccys) + ","]
    for d in days:
        cells = ",".join(str(rates[c].get(d, "N/A")) for c in ccys)
        lines.append(f"{d.isoformat()},{cells},")
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (CACHE_NAME + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(root / CACHE_NAME)


def refresh_cache(
    root: Path,
    currencies: set[str],
    start: date,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> FxTable:
    """Download daily ECB reference rates (only currency codes and dates are sent).

    The result must reach within `FRESH_DAYS` of today and contain every requested currency,
    or it is rejected and the existing cache is left untouched.
    """
    wanted = sorted(c for c in currencies if c != "EUR")
    if not wanted:
        raise FxError("no foreign currencies requested")
    end = today or datetime.now(UTC).date()
    c = client or httpx.Client(timeout=60.0)
    url = f"{FRANKFURTER_URL}/{start.isoformat()}..{end.isoformat()}"
    try:
        r = c.get(url, params={"base": "EUR", "symbols": ",".join(wanted)})
    except httpx.RequestError as exc:
        raise FxError(f"FX download failed ({type(exc).__name__})") from None
    if r.status_code != 200:
        raise FxError(f"FX download failed (HTTP {r.status_code})")
    try:
        table = parse_frankfurter(r.json())
    except ValueError:
        raise FxError("FX response was not valid JSON") from None
    missing = [w for w in wanted if w not in table]
    if missing:
        raise FxError(f"currency not supplied: {', '.join(missing)}")
    if (end - max(max(v) for v in table.values())).days > FRESH_DAYS:
        raise FxError("FX data is stale")
    write_cache(root, table)
    return FxTable(table)
