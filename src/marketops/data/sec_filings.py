"""Read-only SEC EDGAR filing metadata with provenance and deterministic validation.

Source: the SEC's public ``data.sec.gov`` submissions API and ``company_tickers.json``. They need no
key, are free, are updated as filings are disseminated, and are limited to 10 requests per second
per user; automated clients must declare a descriptive User-Agent with a contact address. The SEC
asks for citation as the source. This is official company disclosure metadata (filing form, dates and
links), not a news feed: it contains no headlines and no commentary.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
ATTRIBUTION = "Source: U.S. Securities and Exchange Commission, EDGAR (www.sec.gov)."
LIMITATION = (
    "Filing metadata only: no headline, interpretation or price impact. A filing's presence, "
    "absence or form type says nothing by itself about a company's prospects. Not investment advice."
)
# Material-event and periodic-report forms only; insider, fund and ownership noise is excluded.
FORMS = frozenset({"8-K", "10-K", "10-Q", "6-K", "20-F", "40-F", "DEF 14A"})
MIN_INTERVAL_S = 0.25  # comfortably inside the 10 requests per second fair-access limit
MAX_BYTES = 12_000_000  # the ticker map is a few hundred kB; cap defensively
TICKER_TTL_S = 24 * 3600
FILINGS_TTL_S = 15 * 60

_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_DOCUMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class SecUnavailable(Exception):
    """The source could not be reached or returned data that failed validation."""


@dataclass(frozen=True)
class Filing:
    form: str
    filing_date: str
    accepted_utc: str | None
    accession_number: str
    description: str
    items: list[str]
    url: str


def _clean_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else ""
    return re.sub(r"[\x00-\x1f\x7f]", " ", text).strip()[:limit]


def cik_for_ticker(payload: Any, ticker: str) -> tuple[int, str]:
    """Return (CIK, company title) for ``ticker`` from the SEC ticker map, or raise KeyError."""
    symbol = ticker.upper()
    if not _TICKER.fullmatch(symbol) or not isinstance(payload, dict):
        raise KeyError(symbol)
    for entry in payload.values():
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("ticker"), str)
            and entry["ticker"].upper() == symbol
            and isinstance(entry.get("cik_str"), int)
            and entry["cik_str"] > 0
        ):
            return entry["cik_str"], _clean_text(entry.get("title"), 200)
    raise KeyError(symbol)


def parse_filings(
    payload: Any, cik: int, *, today: date, limit: int = 15
) -> tuple[str, list[Filing]]:
    """Validate a submissions document. Returns (company name, newest-first filings).

    Rows that fail validation are dropped rather than repaired; a structurally wrong document
    raises ``SecUnavailable``.
    """
    try:
        if int(payload["cik"]) != cik:
            raise SecUnavailable("SEC response is for a different company.")
        recent = payload["filings"]["recent"]
        columns = {
            k: recent[k]
            for k in ("accessionNumber", "filingDate", "form", "primaryDocument")
        }
    except (KeyError, TypeError, ValueError):
        raise SecUnavailable("SEC submissions response has an unexpected structure.") from None
    count = len(columns["accessionNumber"])
    if any(len(v) != count for v in columns.values()):
        raise SecUnavailable("SEC submissions columns have inconsistent lengths.")
    optional = {k: recent.get(k) or [] for k in ("acceptanceDateTime", "primaryDocDescription", "items")}
    filings: list[Filing] = []
    for i in range(count):
        form = columns["form"][i]
        accession = columns["accessionNumber"][i]
        document = columns["primaryDocument"][i]
        if form not in FORMS or not isinstance(accession, str) or not isinstance(document, str):
            continue
        if not _ACCESSION.fullmatch(accession) or not _DOCUMENT.fullmatch(document):
            continue
        try:
            filed = date.fromisoformat(columns["filingDate"][i])
        except (TypeError, ValueError):
            continue
        if filed > today:
            continue
        accepted = None
        if len(optional["acceptanceDateTime"]) == count:
            try:
                accepted = (
                    datetime.fromisoformat(optional["acceptanceDateTime"][i].replace("Z", "+00:00"))
                    .astimezone(UTC)
                    .isoformat()
                )
            except (AttributeError, TypeError, ValueError):
                accepted = None
        description = (
            _clean_text(optional["primaryDocDescription"][i], 120)
            if len(optional["primaryDocDescription"]) == count
            else ""
        )
        raw_items = optional["items"][i] if len(optional["items"]) == count else ""
        items = [p for p in _clean_text(raw_items, 80).split(",") if re.fullmatch(r"\d\.\d{2}", p)]
        filings.append(
            Filing(
                form=form,
                filing_date=filed.isoformat(),
                accepted_utc=accepted,
                accession_number=accession,
                description=description,
                items=items,
                url=ARCHIVE_URL.format(
                    cik=cik, accession=accession.replace("-", ""), document=document
                ),
            )
        )
    filings.sort(key=lambda f: (f.filing_date, f.accession_number), reverse=True)
    return _clean_text(payload.get("name"), 200), filings[:limit]


class SecClient:
    """Minimal rate-limited GET client with a short in-process cache. Never writes anywhere."""

    def __init__(self, user_agent: str) -> None:
        self._headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self._lock = threading.Lock()
        self._last = 0.0
        self._cache: dict[str, tuple[float, datetime, Any]] = {}

    def _get_json(self, url: str, ttl: float) -> Any:
        return self._fetch(url, ttl)[1]

    def _fetch(self, url: str, ttl: float) -> tuple[datetime, Any]:
        """Return (UTC time the SEC response was actually fetched, data), serving the cache within ttl."""
        now = time.monotonic()
        hit = self._cache.get(url)
        if hit is not None and now - hit[0] < ttl:
            return hit[1], hit[2]
        with self._lock:
            wait = self._last + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        try:
            with httpx.Client(headers=self._headers, timeout=10.0, follow_redirects=False) as c:
                response = c.get(url)
            response.raise_for_status()
            if len(response.content) > MAX_BYTES:
                raise SecUnavailable("SEC response is unexpectedly large.")
            data = response.json()
        except (httpx.HTTPError, ValueError):
            raise SecUnavailable("SEC EDGAR could not be reached or returned invalid data.") from None
        fetched = datetime.now(UTC)
        self._cache[url] = (time.monotonic(), fetched, data)
        return fetched, data

    def recent_filings(self, ticker: str, limit: int = 15) -> dict[str, Any]:
        try:
            cik, title = cik_for_ticker(self._get_json(TICKERS_URL, TICKER_TTL_S), ticker)
        except KeyError:
            raise LookupError(ticker) from None
        retrieved, payload = self._fetch(SUBMISSIONS_URL.format(cik=cik), FILINGS_TTL_S)
        name, filings = parse_filings(payload, cik, today=retrieved.date(), limit=limit)
        return {
            "symbol": ticker.upper(),
            "cik": cik,
            "company": name or title,
            "forms_included": sorted(FORMS),
            "filings": [asdict(f) for f in filings],
            "source": "SEC EDGAR submissions API",
            "source_url": SUBMISSIONS_URL.format(cik=cik),
            "retrieved_utc": retrieved.isoformat(),
            "attribution": ATTRIBUTION,
            "limitation": LIMITATION,
        }
