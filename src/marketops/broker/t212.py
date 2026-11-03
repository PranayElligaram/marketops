"""Limited read-only Trading 212 API client (Public API v0, HTTP Basic key:secret).

Only fixed GET endpoints are reachable: positions, account summary and the three history
lists (orders, transactions, dividends). There is intentionally no method for placing, editing
or cancelling orders, and no way to pass a path or URL into the client; history pagination
accepts only a validated same-endpoint `nextPagePath` (see `validate_next_page_path`).
Rate limits (docs.trading212.com): positions 1 req/1 s, account summary 1 req/5 s,
history endpoints 20 req/1 min each.
"""

import re
import time
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

import httpx

Environment = Literal["demo", "live"]
HistoryKind = Literal["orders", "transactions", "dividends"]

HISTORY_PATHS: dict[HistoryKind, str] = {
    "orders": "/equity/history/orders",
    "transactions": "/equity/history/transactions",
    "dividends": "/equity/history/dividends",
}
HISTORY_PAGE_LIMIT = 50  # documented maximum
HISTORY_MIN_INTERVAL_S = 3.1  # 60 s / 20 requests, plus margin
MAX_RATE_LIMIT_WAIT_S = 70.0
_API_PREFIX = "/api/v0"
_ALLOWED_QUERY = {"limit", "cursor", "ticker", "time"}
_CURSOR_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


class BrokerError(Exception):
    """A safe-to-display message about brokerage connectivity (never contains credentials).

    `code` is a stable machine-readable reason ("auth", "scope", "rate_limit", "timeout",
    "unavailable", "network", "schema", "pagination", ...) so callers never parse message text.
    `retry_after_s` is set for rate limits when the broker said when to retry.
    """

    def __init__(
        self, message: str, code: str = "error", retry_after_s: float | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after_s = retry_after_s


def require_environment_allowed(environment: str, allow_live: bool) -> None:
    """Demo is the default; live reads need an explicit second opt-in. Never switches silently."""
    if environment == "live" and not allow_live:
        raise BrokerError(
            "Live Trading 212 access is disabled. Set MARKETOPS_T212_ALLOW_LIVE=true to opt in.",
            "live_disabled",
        )


def validate_next_page_path(kind: HistoryKind, next_path: object) -> dict[str, str]:
    """Accept only a relative nextPagePath for the SAME history endpoint; return its query.

    Rejects absolute URLs, other hosts or paths, userinfo, fragments, traversal, repeated or
    unknown query keys and malformed limit/cursor values. The request is rebuilt from the
    configured origin and fixed path, so no broker-supplied string is requested verbatim.
    """
    bad = BrokerError("Trading 212 returned an unsafe or malformed pagination link.", "pagination")
    if not isinstance(next_path, str) or not 0 < len(next_path) <= 512:
        raise bad
    if any(ord(c) < 33 or ord(c) > 126 for c in next_path) or "\\" in next_path:
        raise bad
    parts = urlsplit(next_path)
    if parts.scheme or parts.netloc or parts.fragment:
        raise bad
    if parts.path != f"{_API_PREFIX}{HISTORY_PATHS[kind]}":
        raise bad
    try:
        pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=bool(parts.query))
    except ValueError:
        raise bad from None
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys) or not set(keys) <= _ALLOWED_QUERY:
        raise bad
    query = dict(pairs)
    if "limit" in query and not (query["limit"].isdigit() and 1 <= int(query["limit"]) <= 50):
        raise bad
    if "cursor" in query and not _CURSOR_RE.match(query["cursor"]):
        raise bad
    return query


class HistoryPacer:
    """Client-side spacing between history requests, plus bounded waits after a 429."""

    def __init__(
        self,
        min_interval_s: float = HISTORY_MIN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s, self._clock, self._sleep = min_interval_s, clock, sleep
        self._last: float | None = None

    def wait(self, extra_s: float = 0.0) -> None:
        delay = extra_s
        if self._last is not None:
            delay = max(delay, self.min_interval_s - (self._clock() - self._last))
        if delay > 0:
            self._sleep(delay)
        self._last = self._clock()


class Trading212ReadOnlyClient:
    BASE_URLS = {
        "demo": "https://demo.trading212.com/api/v0",
        "live": "https://live.trading212.com/api/v0",
    }
    _POSITIONS = "/equity/positions"
    _ACCOUNT_SUMMARY = "/equity/account/summary"

    def __init__(self, api_key: str, api_secret: str, environment: Environment):
        if not api_key or not api_secret:
            raise BrokerError("Trading 212 read-only credentials are not configured.", "config")
        if environment not in self.BASE_URLS:
            raise BrokerError("Trading 212 environment must be 'demo' or 'live'.", "config")
        self.environment: Environment = environment
        self.base_url = self.BASE_URLS[environment]
        self._auth = httpx.BasicAuth(api_key, api_secret)

    def __repr__(self) -> str:
        return f"Trading212ReadOnlyClient(environment={self.environment!r})"

    def get_positions(self) -> list[dict[str, Any]]:
        payload = self._get(self._POSITIONS)
        if not isinstance(payload, list) or not all(isinstance(x, dict) for x in payload):
            raise BrokerError("Trading 212 returned an unexpected positions response.", "schema")
        return payload

    def get_account_summary(self) -> dict[str, Any]:
        payload = self._get(self._ACCOUNT_SUMMARY)
        if not isinstance(payload, dict):
            raise BrokerError(
                "Trading 212 returned an unexpected account summary response.", "schema"
            )
        return payload

    def get_history_page(
        self,
        kind: HistoryKind,
        next_page_path: str | None = None,
        pacer: HistoryPacer | None = None,
        retries: int = 2,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One page of a history endpoint -> (items, nextPagePath or None when finished).

        The first page is requested with limit=50. Later pages must come from the previous
        response's nextPagePath, validated by `validate_next_page_path`. 429/408/5xx/network
        failures are retried (paced) up to `retries` times; 401/403 never are.
        """
        query = {"limit": str(HISTORY_PAGE_LIMIT)}
        if next_page_path is not None:
            query = validate_next_page_path(kind, next_page_path)
        pacer = pacer or HistoryPacer()
        payload: Any = None
        for attempt in range(retries + 1):
            pacer.wait()
            try:
                payload = self._get(HISTORY_PATHS[kind], query)
                break
            except BrokerError as exc:
                retryable = exc.code in ("rate_limit", "timeout", "unavailable", "network")
                if not retryable or attempt == retries:
                    raise
                if exc.code == "rate_limit":
                    wait = exc.retry_after_s
                    if wait is not None and wait > MAX_RATE_LIMIT_WAIT_S:
                        raise
                    pacer.wait(extra_s=wait if wait is not None else 20.0)
                else:
                    pacer.wait(extra_s=2.0 * (attempt + 1))
        bad = BrokerError(f"Trading 212 returned an unexpected {kind} history response.", "schema")
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise bad
        items = payload["items"]
        if not all(isinstance(x, dict) for x in items):
            raise bad
        nxt = payload.get("nextPagePath")
        if nxt is not None and not isinstance(nxt, str):
            raise bad
        return items, nxt or None

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        # Fixed paths only. Never accept a URL/path from a request or from an LLM.
        try:
            with httpx.Client(auth=self._auth, timeout=10.0, follow_redirects=False) as client:
                url = f"{self.base_url}{path}"
                response = client.get(url, params=params) if params else client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 401:
                raise BrokerError("Trading 212 rejected the API key or secret.", "auth") from None
            if code == 403:
                raise BrokerError(
                    "Trading 212 API key lacks a required read scope (account, portfolio, "
                    "history:orders, history:transactions, history:dividends).",
                    "scope",
                ) from None
            if code == 408:
                raise BrokerError("Trading 212 timed out. Retry later.", "timeout") from None
            if code == 429:
                wait = _reset_hint(exc.response)
                raise BrokerError(
                    f"Trading 212 rate limit reached. Retry {wait}.",
                    "rate_limit",
                    _retry_after_s(exc.response),
                ) from None
            if code >= 500:
                raise BrokerError(
                    f"Trading 212 is unavailable (HTTP {code}).", "unavailable"
                ) from None
            raise BrokerError(f"Trading 212 returned HTTP {code}.", "http") from None
        except (httpx.RequestError, ValueError) as exc:
            raise BrokerError(
                "Trading 212 is currently unreachable or returned invalid data.", "network"
            ) from exc


def _reset_hint(response: httpx.Response) -> str:
    """x-ratelimit-reset is an epoch timestamp; only a bounded integer is echoed."""
    try:
        reset = int(response.headers.get("x-ratelimit-reset", ""))
    except ValueError:
        return "later"
    return f"after epoch {reset}" if 0 < reset < 10**11 else "later"


def _retry_after_s(response: httpx.Response, now: Callable[[], float] = time.time) -> float | None:
    """Seconds until x-ratelimit-reset (epoch), bounded; None when absent or implausible."""
    try:
        reset = int(response.headers.get("x-ratelimit-reset", ""))
    except ValueError:
        return None
    if not 0 < reset < 10**11:
        return None
    return min(max(reset - now(), 1.0), 3600.0)
