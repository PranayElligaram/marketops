"""Offline tests for the market-data pipeline (synthetic payloads, mock HTTP)."""

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from marketops.data.provider import ProviderError, YahooProvider, parse_chart
from marketops.data.store import ingest_symbol, load_symbol
from marketops.data.validation import DataQualityError, clean_and_validate

STAMPS = [1704205800, 1704292200, 1704378600]  # 2024-01-02..04 14:30 UTC (09:30 NY)


def payload(**over: object) -> dict:
    q = {
        "open": [10.0, 11.0, 12.0],
        "high": [11.0, 12.0, 13.0],
        "low": [9.0, 10.0, 11.0],
        "close": [10.5, 11.5, 12.5],
        "volume": [100, 200, 300],
    }
    q.update(over)  # type: ignore[arg-type]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": STAMPS,
                    "indicators": {"quote": [q], "adjclose": [{"adjclose": [10.4, 11.4, 12.4]}]},
                    "events": {
                        "dividends": {"x": {"date": STAMPS[1], "amount": 0.5}},
                        "splits": {"y": {"date": STAMPS[2], "numerator": 2, "denominator": 1}},
                    },
                }
            ],
        }
    }


def test_parse_chart_maps_events_and_dates() -> None:
    df = parse_chart(payload(), "T")
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert list(df["dividend"]) == [0.0, 0.5, 0.0]
    assert list(df["split"]) == [1.0, 1.0, 2.0]


@pytest.mark.parametrize("bad", [{}, {"chart": {"result": None, "error": None}}, "x"])
def test_parse_chart_rejects_malformed(bad: object) -> None:
    with pytest.raises(ProviderError):
        parse_chart(bad, "T")


def test_parse_chart_rejects_length_mismatch() -> None:
    with pytest.raises(ProviderError):
        parse_chart(payload(volume=[1, 2]), "T")


def test_parse_chart_provider_error() -> None:
    with pytest.raises(ProviderError, match="Not Found"):
        parse_chart({"chart": {"error": {"code": "Not Found"}, "result": None}}, "T")


def good() -> pd.DataFrame:
    return clean_and_validate(parse_chart(payload(), "T"), "T")


def test_validation_drops_all_null_placeholder_row() -> None:
    df = parse_chart(payload(), "T")
    df.loc[2, ["open", "high", "low", "close", "adj_close", "volume"]] = None
    assert len(clean_and_validate(df, "T")) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.assign(close=[10.5, None, 12.5]),
        lambda d: pd.concat([d, d.iloc[[0]]]),
        lambda d: d.iloc[::-1],
        lambda d: d.assign(low=[9.0, -1.0, 11.0]),
        lambda d: d.assign(volume=[100, -5, 300]),
        lambda d: d.assign(high=[11.0, 9.0, 13.0]),  # high < close
        lambda d: d.drop(columns=["volume"]),
    ],
)
def test_validation_rejects_bad_frames(mutate) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DataQualityError):
        clean_and_validate(mutate(good()), "T")


class FakeProvider:
    name = "fake"

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self.df.copy()


def test_ingest_is_idempotent_and_incremental(tmp_path: Path) -> None:
    full = parse_chart(payload(), "T")
    _, added = ingest_symbol(
        FakeProvider(full.iloc[:2]), tmp_path, "T", date(2024, 1, 1), date(2024, 2, 1)
    )
    assert added == 2
    _, added = ingest_symbol(
        FakeProvider(full.iloc[:2]), tmp_path, "T", date(2024, 1, 1), date(2024, 2, 1)
    )
    assert added == 0
    frame, added = ingest_symbol(
        FakeProvider(full), tmp_path, "T", date(2024, 1, 1), date(2024, 2, 1)
    )
    assert added == 1 and len(frame) == 3
    assert load_symbol(tmp_path, "T").equals(frame)


def test_failed_validation_leaves_cache_untouched(tmp_path: Path) -> None:
    full = parse_chart(payload(), "T")
    ingest_symbol(FakeProvider(full), tmp_path, "T", date(2024, 1, 1), date(2024, 2, 1))
    before = (tmp_path / "T.csv").read_text()
    bad = full.assign(close=[10.5, None, 12.5])
    with pytest.raises(DataQualityError):
        ingest_symbol(FakeProvider(bad), tmp_path, "T", date(2024, 1, 1), date(2024, 2, 1))
    assert (tmp_path / "T.csv").read_text() == before


def provider_with(handler) -> YahooProvider:  # type: ignore[no-untyped-def]
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return YahooProvider(client=client, retries=3, backoff_s=0.0, sleep=lambda _s: None)


def test_provider_retries_on_429_then_succeeds() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, json=payload())

    df = provider_with(handler).fetch_daily("T", date(2024, 1, 1), date(2024, 1, 10))
    assert len(df) == 3 and len(calls) == 3


def test_provider_gives_up_after_retries() -> None:
    with pytest.raises(ProviderError, match="giving up"):
        provider_with(lambda r: httpx.Response(503)).fetch_daily(
            "T", date(2024, 1, 1), date(2024, 1, 2)
        )


def test_provider_network_error_and_404_and_bad_json() -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(ProviderError, match="giving up"):
        provider_with(boom).fetch_daily("T", date(2024, 1, 1), date(2024, 1, 2))
    with pytest.raises(ProviderError, match="unknown symbol"):
        provider_with(lambda r: httpx.Response(404)).fetch_daily(
            "T", date(2024, 1, 1), date(2024, 1, 2)
        )
    with pytest.raises(ProviderError, match="JSON"):
        provider_with(lambda r: httpx.Response(200, text="<html>")).fetch_daily(
            "T", date(2024, 1, 1), date(2024, 1, 2)
        )
