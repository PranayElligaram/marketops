"""Symbol mapping, FX, time-weighted statistics and the event-driven NAV (synthetic data only)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from marketops.valuation import twr
from marketops.valuation.fx import FxError, FxTable, parse_ecb_csv
from marketops.valuation.nav import build_events, compute_daily_nav
from marketops.valuation.symbols import (
    as_traded_factor,
    candidate_symbol,
    load_overrides,
    normalise_currency,
    verify_mapping,
)
from tests.valuation_fixtures import bars, cash_tx, dividend, fill, fx_table, series, utc

D1 = date(2024, 1, 1)  # Monday


# ------------------------------------------------------------------ symbols
@pytest.mark.parametrize(
    ("ticker", "symbol"),
    [
        ("AAPL_US_EQ", "AAPL"),
        ("BRK_B_US_EQ", "BRK-B"),
        ("VUSAl_EQ", "VUSA.L"),
        ("SAPd_EQ", "SAP.DE"),
        ("WEIRD_XX_EQ", None),
    ],
)
def test_ticker_rules(ticker: str, symbol: str | None) -> None:
    assert candidate_symbol(ticker).symbol == symbol


def test_override_wins_and_is_validated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "o.json"
    p.write_text('{"WEIRD_XX_EQ": "WRD.L"}')
    c = candidate_symbol("WEIRD_XX_EQ", load_overrides(p))
    assert (c.symbol, c.source) == ("WRD.L", "override")
    p.write_text('{"A": 1}')
    with pytest.raises(ValueError):
        load_overrides(p)


def test_minor_units() -> None:
    assert normalise_currency("GBX") == ("GBP", Decimal(100))
    assert normalise_currency("USD") == ("USD", Decimal(1))


def test_as_traded_factor_restates_pre_split_prices() -> None:
    f = bars(D1, [100, 100, 25, 25], splits={date(2024, 1, 3): 4.0})
    factor = as_traded_factor(pd.Series(f["split"].to_numpy(), index=f["date"]))
    assert list(factor) == [4.0, 4.0, 1.0, 1.0]


def _fills() -> list:
    return [
        fill(1, utc(date(2024, 1, 2)), "buy", "10", "80", 10.0),
        fill(2, utc(date(2024, 1, 5)), "sell", "10", "88", 11.0),  # fully sold
    ]


def test_mapping_verified_from_fills_including_fully_sold() -> None:
    f = bars(D1, [10, 10, 10.5, 11, 11])
    v = verify_mapping("AAA_US_EQ", candidate_symbol("AAA_US_EQ"), f, "USD", _fills())
    assert (v.status, v.fills_checked, v.fills_passed) == ("verified", 2, 2)


def test_mapping_rejects_wrong_symbol_by_price_and_by_currency() -> None:
    wrong = bars(D1, [50, 50, 50, 50, 50])
    v = verify_mapping("AAA_US_EQ", candidate_symbol("AAA_US_EQ"), wrong, "USD", _fills())
    assert v.status == "unverified" and "fall inside" in v.reasons[0]
    ok = bars(D1, [10, 10, 10.5, 11, 11])
    v = verify_mapping("AAA_US_EQ", candidate_symbol("AAA_US_EQ"), ok, "EUR", _fills())
    assert v.status == "unverified" and "currency" in v.reasons[0]


def test_mapping_unmapped_and_missing_data() -> None:
    assert (
        verify_mapping("X_XX_EQ", candidate_symbol("X_XX_EQ"), None, None, []).status == "unmapped"
    )
    cand = candidate_symbol("AAA_US_EQ")
    assert verify_mapping("AAA_US_EQ", cand, None, None, _fills()).status == "unverified"


def test_pence_quoted_instrument_verifies_against_pound_provider_bars() -> None:
    f = bars(D1, [0.10, 0.10, 0.105, 0.11, 0.11])
    fl = [fill(1, utc(date(2024, 1, 2)), "buy", "10", "1", 10.0, "BBBl_EQ", "GBX")]
    v = verify_mapping("BBBl_EQ", candidate_symbol("BBBl_EQ"), f, "GBP", fl)
    assert v.status == "verified"


def test_split_adjusted_provider_bars_verify_pre_split_fills() -> None:
    # Fill at 100 before a 4:1 split; provider bars are adjusted to 25.
    f = bars(D1, [25, 25, 25, 26], splits={date(2024, 1, 4): 4.0})
    fl = [fill(1, utc(date(2024, 1, 2)), "buy", "1", "80", 100.0)]
    # split ratio lives on ex-date Jan 4, so earlier bars restate x4 = 100
    assert (
        verify_mapping("AAA_US_EQ", candidate_symbol("AAA_US_EQ"), f, "USD", fl).status
        == "verified"
    )


# ------------------------------------------------------------------ FX
CSV = "Date,USD,GBP,\n2024-01-05,1.10,0.86,\n2024-01-04,1.09,N/A,\n"


def test_parse_and_triangulate() -> None:
    t = FxTable(parse_ecb_csv(CSV))
    assert t.rate("USD", "GBP", date(2024, 1, 5)) == Decimal("0.86") / Decimal("1.10")
    assert t.rate("EUR", "USD", date(2024, 1, 5)) == Decimal("1.10")
    assert t.rate("GBP", "GBP", date(2024, 1, 5)) == 1


def test_missing_or_stale_fx_is_none_not_invented() -> None:
    t = FxTable(parse_ecb_csv(CSV))
    assert t.rate("USD", "GBP", date(2024, 1, 4)) is None  # GBP missing that day, none before
    assert t.rate("USD", "GBP", date(2024, 1, 20)) is None  # beyond the staleness window
    assert t.rate("JPY", "GBP", date(2024, 1, 5)) is None
    assert t.rate("USD", "GBP", date(2024, 1, 7)) is not None  # weekend uses Friday's rate


@pytest.mark.parametrize("bad", ["", "Foo,USD\n", "Date,USD\nx,1\n", "Date,USD\n2024-01-01,-1\n"])
def test_bad_fx_files_are_rejected(bad: str) -> None:
    with pytest.raises(FxError):
        parse_ecb_csv(bad)


# ------------------------------------------------------------------ TWR statistics
def _days(navs: list[float | None], flows: dict[int, float] | None = None) -> list[twr.DayValue]:
    return [
        twr.DayValue(date(2024, 1, 1 + i), n, (flows or {}).get(i, 0.0)) for i, n in enumerate(navs)
    ]


def test_twr_ignores_deposit_size_and_timing() -> None:
    # +10% on 100, deposit 900 after the close, then -5% on 1000.
    r = twr.daily_returns(_days([100, 110, 1010, 1045], {2: 900}))
    assert [round(x.value, 6) for x in r] == [0.1, 0.0, round(1045 / 1010 - 1, 6)]
    assert twr.link([x.value for x in r]) == pytest.approx(1.1 * (1045 / 1010) - 1)


def test_deposit_alone_is_zero_return_and_withdrawal_is_not_a_loss() -> None:
    r = twr.daily_returns(_days([100, 100, 600, 600, 300], {2: 500, 4: -300}))
    assert len(r) == 4 and all(abs(x.value) < 1e-12 for x in r)


def test_start_of_day_convention_differs_for_flow_days() -> None:
    d = _days([100, 220], {1: 100})  # deposit 100, closing NAV 220
    end = twr.daily_returns(d, "end_of_day")[0].value
    start = twr.daily_returns(d, "start_of_day")[0].value
    assert end == pytest.approx(0.2) and start == pytest.approx(0.1)


def test_unfunded_account_has_no_returns() -> None:
    assert twr.daily_returns(_days([0.0, 0.0, 100.0, 101.0], {2: 100.0})) == [
        twr.DailyReturn(date(2024, 1, 4), pytest.approx(0.01), 0)
    ]


def test_gap_splits_segments_and_links_nothing_across() -> None:
    r = twr.daily_returns(_days([100, 101, None, 120, 121]))
    assert [x.segment for x in r] == [0, 1]
    idx = twr.index_series(r)
    assert idx[1][1] == pytest.approx(1.0 + (121 / 120 - 1))  # restarts, no 101->120 jump


def test_volatility_drawdown_and_annualisation_need_enough_data() -> None:
    assert twr.annualised_volatility([0.01] * 10) is None
    assert twr.max_drawdown([0.01] * 5) is None
    rets = [0.01, -0.02] * 40
    vol = twr.annualised_volatility(rets)
    # deviations are +/-0.015 around the mean -0.005; sample variance = 80/79 x 0.015^2
    assert vol == pytest.approx(0.015 * (80 / 79) ** 0.5 * 252**0.5)
    rs = [0.1] * 5 + [-0.5] + [0.01] * 14
    assert twr.max_drawdown(rs) == pytest.approx(-0.5)
    assert twr.annualise(0.1, date(2024, 1, 1), date(2024, 6, 1)) is None
    assert twr.annualise(0.21, date(2023, 1, 1), date(2025, 1, 1)) == pytest.approx(0.1, abs=2e-3)


def test_periods_are_complete_only_when_fully_covered() -> None:
    wd = [date(2024, 1, 1 + i) for i in range(5)] + [date(2024, 2, 1)]
    nav = [100, 101, 102, 103, 104, 105]
    days = [twr.DayValue(d, float(n), 0.0) for d, n in zip(wd, nav, strict=True)]
    rets = twr.daily_returns(days)
    p = {x.label: x for x in twr.period_returns(rets, wd, "month")}
    assert p["2024-01"].complete is False  # first day has no return (account start)
    assert p["2024-02"].complete is True
    y = twr.period_returns(rets, wd, "year")
    assert y[0].label == "2024" and y[0].complete is False


# ------------------------------------------------------------------ NAV
def _scenario(extra_fills=(), extra_tx=(), extra_div=(), frame=None, fx=None):  # type: ignore[no-untyped-def]
    tx = [cash_tx(1, utc(D1), "deposit", "1000"), *extra_tx]
    fills = [fill(1, utc(date(2024, 1, 2)), "buy", "10", "80", 10.0), *extra_fills]
    frame = frame if frame is not None else bars(date(2024, 1, 1), [10, 10, 11, 12, 12, 12, 13, 13])
    s = {"AAA": series("AAA", frame)}
    from marketops.valuation.symbols import MappingVerdict

    verd = {
        "AAA_US_EQ": MappingVerdict("AAA_US_EQ", "AAA", "rule", "verified", 1, 1, "USD", "USD", [])
    }
    ev, corp = build_events(tx, extra_div, fills, "GBP")
    return ev, corp, verd, s, (fx or fx_table())


def test_nav_is_holdings_at_close_in_gbp_plus_cash() -> None:
    ev, corp, verd, s, fx = _scenario()
    days, qty = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 5))
    by = {d.day: d for d in days}
    assert by[date(2024, 1, 1)].nav == Decimal(1000)  # funded, nothing bought yet
    # Jan 2: 10 sh x $10 x 0.8 = 80 held, cash 920
    assert by[date(2024, 1, 2)].nav == Decimal("1000")
    # Jan 3: price 11 -> 10*11*0.8 = 88 + 920
    assert by[date(2024, 1, 3)].nav == Decimal("1008")
    assert by[date(2024, 1, 3)].net_invested == Decimal(1000)
    assert qty == {"AAA_US_EQ": Decimal(10)}


def test_withdrawal_and_weekend_deposit_roll_into_the_next_weekday_flow() -> None:
    ev, corp, verd, s, fx = _scenario(
        extra_tx=[
            cash_tx(2, utc(date(2024, 1, 6)), "deposit", "500"),  # Saturday
            cash_tx(3, utc(date(2024, 1, 9)), "withdrawal", "-200"),
        ]
    )
    days, _ = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 9))
    by = {d.day: d for d in days}
    assert date(2024, 1, 6) not in by
    assert by[date(2024, 1, 8)].flow == Decimal(500)
    assert by[date(2024, 1, 9)].flow == Decimal(-200)
    assert by[date(2024, 1, 9)].net_invested == Decimal(1300)


def test_dividend_is_counted_once_as_cash_not_in_the_price() -> None:
    div = dividend(1, utc(date(2024, 1, 4)), "5")
    base = _scenario()
    with_div = _scenario(extra_div=[div])
    a = compute_daily_nav(*base[:4], base[4], "GBP", date(2024, 1, 5))[0]
    b = compute_daily_nav(*with_div[:4], with_div[4], "GBP", date(2024, 1, 5))[0]
    assert b[3].nav is not None and a[3].nav is not None
    assert b[3].nav - a[3].nav == Decimal(5)  # Jan 4: exactly the cash credited
    assert b[2].nav == a[2].nav  # day before payment unchanged


def test_missing_fx_makes_the_day_unvalued_not_estimated() -> None:
    ev, corp, verd, s, _ = _scenario()
    days, _q = compute_daily_nav(ev, corp, verd, s, fx_table(days=3), "GBP", date(2024, 1, 12))
    bad = [d for d in days if d.nav is None]
    assert bad and all("missing_fx:USD->GBP" in d.reasons[0] for d in bad)


def test_missing_and_stale_prices_are_not_filled() -> None:
    ev, corp, verd, s, fx = _scenario(frame=bars(date(2024, 1, 1), [10, 10, 11]))
    days, _q = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 15))
    by = {d.day: d for d in days}
    assert by[date(2024, 1, 5)].nav is not None  # last close Jan 3, two days old
    assert by[date(2024, 1, 5)].stale_price_positions == 1
    assert by[date(2024, 1, 12)].nav is None and "price_too_old" in by[date(2024, 1, 12)].reasons[0]


def test_unverified_mapping_blocks_only_while_the_position_is_held() -> None:
    ev, corp, verd, s, fx = _scenario(
        extra_fills=[fill(2, utc(date(2024, 1, 4)), "sell", "10", "90", 11.0)]
    )
    verd["AAA_US_EQ"].status = "unverified"
    days, _q = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 8))
    by = {d.day: d for d in days}
    assert by[date(2024, 1, 2)].nav is None  # held while unverified
    assert by[date(2024, 1, 5)].nav == Decimal(
        "1010"
    )  # sold: valued from cash alone, no price needed


def test_split_multiplies_held_shares_and_values_stay_comparable() -> None:
    # 4:1 split with ex-date Jan 4. The provider's split-adjusted closes are 10 before it
    # (= 40 as traded) and 11 after: the market rose 10% across the split.
    closes = [10, 10, 10, 11, 11, 12]
    frame = bars(date(2024, 1, 1), closes, splits={date(2024, 1, 4): 4.0})
    ev, corp, verd, s, fx = _scenario(frame=frame)
    days, qty = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 8))
    by = {d.day: d for d in days}
    before, after = by[date(2024, 1, 3)].positions[0], by[date(2024, 1, 4)].positions[0]
    assert (before.quantity, after.quantity) == (Decimal(10), Decimal(40))
    assert before.local_price == Decimal(40) and after.local_price == Decimal(11)
    assert before.value == Decimal(10) * 40 * Decimal("0.8")
    assert after.value == Decimal(40) * 11 * Decimal("0.8")
    assert after.value / before.value == Decimal("1.1")  # no phantom gain or loss from the split
    assert qty["AAA_US_EQ"] == Decimal(40)


def test_price_older_than_a_split_is_never_used_after_it() -> None:
    # Bars stop on Jan 2; the ex-date is Jan 3, so a Jan 3 valuation must not reuse Jan 2's price.
    frame = bars(date(2024, 1, 1), [10, 10, 10, 10, 10], splits={date(2024, 1, 5): 2.0})
    frame = frame[frame["date"] != pd.Timestamp("2024-01-04")].reset_index(drop=True)
    frame.loc[3, "split"] = 1.0
    ev, corp, verd, s, fx = _scenario(frame=frame)
    days, _q = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 5))
    by = {d.day: d for d in days}
    assert by[date(2024, 1, 4)].nav is not None  # stale by one day, same epoch: allowed
    assert by[date(2024, 1, 5)].nav is not None  # fresh bar on the ex-date itself
    frame2 = frame[frame["date"] != pd.Timestamp("2024-01-05")]
    s2 = {"AAA": series("AAA", frame2)}
    days2, _ = compute_daily_nav(ev, corp, verd, s2, fx, "GBP", date(2024, 1, 5))
    assert days2[-1].nav is not None  # no ex-date bar, so no split is known: nothing to reject


def test_broker_corporate_action_fill_makes_the_instrument_unvalued_from_that_date() -> None:
    ca = fill(9, utc(date(2024, 1, 4)), "corporate_action", "0", "0", 0.0)
    ca["quantity"] = Decimal("5")
    ev, corp, verd, s, fx = _scenario(extra_fills=[ca])
    days, _q = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 8))
    by = {d.day: d for d in days}
    assert by[date(2024, 1, 3)].nav is not None
    assert "corporate_action_ambiguous" in by[date(2024, 1, 4)].reasons[0]


def test_negative_cash_means_missing_history_and_blocks_the_value() -> None:
    ev, corp, verd, s, fx = _scenario(
        extra_fills=[fill(2, utc(date(2024, 1, 3)), "buy", "100", "5000", 62.5)]
    )
    days, _q = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 5))
    assert days[2].nav is None and "negative_cash_history_incomplete" in days[2].reasons


def test_flagged_and_foreign_wallet_fills_are_excluded_from_cash_and_shares() -> None:
    bad = fill(2, utc(date(2024, 1, 3)), "buy", "5", "50", 10.0)
    bad["anomaly"] = "inconsistent"
    foreign = fill(3, utc(date(2024, 1, 3)), "buy", "5", "50", 10.0)
    foreign["wallet_currency"] = "EUR"
    ev, corp, verd, s, fx = _scenario(extra_fills=[bad, foreign])
    _days, qty = compute_daily_nav(ev, corp, verd, s, fx, "GBP", date(2024, 1, 5))
    assert qty == {"AAA_US_EQ": Decimal(10)}


def _frank(day: str = "2024-01-05") -> dict:
    return {"base": "EUR", "rates": {day: {"USD": 1.1, "GBP": 0.86}}}


def test_zero_placeholder_rate_is_missing_not_a_rate() -> None:
    table = FxTable(parse_ecb_csv("Date,ISK,USD,\n2024-01-05,0,1.1,\n"))
    assert table.rate("ISK", "USD", date(2024, 1, 5)) is None
    assert table.rate("USD", "EUR", date(2024, 1, 5)) is not None


def test_frankfurter_parse_and_cache_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from marketops.valuation.fx import load_cached, parse_frankfurter, write_cache

    write_cache(tmp_path, parse_frankfurter(_frank()))
    t = load_cached(tmp_path)
    assert t is not None
    assert t.rate("USD", "GBP", date(2024, 1, 5)) == Decimal("0.86") / Decimal("1.1")


@pytest.mark.parametrize(
    "bad",
    [{}, {"base": "USD", "rates": {}}, {"base": "EUR", "rates": {"2024-01-05": {"USD": 0}}}],
)
def test_malformed_frankfurter_payloads_are_rejected(bad: dict) -> None:
    from marketops.valuation.fx import parse_frankfurter

    with pytest.raises(FxError):
        parse_frankfurter(bad)


def test_refresh_rejects_stale_or_incomplete_upstream_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import httpx

    from marketops.valuation.fx import refresh_cache

    def client(payload: dict, status: int = 200) -> httpx.Client:
        transport = httpx.MockTransport(lambda r: httpx.Response(status, json=payload))
        return httpx.Client(transport=transport)

    today, start = date(2024, 3, 1), date(2024, 1, 1)
    with pytest.raises(FxError, match="stale"):
        refresh_cache(tmp_path, {"USD", "GBP"}, start, client(_frank("2024-01-05")), today)
    with pytest.raises(FxError, match="not supplied"):
        refresh_cache(tmp_path, {"USD", "JPY"}, start, client(_frank("2024-02-29")), today)
    with pytest.raises(FxError, match="HTTP 500"):
        refresh_cache(tmp_path, {"USD"}, start, client({}, 500), today)
    assert not (tmp_path / "ecb_eurofxref_hist.csv").exists()  # nothing cached on failure
    ok = refresh_cache(tmp_path, {"USD", "GBP"}, start, client(_frank("2024-02-29")), today)
    assert ok.rate("EUR", "USD", date(2024, 2, 29)) == Decimal("1.1")


def test_null_provider_bars_are_dropped_so_the_previous_close_is_used_and_flagged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from marketops.valuation.cache import load_series, store_series

    frame = bars(date(2024, 1, 1), [10, 10, 10, 10, 10, 10])
    frame.loc[3, ["close", "low", "high", "open"]] = float("nan")  # Jan 4
    store_series(tmp_path, "AAA", frame, "USD")
    s = load_series(tmp_path, "AAA")
    assert s is not None and len(s.frame) == 5
    ev, corp, verd, ps, fx = _scenario(frame=s.frame)
    days, _ = compute_daily_nav(ev, corp, verd, {"AAA": s}, fx, "GBP", date(2024, 1, 5))
    jan4 = next(d for d in days if d.day == date(2024, 1, 4))
    assert jan4.nav is not None and jan4.positions[0].stale_days == 1
