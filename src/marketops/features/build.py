"""Point-in-time feature engineering and labels.

Every feature at row date t uses only data with date <= t (end-of-day close known at t).
The label is the forward 5-session total-return sign and uses data up to t+5; it is
never used as a feature. `label_end` records the date the label depends on, which the
training code uses for purging.

All returns use `adj_close` (split+dividend adjusted), i.e. total return. Volume is
split-adjusted by the provider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5
FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",
    "ret_63",
    "ret_126",
    "vol_21",
    "vol_63",
    "price_sma50",
    "dist_high_252",
    "vol_ratio_21",
    "dvol_chg_5",
    "mkt_ret_21",
    "excess_ret_21",
]


def symbol_features(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Features + label for one symbol. `bench` is the benchmark frame (same schema)."""
    d = df.set_index("date")
    px = d["adj_close"]
    lr = np.log(px).diff()
    out = pd.DataFrame(index=d.index)
    for n in (1, 5, 10, 21, 63, 126):
        out[f"ret_{n}"] = np.log(px).diff(n)
    out["vol_21"] = lr.rolling(21).std()
    out["vol_63"] = lr.rolling(63).std()
    out["price_sma50"] = px / px.rolling(50).mean() - 1
    out["dist_high_252"] = px / px.rolling(252).max() - 1
    vol = d["volume"].replace(0, np.nan)
    out["vol_ratio_21"] = vol / vol.rolling(21).mean() - 1
    dvol = (vol * d["close"]).rolling(5).mean()
    out["dvol_chg_5"] = np.log(dvol / dvol.shift(5))
    b = bench.set_index("date")["adj_close"]
    mkt21 = np.log(b).diff(21).reindex(d.index)
    out["mkt_ret_21"] = mkt21
    out["excess_ret_21"] = out["ret_21"] - mkt21
    fwd = px.shift(-HORIZON) / px - 1
    out["fwd_ret"] = fwd
    out["label"] = np.where(fwd.isna(), np.nan, (fwd > 0).astype(float))
    dates = pd.Series(d.index, index=d.index)
    out["label_end"] = dates.shift(-HORIZON)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reset_index()


def build_dataset(
    frames: dict[str, pd.DataFrame], benchmark: str = "SPY", drop_after: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Pooled panel for all non-benchmark symbols, sorted by (date, symbol).

    Rows without a full feature window are dropped. Rows with unknown labels (the last
    HORIZON sessions) are KEPT with NaN label so they can be scored but never trained on.
    `drop_after` removes bars dated after the cutoff (e.g. a possibly incomplete current bar).
    """
    bench = frames[benchmark]
    parts = []
    for sym, f in frames.items():
        if sym == benchmark:
            continue
        if drop_after is not None:
            f = f[f["date"] <= drop_after]
        p = symbol_features(f, bench[bench["date"] <= f["date"].max()])
        p.insert(1, "symbol", sym)
        parts.append(p.dropna(subset=FEATURES))
    ds = pd.concat(parts).sort_values(["date", "symbol"]).reset_index(drop=True)
    return ds
