"""Allowlisted, read-only research tools. The LLM can only name a tool and pass validated args.

No tool touches the filesystem by caller-supplied path, the network, SQL or the broker's write
endpoints. Portfolio access is fixed by the request (the model cannot choose a source).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from marketops.agents.schemas import PortfolioSource
from marketops.broker.t212 import Trading212ReadOnlyClient, require_environment_allowed
from marketops.config import Settings
from marketops.inference import service
from marketops.portfolio.analytics import EXAMPLE_POSITIONS, summarise_positions

SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
MAX_ARGS_CHARS = 300
_TICKER_RE = re.compile(r"^[A-Za-z0-9._\-]{1,20}$")


class ToolError(Exception):
    """Safe-to-show failure (invalid args, missing data, unsupported symbol...)."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Args):
    pass


class SymbolArgs(_Args):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not SYMBOL_RE.match(v):
            raise ValueError("symbol must look like a ticker, e.g. AAPL")
        return v


class PriceArgs(SymbolArgs):
    days: int = Field(default=250, ge=20, le=1500)


class BacktestArgs(_Args):
    period: Literal["test", "dev_oof"] = "test"


@dataclass
class ToolDef:
    name: str
    description: str
    args: type[_Args]
    kind: Literal["fact", "model"]
    fn: Callable[[Toolbox, Any], tuple[dict[str, Any], str]]
    sensitive: bool = False


def _r(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v, nd) for v in x]
    return x


class Toolbox:
    def __init__(self, settings: Settings, portfolio_source: PortfolioSource = "none") -> None:
        self.settings = settings
        self.portfolio_source = portfolio_source
        self._lm: service.LoadedModel | None = None
        self.data_root = Path(settings.data_dir)

    @property
    def lm(self) -> service.LoadedModel:
        if self._lm is None:
            try:
                self._lm = service.load_latest(Path(self.settings.model_dir))
            except service.ModelNotAvailable as exc:
                raise ToolError(str(exc)) from None
        return self._lm

    def universe(self) -> set[str]:
        try:
            return set(self.lm.meta["symbols"])
        except ToolError:
            return set()

    # ------------------------------------------------------------------ registry
    def defs(self) -> dict[str, ToolDef]:
        d = {t.name: t for t in _TOOLS}
        if self.portfolio_source == "none":
            d.pop("get_portfolio_exposure")
        return d

    def schemas(self) -> list[dict[str, Any]]:
        out = []
        for t in self.defs().values():
            schema = t.args.model_json_schema()
            schema.pop("title", None)
            for p in schema.get("properties", {}).values():
                p.pop("title", None)
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": schema,
                    },
                }
            )
        return out

    def execute(
        self, name: Any, raw_args: Any
    ) -> tuple[ToolDef, dict[str, Any], dict[str, Any], str]:
        """Validate and run. Returns (def, validated args, data, source) or raises ToolError."""
        tools = self.defs()
        if not isinstance(name, str) or name not in tools:
            raise ToolError(f"Unknown or unavailable tool {str(name)[:40]!r}.")
        if raw_args is None:
            raw_args = {}
        if isinstance(raw_args, str):
            if len(raw_args) > MAX_ARGS_CHARS:
                raise ToolError("Tool arguments too large.")
            try:
                raw_args = json.loads(raw_args or "{}")
            except ValueError:
                raise ToolError("Tool arguments are not valid JSON.") from None
        if not isinstance(raw_args, dict):
            raise ToolError("Tool arguments must be an object.")
        if len(json.dumps(raw_args, default=str)) > MAX_ARGS_CHARS:
            raise ToolError("Tool arguments too large.")
        t = tools[name]
        try:
            args = t.args(**raw_args)
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments: {exc.errors()[0].get('msg')}") from None
        except TypeError:
            raise ToolError("Invalid arguments.") from None
        try:
            data, source = t.fn(self, args)
        except ToolError:
            raise
        except service.UnsupportedSymbol as exc:
            raise ToolError(str(exc)) from None
        except (service.ModelNotAvailable, FileNotFoundError) as exc:
            raise ToolError(f"Data or model artefact unavailable: {exc}") from None
        except ValueError as exc:
            raise ToolError(f"Cached data invalid: {exc}") from None
        return t, args.model_dump(), _r(data), source

    # ------------------------------------------------------------------ implementations
    def _symbols(self, _: NoArgs) -> tuple[dict[str, Any], str]:
        lm = self.lm
        return (
            {"model_universe": lm.meta["symbols"], "benchmark": "SPY (history only, not scored)"},
            f"model:{lm.version}/metadata.json",
        )

    def _price(self, a: PriceArgs) -> tuple[dict[str, Any], str]:
        df = service.history(self.data_root, self.lm, a.symbol, a.days)
        if len(df) < 2:
            raise ToolError(f"Not enough cached bars for {a.symbol}.")
        first, last = df.iloc[0], df.iloc[-1]
        age = (datetime.now(UTC).date() - last["date"].date()).days
        return (
            {
                "symbol": a.symbol,
                "bars": int(len(df)),
                "first_date": str(first["date"].date()),
                "last_date": str(last["date"].date()),
                "last_close": float(last["close"]),
                "window_total_return_adj": float(last["adj_close"] / first["adj_close"] - 1),
                "window_high_close": float(df["close"].max()),
                "window_low_close": float(df["close"].min()),
                "avg_daily_volume": float(df["volume"].mean()),
                "data_age_days": age,
                "provider": "yahoo_chart (unofficial; personal research use)",
            },
            f"data/raw/{a.symbol}.csv",
        )

    def _signal(self, a: SymbolArgs) -> tuple[dict[str, Any], str]:
        r = service.analyse(self.data_root, self.lm, a.symbol)
        r.pop("warnings", None)
        r["horizon"] = "next 5 sessions"
        r["note"] = "Model estimate. Not calibrated out of sample; no demonstrated edge."
        return r, f"model:{self.lm.version}+data/raw/{a.symbol}.csv"

    def _metadata(self, _: NoArgs) -> tuple[dict[str, Any], str]:
        m = self.lm.meta
        keys = (
            "version",
            "target",
            "selected_model",
            "threshold",
            "dev_end",
            "test_start",
            "symbols",
            "train_rows_final",
            "test_rows",
            "caveats",
        )
        return {k: m.get(k) for k in keys}, f"model:{self.lm.version}/metadata.json"

    def _evaluation(self, _: NoArgs) -> tuple[dict[str, Any], str]:
        m = self.lm.meta
        t = m["test"]
        keep = (
            "n",
            "base_rate",
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "f1",
            "balanced_accuracy",
        )
        out = {
            "model_version": self.lm.version,
            "held_out_test": {
                k: {x: v.get(x) for x in keep if x in v}
                for k, v in t.items()
                if isinstance(v, dict) and "roc_auc" in v
            },
            "calibration_bins_model": (t.get("model") or {}).get("calibration"),
            "walk_forward_cv": m.get("walk_forward_cv"),
            "caveats": m.get("caveats"),
        }
        return out, f"model:{self.lm.version}/metadata.json"

    def _backtest(self, a: BacktestArgs) -> tuple[dict[str, Any], str]:
        bt = service.load_backtest(self.lm)
        dates: list[str] = []
        for curve in ((bt.get("equity_curves") or {}).get(a.period) or {}).values():
            dates += [str(min(curve)), str(max(curve))] if curve else []
        end = max(dates) if dates else None
        dev_end = self.lm.meta.get("dev_end")
        overlap = a.period == "dev_oof" and end and dev_end and end > str(dev_end)
        return (
            {
                "period": a.period,
                "period_start": min(dates) if dates else None,
                "period_end": end,
                "period_warning": (
                    f"This dev_oof curve ends {end}, after the development end date {dev_end}: it "
                    "overlaps the held-out test period and is not development-only evidence."
                )
                if overlap
                else None,
                "metrics": bt[a.period],
                "cost_bps_one_way": bt["cost_bps_one_way"],
                "rebalance_every_sessions": bt["rebalance_every_sessions"],
                "assumptions": bt["assumptions"],
            },
            f"model:{self.lm.version}/backtest.json",
        )

    def _shap(self, _: NoArgs) -> tuple[dict[str, Any], str]:
        p = self.lm.directory / "shap.json"
        if not p.exists():
            raise ToolError("No SHAP report for this model.")
        raw = json.loads(p.read_text())["mean_abs_shap_logodds"]
        top = dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:8])
        return (
            {"mean_abs_shap_logodds_top8": top, "note": "Describes the model, not the market."},
            f"model:{self.lm.version}/shap.json",
        )

    def _portfolio(self, _: NoArgs) -> tuple[dict[str, Any], str]:
        s = self.settings
        if self.portfolio_source == "example":
            ov = summarise_positions(EXAMPLE_POSITIONS, source="synthetic_example")
            label = "SYNTHETIC DEMONSTRATION DATA, not a real account"
        else:
            if not s.agent_allow_broker_portfolio:
                raise ToolError("Broker portfolio access is disabled for the agent.")
            if not s.broker_configured:
                raise ToolError("Trading 212 read-only keys are not configured.")
            try:
                require_environment_allowed(s.t212_environment, s.t212_allow_live)
                rows = Trading212ReadOnlyClient(
                    api_key=s.t212_api_key,
                    api_secret=s.t212_api_secret,
                    environment=s.t212_environment,
                ).get_positions()
                ov = summarise_positions(rows, source="trading212")
            except Exception as exc:  # broker/validation details may be sensitive
                raise ToolError(f"Portfolio unavailable ({type(exc).__name__}).") from None
            label = "Trading 212 open positions (read-only snapshot)"
        w = sorted((h.weight_pct for h in ov.holdings), reverse=True)
        hhi = sum((x / 100) ** 2 for x in w)
        holdings = [
            {
                "ticker": h.ticker if _TICKER_RE.match(h.ticker) else "UNKNOWN",
                "weight_pct": h.weight_pct,
            }
            for h in ov.holdings
        ]
        return (
            {
                "data_label": label,
                "holdings_count": len(w),
                "weights": holdings,
                "largest_weight_pct": w[0] if w else 0,
                "top3_weight_pct": round(sum(w[:3]), 2),
                "herfindahl_index": hhi,
                "effective_number_of_holdings": (1 / hhi) if hhi else 0,
                "excludes": "cash, fees, realised P/L, history; absolute values withheld",
            },
            f"portfolio:{ov.source}",
        )


_TOOLS: list[ToolDef] = [
    ToolDef(
        "list_supported_symbols",
        "List stock symbols the research model covers.",
        NoArgs,
        "fact",
        Toolbox._symbols,
    ),
    ToolDef(
        "get_price_summary",
        "Summary statistics of cached daily price history for a symbol.",
        PriceArgs,
        "fact",
        Toolbox._price,
    ),
    ToolDef(
        "get_ml_signal",
        "Latest ML model probability that the stock rises over the next 5 sessions "
        "(a model estimate with no demonstrated edge).",
        SymbolArgs,
        "model",
        Toolbox._signal,
    ),
    ToolDef(
        "get_model_metadata",
        "ML model version, target, threshold and training window.",
        NoArgs,
        "fact",
        Toolbox._metadata,
    ),
    ToolDef(
        "get_model_evaluation",
        "Held-out and walk-forward evaluation of the ML model vs baselines, incl. calibration.",
        NoArgs,
        "fact",
        Toolbox._evaluation,
    ),
    ToolDef(
        "get_backtest",
        "Historical strategy backtest metrics (model vs baselines).",
        BacktestArgs,
        "fact",
        Toolbox._backtest,
    ),
    ToolDef(
        "get_shap_importance",
        "Top SHAP feature importances of the ML model.",
        NoArgs,
        "model",
        Toolbox._shap,
    ),
    ToolDef(
        "get_portfolio_exposure",
        "Portfolio concentration and weights (read-only, no absolute values).",
        NoArgs,
        "fact",
        Toolbox._portfolio,
        sensitive=True,
    ),
]
