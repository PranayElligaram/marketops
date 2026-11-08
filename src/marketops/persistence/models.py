"""PostgreSQL tables (SQLAlchemy Core). Created idempotently with create_all."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)
metadata = MetaData()

model_registry = Table(
    "model_registry",
    metadata,
    Column("version", String(32), primary_key=True),
    Column("selected_model", String(64), nullable=False),
    Column("registered_utc", DateTime(timezone=True), nullable=False),
    Column("metadata", JSON, nullable=False),
)

prediction_log = Table(
    "prediction_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_utc", DateTime(timezone=True), nullable=False),
    Column("model_version", String(32), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("as_of_date", String(10), nullable=False),
    Column("probability_up", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("signal", String(16), nullable=False),
    Column("features", JSON, nullable=False),
)

agent_run = Table(
    "agent_run",
    metadata,
    Column("run_id", String(32), primary_key=True),
    Column("created_utc", DateTime(timezone=True), nullable=False),
    Column("status", String(32), nullable=False),
    Column("mode", String(32), nullable=False),
    Column("llm_model", String(64), nullable=False),
    Column("llm_status", String(64), nullable=False),
    Column("question_sha256", String(64), nullable=False),
    Column("question_preview", String(200), nullable=False),
    Column("portfolio_source", String(16), nullable=False),
    Column("iterations", Integer, nullable=False),
    Column("tool_calls", Integer, nullable=False),
    Column("duration_ms", Float, nullable=False),
    Column("rejected_claims", Integer, nullable=False),
    Column("flags", JSON, nullable=False),
    Column("tool_trace", JSON, nullable=False),
    Column("evidence_refs", JSON, nullable=False),
)


# ---- Trading 212 account history (read-only imports). Every row is bound to an environment
# (demo/live) and an opaque account_key = sha256(environment:account_id)[:16], so the raw account
# number is never stored and demo and live data can never mix. Raw broker payloads are not kept.
_MONEY = Numeric(24, 8)


def _scope_cols() -> list[Column[Any]]:
    return [
        Column("environment", String(8), nullable=False),
        Column("account_key", String(16), nullable=False),
    ]


broker_sync_run = Table(
    "broker_sync_run",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    *_scope_cols(),
    Column("kind", String(16), nullable=False),  # orders | transactions | dividends
    Column("mode", String(12), nullable=False),  # full | incremental
    # running | complete | interrupted | failed
    Column("status", String(16), nullable=False),
    Column("started_utc", DateTime(timezone=True), nullable=False),
    Column("finished_utc", DateTime(timezone=True)),
    Column("reached_end", Boolean, nullable=False, default=False),  # nextPagePath was null
    Column("pages_fetched", Integer, nullable=False, default=0),
    Column("records_seen", Integer, nullable=False, default=0),
    Column("records_inserted", Integer, nullable=False, default=0),
    Column("records_duplicate", Integer, nullable=False, default=0),
    Column("records_rejected", Integer, nullable=False, default=0),
    Column("records_conflicting", Integer, nullable=False, default=0),
    Column("reject_reasons", JSON, nullable=False, default=dict),  # {reason_code: count}
    Column("error_code", String(32)),
    Column("account_currency", String(8)),
    Column("broker_realised_pnl", _MONEY),  # account summary value at orders sync time
    Index("ix_sync_scope_kind", "environment", "account_key", "kind", "id"),
)

broker_order_fill = Table(
    "broker_order_fill",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    *_scope_cols(),
    Column("source_id", String(48), nullable=False),  # "<order id>:<fill id or ->"
    Column("order_id", BigInteger, nullable=False),
    Column("fill_id", BigInteger),
    Column("ticker", String(48), nullable=False),
    # buy | sell | corporate_action | unclassified_fill | unfilled
    Column("category", String(24), nullable=False),
    Column("side", String(8)),
    Column("order_type", String(16)),
    Column("order_status", String(24)),
    Column("fill_type", String(40)),
    Column("trading_method", String(8)),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("quantity", _MONEY),
    Column("price", _MONEY),  # instrument currency
    Column("order_currency", String(8)),
    Column("instrument_currency", String(8)),
    Column("wallet_currency", String(8)),
    Column("net_value", _MONEY),  # wallet currency, sign as reported
    Column("fx_rate", _MONEY),
    Column("realised_pnl", _MONEY),  # wallet currency, broker-reported per fill
    Column("fees_total", _MONEY),  # only when all fee entries share the wallet currency
    Column("taxes", JSON, nullable=False, default=list),
    Column("anomaly", String(48)),
    Column("content_hash", String(64), nullable=False),
    Column("first_seen_utc", DateTime(timezone=True), nullable=False),
    Column("sync_run_id", Integer, nullable=False),
    UniqueConstraint("environment", "account_key", "source_id", name="uq_order_fill_source"),
    Index("ix_order_fill_time", "environment", "account_key", "occurred_at"),
)

broker_cash_transaction = Table(
    "broker_cash_transaction",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    *_scope_cols(),
    Column("source_id", String(64), nullable=False),  # broker `reference`
    Column("tx_type", String(32), nullable=False),  # as reported
    # deposit | withdrawal | fee | interest | transfer | unclassified
    Column("category", String(16), nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("amount", _MONEY, nullable=False),  # sign as reported
    Column("currency", String(8), nullable=False),
    Column("anomaly", String(48)),
    Column("content_hash", String(64), nullable=False),
    Column("first_seen_utc", DateTime(timezone=True), nullable=False),
    Column("sync_run_id", Integer, nullable=False),
    UniqueConstraint("environment", "account_key", "source_id", name="uq_cash_tx_source"),
    Index("ix_cash_tx_time", "environment", "account_key", "occurred_at"),
)

broker_dividend = Table(
    "broker_dividend",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    *_scope_cols(),
    Column("source_id", String(64), nullable=False),  # broker `reference`
    Column("ticker", String(48), nullable=False),
    Column("div_type", String(80), nullable=False),  # as reported
    Column("category", String(24), nullable=False),  # dividend | interest | other_distribution
    Column("occurred_at", DateTime(timezone=True), nullable=False),  # paidOn
    Column("amount", _MONEY, nullable=False),  # account currency, as credited by the broker
    Column("currency", String(8), nullable=False),
    Column("amount_eur", _MONEY),
    Column("gross_per_share", _MONEY),  # instrument currency
    Column("quantity", _MONEY),
    Column("ticker_currency", String(8)),
    Column("instrument_currency", String(8)),
    Column("anomaly", String(48)),
    Column("content_hash", String(64), nullable=False),
    Column("first_seen_utc", DateTime(timezone=True), nullable=False),
    Column("sync_run_id", Integer, nullable=False),
    UniqueConstraint("environment", "account_key", "source_id", name="uq_dividend_source"),
    Index("ix_dividend_time", "environment", "account_key", "occurred_at"),
)


# Manual hypothetical research journal. NEVER broker executions or inferred fills.
# Protected by the private API token and held only in the local PostgreSQL database.
research_journal = Table(
    "research_journal",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("recorded_utc", DateTime(timezone=True), nullable=False),
    Column("decision_day", Date, nullable=False),
    Column("symbol", String(20), nullable=False),
    Column("thesis", String(1200), nullable=False),
    Column("risk_note", String(600), nullable=False),
    Column("hypothetical_entry_gbp", _MONEY, nullable=False),
    Column("hypothetical_units", _MONEY, nullable=False),
    Column("planned_stop_gbp", _MONEY),
    Column("retrospective", Boolean, nullable=False),
    Column("status", String(16), nullable=False, default="open"),
    Column("outcome_day", Date),
    Column("hypothetical_exit_gbp", _MONEY),
    Column("outcome_note", String(1200)),
    Column("closed_utc", DateTime(timezone=True)),
    Index("ix_research_journal_recorded", "recorded_utc"),
)


def init_schema(engine: Engine) -> None:
    metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        # create_all never alters an existing table; journals created before closed_utc existed.
        with engine.begin() as c:
            c.execute(
                text("ALTER TABLE research_journal ADD COLUMN IF NOT EXISTS closed_utc TIMESTAMPTZ")
            )


def register_model(engine: Engine, meta: dict[str, Any]) -> None:
    init_schema(engine)
    with engine.begin() as c:
        exists = c.execute(
            select(model_registry.c.version).where(model_registry.c.version == meta["version"])
        ).first()
        if not exists:
            c.execute(
                insert(model_registry).values(
                    version=meta["version"],
                    selected_model=meta["selected_model"],
                    registered_utc=datetime.now(UTC),
                    metadata=meta,
                )
            )


def log_prediction(engine: Engine, row: dict[str, Any]) -> bool:
    """Best-effort: a DB outage must not break read-only analysis. Returns success."""
    try:
        with engine.begin() as c:
            c.execute(insert(prediction_log).values(created_utc=datetime.now(UTC), **row))
        return True
    except SQLAlchemyError as exc:
        logger.warning("prediction persistence failed: %s", type(exc).__name__)
        return False


def recent_predictions(engine: Engine, limit: int = 50) -> list[dict[str, Any]]:
    with engine.connect() as c:
        rows = c.execute(
            select(prediction_log).order_by(prediction_log.c.id.desc()).limit(limit)
        ).mappings()
        return [{**r, "created_utc": r["created_utc"].isoformat()} for r in rows]


def log_agent_run(engine: Engine, row: dict[str, Any]) -> bool:
    """Best-effort audit write; never raises. Rows hold hashes/refs, never tool payloads."""
    try:
        init_schema(engine)
        with engine.begin() as c:
            c.execute(insert(agent_run).values(created_utc=datetime.now(UTC), **row))
        return True
    except SQLAlchemyError as exc:
        logger.warning("agent audit persistence failed: %s", type(exc).__name__)
        return False


def recent_agent_runs(engine: Engine, limit: int = 20) -> list[dict[str, Any]]:
    with engine.connect() as c:
        rows = c.execute(
            select(agent_run).order_by(agent_run.c.created_utc.desc()).limit(limit)
        ).mappings()
        return [{**r, "created_utc": r["created_utc"].isoformat()} for r in rows]
