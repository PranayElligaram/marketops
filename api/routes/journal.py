"""Private, local-only hypothetical research journal. No execution or broker interaction.

All entries are manually supplied; there are no market-data lookups or fabricated fills.
Backward-dated ideas are tagged retrospective. Returns are illustrative gross arithmetic.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from api.security import private_access
from marketops.persistence import database
from marketops.persistence.models import init_schema, research_journal

router = APIRouter(prefix="/journal", tags=["journal"], dependencies=[Depends(private_access)])

SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,19}$")
NOTE = (
    "Manual hypothetical research record, not an actual order or verified execution. "
    "P/L excludes spreads, FX, fees, taxes and slippage."
)


class NewIdea(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    decision_day: date
    thesis: str = Field(min_length=5, max_length=1200)
    risk_note: str = Field(min_length=5, max_length=600)
    hypothetical_entry_gbp: Decimal = Field(gt=0, le=10000000)
    hypothetical_units: Decimal = Field(gt=0, le=100000000)
    planned_stop_gbp: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check(self) -> NewIdea:
        self.symbol = self.symbol.upper()
        if not SYMBOL.fullmatch(self.symbol):
            raise ValueError("Invalid ticker")
        if self.decision_day > datetime.now(UTC).date():
            raise ValueError("Decision date must not be in the future")
        if (
            self.planned_stop_gbp is not None
            and self.planned_stop_gbp >= self.hypothetical_entry_gbp
        ):
            raise ValueError("For a long-only research idea the stop must be below entry")
        return self


class CloseIdea(BaseModel):
    outcome_day: date
    hypothetical_exit_gbp: Decimal = Field(gt=0, le=10000000)
    outcome_note: str = Field(min_length=5, max_length=1200)


class JournalEntry(BaseModel):
    id: int
    recorded_utc: datetime
    decision_day: date
    symbol: str
    thesis: str
    risk_note: str
    hypothetical_entry_gbp: Decimal
    hypothetical_units: Decimal
    planned_stop_gbp: Decimal | None
    retrospective: bool
    status: str
    outcome_day: date | None
    hypothetical_exit_gbp: Decimal | None
    outcome_note: str | None
    hypothetical_gross_pnl_gbp: Decimal | None
    limitation: str = NOTE


def _engine() -> Engine:
    try:
        engine = database.get_engine()
        init_schema(engine)
        return engine
    except (RuntimeError, SQLAlchemyError):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Local journal database is unavailable."
        ) from None


def _entry(row: dict[str, object]) -> JournalEntry:
    price = row.get("hypothetical_exit_gbp")
    pnl = (
        (Decimal(str(price)) - Decimal(str(row["hypothetical_entry_gbp"])))
        * Decimal(str(row["hypothetical_units"]))
        if price is not None
        else None
    )
    return JournalEntry(**row, hypothetical_gross_pnl_gbp=pnl)


@router.post("/ideas", response_model=JournalEntry, status_code=201)
def create_idea(idea: NewIdea) -> JournalEntry:
    engine = _engine()
    recorded = datetime.now(UTC)
    values = idea.model_dump()
    values.update(
        recorded_utc=recorded,
        retrospective=idea.decision_day < recorded.date(),
        status="open",
    )
    try:
        with engine.begin() as connection:
            result = connection.execute(insert(research_journal).values(**values))
            key = result.inserted_primary_key[0]
            row = (
                connection.execute(select(research_journal).where(research_journal.c.id == key))
                .mappings()
                .one()
            )
            return _entry(dict(row))
    except SQLAlchemyError:
        raise HTTPException(503, "Local journal write failed.") from None


@router.get("/ideas", response_model=list[JournalEntry])
def list_ideas(limit: Annotated[int, Query(ge=1, le=100)] = 50) -> list[JournalEntry]:
    engine = _engine()
    try:
        with engine.connect() as connection:
            rows = (
                connection.execute(
                    select(research_journal).order_by(research_journal.c.id.desc()).limit(limit)
                )
                .mappings()
                .all()
            )
        return [_entry(dict(row)) for row in rows]
    except SQLAlchemyError:
        raise HTTPException(503, "Local journal read failed.") from None


@router.post("/ideas/{idea_id}/close", response_model=JournalEntry)
def close_idea(idea_id: int, outcome: CloseIdea) -> JournalEntry:
    engine = _engine()
    if outcome.outcome_day > datetime.now(UTC).date():
        raise HTTPException(422, "Outcome date must not be in the future.")
    try:
        with engine.begin() as connection:
            old = (
                connection.execute(select(research_journal).where(research_journal.c.id == idea_id))
                .mappings()
                .first()
            )
            if old is None:
                raise HTTPException(404, "Journal idea not found.")
            if old["status"] != "open":
                raise HTTPException(409, "This research idea already has an outcome.")
            if outcome.outcome_day < old["decision_day"]:
                raise HTTPException(422, "Outcome cannot predate the original decision.")
            connection.execute(
                update(research_journal)
                .where(research_journal.c.id == idea_id)
                .values(
                    status="closed",
                    outcome_day=outcome.outcome_day,
                    hypothetical_exit_gbp=outcome.hypothetical_exit_gbp,
                    outcome_note=outcome.outcome_note,
                )
            )
            row = (
                connection.execute(select(research_journal).where(research_journal.c.id == idea_id))
                .mappings()
                .one()
            )
            return _entry(dict(row))
    except SQLAlchemyError:
        raise HTTPException(503, "Local journal update failed.") from None
