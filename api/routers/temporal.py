"""
/temporal router — Blix v0.3.7

Endpoints
---------
GET  /temporal/state/{entity}/{attribute}        — current tracked value
GET  /temporal/state/{entity}/{attribute}/history — full value history
POST /temporal/query                              — natural-language temporal query
GET  /temporal/evolution/{entity}                 — interest/skill/project/identity evolution
GET  /temporal/beliefs                            — list active beliefs
GET  /temporal/truth/{record_id}                  — truth status of a record
POST /temporal/resolve                             — manually resolve a contradiction pair
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from api.context import BlixContext
from api.deps import get_context

router = APIRouter(prefix="/temporal", tags=["Temporal State"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TemporalQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural-language temporal query.")


class ResolveRequest(BaseModel):
    record_a_id: str
    record_b_id: str
    text_a: str
    text_b: str
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    confidence_a: float = 0.5
    confidence_b: float = 0.5
    newer_id: Optional[str] = None


# ---------------------------------------------------------------------------
# State endpoints
# ---------------------------------------------------------------------------


@router.get("/state/{entity}/{attribute}", summary="Current tracked value")
async def get_current_state(
    entity: str, attribute: str, ctx: BlixContext = Depends(get_context),
) -> dict:
    """Return the currently-active value for (entity, attribute), if tracked."""
    snap = ctx.state_tracker.current(entity, attribute)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No tracked state for {entity}.{attribute}.")
    truth_status = ctx.truth_manager.status_of(snap.snapshot_id)
    return {**snap.to_dict(), "truth_status": truth_status.value}


@router.get("/state/{entity}/{attribute}/history", summary="Full value history")
async def get_state_history(
    entity: str, attribute: str, ctx: BlixContext = Depends(get_context),
) -> dict:
    """Return the full chronological history of values for (entity, attribute)."""
    history = ctx.state_tracker.history(entity, attribute)
    if not history:
        raise HTTPException(status_code=404, detail=f"No history for {entity}.{attribute}.")
    return {
        "entity": entity, "attribute": attribute,
        "history": [
            {**s.to_dict(), "truth_status": ctx.truth_manager.status_of(s.snapshot_id).value}
            for s in history
        ],
    }


@router.get("/state/{entity}/{attribute}/at", summary="Value at a specific point in time")
async def get_state_at_time(
    entity: str, attribute: str,
    timestamp: str = Query(..., description="ISO 8601 timestamp."),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Return the value that was active for (entity, attribute) at a given time."""
    snap = ctx.state_tracker.at_time(entity, attribute, timestamp)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No recorded value for {entity}.{attribute} at {timestamp}.")
    return snap.to_dict()


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


@router.post("/query", summary="Natural-language temporal query")
async def temporal_query(
    req: TemporalQueryRequest, ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Answer a temporal query: current state, historical state, transitions,
    or evolution.

    Examples: "What was my favorite language in 2024?", "How has my
    research evolved?", "When did Blix adopt FastAPI?", "What changed
    during the last month?"
    """
    result = ctx.temporal_query_engine.query(req.query)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Evolution endpoint
# ---------------------------------------------------------------------------


@router.get("/evolution/{entity}", summary="Interest/skill/project/identity evolution")
async def get_evolution(entity: str, ctx: BlixContext = Depends(get_context)) -> dict:
    """Return the full four-dimension evolution report for an entity."""
    report = ctx.state_reflection.generate(entity)
    return report.to_dict()


@router.get("/evolution/{entity}/recent", summary="Recent cross-dimension shifts")
async def get_recent_shifts(
    entity: str,
    days: int = Query(default=30, ge=1, le=3650),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Return all attribute changes for an entity within the last N days."""
    shifts = ctx.state_reflection.recent_shifts(entity, days=days)
    return {"entity": entity, "days": days, "shifts": shifts}


# ---------------------------------------------------------------------------
# Beliefs endpoint
# ---------------------------------------------------------------------------


@router.get("/beliefs", summary="List beliefs")
async def list_beliefs(
    status: Optional[str] = Query(default=None, description="Filter by TruthStatus (active, conflicting, etc.)"),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """List beliefs, optionally filtered by TruthStatus."""
    from core.truth_manager import TruthStatus
    if status:
        try:
            status_enum = TruthStatus(status.lower())
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown status '{status}'.")
        beliefs = ctx.belief_store.all_with_status(status_enum)
    else:
        beliefs = ctx.belief_store.all_active()
    return {"total": len(beliefs), "beliefs": [b.to_dict() for b in beliefs]}


# ---------------------------------------------------------------------------
# Truth + resolution endpoints
# ---------------------------------------------------------------------------


@router.get("/truth/{record_id}", summary="Truth status of a record")
async def get_truth_status(record_id: str, ctx: BlixContext = Depends(get_context)) -> dict:
    """Return the TruthRecord (status + history) for a belief or snapshot id."""
    record = ctx.truth_manager.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No truth record for '{record_id}'.")
    return record.to_dict()


@router.post("/resolve", summary="Resolve a contradiction between two records")
async def resolve_contradiction(req: ResolveRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Classify and resolve a contradiction between two competing claims
    (Replacement / Parallel Truth / Merge / Conflict).
    """
    result = ctx.contradiction_resolver.resolve(
        record_a_id=req.record_a_id, record_b_id=req.record_b_id,
        text_a=req.text_a, text_b=req.text_b,
        value_a=req.value_a, value_b=req.value_b,
        confidence_a=req.confidence_a, confidence_b=req.confidence_b,
        newer_id=req.newer_id,
    )
    return result.to_dict()
