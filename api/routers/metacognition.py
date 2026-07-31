"""
/metacognition router — Blix v0.3.8

Endpoints
---------
GET  /metacognition/self-model                — current SelfModel snapshot
GET  /metacognition/capabilities                — capability tracker domains
GET  /metacognition/confidence/{namespace}        — confidence records in a namespace
POST /metacognition/strategy/decide                 — get a strategy decision for a situation
GET  /metacognition/skills                              — list learned procedural skills
POST /metacognition/skills/match                          — find a matching skill for a goal
GET  /metacognition/behavior-insights                       — recent meta-reflection insights
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.context import BlixContext
from api.deps import get_context

router = APIRouter(prefix="/metacognition", tags=["Metacognition"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StrategyDecideRequest(BaseModel):
    ref_key: str = Field(..., min_length=1, max_length=200)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SkillMatchRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=1000)


# ---------------------------------------------------------------------------
# Self Model
# ---------------------------------------------------------------------------


@router.get("/self-model", summary="Current self-model snapshot")
async def get_self_model(ctx: BlixContext = Depends(get_context)) -> dict:
    """Return Blix's current self-model: capabilities, weaknesses, strengths, known limits."""
    return ctx.self_model.model.to_dict()


@router.get("/capabilities", summary="Capability tracker domains")
async def get_capabilities(ctx: BlixContext = Depends(get_context)) -> dict:
    """Return raw per-domain accuracy track record."""
    records = ctx.capability_tracker.all_records()
    return {
        "tracked_domains": len(records),
        "domains": [r.to_dict() for r in sorted(records, key=lambda r: -r.accuracy)],
    }


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


@router.get("/confidence/{namespace}", summary="Confidence records in a namespace")
async def get_confidence_namespace(
    namespace: str,
    threshold: Optional[float] = Query(default=None, description="Only return records below this confidence."),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """List confidence records for a namespace (e.g. 'belief', 'plan', 'tool')."""
    if threshold is not None:
        records = ctx.confidence_manager.low_confidence(namespace=namespace, threshold=threshold)
    else:
        records = ctx.confidence_manager.all_in_namespace(namespace)
    return {
        "namespace": namespace,
        "mean_confidence": round(ctx.confidence_manager.mean_confidence(namespace), 4),
        "records": [r.to_dict() for r in records],
    }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@router.post("/strategy/decide", summary="Decide a reasoning strategy for a situation")
async def decide_strategy(req: StrategyDecideRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Decide the appropriate reasoning strategy for ``ref_key`` based on
    confidence and observed failure history.
    """
    decision = ctx.strategy_manager.decide(req.ref_key, confidence=req.confidence)
    return decision.to_dict()


# ---------------------------------------------------------------------------
# Procedural memory / skills
# ---------------------------------------------------------------------------


@router.get("/skills", summary="List learned procedural skills")
async def list_skills(ctx: BlixContext = Depends(get_context)) -> dict:
    """Return all learned skills with usage/success statistics."""
    skills = ctx.procedural_memory.all_skills()
    return {"total": len(skills), "skills": [s.to_dict() for s in skills]}


@router.post("/skills/match", summary="Find a matching skill for a goal")
async def match_skill(req: SkillMatchRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Find the best-matching learned skill for a new goal, if any."""
    skill = ctx.procedural_memory.find_matching_skill(req.goal)
    if skill is None:
        return {"matched": False, "skill": None}
    return {"matched": True, "skill": skill.to_dict()}


# ---------------------------------------------------------------------------
# Behavior insights
# ---------------------------------------------------------------------------


@router.get("/behavior-insights", summary="Recent meta-reflection behavior insights")
async def behavior_insights(
    limit: int = Query(default=10, ge=1, le=100),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Return recent behavior-change insights persisted by
    ``MetaReflectionEngine`` under ``ReflectionScope.BEHAVIOR``.
    """
    from reflection.reflection_engine import ReflectionScope
    insights = ctx.reflection.get_recent_insights(scope=ReflectionScope.BEHAVIOR, limit=limit)
    return {"total": len(insights), "insights": [i.to_dict() for i in insights]}
