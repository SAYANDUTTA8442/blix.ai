"""
/reflection router — Blix v0.3.3

Endpoints
---------
GET  /reflection/insights              — recent reflection insights
GET  /reflection/insights/actionable   — actionable insights (v0.3.3 upgrade)
POST /reflection/run                   — trigger a reflection pass
POST /reflection/insights/generate     — run full insight generation pass
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.context import BlixContext
from api.deps import get_context
from api.models import (
    InsightItem, InsightsResponse, ReflectionRunRequest, ReflectionRunResponse,
)
from reflection.reflection_engine import ReflectionScope

router = APIRouter(prefix="/reflection", tags=["Reflection"])


def _insight_item(i: object) -> InsightItem:
    return InsightItem(
        insight=getattr(i, "insight"),
        confidence=getattr(i, "confidence", 0.5),
        category=getattr(i, "category", "trend") if hasattr(i, "category") else
                 getattr(i, "scope", "session") if hasattr(i, "scope") else "trend",
        evidence=list(getattr(i, "evidence", [])),
        recommendation=getattr(i, "recommendation", ""),
        created_at=getattr(i, "created_at", ""),
    )


@router.get("/insights", response_model=InsightsResponse, summary="Recent reflection insights")
async def list_insights(
    limit: int = Query(default=10, ge=1, le=100),
    scope: str | None = Query(default=None, description="Filter by scope: session|daily|weekly|project|behavior|learning"),
    ctx: BlixContext = Depends(get_context),
) -> InsightsResponse:
    """Return the most recent reflection insights, optionally filtered by scope."""
    scope_enum = None
    if scope is not None:
        try:
            scope_enum = ReflectionScope(scope)
        except ValueError:
            pass
    insights = ctx.reflection.get_recent_insights(scope=scope_enum, limit=limit)
    items = [_insight_item(i) for i in insights]
    return InsightsResponse(insights=items, total=len(items))


@router.get("/insights/actionable", response_model=InsightsResponse, summary="Actionable insights")
async def list_actionable_insights(
    category: str | None = Query(default=None, description="trend|bottleneck|research_interest|project_pattern"),
    limit: int = Query(default=10, ge=1, le=100),
    ctx: BlixContext = Depends(get_context),
) -> InsightsResponse:
    """
    Return actionable insights (v0.3.3 Insight Generation Engine).

    Each insight includes evidence and a concrete recommendation.
    """
    from reflection.insight_engine import InsightCategory

    cat = None
    if category is not None:
        try:
            cat = InsightCategory(category)
        except ValueError:
            pass

    insights = ctx.insight_engine.list_insights(category=cat)
    # Sort newest first; cap at limit
    insights = sorted(insights, key=lambda i: i.created_at, reverse=True)[:limit]
    items = [_insight_item(i) for i in insights]
    return InsightsResponse(insights=items, total=len(items))


@router.post("/run", response_model=ReflectionRunResponse, summary="Run a reflection pass")
async def run_reflection(
    req: ReflectionRunRequest,
    ctx: BlixContext = Depends(get_context),
) -> ReflectionRunResponse:
    """
    Trigger a reflection pass at the requested scope.

    If ``material`` is not provided, the engine auto-collects the most
    relevant content:
    * session   → latest session summary text
    * daily     → latest daily summary text
    * weekly    → latest weekly summary text
    * behavior  → text of last 50 memories
    * project   → project summary for scope_ref
    * learning  → learning state topics
    """
    material = req.material

    if material is None:
        scope = req.scope
        if scope in ("session",):
            sessions = ctx.hierarchy.get_latest_sessions(1)
            material = sessions[0].summary if sessions else ""
        elif scope == "daily":
            from datetime import date
            ds = ctx.hierarchy.get_daily(date.today().isoformat())
            material = ds.summary if ds else ""
        elif scope == "weekly":
            from datetime import date
            from core.hierarchy_manager import _date_to_week
            ws = ctx.hierarchy.get_weekly(_date_to_week(date.today().isoformat()))
            material = ws.summary if ws else ""
        elif scope == "project":
            ps = ctx.project_manager.get(req.scope_ref)
            if ps:
                material = f"{ps.project_name}: {', '.join(ps.goals)}"
            else:
                material = req.scope_ref
        elif scope == "learning":
            ls = ctx.memory_manager.learning_state
            topics = getattr(ls, "topics", {})
            lines = [f"{t}: {v.get('count', 0) if isinstance(v, dict) else getattr(v, 'count', 0)} mentions"
                     for t, v in list(topics.items())[:20]]
            material = "\n".join(lines)
        else:  # behavior / fallback
            memories = ctx.memory_manager.get_all_memories()[-50:]
            material = " ".join(m.output[:100] for m in memories)

    try:
        scope_enum = ReflectionScope(req.scope)
    except ValueError:
        scope_enum = ReflectionScope.BEHAVIOR

    record = ctx.reflection.reflect(scope_enum, req.scope_ref, material or "")
    items = [_insight_item(i) for i in record.insights]
    return ReflectionRunResponse(scope=req.scope, scope_ref=req.scope_ref, insights=items)


@router.post("/insights/generate", response_model=InsightsResponse, summary="Generate actionable insights")
async def generate_insights(ctx: BlixContext = Depends(get_context)) -> InsightsResponse:
    """
    Run the full InsightGenerationEngine pass over current memories, goals,
    and project states, producing trend/bottleneck/research_interest/project_pattern
    insights with concrete recommendations.
    """
    memories = ctx.memory_manager.get_all_memories()
    goals = ctx.goals.list_goals()
    project_states = ctx.project_intelligence.list_all()

    new_insights = ctx.insight_engine.generate_all(
        memories=memories, goals=goals, project_states=project_states
    )
    items = [_insight_item(i) for i in new_insights]
    return InsightsResponse(insights=items, total=len(items))
