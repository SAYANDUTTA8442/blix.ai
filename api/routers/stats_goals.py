"""
/stats router — Dashboard statistics endpoint — Blix v0.3.3
/goals router — Goal tracking CRUD endpoints
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.context import BlixContext
from api.deps import get_context
from api.models import (
    CreateGoalRequest, GoalBlockerRequest, GoalItem, GoalListResponse,
    GoalMilestoneRequest, GoalProgressRequest, StatsResponse,
)

# ---------------------------------------------------------------------------
# Stats router
# ---------------------------------------------------------------------------

stats_router = APIRouter(prefix="/stats", tags=["Dashboard"])


@stats_router.get("", response_model=StatsResponse, summary="Dashboard statistics")
async def dashboard_stats(ctx: BlixContext = Depends(get_context)) -> StatsResponse:
    """
    Aggregate counts across every Blix subsystem — suitable for a
    dashboard, demo, or research paper table.
    """
    data = ctx.dashboard_stats()
    return StatsResponse(**data)


# ---------------------------------------------------------------------------
# Goals router
# ---------------------------------------------------------------------------

goals_router = APIRouter(prefix="/goals", tags=["Goals"])


def _goal_item(g: object) -> GoalItem:
    return GoalItem(
        goal_id=getattr(g, "goal_id"),
        title=getattr(g, "title"),
        description=getattr(g, "description", ""),
        status=getattr(g, "status").value if hasattr(getattr(g, "status"), "value") else str(getattr(g, "status")),
        priority=getattr(g, "priority"),
        progress=getattr(g, "progress"),
        related_project=getattr(g, "related_project", ""),
        blockers=[b.description for b in getattr(g, "active_blockers", [])],
        milestones=[{"title": m.title, "status": m.status.value} for m in getattr(g, "milestones", [])],
        tasks=[{"title": t.title, "status": t.status.value} for t in getattr(g, "tasks", [])],
    )


@goals_router.get("", response_model=GoalListResponse, summary="List goals")
async def list_goals(
    status: str | None = Query(default=None, description="Filter: active|paused|completed|abandoned"),
    ctx: BlixContext = Depends(get_context),
) -> GoalListResponse:
    from reflection.goal_tracker import GoalStatus

    status_enum = None
    if status is not None:
        try:
            status_enum = GoalStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown status: {status!r}")

    goals = ctx.goals.list_goals(status=status_enum)
    return GoalListResponse(goals=[_goal_item(g) for g in goals], total=len(goals))


@goals_router.post("", response_model=GoalItem, summary="Create goal")
async def create_goal(
    req: CreateGoalRequest, ctx: BlixContext = Depends(get_context)
) -> GoalItem:
    goal = ctx.goals.create_goal(
        title=req.title,
        description=req.description,
        priority=req.priority,
        related_project=req.related_project,
    )
    return _goal_item(goal)


@goals_router.get("/{goal_id}", response_model=GoalItem, summary="Get goal by id")
async def get_goal(goal_id: str, ctx: BlixContext = Depends(get_context)) -> GoalItem:
    goal = ctx.goals.get(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found.")
    return _goal_item(goal)


@goals_router.patch("/{goal_id}/progress", response_model=GoalItem, summary="Set goal progress")
async def set_progress(
    goal_id: str, req: GoalProgressRequest, ctx: BlixContext = Depends(get_context)
) -> GoalItem:
    goal = ctx.goals.set_progress_override(goal_id, req.progress)
    if goal is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found.")
    return _goal_item(goal)


@goals_router.post("/{goal_id}/blockers", response_model=GoalItem, summary="Add blocker")
async def add_blocker(
    goal_id: str, req: GoalBlockerRequest, ctx: BlixContext = Depends(get_context)
) -> GoalItem:
    goal = ctx.goals.add_blocker(goal_id, req.description)
    if goal is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found.")
    return _goal_item(goal)


@goals_router.delete("/{goal_id}/blockers", response_model=GoalItem, summary="Resolve blocker")
async def resolve_blocker(
    goal_id: str, req: GoalBlockerRequest, ctx: BlixContext = Depends(get_context)
) -> GoalItem:
    goal = ctx.goals.resolve_blocker(goal_id, req.description)
    if goal is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found.")
    return _goal_item(goal)


@goals_router.post("/{goal_id}/milestones", response_model=GoalItem, summary="Add milestone")
async def add_milestone(
    goal_id: str, req: GoalMilestoneRequest, ctx: BlixContext = Depends(get_context)
) -> GoalItem:
    goal = ctx.goals.add_milestone(goal_id, req.title)
    if goal is None:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found.")
    return _goal_item(goal)
