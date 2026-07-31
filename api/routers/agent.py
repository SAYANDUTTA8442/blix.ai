"""
/agent router — Blix v0.3.5 / v0.3.6

Endpoints
---------
POST /agent/run            — execute a goal end-to-end (plan + execute + verify + replan)
POST /agent/plan            — plan only (no execution) — preview the TaskGraph
POST /agent/critique        — plan + run PlanCritic without executing (v0.3.6)
GET  /agent/history         — recent execution history entries
GET  /agent/sessions         — recent agent run summaries
GET  /agent/tools             — list registered tools
GET  /agent/failures           — known failure patterns (v0.3.6)
GET  /agent/tool-reliability     — cross-run tool reliability stats (v0.3.6)
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.context import BlixContext
from api.deps import get_context

router = APIRouter(prefix="/agent", tags=["Agent"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000, description="Natural-language goal for the agent.")


class AgentPlanRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run", summary="Execute a goal end-to-end")
async def run_agent(
    req: AgentRunRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Plan and execute a natural-language goal using the full adaptive agent loop:

        Goal → Plan → Critic → Act → Verify → Observe → Reflect → Replan → Learn

    Runs synchronously (in a thread pool) since tool calls may block.
    Returns the full ``AgentRunResult`` summary including task-by-task
    history, replan count, plan critique, plan-level reflection, and
    the final assembled output.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, ctx.agent_session.run, req.goal)
    return {
        **result.to_dict(),
        "history": result.history,
    }


@router.post("/plan", summary="Plan a goal without executing")
async def plan_agent(
    req: AgentPlanRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Preview the planned ``TaskGraph`` for a goal without executing it.

    Useful for inspecting what the agent intends to do before committing
    to a (potentially slow/costly) full execution run.
    """
    loop = asyncio.get_event_loop()
    parsed_goal, graph = await loop.run_in_executor(None, ctx.planner.plan, req.goal)
    return {
        "parsed_goal": {
            "title": parsed_goal.title,
            "description": parsed_goal.description,
            "domain": parsed_goal.domain,
            "complexity": parsed_goal.complexity,
            "requires_web": parsed_goal.requires_web,
            "requires_code": parsed_goal.requires_code,
            "requires_files": parsed_goal.requires_files,
        },
        "task_graph": graph.to_dict(),
    }


@router.post("/critique", summary="Plan a goal and critique it without executing")
async def critique_agent(
    req: AgentPlanRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Plan a goal and run the PlanCritic ("think before acting") without
    executing anything. Surfaces circular dependencies, missing tools,
    risky/unreliable tool choices, known-failure matches, and missing
    steps before any tool call happens.
    """
    loop = asyncio.get_event_loop()
    parsed_goal, graph = await loop.run_in_executor(None, ctx.planner.plan, req.goal)
    critique = await loop.run_in_executor(None, ctx.plan_critic.critique, graph)
    return {
        "task_graph": graph.to_dict(),
        "critique": critique.to_dict(),
    }


@router.get("/history", summary="Recent execution history")
async def agent_history(
    goal: str | None = Query(default=None, description="Filter by goal substring."),
    limit: int = Query(default=20, ge=1, le=200),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Return recent ``ExecutionHistoryEntry`` records."""
    entries = ctx.reflection_loop.get_history(goal=goal, limit=limit)
    return {
        "total": len(entries),
        "success_rate": round(ctx.reflection_loop.success_rate(), 3),
        "mean_quality": round(ctx.reflection_loop.mean_quality(), 3),
        "entries": [e.to_dict() for e in entries],
    }


@router.get("/sessions", summary="Recent agent run sessions")
async def agent_sessions(
    limit: int = Query(default=5, ge=1, le=50),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Return summaries of recent ``AgentSession.run()`` calls."""
    sessions = ctx.agent_session.recent_sessions(limit)
    return {
        "total_sessions": ctx.agent_session.session_count,
        "sessions": [s.to_dict() for s in sessions],
    }


@router.get("/tools", summary="List registered agent tools")
async def list_tools(ctx: BlixContext = Depends(get_context)) -> dict:
    """Return the schema of all tools available to the agent."""
    return {"tools": ctx.tool_registry.schema()}


@router.get("/failures", summary="Known failure patterns")
async def list_failures(
    limit: int = Query(default=10, ge=1, le=100),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Return the most common recorded failure patterns from ``FailureMemory``.

    Each record reflects a task/tool combination that has failed before,
    optionally with a known fix discovered by a prior replan or
    plan-level reflection.
    """
    records = ctx.failure_memory.most_common_failures(top_k=limit)
    return {
        "total_recorded": ctx.failure_memory.count,
        "failures": [r.to_dict() for r in records],
    }


@router.get("/tool-reliability", summary="Cross-run tool reliability stats")
async def tool_reliability(ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Return cross-run reliability statistics for every tool that has been
    executed at least once, as tracked by ``ToolReliabilityRegistry``.
    """
    records = ctx.tool_reliability_registry.all_records()
    return {
        "tracked_tools": len(records),
        "tools": [r.to_dict() for r in sorted(records, key=lambda r: -r.success_rate)],
    }
