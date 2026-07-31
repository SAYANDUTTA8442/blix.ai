"""
/agents router — Blix v0.3.14
POST /agents/failure/record           — record a tool failure
GET  /agents/failure/common           — most common failure patterns
GET  /agents/failure/count            — total failure count
POST /agents/tool/record              — record tool execution outcome
GET  /agents/tool/all                 — all tools and reliability
GET  /agents/tool/{tool_name}/stats   — single tool reliability
"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from api.context import BlixContext
from api.deps import get_context

router = APIRouter(prefix="/agents", tags=["Agents"])

class RecordFailureRequest(BaseModel):
    task_title: str = Field(..., min_length=1, max_length=300)
    tool: str = Field(..., min_length=1, max_length=100)
    failure: str = Field(..., min_length=1, max_length=500)

class RecordToolOutcomeRequest(BaseModel):
    tool_name: str = Field(..., min_length=1, max_length=100)
    success: bool
    duration_ms: float = Field(default=0.0, ge=0.0)

@router.post("/failure/record")
async def record_failure(req: RecordFailureRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    ctx.failure_memory.record(req.task_title, req.tool, req.failure)
    return {"recorded": True, "total_failures": ctx.failure_memory.count}

@router.get("/failure/common")
async def common_failures(
    top_k: int = Query(default=10, ge=1, le=100),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    records = ctx.failure_memory.most_common_failures(top_k=top_k)
    return {"total": len(records), "failures": [r.to_dict() for r in records]}

@router.get("/failure/count")
async def failure_count(ctx: BlixContext = Depends(get_context)) -> dict:
    return {"count": ctx.failure_memory.count}

@router.post("/tool/record")
async def record_tool_outcome(req: RecordToolOutcomeRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    ctx.tool_reliability_registry.record(req.tool_name, success=req.success, duration_ms=req.duration_ms)
    return {
        "recorded": True,
        "tool": req.tool_name,
        "current_success_rate": round(ctx.tool_reliability_registry.success_rate(req.tool_name), 4),
    }

@router.get("/tool/all")
async def all_tools(ctx: BlixContext = Depends(get_context)) -> dict:
    reg = ctx.tool_reliability_registry
    names = reg.all_tools() if hasattr(reg, "all_tools") else list(reg._tools.keys()) if hasattr(reg, "_tools") else []
    result = [{"tool": n, "success_rate": round(reg.success_rate(n), 4)} for n in names]
    return {"total": len(result), "tools": result}

@router.get("/tool/{tool_name}/stats")
async def tool_stats(tool_name: str, ctx: BlixContext = Depends(get_context)) -> dict:
    rate = ctx.tool_reliability_registry.success_rate(tool_name)
    return {"tool": tool_name, "success_rate": round(rate, 4)}
