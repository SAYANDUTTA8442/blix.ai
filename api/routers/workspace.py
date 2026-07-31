"""
/workspace router — Blix v0.3.9

Endpoints
---------
GET  /workspace/state                       — current workspace contents (items, active goal, attention focus)
POST /workspace/submit                          — submit a candidate item for the next attention cycle
POST /workspace/cycle                               — run one attention -> entry -> broadcast cycle
GET  /workspace/snapshots                               — list saved workspace snapshots
POST /workspace/snapshots                                   — capture a new snapshot of current state
POST /workspace/snapshots/{snapshot_id}/restore               — restore a snapshot's goal/focus context
POST /workspace/specialists/consult                               — poll all specialists + consensus for a topic
POST /workspace/inner-dialogue                                       — run an inner-dialogue pass over a topic
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.context import BlixContext
from api.deps import get_context
from workspace.attention_manager import AttentionCandidate

router = APIRouter(prefix="/workspace", tags=["Workspace"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SubmitCandidateRequest(BaseModel):
    ref_id: str = Field(..., min_length=1, max_length=200)
    source: str = Field(..., min_length=1, max_length=100)
    content_summary: str = Field(..., min_length=1, max_length=500)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RunCycleRequest(BaseModel):
    active_goal: Optional[str] = Field(default=None, max_length=500)


class CaptureSnapshotRequest(BaseModel):
    important_beliefs: list[str] = Field(default_factory=list)
    current_plan_graph_id: Optional[str] = None
    current_plan_summary: str = ""
    current_failures: list[str] = Field(default_factory=list)


class SpecialistConsultRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=1000)


class InnerDialogueRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=1000)


# ---------------------------------------------------------------------------
# Workspace state / cycle
# ---------------------------------------------------------------------------


@router.get("/state", summary="Current workspace contents")
async def get_workspace_state(ctx: BlixContext = Depends(get_context)) -> dict:
    """Return the current contents of the Global Workspace."""
    return ctx.global_workspace.memory.to_dict()


@router.post("/submit", summary="Submit a candidate for the next attention cycle")
async def submit_candidate(req: SubmitCandidateRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Submit one candidate item for consideration in the next workspace cycle."""
    candidate = AttentionCandidate(
        ref_id=req.ref_id, source=req.source, content_summary=req.content_summary,
        relevance=req.relevance, urgency=req.urgency, novelty=req.novelty, confidence=req.confidence,
    )
    ctx.global_workspace.submit_candidate(candidate)
    return {"submitted": True, "pending_count": ctx.global_workspace.pending_count}


@router.post("/cycle", summary="Run one attention -> entry -> broadcast cycle")
async def run_cycle(req: RunCycleRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Run one full workspace cycle over all pending candidates."""
    result = ctx.global_workspace.run_cycle(active_goal=req.active_goal)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@router.get("/snapshots", summary="List saved workspace snapshots")
async def list_snapshots(ctx: BlixContext = Depends(get_context)) -> dict:
    """Return all saved workspace snapshots."""
    snapshots = ctx.workspace_snapshots.all_snapshots()
    return {"total": len(snapshots), "snapshots": [s.to_dict() for s in snapshots]}


@router.post("/snapshots", summary="Capture a new workspace snapshot")
async def capture_snapshot(req: CaptureSnapshotRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Capture the current workspace state for later suspension/resumption."""
    snapshot = ctx.workspace_snapshots.capture(
        ctx.global_workspace,
        important_beliefs=req.important_beliefs,
        current_plan_graph_id=req.current_plan_graph_id,
        current_plan_summary=req.current_plan_summary,
        current_failures=req.current_failures,
    )
    return snapshot.to_dict()


@router.post("/snapshots/{snapshot_id}/restore", summary="Restore a snapshot's goal/focus context")
async def restore_snapshot(snapshot_id: str, ctx: BlixContext = Depends(get_context)) -> dict:
    """Restore a previously-captured snapshot's active goal back into the live workspace."""
    snapshot = ctx.workspace_snapshots.restore(snapshot_id, ctx.global_workspace)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found.")
    return {"restored": True, "snapshot": snapshot.to_dict()}


# ---------------------------------------------------------------------------
# Specialists / consensus
# ---------------------------------------------------------------------------


@router.post("/specialists/consult", summary="Poll all specialists and aggregate consensus for a topic")
async def consult_specialists(req: SpecialistConsultRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Poll every registered internal specialist about a topic and return the aggregated consensus."""
    result = ctx.specialist_consensus.decide(req.topic)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Inner dialogue
# ---------------------------------------------------------------------------


@router.post("/inner-dialogue", summary="Run an inner-dialogue pass over a topic")
async def run_inner_dialogue(req: InnerDialogueRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Consult every registered inner-dialogue voice about a topic and return the transcript."""
    transcript = ctx.inner_dialogue.run(req.topic)
    return transcript.to_dict()
