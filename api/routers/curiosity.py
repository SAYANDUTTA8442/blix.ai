"""
/curiosity router — Blix v0.3.13

Endpoints
---------
GET  /curiosity/signals                     — generate ranked curiosity signals from current cognitive state
POST /curiosity/hypotheses                      — propose a new hypothesis
GET  /curiosity/hypotheses                          — list hypotheses (filterable by status)
POST /curiosity/hypotheses/{id}/evidence                — add evidence to a hypothesis
POST /curiosity/experiments                                 — plan an experiment for a hypothesis
POST /curiosity/experiments/from-signal                         — plan an experiment from a curiosity signal + hypothesis
GET  /curiosity/experiments                                         — list experiments
POST /curiosity/experiments/{id}/outcome                                — record an experiment outcome
GET  /curiosity/knowledge-gaps                                              — list current knowledge gaps
POST /curiosity/knowledge-gaps/discover                                         — discover gaps from self-model/failures/cause-graph
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.context import BlixContext
from api.deps import get_context
from hypothesis.hypothesis_manager import HypothesisStatus
from knowledge.knowledge_gap_tracker import GapSeverity

router = APIRouter(prefix="/curiosity", tags=["Curiosity & Experimentation"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ProposeHypothesisRequest(BaseModel):
    statement: str = Field(..., min_length=1, max_length=500)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    source: str = Field(default="", max_length=200)


class AddEvidenceRequest(BaseModel):
    evidence: str = Field(..., min_length=1, max_length=500)
    confidence_delta: float = Field(default=0.1, ge=-1.0, le=1.0)


class PlanExperimentRequest(BaseModel):
    hypothesis_id: str = Field(..., min_length=1)
    actions: list[str] = Field(..., min_length=1)
    expected_result: str = Field(..., min_length=1, max_length=500)
    success_criteria: list[str] = Field(..., min_length=1)


class PlanFromSignalRequest(BaseModel):
    hypothesis_id: str = Field(..., min_length=1)
    signal_target: str = Field(..., min_length=1, max_length=300)
    signal_trigger: str = Field(..., min_length=1, max_length=50)
    signal_reason: str = Field(default="", max_length=500)


class RecordOutcomeRequest(BaseModel):
    outcome: str = Field(..., min_length=1, max_length=500)
    success: bool
    confidence_delta: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


# ---------------------------------------------------------------------------
# Curiosity signals
# ---------------------------------------------------------------------------


@router.get("/signals", summary="Generate ranked curiosity signals")
async def get_curiosity_signals(
    top_k: int = Query(default=10, ge=1, le=50),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Scan current cognitive state and return the top-k exploration targets."""
    signals = ctx.curiosity_engine.generate_signals(top_k=top_k)
    return {"total": len(signals), "signals": [s.to_dict() for s in signals]}


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------


@router.post("/hypotheses", summary="Propose a new hypothesis")
async def propose_hypothesis(req: ProposeHypothesisRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    h = ctx.hypothesis_manager.propose(req.statement, confidence=req.confidence, source=req.source)
    return h.to_dict()


@router.get("/hypotheses", summary="List hypotheses")
async def list_hypotheses(
    status: Optional[str] = Query(default=None, description="Filter by status: pending, supported, rejected, unknown"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    if status:
        try:
            hyp_status = HypothesisStatus(status)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status '{status}'.")
        hypotheses = ctx.hypothesis_manager.by_status(hyp_status)
    else:
        hypotheses = list(ctx.hypothesis_manager._hypotheses.values())
    total = len(hypotheses)
    page = hypotheses[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "hypotheses": [h.to_dict() for h in page]}


@router.post("/hypotheses/{hypothesis_id}/evidence", summary="Add evidence to a hypothesis")
async def add_evidence(hypothesis_id: str, req: AddEvidenceRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    h = ctx.hypothesis_manager.add_evidence(hypothesis_id, req.evidence, confidence_delta=req.confidence_delta)
    if h is None:
        raise HTTPException(status_code=404, detail=f"Hypothesis '{hypothesis_id}' not found.")
    return h.to_dict()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@router.post("/experiments", summary="Plan an experiment for a hypothesis")
async def plan_experiment(req: PlanExperimentRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    try:
        exp = ctx.experiment_planner.plan(
            req.hypothesis_id, req.actions, req.expected_result, req.success_criteria,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return exp.to_dict()


@router.post("/experiments/from-signal", summary="Plan an experiment from a curiosity signal")
async def plan_from_signal(req: PlanFromSignalRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    from curiosity.curiosity_engine import CuriositySignal, CuriosityTrigger
    try:
        trigger = CuriosityTrigger(req.signal_trigger)
    except ValueError:
        trigger = CuriosityTrigger.UNKNOWN_DOMAIN
    signal = CuriositySignal(
        target=req.signal_target, trigger=trigger, reason=req.signal_reason,
        novelty=0.5, uncertainty=0.5, expected_information_gain=0.5,
    )
    try:
        exp = ctx.experiment_planner.plan_from_signal(signal, req.hypothesis_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return exp.to_dict()


@router.get("/experiments", summary="List experiments")
async def list_experiments(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    all_exps = list(ctx.experiment_planner._experiments.values())
    total = len(all_exps)
    page = all_exps[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "experiments": [e.to_dict() for e in page]}


@router.post("/experiments/{experiment_id}/outcome", summary="Record an experiment outcome")
async def record_outcome(experiment_id: str, req: RecordOutcomeRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    exp = ctx.experiment_planner.record_outcome(
        experiment_id, req.outcome, req.success, confidence_delta=req.confidence_delta,
    )
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return exp.to_dict()


# ---------------------------------------------------------------------------
# Knowledge gaps
# ---------------------------------------------------------------------------


@router.get("/knowledge-gaps", summary="List current knowledge gaps")
async def list_knowledge_gaps(
    min_severity: Optional[str] = Query(default=None, description="Filter: low, medium, high, critical"),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    sev = None
    if min_severity:
        try:
            sev = GapSeverity(min_severity)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid severity '{min_severity}'.")
    gaps = ctx.knowledge_gap_tracker.gaps(min_severity=sev)
    return {"total": len(gaps), "gaps": [g.to_dict() for g in gaps]}


@router.post("/knowledge-gaps/discover", summary="Discover knowledge gaps from existing infrastructure")
async def discover_knowledge_gaps(ctx: BlixContext = Depends(get_context)) -> dict:
    """Scan SelfModel capability scores, FailureMemory, and CauseGraph for knowledge gaps."""
    from_self_model = ctx.knowledge_gap_tracker.discover_from_self_model(ctx.self_model)
    from_failures = ctx.knowledge_gap_tracker.discover_from_failure_memory(ctx.failure_memory)
    from_causes = ctx.knowledge_gap_tracker.discover_from_cause_graph(ctx.cause_graph)
    total = len(from_self_model) + len(from_failures) + len(from_causes)
    return {
        "total_discovered": total,
        "from_self_model": len(from_self_model),
        "from_failures": len(from_failures),
        "from_causes": len(from_causes),
        "gaps": [g.to_dict() for g in ctx.knowledge_gap_tracker.gaps()],
    }
