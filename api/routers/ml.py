"""
/ml router — Blix v0.3.10

Endpoints
---------
GET  /ml/status                        — training status of every learned model (cold-start vs. learned)
POST /ml/world-model/predict               — predict plan success / tool failure / confidence decay from a latent state
POST /ml/value/estimate                        — estimate V(state) for a latent state
POST /ml/scenarios/rank                            — rank candidate scenarios by value-network score
POST /ml/tool-success/predict                          — predict tool success probability
POST /ml/confidence/predict                                — predict P(answer_correct)
GET  /ml/failure-clusters                                      — discover recurring failure patterns
POST /ml/future/predict                                            — record a prediction about a future state
GET  /ml/future/pending                                                — list unresolved future predictions
POST /ml/future/{id}/resolve                                               — resolve a future prediction with its actual outcome
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.context import BlixContext
from api.deps import get_context
from world_model.latent_world_model import LatentState
from world_model.scenario_ranker import Scenario

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LatentStateInput(BaseModel):
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    capability_estimate: float = Field(default=0.5, ge=0.0, le=1.0)
    recent_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    dependency_density: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_latent_state(self) -> LatentState:
        return LatentState(**self.model_dump())


class ScenarioInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)
    state: LatentStateInput


class RankScenariosRequest(BaseModel):
    scenarios: list[ScenarioInput] = Field(..., min_length=1)


class ToolSuccessRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=500)
    tool: str = Field(..., min_length=1, max_length=200)
    task_complexity_hint: float = Field(default=0.5, ge=0.0, le=1.0)
    is_repeated_attempt: bool = False
    context_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ConfidencePredictRequest(BaseModel):
    evidence_count: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    contradicting_evidence_count: int = Field(default=0, ge=0)
    verification_passed: Optional[bool] = None


class FuturePredictRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_date: Optional[str] = None
    rationale: str = Field(default="", max_length=500)


class FutureResolveRequest(BaseModel):
    actual_outcome: bool


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status", summary="Training status of every learned model")
async def ml_status(ctx: BlixContext = Depends(get_context)) -> dict:
    """Report whether each v0.3.10 learned model is still cold-starting or has trained."""
    return {
        "world_model": {"is_trained": ctx.latent_world_model.is_trained, "sample_count": ctx.latent_world_model.sample_count},
        "value_network": {"is_trained": ctx.value_network.is_trained, "sample_count": ctx.value_network.sample_count},
        "cross_encoder_reranker": {"is_using_real_model": ctx.cross_encoder_reranker.is_using_real_model},
        "confidence_model": {"is_trained": ctx.confidence_model.is_trained, "sample_count": ctx.confidence_model.sample_count},
        "tool_success_predictor": {"is_trained": ctx.tool_success_predictor.is_trained, "sample_count": ctx.tool_success_predictor.sample_count},
        "neural_attention_scorer": {"is_trained": ctx.neural_attention_scorer.is_trained, "sample_count": ctx.neural_attention_scorer.sample_count},
        "strategy_selector": {"is_trained": ctx.strategy_selector.is_trained, "sample_count": ctx.strategy_selector.sample_count},
        "memory_importance_predictor": {"is_trained": ctx.memory_importance_predictor.is_trained, "sample_count": ctx.memory_importance_predictor.sample_count},
        "continual_learning": ctx.continual_learning.learning_status(),
    }


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------


@router.post("/world-model/predict", summary="Predict plan success / tool failure / confidence decay")
async def world_model_predict(req: LatentStateInput, ctx: BlixContext = Depends(get_context)) -> dict:
    """Predict outcomes for a latent state via the Latent World Model."""
    return ctx.latent_world_model.predict(req.to_latent_state()).to_dict()


@router.post("/value/estimate", summary="Estimate V(state)")
async def value_estimate(req: LatentStateInput, ctx: BlixContext = Depends(get_context)) -> dict:
    """Estimate the value network's V(state) for a latent state."""
    value = ctx.value_network.value(req.to_latent_state())
    return {"value": round(value, 4), "is_trained": ctx.value_network.is_trained}


@router.post("/scenarios/rank", summary="Rank candidate scenarios by value")
async def rank_scenarios(req: RankScenariosRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Rank candidate scenarios best-first using the Scenario Ranker."""
    scenarios = [Scenario(name=s.name, description=s.description, state=s.state.to_latent_state()) for s in req.scenarios]
    ranked = ctx.scenario_ranker.rank(scenarios)
    return {"ranked": [r.to_dict() for r in ranked]}


# ---------------------------------------------------------------------------
# Tool success / confidence prediction
# ---------------------------------------------------------------------------


@router.post("/tool-success/predict", summary="Predict tool success probability")
async def tool_success_predict(req: ToolSuccessRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Predict P(success) for invoking a tool on a task."""
    result = ctx.tool_success_predictor.predict(
        req.task, req.tool, task_complexity_hint=req.task_complexity_hint,
        is_repeated_attempt=req.is_repeated_attempt, context_confidence=req.context_confidence,
    )
    return result.to_dict()


@router.post("/confidence/predict", summary="Predict P(answer_correct)")
async def confidence_predict(req: ConfidencePredictRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Predict P(answer_correct) given evidence/verification features."""
    result = ctx.confidence_model.predict_correctness(
        evidence_count=req.evidence_count, source_count=req.source_count,
        contradicting_evidence_count=req.contradicting_evidence_count, verification_passed=req.verification_passed,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Failure pattern mining
# ---------------------------------------------------------------------------


@router.get("/failure-clusters", summary="Discover recurring failure patterns")
async def failure_clusters(ctx: BlixContext = Depends(get_context)) -> dict:
    """Cluster current failure records to discover recurring patterns."""
    clusters = ctx.failure_clusterer.recurring_clusters()
    return {"total": len(clusters), "clusters": [c.to_dict() for c in clusters]}


# ---------------------------------------------------------------------------
# Predictive memory
# ---------------------------------------------------------------------------


@router.post("/future/predict", summary="Record a prediction about a future state")
async def future_predict(req: FuturePredictRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Record a new ExpectedState prediction."""
    state = ctx.future_memory.predict(
        subject=req.subject, confidence=req.confidence, predicted_date=req.predicted_date, rationale=req.rationale,
    )
    return state.to_dict()


@router.get("/future/pending", summary="List unresolved future predictions")
async def future_pending(ctx: BlixContext = Depends(get_context)) -> dict:
    """List all predictions awaiting resolution."""
    pending = ctx.future_memory.pending()
    return {"total": len(pending), "predictions": [p.to_dict() for p in pending]}


@router.post("/future/{expected_state_id}/resolve", summary="Resolve a future prediction")
async def future_resolve(expected_state_id: str, req: FutureResolveRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Resolve a prediction with its actual outcome, for later calibration scoring."""
    state = ctx.future_memory.resolve(expected_state_id, actual_outcome=req.actual_outcome)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Prediction '{expected_state_id}' not found.")
    return state.to_dict()
