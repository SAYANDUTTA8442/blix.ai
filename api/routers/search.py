"""
/search router — Blix v0.3.12

Endpoints
---------
POST /search/beam                          — run beam search from a start state toward a goal
POST /search/explain                           — explain a previously-run beam search result
POST /search/counterfactual/trajectories           — counterfactual exploration with full Scenario(actions, predicted_outcome, confidence, trajectory)
GET  /search/predictions/calibration                   — full calibration report (Brier/ECE/over-under-confidence)
GET  /search/predictions/drift                             — prediction calibration drift over time
GET  /search/predictions/calibration/{subject}                 — calibration scoped to one subject
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.context import BlixContext
from api.deps import get_context
from causality.counterfactual_engine import CounterfactualAlternative
from planning.beam_search import BeamSearchResult
from world_model.latent_world_model import LatentState

router = APIRouter(prefix="/search", tags=["Imagination & Search"])


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


class CandidateActionInput(BaseModel):
    action: str = Field(..., min_length=1, max_length=200)
    resulting_state: LatentStateInput


class BeamSearchRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=500)
    start_state: LatentStateInput
    candidate_actions: list[CandidateActionInput] = Field(
        ..., min_length=1,
        description="Candidate (action, resulting_state) pairs applied uniformly at every search depth.",
    )
    beam_width: int = Field(default=3, ge=1, le=10)
    max_depth: int = Field(default=3, ge=1, le=6)


class ExplainBeamSearchRequest(BaseModel):
    """Re-supplies the same search inputs so the critic can re-run and explain the result (stateless API boundary)."""
    goal: str = Field(..., min_length=1, max_length=500)
    start_state: LatentStateInput
    candidate_actions: list[CandidateActionInput] = Field(..., min_length=1)
    beam_width: int = Field(default=3, ge=1, le=10)
    max_depth: int = Field(default=3, ge=1, le=6)


class CounterfactualAlternativeInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)
    resulting_state: LatentStateInput


class CounterfactualTrajectoriesRequest(BaseModel):
    current_state: LatentStateInput
    alternatives: list[CounterfactualAlternativeInput] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------


def _make_action_generator(candidates: list[CandidateActionInput]):
    """Uniform action generator: the same candidate (action, resulting_state) set at every depth."""
    pairs = [(c.action, c.resulting_state.to_latent_state()) for c in candidates]

    def _generator(state: LatentState) -> list[tuple[str, LatentState]]:
        return pairs

    return _generator


@router.post("/beam", summary="Run beam search from a start state toward a goal")
async def run_beam_search(req: BeamSearchRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Search for the best action sequence toward a goal via beam search.
    No MCTS — top-K beams pruned by ValueNetwork score at each depth.
    """
    from planning.beam_search import BeamSearchPlanner
    planner = BeamSearchPlanner(ctx.value_network, beam_width=req.beam_width, max_depth=req.max_depth)
    action_generator = _make_action_generator(req.candidate_actions)
    result = planner.search(req.goal, req.start_state.to_latent_state(), action_generator)
    return result.to_dict()


@router.post("/explain", summary="Explain a beam search decision")
async def explain_beam_search(req: ExplainBeamSearchRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Re-run the same search and produce a DecisionExplanation: why this
    branch was chosen, what assumptions it rests on, and what's risky.
    """
    from planning.beam_search import BeamSearchPlanner
    planner = BeamSearchPlanner(ctx.value_network, beam_width=req.beam_width, max_depth=req.max_depth)
    action_generator = _make_action_generator(req.candidate_actions)
    result = planner.search(req.goal, req.start_state.to_latent_state(), action_generator)
    explanation = ctx.search_critic.explain(result)
    return explanation.to_dict()


# ---------------------------------------------------------------------------
# Counterfactual trajectories (v0.3.12 extension of v0.3.11's engine)
# ---------------------------------------------------------------------------


@router.post("/counterfactual/trajectories", summary="Counterfactual exploration with full trajectories")
async def explore_counterfactual_trajectories(req: CounterfactualTrajectoriesRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Rank what-if alternatives, same as
    /causality/counterfactual/explore, but each result also carries a
    full Scenario(actions, predicted_outcome, confidence, trajectory)
    — a first-class current_state -> action -> resulting_state chain.
    """
    current_state = req.current_state.to_latent_state()
    alternatives = [
        CounterfactualAlternative(name=a.name, description=a.description, resulting_state=a.resulting_state.to_latent_state())
        for a in req.alternatives
    ]
    results = ctx.counterfactual_engine.explore_with_trajectories(current_state, alternatives, top_k=req.top_k)
    return {"total": len(results), "scenarios": [r.to_dict() for r in results]}


# ---------------------------------------------------------------------------
# Prediction calibration (Items 6 + 10 wiring)
# ---------------------------------------------------------------------------


@router.get("/predictions/calibration", summary="Full prediction calibration report")
async def prediction_calibration(ctx: BlixContext = Depends(get_context)) -> dict:
    """Brier score, expected calibration error, over/under-confidence rate, per-bucket breakdown."""
    return ctx.prediction_evaluator.calibration_report()


@router.get("/predictions/drift", summary="Prediction calibration drift over time")
async def prediction_drift(
    min_samples_per_half: int = Query(default=3, ge=1, le=50), ctx: BlixContext = Depends(get_context),
) -> dict:
    """Compare Brier score between earlier and more recent resolved predictions."""
    drift = ctx.prediction_evaluator.prediction_drift(min_samples_per_half=min_samples_per_half)
    if drift is None:
        return {"available": False, "reason": "Not enough resolved predictions yet for a meaningful drift comparison."}
    return {"available": True, **drift.to_dict()}


@router.get("/predictions/calibration/{subject}", summary="Calibration scoped to one subject")
async def prediction_calibration_for_subject(subject: str, ctx: BlixContext = Depends(get_context)) -> dict:
    """Calibration report for predictions about one specific subject."""
    return ctx.prediction_evaluator.calibration_for_subject(subject)
