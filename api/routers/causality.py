"""
/causality router — Blix v0.3.11

Endpoints
---------
POST /causality/cause/record                — record a cause-effect observation
GET  /causality/cause/effects/{trigger}          — known effects of a trigger
GET  /causality/cause/causes/{effect}                — known causes of an effect
POST /causality/belief-graph/dependency                  — declare a supports/weakens belief dependency
POST /causality/belief-graph/propagate                       — propagate a confidence change through the belief DAG
GET  /causality/principles                                       — list synthesized principles
POST /causality/principles/synthesize                                — synthesize principles from current evidence
POST /causality/reflect/causal                                           — prescriptive reflection on a failed topic
GET  /causality/reflect/why                                                  — why do I repeatedly fail in <domain>?
GET  /causality/reflect/causes-of                                               — what causes <effect>?
POST /causality/strategy/evolve                                                     — propose an explainable strategy change
POST /causality/counterfactual/explore                                                  — rank what-if alternatives

Every COUNTERFACTUAL-tagged response includes confidence, evidence_count,
basis, and validated_causally=False — never silently promoted to a belief.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.context import BlixContext
from api.deps import get_context
from causality.cause_graph import CauseRelation
from causality.belief_dependency_graph import DependencyRelation
from causality.counterfactual_engine import CounterfactualAlternative
from metacognition.strategy_manager import ReasoningStrategy
from world_model.latent_world_model import LatentState

router = APIRouter(prefix="/causality", tags=["Causality"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RecordCauseRequest(BaseModel):
    trigger: str = Field(..., min_length=1, max_length=300)
    effect: str = Field(..., min_length=1, max_length=300)
    relation: CauseRelation
    initial_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AddBeliefDependencyRequest(BaseModel):
    source_belief_id: str = Field(..., min_length=1)
    target_belief_id: str = Field(..., min_length=1)
    relation: DependencyRelation
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class PropagateBeliefRequest(BaseModel):
    changed_belief_id: str = Field(..., min_length=1)
    confidence_delta: float = Field(..., ge=-1.0, le=1.0)


class CausalReflectRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    alternative_strategy: Optional[ReasoningStrategy] = None
    latent_state: Optional[dict] = None


class StrategyEvolveRequest(BaseModel):
    ref_key: str = Field(..., min_length=1, max_length=200)
    failure_topic: str = Field(..., min_length=1, max_length=500)


class CounterfactualAlternativeInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)
    resulting_state: dict = Field(default_factory=dict)


class CounterfactualExploreRequest(BaseModel):
    current_state: dict = Field(default_factory=dict)
    alternatives: list[CounterfactualAlternativeInput] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# CauseGraph
# ---------------------------------------------------------------------------


@router.post("/cause/record", summary="Record a cause-effect observation")
async def record_cause(req: RecordCauseRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Record one observed (trigger, effect, relation) co-occurrence."""
    edge = ctx.cause_graph.record_observation(req.trigger, req.effect, req.relation, initial_confidence=req.initial_confidence)
    return edge.to_dict()


@router.get("/cause/effects/{trigger}", summary="Known effects of a trigger")
async def effects_of(trigger: str, ctx: BlixContext = Depends(get_context)) -> dict:
    """All known effects for a given trigger, highest confidence first."""
    edges = ctx.cause_graph.effects_of(trigger)
    return {"total": len(edges), "edges": [e.to_dict() for e in edges]}


@router.get("/cause/causes/{effect}", summary="Known causes of an effect")
async def causes_of(effect: str, ctx: BlixContext = Depends(get_context)) -> dict:
    """All known causes for a given effect, highest confidence first."""
    edges = ctx.cause_graph.causes_of(effect)
    return {"total": len(edges), "edges": [e.to_dict() for e in edges]}


# ---------------------------------------------------------------------------
# Belief Dependency Graph
# ---------------------------------------------------------------------------


@router.post("/belief-graph/dependency", summary="Declare a supports/weakens belief dependency")
async def add_belief_dependency(req: AddBeliefDependencyRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Declare that target_belief_id depends on source_belief_id."""
    edge = ctx.belief_dependency_graph.add_dependency(
        req.source_belief_id, req.target_belief_id, req.relation, strength=req.strength,
    )
    return edge.to_dict()


@router.post("/belief-graph/propagate", summary="Propagate a confidence change through the belief DAG")
async def propagate_belief(req: PropagateBeliefRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Propagate a confidence delta from one belief through its dependents."""
    results = ctx.belief_dependency_graph.propagate(req.changed_belief_id, req.confidence_delta)
    return {"total_affected": len(results), "results": [r.to_dict() for r in results]}


# ---------------------------------------------------------------------------
# Principles
# ---------------------------------------------------------------------------


@router.get("/principles", summary="List synthesized principles")
async def list_principles(
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """List all synthesized principles, optionally filtered to a minimum confidence."""
    principles = ctx.principle_store.high_confidence(min_confidence) if min_confidence is not None else ctx.principle_store.all_principles()
    return {"total": len(principles), "principles": [p.to_dict() for p in principles]}


@router.post("/principles/synthesize", summary="Synthesize principles from current evidence")
async def synthesize_principles(ctx: BlixContext = Depends(get_context)) -> dict:
    """Synthesize new principles from currently well-evidenced CauseGraph edges and failure clusters."""
    principles = ctx.principle_synthesizer.synthesize_all()
    return {"synthesized": len(principles), "principles": [p.to_dict() for p in principles]}


# ---------------------------------------------------------------------------
# Causal / Meta-Causal Reflection
# ---------------------------------------------------------------------------


@router.post("/reflect/causal", summary="Prescriptive reflection on a failed topic")
async def reflect_causal(req: CausalReflectRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Look up relevant principles for a failed topic and (optionally) estimate an alternative strategy's success."""
    latent_state = LatentState(**req.latent_state) if req.latent_state else None
    result = ctx.causal_reflection.reflect_on_failure(
        topic=req.topic, alternative_strategy=req.alternative_strategy, latent_state_for_alternative=latent_state,
    )
    return result.to_dict()


@router.get("/reflect/why", summary="Why do I repeatedly fail in <domain>?")
async def reflect_why(domain: str = Query(..., min_length=1), ctx: BlixContext = Depends(get_context)) -> dict:
    """Answer 'why do I repeatedly fail in <domain> tasks?' from CauseGraph evidence."""
    answer = ctx.meta_causal_reflection.why_repeated_failures(domain)
    return answer.to_dict()


@router.get("/reflect/causes-of", summary="What causes <effect>?")
async def reflect_causes_of(
    effect: str = Query(..., min_length=1),
    relation: Optional[CauseRelation] = Query(default=None),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Answer 'what causes <effect>?' from CauseGraph evidence."""
    answer = ctx.meta_causal_reflection.what_causes(effect, relation=relation)
    return answer.to_dict()


# ---------------------------------------------------------------------------
# Strategy Evolution
# ---------------------------------------------------------------------------


@router.post("/strategy/evolve", summary="Propose an explainable strategy change")
async def evolve_strategy(req: StrategyEvolveRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """Propose a strategy change grounded in CauseGraph/Principle evidence, with an explicit explanation."""
    decision = ctx.strategy_evolution.evolve_strategy(req.ref_key, req.failure_topic)
    return decision.to_dict()


# ---------------------------------------------------------------------------
# Counterfactual Scenario Engine
# ---------------------------------------------------------------------------


@router.post("/counterfactual/explore", summary="Rank what-if alternatives")
async def explore_counterfactuals(req: CounterfactualExploreRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Rank candidate 'what if' alternatives by estimated value. Every
    result is tagged epistemic_status=counterfactual,
    validated_causally=false — these are estimates, not observations,
    and are never written to the belief system by this endpoint.
    """
    current_state = LatentState(**req.current_state) if req.current_state else LatentState()
    alternatives = [
        CounterfactualAlternative(
            name=a.name, description=a.description,
            resulting_state=LatentState(**a.resulting_state) if a.resulting_state else LatentState(),
        )
        for a in req.alternatives
    ]
    results = ctx.counterfactual_engine.explore(current_state, alternatives, top_k=req.top_k)
    return {"total": len(results), "scenarios": [r.to_dict() for r in results]}
