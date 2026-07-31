"""
Strategy Evolution — Blix v0.3.11  (New module 8, Phase 3)

Upgrades strategy adaptation from confidence/complexity-threshold-only
(``metacognition.strategy_manager.StrategyManager``, v0.3.8) and
learned-but-unexplained (``metacognition.strategy_selector.StrategySelectorNetwork``,
v0.3.10) to EXPLAINABLE, cause-informed evolution:

    Low confidence
      ↓
    ToT
    (v0.3.8 — a threshold fired, no causal explanation)

becomes:

    Failure cluster
      ↓
    Cause analysis
      ↓
    New strategy
    (v0.3.11 — the strategy switch cites WHICH cause/principle motivated it)

``StrategyEvolution`` does not replace ``StrategySelectorNetwork`` — it
wraps it, and its job is purely to attach an explanation: given a
``causality.cause_graph.CauseGraph`` lookup or a relevant
``causality.principle.Principle``, propose a strategy change AND state
the causal reasoning behind it, then optionally feed the outcome back
into ``StrategySelectorNetwork.observe_outcome()`` so the learned model
keeps improving from the same evidence.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from causality.cause_graph import CauseGraph, CauseRelation
from causality.epistemic_status import EpistemicStatus
from causality.principle import Principle, PrincipleStore
from metacognition.strategy_manager import ReasoningStrategy
from metacognition.strategy_selector import StrategySelectorNetwork
from planning.plan_evaluator import PlanQualityScore
from utils.logger import get_logger

log = get_logger(__name__)

_STRATEGY_FOR_RELATION = {
    CauseRelation.BLOCKS: ReasoningStrategy.DECOMPOSE_FURTHER,
    CauseRelation.CAUSES: ReasoningStrategy.CRITIC_FIRST,
    CauseRelation.DECREASES: ReasoningStrategy.CRITIC_FIRST,
    CauseRelation.ENABLES: ReasoningStrategy.TREE_OF_THOUGHT,
    CauseRelation.INCREASES: ReasoningStrategy.TREE_OF_THOUGHT,
}


@dataclass
class StrategyEvolutionDecision:
    """A strategy recommendation with its explicit causal/principle justification."""

    ref_key: str
    recommended_strategy: ReasoningStrategy
    explanation: str
    cited_principle: Optional[Principle] = None
    epistemic_status: EpistemicStatus = EpistemicStatus.DERIVED
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "ref_key": self.ref_key, "recommended_strategy": self.recommended_strategy.value,
            "explanation": self.explanation,
            "cited_principle": self.cited_principle.to_dict() if self.cited_principle else None,
            "epistemic_status": self.epistemic_status.value, "decided_at": self.decided_at,
        }


class StrategyEvolution:
    """
    Proposes explainable strategy changes grounded in CauseGraph
    edges/Principles, and (optionally) feeds outcomes back into the
    learned StrategySelectorNetwork.

    Parameters
    ----------
    cause_graph:
        ``CauseGraph`` — source of typed cause-effect edges.
    principle_store:
        Optional ``PrincipleStore`` — source of synthesized principles.
    strategy_selector:
        Optional ``StrategySelectorNetwork`` — outcomes observed here
        are also reported there, so the learned model benefits too.
    """

    def __init__(
        self,
        cause_graph: CauseGraph,
        principle_store: Optional[PrincipleStore] = None,
        strategy_selector: Optional[StrategySelectorNetwork] = None,
    ) -> None:
        self._cause_graph = cause_graph
        self._principle_store = principle_store
        self._strategy_selector = strategy_selector

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve_strategy(self, ref_key: str, failure_topic: str) -> StrategyEvolutionDecision:
        """
        Propose a strategy change for ``ref_key``, explained by the
        highest-confidence relevant CauseGraph edge (or Principle, if
        one exists) touching ``failure_topic``.
        """
        causes = [
            e for e in self._cause_graph.all_edges()
            if failure_topic.lower() in e.trigger.lower() or failure_topic.lower() in e.effect.lower()
        ]
        causes.sort(key=lambda e: -e.confidence)

        cited_principle = None
        if self._principle_store is not None:
            from causality.causal_memory import _jaccard
            scored = [(p, _jaccard(failure_topic, p.statement)) for p in self._principle_store.all_principles()]
            relevant = sorted([p for p, s in scored if s > 0], key=lambda p: -p.confidence)
            cited_principle = relevant[0] if relevant else None

        if causes:
            top_cause = causes[0]
            strategy = _STRATEGY_FOR_RELATION.get(top_cause.relation, ReasoningStrategy.CRITIC_FIRST)
            explanation = (
                f"Recommending {strategy.value} because '{top_cause.trigger}' {top_cause.relation.value} "
                f"'{top_cause.effect}' (confidence {top_cause.confidence:.2f}, {top_cause.evidence_count} observation(s))."
            )
        elif cited_principle is not None:
            strategy = ReasoningStrategy.CRITIC_FIRST
            explanation = f"Recommending {strategy.value} per principle: \"{cited_principle.statement}\" (confidence {cited_principle.confidence:.2f})."
        else:
            strategy = ReasoningStrategy.DIRECT
            explanation = f"No causal pattern or principle found for '{failure_topic}' yet — no evolution warranted."

        return StrategyEvolutionDecision(
            ref_key=ref_key, recommended_strategy=strategy, explanation=explanation, cited_principle=cited_principle,
        )

    # ------------------------------------------------------------------
    # Feedback loop into StrategySelectorNetwork
    # ------------------------------------------------------------------

    def record_outcome(
        self, ref_key: str, quality: Optional[PlanQualityScore], strategy_used: ReasoningStrategy, succeeded: bool,
    ) -> None:
        """Report an outcome to the wrapped StrategySelectorNetwork, if one is configured."""
        if self._strategy_selector is not None:
            self._strategy_selector.observe_outcome(ref_key, quality, strategy_used, succeeded)
