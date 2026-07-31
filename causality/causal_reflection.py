"""
Causal Reflection — Blix v0.3.11  (New module 6, Phase 3)

Upgrades reflection from descriptive to PRESCRIPTIVE, and — per
explicit design direction — operates over PRINCIPLES rather than raw
failures directly:

    Task failed.
      ↓
    (descriptive, v0.3.8 reflection.meta_reflection.MetaReflectionEngine)

becomes:

    Task failed.

    Relevant principle: "Always evaluate before optimizing" (confidence 0.8)
    Alternative strategy: Tree-of-Thought
    Estimated success: 0.81

``CausalReflection`` extends ``reflection.meta_reflection.MetaReflectionEngine``
rather than replacing it — every existing pattern-detection check
(frequent replanning, low confidence, tool bottlenecks) still runs and
still feeds ``reflection.reflection_engine.ReflectionEngine`` exactly
as before. What's new is a second pass: for a given failure/topic,
look up relevant ``causality.principle.Principle`` objects (not raw
``CauseGraph``/``FailureMemory`` records directly — that lookup
happens once, upstream, in ``PrincipleSynthesizer``, and reflection
consumes the already-synthesized generalization) and turn them into a
concrete strategy recommendation.

The "estimated success" figure reuses
``world_model.value_network.ValueNetwork`` (v0.3.10) — it is a value
estimate from Blix's own learned/heuristic value function, not a
validated causal claim; see this module's ``CausalReflectionResult``
for the same epistemic-honesty fields used everywhere else in v0.3.11.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from causality.epistemic_status import EpistemicStatus
from causality.principle import Principle, PrincipleStore
from metacognition.strategy_manager import ReasoningStrategy
from reflection.meta_reflection import MetaReflectionEngine
from reflection.reflection_engine import ReflectionEngine
from world_model.value_network import ValueNetwork
from world_model.latent_world_model import LatentState
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CausalReflectionResult:
    """One prescriptive reflection: what failed, the relevant principle, and a recommended alternative."""

    topic: str
    relevant_principles: list[Principle] = field(default_factory=list)
    suggested_strategy: Optional[ReasoningStrategy] = None
    estimated_success: Optional[float] = None
    epistemic_status: EpistemicStatus = EpistemicStatus.DERIVED
    basis: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "relevant_principles": [p.to_dict() for p in self.relevant_principles],
            "suggested_strategy": self.suggested_strategy.value if self.suggested_strategy else None,
            "estimated_success": round(self.estimated_success, 4) if self.estimated_success is not None else None,
            "epistemic_status": self.epistemic_status.value,
            "basis": self.basis,
            "generated_at": self.generated_at,
        }


class CausalReflection(MetaReflectionEngine):
    """
    Extends ``MetaReflectionEngine`` with principle-grounded,
    prescriptive reflection.

    Parameters
    ----------
    reflection_engine:
        Passed through to ``MetaReflectionEngine``.
    principle_store:
        ``PrincipleStore`` — source of relevant principles for a topic.
    value_network:
        Optional ``ValueNetwork`` (v0.3.10) — supplies the "estimated
        success" figure for a suggested alternative strategy.
    """

    def __init__(
        self,
        reflection_engine: Optional[ReflectionEngine] = None,
        principle_store: Optional[PrincipleStore] = None,
        value_network: Optional[ValueNetwork] = None,
    ) -> None:
        super().__init__(reflection_engine=reflection_engine)
        self._principle_store = principle_store
        self._value_network = value_network

    # ------------------------------------------------------------------
    # Principle lookup
    # ------------------------------------------------------------------

    def _relevant_principles(self, topic: str, top_k: int = 3) -> list[Principle]:
        if self._principle_store is None:
            return []
        from causality.causal_memory import _jaccard  # reuse the project's existing token-overlap matcher

        scored = [
            (p, _jaccard(topic, p.statement))
            for p in self._principle_store.all_principles()
        ]
        relevant = [p for p, score in scored if score > 0]
        relevant.sort(key=lambda p: -p.confidence)
        return relevant[:top_k]

    # ------------------------------------------------------------------
    # Prescriptive reflection
    # ------------------------------------------------------------------

    def reflect_on_failure(
        self,
        topic: str,
        current_strategy: Optional[ReasoningStrategy] = None,
        alternative_strategy: Optional[ReasoningStrategy] = None,
        latent_state_for_alternative: Optional[LatentState] = None,
    ) -> CausalReflectionResult:
        """
        Produce a prescriptive reflection for one failed topic/task:
        look up relevant principles, and (if an alternative strategy
        and a latent state are supplied) estimate its likely success
        via the Value Network.

        ``estimated_success`` is explicitly an ``EpistemicStatus.PREDICTED``
        figure from Blix's own value function — not a validated causal claim.
        """
        principles = self._relevant_principles(topic)

        estimated_success = None
        basis = ""
        if alternative_strategy is not None and latent_state_for_alternative is not None and self._value_network is not None:
            estimated_success = self._value_network.value(latent_state_for_alternative)
            basis = f"ValueNetwork estimate (is_trained={self._value_network.is_trained})"

        return CausalReflectionResult(
            topic=topic, relevant_principles=principles, suggested_strategy=alternative_strategy,
            estimated_success=estimated_success, epistemic_status=EpistemicStatus.PREDICTED if estimated_success is not None else EpistemicStatus.DERIVED,
            basis=basis,
        )
