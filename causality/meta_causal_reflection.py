"""
Meta-Causal Reflection — Blix v0.3.11  (New module 7, Phase 3)

Goes one level beyond ``causality.causal_reflection.CausalReflection``
(which reflects on ONE failure/topic). ``MetaCausalReflection`` asks
the deeper, recurring questions the spec calls out explicitly:

    Instead of:  "Why did task X fail?"
    Blix asks:   "Why do I repeatedly fail in research tasks?"
                 "What causes low confidence?"
                 "Which strategies cause success?"

These are aggregate queries over ``causality.cause_graph.CauseGraph``
and ``causality.principle.PrincipleStore`` — real graph/store queries
(filter, group, rank by confidence/evidence), not new inference. This
module does not discover anything CauseGraph/PrincipleStore don't
already contain; it answers the higher-level QUESTIONS a reflective
system would ask of that data.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from causality.cause_graph import CauseEdge, CauseGraph, CauseRelation
from causality.epistemic_status import EpistemicStatus
from causality.principle import Principle, PrincipleStore
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MetaCausalAnswer:
    """One answer to a recurring meta-causal question, with its evidence trail."""

    question: str
    answer_summary: str
    supporting_edges: list[CauseEdge] = field(default_factory=list)
    epistemic_status: EpistemicStatus = EpistemicStatus.DERIVED
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer_summary": self.answer_summary,
            "supporting_edges": [e.to_dict() for e in self.supporting_edges],
            "epistemic_status": self.epistemic_status.value, "generated_at": self.generated_at,
        }


class MetaCausalReflection:
    """
    Answers recurring, aggregate causal questions by querying
    ``CauseGraph`` and ``PrincipleStore``.

    Parameters
    ----------
    cause_graph:
        ``CauseGraph`` — source of typed cause-effect edges.
    principle_store:
        Optional ``PrincipleStore`` — source of synthesized principles.
    """

    def __init__(self, cause_graph: CauseGraph, principle_store: Optional[PrincipleStore] = None) -> None:
        self._cause_graph = cause_graph
        self._principle_store = principle_store

    # ------------------------------------------------------------------
    # "Why do I repeatedly fail in <domain> tasks?"
    # ------------------------------------------------------------------

    def why_repeated_failures(self, domain_keyword: str) -> MetaCausalAnswer:
        """Find recurring BLOCKS/CAUSES edges whose trigger or effect mentions ``domain_keyword``."""
        matching = [
            e for e in self._cause_graph.all_edges()
            if e.relation in (CauseRelation.BLOCKS, CauseRelation.CAUSES)
            and (domain_keyword.lower() in e.trigger.lower() or domain_keyword.lower() in e.effect.lower())
        ]
        matching.sort(key=lambda e: -e.evidence_count)

        if not matching:
            summary = f"No recurring causal pattern found yet for '{domain_keyword}'."
        else:
            top = matching[0]
            summary = (
                f"The most evidenced recurring pattern for '{domain_keyword}' is: "
                f"'{top.trigger}' {top.relation.value} '{top.effect}' "
                f"(seen {top.evidence_count} time(s), confidence {top.confidence:.2f})."
            )

        return MetaCausalAnswer(
            question=f"Why do I repeatedly fail in {domain_keyword} tasks?",
            answer_summary=summary, supporting_edges=matching[:5],
        )

    # ------------------------------------------------------------------
    # "What causes low confidence?"
    # ------------------------------------------------------------------

    def what_causes(self, effect_keyword: str, relation: Optional[CauseRelation] = None) -> MetaCausalAnswer:
        """All known causes of an effect matching ``effect_keyword``, ranked by confidence."""
        matching = [e for e in self._cause_graph.all_edges() if effect_keyword.lower() in e.effect.lower()]
        if relation is not None:
            matching = [e for e in matching if e.relation == relation]
        matching.sort(key=lambda e: -e.confidence)

        if not matching:
            summary = f"No known causes recorded yet for '{effect_keyword}'."
        else:
            causes = ", ".join(f"'{e.trigger}' ({e.relation.value}, conf={e.confidence:.2f})" for e in matching[:3])
            summary = f"Known contributors to '{effect_keyword}': {causes}."

        return MetaCausalAnswer(question=f"What causes {effect_keyword}?", answer_summary=summary, supporting_edges=matching[:5])

    # ------------------------------------------------------------------
    # "Which strategies cause success?"
    # ------------------------------------------------------------------

    def which_strategies_cause_success(self) -> MetaCausalAnswer:
        """All ENABLES/INCREASES edges whose effect mentions success/confidence/improvement (and isn't itself a negative outcome)."""
        success_keywords = ("success", "confidence", "improv", "reliab", "faster", "better", "higher")
        negative_qualifiers = (
            "low", "poor", "lack of", "reduced", "decreased", "no ",
            "unreliable", "drops", "declines", "worse", "slower", "higher risk",
            "failure", "error", "timeout", "crash",
        )
        matching = [
            e for e in self._cause_graph.all_edges()
            if e.relation in (CauseRelation.ENABLES, CauseRelation.INCREASES)
            and any(k in e.effect.lower() for k in success_keywords)
            and not any(neg in e.effect.lower() for neg in negative_qualifiers)
        ]
        matching.sort(key=lambda e: -e.confidence)

        if not matching:
            summary = "No strategy-to-success causal pattern recorded yet."
        else:
            strategies = ", ".join(f"'{e.trigger}'" for e in matching[:3])
            summary = f"Strategies/conditions most associated with success: {strategies}."

        return MetaCausalAnswer(question="Which strategies cause success?", answer_summary=summary, supporting_edges=matching[:5])

    # ------------------------------------------------------------------
    # Principle-level aggregate view
    # ------------------------------------------------------------------

    def top_principles_for_domain(self, domain_keyword: str, top_k: int = 5) -> list[Principle]:
        """Highest-confidence principles whose statement mentions ``domain_keyword``."""
        if self._principle_store is None:
            return []
        matching = [p for p in self._principle_store.all_principles() if domain_keyword.lower() in p.statement.lower()]
        return sorted(matching, key=lambda p: -p.confidence)[:top_k]

    # ------------------------------------------------------------------
    # v0.3.13 — Hypothesis failure aggregate (cross-run)
    # ------------------------------------------------------------------

    def which_hypotheses_failed_repeatedly(self, hypothesis_manager, min_evidence: int = 2) -> "MetaCausalAnswer":
        """
        v0.3.13 — Cross-run aggregate: which hypotheses were rejected
        despite multiple pieces of evidence? These represent recurring
        incorrect beliefs that the system keeps forming, flagging a
        systematic gap in prior knowledge or inference.
        """
        failed = hypothesis_manager.repeatedly_failed(min_evidence=min_evidence)
        if not failed:
            summary = "No hypotheses have been repeatedly rejected yet — not enough experimental history."
        else:
            examples = "; ".join(f"'{h.statement[:50]}' ({len(h.evidence)} evidence pieces)" for h in failed[:3])
            summary = (
                f"{len(failed)} hypothesis/hypotheses repeatedly formed but rejected: {examples}. "
                f"These may reflect systematic gaps in prior knowledge or persistent incorrect inference."
            )
        return MetaCausalAnswer(
            question="Which hypotheses failed repeatedly?",
            answer_summary=summary, supporting_edges=[],
        )
