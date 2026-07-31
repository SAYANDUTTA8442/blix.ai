"""
Belief Store — Blix v0.3.7  (New module 4)

A ``Belief`` is a standalone factual claim Blix holds, distinct from a
``StateSnapshot`` (which is scoped to one entity/attribute/value triple
with a time window). Beliefs are free-text statements — "User prefers
dark mode", "AI and Artificial Intelligence refer to the same concept" —
that accumulate evidence over multiple observations and whose
``TruthStatus`` is owned by ``core.truth_manager.TruthManager``.

    Belief(
        statement="User's favorite language is Rust",
        confidence=0.85,
        evidence_count=4,
        source_count=2,
        status=TruthStatus.ACTIVE,
    )

This is the substrate ``core.contradiction_resolver.ContradictionResolver``
operates on for the MERGE and CONFLICT cases — StateSnapshot/transition
semantics handle the REPLACEMENT and PARALLEL-TRUTH cases for tracked
attributes, but free-form beliefs (not yet modeled as entity/attribute)
need their own store.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from causality.epistemic_status import EpistemicStatus

from core.truth_manager import TruthStatus
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------


@dataclass
class Belief:
    """
    One factual claim with accumulating evidence.

    Fields
    ------
    statement:
        The natural-language claim.
    confidence:
        0–1, increases with corroborating evidence, decreases on conflict.
    evidence_count:
        How many times this belief has been (re)observed.
    source_count:
        How many DISTINCT sources (memory ids) support it — distinct
        from evidence_count, since the same source can reinforce a
        belief more than once but distinct sources matter more for
        confidence.
    status:
        ``TruthStatus`` — owned by ``TruthManager`` but cached here for
        convenient filtering without a manager round-trip.
    epistemic_status:
        v0.3.11 — ``causality.epistemic_status.EpistemicStatus``, ORTHOGONAL
        to ``status``: ``status`` answers "is this currently true",
        ``epistemic_status`` answers "how did we come to hold this."
        Defaults to OBSERVED (the v0.3.7-and-earlier assumption — every
        prior caller of ``add_or_reinforce()`` was reporting something
        actually witnessed). A belief with ``epistemic_status=HYPOTHESIS``
        is not yet trustworthy and must go through
        ``BeliefStore.confirm_observation()`` before other modules
        should treat it as OBSERVED.
    topic:
        Optional topic label for retrieval/filtering.
    """

    statement: str
    belief_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    confidence: float = 0.5
    evidence_count: int = 1
    source_memory_ids: list[int] = field(default_factory=list)
    status: TruthStatus = TruthStatus.ACTIVE
    epistemic_status: EpistemicStatus = EpistemicStatus.OBSERVED
    topic: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def source_count(self) -> int:
        return len(set(self.source_memory_ids))

    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id,
            "statement": self.statement,
            "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "source_memory_ids": self.source_memory_ids,
            "status": self.status.value,
            "epistemic_status": self.epistemic_status.value,
            "topic": self.topic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Belief":
        return cls(
            statement=d["statement"],
            belief_id=d.get("belief_id", uuid.uuid4().hex[:8]),
            confidence=d.get("confidence", 0.5),
            evidence_count=d.get("evidence_count", 1),
            source_memory_ids=d.get("source_memory_ids", []),
            status=TruthStatus(d.get("status", "active")),
            epistemic_status=EpistemicStatus(d.get("epistemic_status", EpistemicStatus.OBSERVED.value)),
            topic=d.get("topic", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Belief Store
# ---------------------------------------------------------------------------

_STOP = {"a", "an", "the", "is", "are", "to", "for", "of", "in", "on", "and",
         "or", "with", "user", "users", "i", "my"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


class BeliefStore:
    """
    Persists and queries ``Belief`` objects.

    Parameters
    ----------
    beliefs_file:
        Path to ``beliefs.json``.
    similarity_threshold:
        Jaccard token-overlap threshold for considering two statements
        "the same belief" when reinforcing or detecting conflicts.
    confidence_increment:
        How much corroborating evidence boosts confidence.
    confidence_decrement:
        How much conflicting evidence reduces confidence.
    """

    def __init__(
        self,
        beliefs_file: Path,
        similarity_threshold: float = 0.5,
        confidence_increment: float = 0.1,
        confidence_decrement: float = 0.15,
    ) -> None:
        self._file = beliefs_file
        self._threshold = similarity_threshold
        self._inc = confidence_increment
        self._dec = confidence_decrement
        self._beliefs: dict[str, Belief] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                b = Belief.from_dict(item)
                self._beliefs[b.belief_id] = b
            log.info("BeliefStore: loaded %d belief(s).", len(self._beliefs))
        except Exception as exc:
            log.warning("BeliefStore: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([b.to_dict() for b in self._beliefs.values()], fh, indent=2, ensure_ascii=False)

    def persist(self) -> None:
        """
        Public save hook for callers (e.g. v0.3.11's
        ``causality.belief_dependency_graph.BeliefDependencyGraph``)
        that legitimately mutate a ``Belief`` obtained via ``get()`` in
        place and need to flush the change to disk, without reaching
        into the private ``_save()`` method directly.
        """
        self._save()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_or_reinforce(
        self,
        statement: str,
        confidence: float = 0.5,
        source_memory_id: Optional[int] = None,
        topic: str = "",
        epistemic_status: EpistemicStatus = EpistemicStatus.OBSERVED,
    ) -> Belief:
        """
        Add a new belief, or reinforce an existing similar one
        (incrementing evidence_count and bumping confidence).

        ``epistemic_status`` defaults to OBSERVED, matching every
        pre-v0.3.11 caller's assumption (this is the general-purpose
        entry point for things Blix actually witnessed). Callers
        working with anything less trustworthy (a prediction, a
        counterfactual, an unconfirmed hypothesis) should use
        ``add_hypothesis()`` instead — ``add_or_reinforce()`` does not
        accept COUNTERFACTUAL or PREDICTED here by convention, enforced
        by every caller in this codebase using the narrower method for
        those cases (see module docstring's pipeline safeguard).
        """
        existing = self.find_similar(statement)
        if existing:
            existing.evidence_count += 1
            existing.confidence = min(1.0, existing.confidence + self._inc)
            if source_memory_id is not None:
                existing.source_memory_ids.append(source_memory_id)
            existing.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            log.debug("BeliefStore: reinforced %s (confidence=%.2f)", existing.belief_id, existing.confidence)
            return existing

        belief = Belief(
            statement=statement, confidence=confidence, topic=topic,
            source_memory_ids=[source_memory_id] if source_memory_id is not None else [],
            epistemic_status=epistemic_status,
        )
        self._beliefs[belief.belief_id] = belief
        self._save()
        log.info("BeliefStore: new belief '%s' (id=%s)", statement[:60], belief.belief_id)
        return belief

    def add_hypothesis(
        self, statement: str, confidence: float = 0.3, basis: str = "", topic: str = "",
    ) -> Belief:
        """
        Add a candidate belief with ``epistemic_status=HYPOTHESIS`` —
        the ONLY entry point intended for low-trust sources (a
        counterfactual estimate, a model prediction someone wants to
        track toward possible confirmation, a guess). A hypothesis is
        NOT treated as OBSERVED by any other v0.3.x module's filtering
        logic (e.g. ``all_with_status`` callers should also check
        ``epistemic_status`` if they care about trust level) until
        ``confirm_observation()`` is explicitly called on it.

        ``basis`` is a free-text note on why this hypothesis was
        proposed (e.g. "counterfactual: ToT strategy on similar past
        failures") — stored as the belief's topic if no topic is given.
        """
        belief = Belief(
            statement=statement, confidence=max(0.0, min(1.0, confidence)),
            topic=topic or basis, epistemic_status=EpistemicStatus.HYPOTHESIS,
        )
        self._beliefs[belief.belief_id] = belief
        self._save()
        log.info("BeliefStore: new HYPOTHESIS '%s' (id=%s, basis=%r)", statement[:60], belief.belief_id, basis)
        return belief

    def confirm_observation(self, belief_id: str, source_memory_id: Optional[int] = None) -> Optional[Belief]:
        """
        Promote a HYPOTHESIS to OBSERVED. This is the ONLY function in
        the codebase that moves a belief's epistemic_status toward
        trusted — it requires the caller to explicitly assert "this was
        actually confirmed", it never happens as a side effect of
        anything else (most importantly: never as a side effect of
        ``causality.counterfactual_engine`` producing a scenario estimate).
        """
        belief = self._beliefs.get(belief_id)
        if belief is None or belief.epistemic_status != EpistemicStatus.HYPOTHESIS:
            return None
        belief.epistemic_status = EpistemicStatus.OBSERVED
        belief.evidence_count += 1
        belief.confidence = min(1.0, belief.confidence + self._inc)
        if source_memory_id is not None:
            belief.source_memory_ids.append(source_memory_id)
        belief.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        log.info("BeliefStore: confirmed HYPOTHESIS -> OBSERVED for %s", belief_id)
        return belief

    def weaken(self, belief_id: str, reason: str = "") -> Optional[Belief]:
        """Reduce confidence in a belief (e.g. due to conflicting evidence)."""
        belief = self._beliefs.get(belief_id)
        if belief is None:
            return None
        belief.confidence = max(0.0, belief.confidence - self._dec)
        belief.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        log.info("BeliefStore: weakened %s to confidence=%.2f (%s)", belief_id, belief.confidence, reason)
        return belief

    def set_status(self, belief_id: str, status: TruthStatus) -> Optional[Belief]:
        belief = self._beliefs.get(belief_id)
        if belief is None:
            return None
        belief.status = status
        belief.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return belief

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, belief_id: str) -> Optional[Belief]:
        return self._beliefs.get(belief_id)

    def find_similar(self, statement: str, exclude_id: Optional[str] = None) -> Optional[Belief]:
        """Find the most similar existing belief above the similarity threshold."""
        best: Optional[Belief] = None
        best_score = 0.0
        for b in self._beliefs.values():
            if b.belief_id == exclude_id:
                continue
            score = _jaccard(statement, b.statement)
            if score >= self._threshold and score > best_score:
                best, best_score = b, score
        return best

    def find_conflicting_candidates(self, statement: str, min_overlap: float = 0.3) -> list[Belief]:
        """
        Find beliefs that share substantial topic overlap with ``statement``
        but aren't similar enough to be the same belief — i.e. potential
        CONFLICT or PARALLEL TRUTH candidates for the ContradictionResolver.
        """
        candidates = []
        for b in self._beliefs.values():
            score = _jaccard(statement, b.statement)
            if min_overlap <= score < self._threshold:
                candidates.append(b)
        return candidates

    def all_active(self) -> list[Belief]:
        return [b for b in self._beliefs.values() if b.status == TruthStatus.ACTIVE]

    def all_with_status(self, status: TruthStatus) -> list[Belief]:
        return [b for b in self._beliefs.values() if b.status == status]

    def by_topic(self, topic: str) -> list[Belief]:
        topic = topic.lower()
        return [b for b in self._beliefs.values() if b.topic.lower() == topic]

    def low_confidence(self, threshold: float = 0.3) -> list[Belief]:
        return [b for b in self._beliefs.values() if b.confidence < threshold]

    @property
    def count(self) -> int:
        return len(self._beliefs)
