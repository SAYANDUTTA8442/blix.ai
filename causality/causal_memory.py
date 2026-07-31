"""
Causal Memory — Blix v0.3.11  (New module 3, Phase 1)

Existing memory layers store WHAT happened: episodes
(``core.memory_manager``), beliefs (``memory.beliefs``), and tracked
state (``core.state_tracker``). None of them store a reusable WHY.
``CausalMemory`` fills that gap with a dedicated record type:

    CauseMemory(
        trigger="skipping benchmarks",
        effect="poor optimization",
        confidence=0.7,
    )

These become reusable principles Blix can recall directly — "have I
seen this trigger before, and what tends to follow?" — without
re-deriving the pattern from scratch each time. ``CausalMemory`` is
deliberately a separate, lighter-weight store from
``causality.cause_graph.CauseGraph``: the graph is the queryable
NETWORK structure (typed relations, multi-hop traversal); CausalMemory
is the flat, fast-lookup RECALL store, keyed for "what do I remember
about trigger X" lookups. Both are fed by the same kinds of real
co-occurrence observations and both carry ``EpistemicStatus.DERIVED``.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from causality.epistemic_status import EpistemicStatus
from utils.logger import get_logger

log = get_logger(__name__)

_STOP = {"a", "an", "the", "is", "are", "to", "for", "of", "in", "on", "and", "or", "with"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class CauseMemory:
    """One reusable (trigger, effect) principle with accumulated confidence."""

    trigger: str
    effect: str
    confidence: float = 0.5
    evidence_count: int = 1
    epistemic_status: EpistemicStatus = EpistemicStatus.DERIVED
    cause_memory_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.cause_memory_id:
            self.cause_memory_id = f"{self.trigger}->{self.effect}"

    def to_dict(self) -> dict:
        return {
            "cause_memory_id": self.cause_memory_id, "trigger": self.trigger, "effect": self.effect,
            "confidence": round(self.confidence, 4), "evidence_count": self.evidence_count,
            "epistemic_status": self.epistemic_status.value,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CauseMemory":
        return cls(
            trigger=d["trigger"], effect=d["effect"], confidence=d.get("confidence", 0.5),
            evidence_count=d.get("evidence_count", 1),
            epistemic_status=EpistemicStatus(d.get("epistemic_status", EpistemicStatus.DERIVED.value)),
            cause_memory_id=d.get("cause_memory_id", ""), created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
        )


class CausalMemoryStore:
    """
    Stores and recalls ``CauseMemory`` principles.

    Parameters
    ----------
    causal_memory_file:
        Path to ``causal_memory.json``.
    similarity_threshold:
        Jaccard threshold for considering a new (trigger, effect) pair
        a reinforcement of an existing one rather than a new record.
    confidence_increment:
        Confidence boost per corroborating observation.
    """

    def __init__(self, causal_memory_file: Path, similarity_threshold: float = 0.5, confidence_increment: float = 0.1) -> None:
        self._file = causal_memory_file
        self._threshold = similarity_threshold
        self._increment = confidence_increment
        self._memories: dict[str, CauseMemory] = {}
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
                m = CauseMemory.from_dict(item)
                self._memories[m.cause_memory_id] = m
        except Exception as exc:
            log.warning("CausalMemoryStore: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([m.to_dict() for m in self._memories.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, trigger: str, effect: str, confidence: float = 0.5) -> CauseMemory:
        """Record one observed (trigger, effect) co-occurrence, reinforcing a matching existing memory if found."""
        existing = self.recall(trigger, effect)
        if existing is not None:
            existing.evidence_count += 1
            existing.confidence = min(1.0, existing.confidence + self._increment)
            existing.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            return existing

        memory = CauseMemory(trigger=trigger, effect=effect, confidence=confidence)
        self._memories[memory.cause_memory_id] = memory
        self._save()
        return memory

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(self, trigger: str, effect: Optional[str] = None) -> Optional[CauseMemory]:
        """Find the best-matching existing CauseMemory for a (trigger[, effect]) pair, by Jaccard similarity."""
        best: Optional[CauseMemory] = None
        best_score = 0.0
        for m in self._memories.values():
            trigger_score = _jaccard(trigger, m.trigger)
            if trigger_score < self._threshold:
                continue
            combined_score = trigger_score
            if effect is not None:
                effect_score = _jaccard(effect, m.effect)
                if effect_score < self._threshold:
                    continue
                combined_score = (trigger_score + effect_score) / 2
            if combined_score > best_score:
                best, best_score = m, combined_score
        return best

    def effects_of_trigger(self, trigger: str, top_k: int = 5) -> list[CauseMemory]:
        """All remembered effects for triggers similar to ``trigger``, highest confidence first."""
        matches = [m for m in self._memories.values() if _jaccard(trigger, m.trigger) >= self._threshold]
        return sorted(matches, key=lambda m: -m.confidence)[:top_k]

    def all_memories(self) -> list[CauseMemory]:
        return list(self._memories.values())

    def high_confidence_principles(self, threshold: float = 0.7, min_evidence: int = 2) -> list[CauseMemory]:
        """Memories confident and well-evidenced enough to be treated as reusable principles."""
        return [m for m in self._memories.values() if m.confidence >= threshold and m.evidence_count >= min_evidence]

    @property
    def count(self) -> int:
        return len(self._memories)
