"""
Memory Consolidation Engine — Blix v0.3.2  (Feature 2)

Converts repeated/similar memories into stable ``CanonicalFact`` objects:

    100 similar memories → 1 Canonical Fact (confidence=0.98, evidence_count=37)

Pipeline
--------
1. **Duplicate detection** — group extracted_facts strings by semantic
   similarity (embedding cosine sim) or lexical overlap fallback.
2. **Similar memory merging** — within each group, pick the most
   representative phrasing (highest-importance source, or shortest
   canonical form).
3. **Fact strengthening** — confidence accumulates with each new
   corroborating memory using a saturating update rule.
4. **Confidence accumulation** — ``confidence = 1 - (1-base)^evidence_count``
   (each new piece of evidence multiplicatively reduces doubt).
5. **Memory compression** — corroborated source memories can be handed to
   ``MemoryLifecycleManager.compress()`` since their content now lives in
   the canonical fact.

Output example
---------------
    {
      "fact": "User prefers PyTorch for AI development",
      "confidence": 0.98,
      "evidence_count": 37
    }

Python 3.10 compatible.
"""
# DEPRECATED — reflection.consolidation_engine (ISSUE-009)
#
# This module is superseded by memory.hybrid.consolidation.consolidation_engine.
# The class ``ConsolidationEngine`` here is the v0.3.x implementation;
# ``memory.hybrid.consolidation.consolidation_engine.ConsolidationEngine`` is the v0.3.15+ HGSHM implementation.
#
# These are different classes with different APIs. Callers that need
# the v0.3.15+ version must update their imports:
#
#     # Old (this file — legacy):
#     from reflection.consolidation_engine import ConsolidationEngine
#
#     # New (HGSHM-backed):
#     from memory.hybrid.consolidation.consolidation_engine import ConsolidationEngine
#
# This file will be removed in v0.4. Do not add new callers.
# Issue: https://github.com/blix/blix/issues/9
#


from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical fact model
# ---------------------------------------------------------------------------


@dataclass
class CanonicalFact:
    """
    A consolidated, stable fact derived from multiple corroborating memories.

    Fields
    ------
    fact_id:
        Stable identifier, e.g. ``"fact_7"``.
    fact:
        Canonical natural-language statement.
    confidence:
        ``1 - (1 - base_confidence) ** evidence_count``, capped at 0.99.
    evidence_count:
        Number of source memories supporting this fact.
    source_memory_ids:
        All MemoryEntry ids that contributed evidence.
    variants:
        Alternative phrasings seen across source memories (for audit).
    topic:
        Primary topic tag, if known.
    """

    fact_id: str
    fact: str
    confidence: float = 0.5
    evidence_count: int = 1
    source_memory_ids: list[int] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    topic: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "fact": self.fact,
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "source_memory_ids": self.source_memory_ids,
            "variants": self.variants,
            "topic": self.topic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalFact":
        return cls(
            fact_id=d["fact_id"],
            fact=d["fact"],
            confidence=d.get("confidence", 0.5),
            evidence_count=d.get("evidence_count", 1),
            source_memory_ids=d.get("source_memory_ids", []),
            variants=d.get("variants", []),
            topic=d.get("topic", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Consolidation engine
# ---------------------------------------------------------------------------


class ConsolidationEngine:
    """
    Detects duplicate/similar facts across memories and consolidates them
    into ``CanonicalFact`` objects.

    Parameters
    ----------
    facts_file:
        Path to ``canonical_facts.json``.
    similarity_threshold:
        Cosine similarity (when embeddings provided) or token-overlap
        ratio (fallback) above which two facts are considered the "same".
    base_confidence:
        Starting confidence assigned to a fact's first observation.
    confidence_growth:
        Per-evidence confidence growth rate, used in
        ``confidence = 1 - (1 - base_confidence) ** evidence_count``
        when ``confidence_growth=1.0`` (default). Values < 1 slow growth.
    """

    def __init__(
        self,
        facts_file: Path,
        similarity_threshold: float = 0.75,
        base_confidence: float = 0.5,
        confidence_growth: float = 1.0,
    ) -> None:
        self._file = facts_file
        self._threshold = similarity_threshold
        self._base = base_confidence
        self._growth = confidence_growth
        self._facts: dict[str, CanonicalFact] = {}
        self._next_id = 0
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
                cf = CanonicalFact.from_dict(item)
                self._facts[cf.fact_id] = cf
            if self._facts:
                self._next_id = max(
                    int(fid.replace("fact_", "")) for fid in self._facts
                ) + 1
            log.info("ConsolidationEngine: loaded %d canonical facts.", len(self._facts))
        except Exception as exc:
            log.warning("ConsolidationEngine: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(
                [f.to_dict() for f in self._facts.values()], fh, indent=2, ensure_ascii=False
            )

    # ------------------------------------------------------------------
    # Core consolidation
    # ------------------------------------------------------------------

    def consolidate(
        self,
        fact_text: str,
        source_memory_id: int,
        topic: str = "",
        embedding: Optional[np.ndarray] = None,
    ) -> CanonicalFact:
        """
        Add one fact observation, merging into an existing CanonicalFact
        if sufficiently similar, else creating a new one.

        Returns the resulting (possibly updated) ``CanonicalFact``.
        """
        match = self._find_match(fact_text, embedding)

        if match is not None:
            self._merge(match, fact_text, source_memory_id)
            self._save()
            log.debug(
                "ConsolidationEngine: merged into %s (evidence=%d, conf=%.3f)",
                match.fact_id, match.evidence_count, match.confidence,
            )
            return match

        cf = CanonicalFact(
            fact_id=f"fact_{self._next_id}",
            fact=fact_text,
            confidence=self._base,
            evidence_count=1,
            source_memory_ids=[source_memory_id],
            variants=[fact_text],
            topic=topic,
        )
        self._next_id += 1
        self._facts[cf.fact_id] = cf
        self._save()
        log.debug("ConsolidationEngine: new canonical fact %s", cf.fact_id)
        return cf

    def consolidate_batch(
        self,
        facts: list[tuple[str, int, str]],  # (fact_text, source_memory_id, topic)
        embeddings: Optional[dict[int, np.ndarray]] = None,
    ) -> list[CanonicalFact]:
        """Consolidate a batch of (fact, memory_id, topic) tuples."""
        results = []
        for fact_text, mem_id, topic in facts:
            emb = embeddings.get(mem_id) if embeddings else None
            results.append(self.consolidate(fact_text, mem_id, topic, emb))
        return results

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _find_match(
        self, fact_text: str, embedding: Optional[np.ndarray]
    ) -> Optional[CanonicalFact]:
        best: Optional[CanonicalFact] = None
        best_sim = 0.0
        for cf in self._facts.values():
            sim = self._similarity(fact_text, cf, embedding)
            if sim > best_sim:
                best_sim = sim
                best = cf
        return best if best_sim >= self._threshold else None

    def _similarity(
        self, fact_text: str, cf: CanonicalFact, embedding: Optional[np.ndarray]
    ) -> float:
        """
        Compute similarity between a new fact and an existing canonical fact.

        Uses token-overlap (Jaccard) on normalised text — embedding-based
        similarity can be layered in by a caller that precomputes and
        caches per-fact embeddings (not stored here to keep this module
        dependency-light).
        """
        return _jaccard(fact_text, cf.fact)

    # ------------------------------------------------------------------
    # Merging / confidence accumulation
    # ------------------------------------------------------------------

    def _merge(self, cf: CanonicalFact, fact_text: str, source_memory_id: int) -> None:
        """Merge a new observation into an existing CanonicalFact."""
        if source_memory_id not in cf.source_memory_ids:
            cf.source_memory_ids.append(source_memory_id)
            cf.evidence_count += 1

        if fact_text not in cf.variants:
            cf.variants.append(fact_text)
            # Prefer the shortest variant as canonical (most concise)
            cf.fact = min(cf.variants, key=len)

        cf.confidence = self._accumulate_confidence(cf.evidence_count)
        cf.updated_at = datetime.now(timezone.utc).isoformat()

    def _accumulate_confidence(self, evidence_count: int) -> float:
        """
        confidence = 1 - (1 - base)^(evidence_count * growth)

        With base=0.5 and growth=1.0:
            n=1  → 0.50
            n=2  → 0.75
            n=5  → 0.969
            n=37 → ~1.0 (capped at 0.99)
        """
        exponent = evidence_count * self._growth
        conf = 1.0 - (1.0 - self._base) ** exponent
        return round(min(0.99, conf), 4)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_fact(self, fact_id: str) -> Optional[CanonicalFact]:
        return self._facts.get(fact_id)

    def list_facts(
        self,
        topic: Optional[str] = None,
        min_confidence: float = 0.0,
        min_evidence: int = 1,
    ) -> list[CanonicalFact]:
        facts = list(self._facts.values())
        if topic:
            facts = [f for f in facts if f.topic == topic]
        facts = [f for f in facts if f.confidence >= min_confidence and f.evidence_count >= min_evidence]
        return sorted(facts, key=lambda f: -f.confidence)

    def strongest_facts(self, top_k: int = 10) -> list[CanonicalFact]:
        """Return the top-k facts by confidence (then evidence count)."""
        return sorted(
            self._facts.values(),
            key=lambda f: (-f.confidence, -f.evidence_count),
        )[:top_k]

    def consolidatable_memory_ids(self, min_evidence: int = 3) -> set[int]:
        """
        Return memory ids that are part of a well-corroborated canonical
        fact (evidence_count >= min_evidence) and are therefore candidates
        for ``MemoryLifecycleManager.compress()``.

        Always keeps the FIRST source memory (the original observation)
        out of the compression set, so provenance is preserved.
        """
        ids: set[int] = set()
        for cf in self._facts.values():
            if cf.evidence_count >= min_evidence and len(cf.source_memory_ids) > 1:
                ids.update(cf.source_memory_ids[1:])
        return ids

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    def summary(self) -> str:
        if not self._facts:
            return "No canonical facts yet."
        top = self.strongest_facts(3)
        parts = [f'"{f.fact}" ({f.confidence:.2f}, n={f.evidence_count})' for f in top]
        return f"{self.fact_count} canonical facts. Top: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


_STOP = {"a", "an", "the", "is", "are", "was", "were", "to", "for", "of",
         "in", "on", "and", "or", "with", "i", "user", "prefers"}


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in _STOP and len(w) > 1
    }
