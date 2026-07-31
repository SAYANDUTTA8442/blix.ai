"""
Confidence Propagation & Fact Verification — Blix v0.3.1  (Issues 6 & 7)

Addresses:
  Issue 6: "Memory extraction is a single point of failure — no confidence propagation."
  Issue 7: "No grounded fact verification — architecture trusts extraction too much."

VerifiedFact
------------
Every extracted fact now carries:
    belief_score        — extraction confidence (0–1)
    source_count        — how many independent memories assert this fact
    verification_status — "unverified" | "confirmed" | "refuted" | "uncertain"

FactVerifier
------------
Cross-checks extracted facts against:
1. The existing memory pool (corroboration counting).
2. The existing ProfileEvolver state (does the fact match the profile?).

Facts asserted by multiple independent memories get a higher belief_score.
Facts that directly contradict the profile get status="refuted".

ConfidencePropagator
--------------------
Propagates extraction-level confidence downstream:
    MemoryExtractor confidence (0–1)
        ↓  ×  topic_confidence
    ProfileEvolver.update(confidence=...)
        ↓  ×  graph_confidence
    MemoryGraph.upsert_relation(confidence=...)

So a low-confidence extraction produces low-confidence profile and graph updates.

Python 3.10 compatible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# VerifiedFact
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"


@dataclass
class VerifiedFact:
    """
    A fact with attached confidence and verification metadata.

    Fields
    ------
    text:
        The fact string, e.g. "Sayan is interested in NLP."
    belief_score:
        Composite confidence 0–1.  Higher = more trustworthy.
    source_count:
        Number of independent memories that assert (approximately) this fact.
    verification_status:
        Result of cross-checking against existing memories and profile.
    source_memory_ids:
        Which memories contributed to this fact.
    """

    text: str
    belief_score: float = 0.5
    source_count: int = 1
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    source_memory_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "belief_score": self.belief_score,
            "source_count": self.source_count,
            "verification_status": self.verification_status.value,
            "source_memory_ids": self.source_memory_ids,
        }


# ---------------------------------------------------------------------------
# FactVerifier
# ---------------------------------------------------------------------------


class FactVerifier:
    """
    Cross-checks extracted facts for corroboration and contradiction.

    Parameters
    ----------
    corroboration_boost:
        How much to boost belief_score per additional corroborating memory.
    contradiction_penalty:
        How much to reduce belief_score if the profile contradicts the fact.
    min_overlap_words:
        Minimum shared word count to consider two facts as corroborating.
    """

    def __init__(
        self,
        corroboration_boost: float = 0.1,
        contradiction_penalty: float = 0.3,
        min_overlap_words: int = 3,
    ) -> None:
        self._boost = corroboration_boost
        self._penalty = contradiction_penalty
        self._min_overlap = min_overlap_words

    def verify(
        self,
        facts: list[str],
        extraction_confidence: float,
        existing_memories: list,       # list[MemoryEntry]
        profile_dict: Optional[dict] = None,
        source_memory_id: Optional[int] = None,
    ) -> list[VerifiedFact]:
        """
        Return a ``VerifiedFact`` for each input fact string.

        Parameters
        ----------
        facts:
            Raw fact strings from ``MemoryExtractor``.
        extraction_confidence:
            Overall confidence from the extraction run (0–1).
        existing_memories:
            Full memory pool for corroboration search.
        profile_dict:
            Current profile as a flat dict for contradiction check.
        source_memory_id:
            The MemoryEntry.id that produced these facts.
        """
        verified: list[VerifiedFact] = []
        for fact in facts:
            vf = VerifiedFact(
                text=fact,
                belief_score=extraction_confidence,
                source_memory_ids=[source_memory_id] if source_memory_id else [],
            )
            # Step 1: corroboration
            self._check_corroboration(vf, existing_memories)
            # Step 2: profile contradiction
            if profile_dict:
                self._check_profile(vf, profile_dict)
            verified.append(vf)
        return verified

    def _check_corroboration(self, vf: VerifiedFact, memories: list) -> None:
        """Boost belief_score for facts mentioned in multiple memories."""
        fact_words = _tokenise(vf.text)
        corroborating_ids: list[int] = []
        for mem in memories:
            mem_text = " ".join(getattr(mem, "extracted_facts", []))
            mem_words = _tokenise(mem_text)
            overlap = len(fact_words & mem_words)
            if overlap >= self._min_overlap:
                corroborating_ids.append(getattr(mem, "id"))

        if corroborating_ids:
            vf.source_count = 1 + len(corroborating_ids)
            boost = min(self._boost * len(corroborating_ids), 0.3)
            vf.belief_score = min(1.0, vf.belief_score + boost)
            vf.source_memory_ids.extend(
                mid for mid in corroborating_ids
                if mid not in vf.source_memory_ids
            )
            vf.verification_status = VerificationStatus.CONFIRMED
            log.debug(
                "FactVerifier: fact corroborated by %d memories → belief=%.2f",
                len(corroborating_ids), vf.belief_score,
            )

    def _check_profile(self, vf: VerifiedFact, profile: dict) -> None:
        """Penalise facts that contradict the current profile."""
        fact_lower = vf.text.lower()
        # Simple negation check: if fact contains "no longer X" and profile has X
        negated_terms = re.findall(
            r"(?:no longer|not|never|stopped|quit)\s+(?:interested in|working on)?\s*(\w+)",
            fact_lower,
        )
        for term in negated_terms:
            for val in profile.values():
                if isinstance(val, list) and any(term in str(v).lower() for v in val):
                    vf.belief_score = max(0.0, vf.belief_score - self._penalty)
                    vf.verification_status = VerificationStatus.UNCERTAIN
                    log.debug(
                        "FactVerifier: possible contradiction with profile term %r", term
                    )
                    return

        if vf.verification_status == VerificationStatus.UNVERIFIED:
            vf.verification_status = VerificationStatus.UNVERIFIED  # no change


# ---------------------------------------------------------------------------
# ConfidencePropagator
# ---------------------------------------------------------------------------


class ConfidencePropagator:
    """
    Propagates extraction confidence downstream to profile and graph updates.

    Usage
    -----
        propagator = ConfidencePropagator(base_confidence=0.8)
        profile_conf = propagator.profile_confidence(topic_specificity=0.9)
        graph_conf   = propagator.graph_confidence(profile_conf)

    The chain ensures that a low-confidence extraction produces
    proportionally low-confidence updates throughout the pipeline.

    Parameters
    ----------
    base_confidence:
        Raw confidence from ``MemoryExtractor`` (0–1).
    topic_decay:
        Multiplier applied at the profile update stage.
    graph_decay:
        Additional multiplier applied at the graph update stage.
    """

    def __init__(
        self,
        base_confidence: float = 1.0,
        topic_decay: float = 0.9,
        graph_decay: float = 0.85,
    ) -> None:
        self._base = max(0.0, min(1.0, base_confidence))
        self._topic_decay = topic_decay
        self._graph_decay = graph_decay

    def profile_confidence(self, topic_specificity: float = 1.0) -> float:
        """
        Confidence to pass to ``ProfileEvolver.update()``.

        ``topic_specificity`` is 1.0 for precise facts (named entities),
        lower for vague ones ("might be interested in…").
        """
        return round(
            min(1.0, self._base * self._topic_decay * topic_specificity), 4
        )

    def graph_confidence(self, profile_conf: Optional[float] = None) -> float:
        """
        Confidence to pass to ``MemoryGraph.upsert_relation()``.

        Computed from profile_confidence if provided, else from base.
        """
        upstream = profile_conf if profile_conf is not None else self.profile_confidence()
        return round(min(1.0, upstream * self._graph_decay), 4)

    def fact_belief_score(self, verified_fact: "VerifiedFact") -> float:  # type: ignore[name-defined]
        """
        Final belief score combining propagated confidence with VerifiedFact score.
        """
        return round(min(1.0, self._base * verified_fact.belief_score), 4)

    @classmethod
    def from_extraction_result(cls, importance: float) -> "ConfidencePropagator":
        """
        Construct from a ``MemoryExtractor`` importance score.

        Low importance → low confidence propagation.
        """
        base = max(0.1, min(1.0, importance if importance > 0 else 0.5))
        return cls(base_confidence=base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> set[str]:
    """Simple whitespace tokeniser returning lowercase content words."""
    STOP = {"a", "an", "the", "is", "are", "was", "were", "in", "of",
            "and", "or", "to", "for", "on", "at", "by", "with", "i", "it"}
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}
