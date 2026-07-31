"""
Contradiction Resolver — Blix v0.3.7  (New module 5)

Fixes the core bug the spec calls out: v0.3.1's ``ContradictionDetector``
treats every conflict as winner-take-all. Two new facts about the same
topic don't always mean one is wrong — they fall into four distinct
cases, and getting the classification right is the whole point of this
module:

    Replacement      Delhi → Kolkata          (old value stops being true)
    Parallel Truth   Python AND Rust           (both still true, different scope/time)
    Merge            "AI" == "Artificial Intelligence"   (same fact, different wording)
    Conflict         Genuinely contradictory, insufficient evidence to resolve

``ContradictionResolver`` classifies a candidate pair and delegates to
the right downstream mechanism:

    Replacement → core.state_transition.StateTransitionEngine.transition()
                   (or core.truth_manager.TruthManager.replace() for beliefs)
    Parallel    → core.truth_manager.TruthManager — both stay ACTIVE, no merge
    Merge       → core.truth_manager.TruthManager.merge()
    Conflict    → core.truth_manager.TruthManager.mark_conflicting()
                   (needs evidence comparison before it can resolve further)

Python 3.10 compatible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from core.truth_manager import TruthManager, TruthStatus
from memory.beliefs import Belief, BeliefStore, _jaccard
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Case classification
# ---------------------------------------------------------------------------


class ContradictionCase(str, Enum):
    REPLACEMENT = "replacement"
    PARALLEL_TRUTH = "parallel_truth"
    MERGE = "merge"
    CONFLICT = "conflict"
    NONE = "none"          # not actually a contradiction


@dataclass
class ResolutionResult:
    """Outcome of resolving one candidate pair."""

    case: ContradictionCase
    record_a_id: str
    record_b_id: str
    explanation: str = ""
    winner_id: Optional[str] = None   # for REPLACEMENT/MERGE

    def to_dict(self) -> dict:
        return {
            "case": self.case.value,
            "record_a_id": self.record_a_id,
            "record_b_id": self.record_b_id,
            "explanation": self.explanation,
            "winner_id": self.winner_id,
        }


# ---------------------------------------------------------------------------
# Heuristic signal patterns
# ---------------------------------------------------------------------------

# Markers that suggest the SAME underlying entity changed (replacement),
# as opposed to two genuinely separate facts coexisting.
_REPLACEMENT_MARKERS = re.compile(
    r"\b(now|moved to|switched to|changed to|relocated to|"
    r"no longer|instead of|used to|previously|formerly)\b", re.IGNORECASE,
)

# Markers suggesting BOTH things remain true (parallel truth), often
# scoped differently ("for work" vs "for hobby", "and also").
_PARALLEL_MARKERS = re.compile(
    r"\b(also|as well as|in addition|both|and also|"
    r"for work|for personal|on weekends|sometimes)\b", re.IGNORECASE,
)

# Acronym/synonym pattern: short form vs long form of the same concept.
_ACRONYM_RE = re.compile(r"^[A-Z]{2,6}$")


class ContradictionResolver:
    """
    Classifies and resolves contradictions between two competing claims
    about the same topic/entity.

    Parameters
    ----------
    truth_manager:
        ``TruthManager`` — owns TruthStatus transitions for both
        REPLACEMENT and MERGE/CONFLICT outcomes.
    belief_store:
        Optional ``BeliefStore`` — used for MERGE confidence comparison
        and to look up statement text from belief ids.
    merge_similarity_threshold:
        Statement similarity above this is treated as the same fact in
        different words (MERGE), not separate competing claims.
    """

    def __init__(
        self,
        truth_manager: TruthManager,
        belief_store: Optional[BeliefStore] = None,
        merge_similarity_threshold: float = 0.55,
    ) -> None:
        self._truth = truth_manager
        self._beliefs = belief_store
        self._merge_threshold = merge_similarity_threshold

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        text_a: str,
        text_b: str,
        value_a: Optional[str] = None,
        value_b: Optional[str] = None,
    ) -> ContradictionCase:
        """
        Determine which of the four cases applies to a pair of competing
        claims.

        Parameters
        ----------
        text_a / text_b:
            Full claim text (for marker detection), e.g. the original
            memory sentences.
        value_a / value_b:
            The specific competing values, if this concerns a tracked
            attribute (e.g. "Delhi" vs "Kolkata"). If provided and
            textually very similar to each other (acronym/synonym),
            classified as MERGE.
        """
        # 1. Acronym/synonym check on the values themselves
        if value_a and value_b and self._is_acronym_pair(value_a, value_b):
            return ContradictionCase.MERGE

        # 2. Statement-level similarity → MERGE (same fact, different words)
        if _jaccard(text_a, text_b) >= self._merge_threshold and value_a is None:
            return ContradictionCase.MERGE

        # 3. Replacement markers → one supersedes the other
        if _REPLACEMENT_MARKERS.search(text_a) or _REPLACEMENT_MARKERS.search(text_b):
            return ContradictionCase.REPLACEMENT

        # 4. Parallel-truth markers → both remain true
        if _PARALLEL_MARKERS.search(text_a) or _PARALLEL_MARKERS.search(text_b):
            return ContradictionCase.PARALLEL_TRUTH

        # 5. No markers either way — genuinely ambiguous → CONFLICT
        return ContradictionCase.CONFLICT

    def _is_acronym_pair(self, value_a: str, value_b: str) -> bool:
        """True if one value is an acronym/short-form of the other (AI / Artificial Intelligence)."""
        a, b = value_a.strip(), value_b.strip()
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        if not _ACRONYM_RE.match(short.replace(" ", "")):
            return False
        long_initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", long)).upper()
        return long_initials == short.upper()

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        record_a_id: str,
        record_b_id: str,
        text_a: str,
        text_b: str,
        value_a: Optional[str] = None,
        value_b: Optional[str] = None,
        confidence_a: float = 0.5,
        confidence_b: float = 0.5,
        newer_id: Optional[str] = None,
    ) -> ResolutionResult:
        """
        Classify and resolve a contradiction between two records (belief
        ids, state snapshot ids, or any TruthManager-tracked id).

        Parameters
        ----------
        newer_id:
            Which of the two ids is chronologically newer — used to break
            ties for REPLACEMENT (the newer claim wins by default; this
            mirrors a recency prior, not strict evidence comparison).
        """
        case = self.classify(text_a, text_b, value_a, value_b)

        if case == ContradictionCase.MERGE:
            survivor = record_a_id
            self._truth.merge(record_a_id, record_b_id, surviving_id=survivor)
            return ResolutionResult(
                case=case, record_a_id=record_a_id, record_b_id=record_b_id,
                winner_id=survivor,
                explanation=f"'{text_a[:40]}' and '{text_b[:40]}' refer to the same fact; merged.",
            )

        if case == ContradictionCase.REPLACEMENT:
            # Prefer explicit recency signal; fall back to confidence.
            if newer_id is not None:
                old_id = record_b_id if newer_id == record_a_id else record_a_id
                new_id = newer_id
            else:
                new_id = record_a_id if confidence_a >= confidence_b else record_b_id
                old_id = record_b_id if new_id == record_a_id else record_a_id
            self._truth.replace(old_id, new_id)
            return ResolutionResult(
                case=case, record_a_id=record_a_id, record_b_id=record_b_id,
                winner_id=new_id,
                explanation=f"'{old_id}' superseded by '{new_id}'.",
            )

        if case == ContradictionCase.PARALLEL_TRUTH:
            # Both remain ACTIVE — no winner. Just make sure both are
            # registered so they don't default to an unintended status.
            self._truth.ensure(record_a_id, TruthStatus.ACTIVE)
            self._truth.ensure(record_b_id, TruthStatus.ACTIVE)
            return ResolutionResult(
                case=case, record_a_id=record_a_id, record_b_id=record_b_id,
                explanation=f"'{text_a[:40]}' and '{text_b[:40]}' both remain true (parallel truth).",
            )

        # CONFLICT — flag for evidence comparison, don't pick a winner yet.
        self._truth.mark_conflicting(record_a_id, record_b_id)
        return ResolutionResult(
            case=case, record_a_id=record_a_id, record_b_id=record_b_id,
            explanation=(
                f"'{text_a[:40]}' and '{text_b[:40]}' conflict; insufficient evidence "
                "to resolve automatically."
            ),
        )

    # ------------------------------------------------------------------
    # Evidence comparison (for CONFLICT cases)
    # ------------------------------------------------------------------

    def compare_evidence(
        self,
        evidence_count_a: int,
        evidence_count_b: int,
        source_count_a: int,
        source_count_b: int,
        confidence_a: float,
        confidence_b: float,
    ) -> Optional[str]:
        """
        Given evidence statistics for two conflicting claims, determine
        whether one now clearly dominates and should be promoted out of
        CONFLICT.

        Returns "a", "b", or None if still genuinely ambiguous.

        Heuristic: a claim wins if it has BOTH more distinct sources AND
        higher confidence than the other; otherwise stays unresolved
        (this deliberately requires convergent evidence, not just one
        signal, to avoid flip-flopping on noisy single observations).
        """
        a_dominates = source_count_a > source_count_b and confidence_a > confidence_b
        b_dominates = source_count_b > source_count_a and confidence_b > confidence_a
        if a_dominates and not b_dominates:
            return "a"
        if b_dominates and not a_dominates:
            return "b"
        return None

    def try_resolve_conflict(
        self,
        record_a_id: str,
        record_b_id: str,
        evidence_count_a: int,
        evidence_count_b: int,
        source_count_a: int,
        source_count_b: int,
        confidence_a: float,
        confidence_b: float,
    ) -> Optional[ResolutionResult]:
        """
        Attempt to promote a CONFLICT into a resolved REPLACEMENT once
        enough evidence has accumulated on one side.

        Returns ``None`` if still ambiguous (record stays CONFLICTING).
        """
        winner = self.compare_evidence(
            evidence_count_a, evidence_count_b,
            source_count_a, source_count_b,
            confidence_a, confidence_b,
        )
        if winner is None:
            return None

        winner_id = record_a_id if winner == "a" else record_b_id
        loser_id = record_b_id if winner == "a" else record_a_id
        self._truth.replace(loser_id, winner_id, note="resolved from conflict via evidence comparison")
        return ResolutionResult(
            case=ContradictionCase.REPLACEMENT,
            record_a_id=record_a_id, record_b_id=record_b_id,
            winner_id=winner_id,
            explanation=f"Conflict resolved: '{winner_id}' now has stronger evidence.",
        )
