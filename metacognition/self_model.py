"""
Self Model — Blix v0.3.8  (New module 2)

Gives the Planner (and every other cognitive module) a persistent,
queryable answer to "what am I actually good at?" instead of treating
every domain as equally capable.

    SelfModel.capabilities["coding"]          -> 0.93
    SelfModel.capabilities["legal_reasoning"]  -> 0.52

This is intentionally a thin, storage-backed record — NOT a re-derivation
of capability scores (that's ``metacognition.capability_tracker.CapabilityTracker``'s
job). ``SelfModel`` is the place those scores (and qualitative
weaknesses/strengths/preferences/known_limits) live so other modules can
read a stable snapshot without recomputing anything.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

# Domains scoring below this are surfaced as "known limits" automatically.
_WEAKNESS_THRESHOLD = 0.6
_STRENGTH_THRESHOLD = 0.85


@dataclass
class SelfModel:
    """
    Blix's model of its own capabilities.

    Fields
    ------
    capabilities:
        domain -> score (0-1), e.g. {"coding": 0.93, "legal_reasoning": 0.52}.
    weaknesses:
        Domains explicitly flagged as weak (may be a superset of what
        ``low_capability_domains()`` would derive — e.g. flagged manually
        even before enough evidence accumulates).
    strengths:
        Domains explicitly flagged as strong.
    preferences:
        Free-form key -> value preferences (e.g. {"preferred_tool_for_research": "web_search"}).
    known_limits:
        Free-text statements of hard limits (e.g. "Cannot verify real-time facts after cutoff").
    """

    capabilities: dict[str, float] = field(default_factory=dict)
    weaknesses: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)
    known_limits: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "capabilities": {k: round(v, 3) for k, v in self.capabilities.items()},
            "weaknesses": self.weaknesses,
            "strengths": self.strengths,
            "preferences": self.preferences,
            "known_limits": self.known_limits,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        return cls(
            capabilities=d.get("capabilities", {}),
            weaknesses=d.get("weaknesses", []),
            strengths=d.get("strengths", []),
            preferences=d.get("preferences", {}),
            known_limits=d.get("known_limits", []),
            updated_at=d.get("updated_at", ""),
        )


class SelfModelStore:
    """
    Persists and maintains the live ``SelfModel``.

    Parameters
    ----------
    self_model_file:
        Path to ``self_model.json``.
    """

    def __init__(self, self_model_file: Path) -> None:
        self._file = self_model_file
        self._model = SelfModel()
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
            self._model = SelfModel.from_dict(raw)
            log.info("SelfModelStore: loaded model with %d tracked domain(s).", len(self._model.capabilities))
        except Exception as exc:
            log.warning("SelfModelStore: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(self._model.to_dict(), fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def model(self) -> SelfModel:
        return self._model

    def capability(self, domain: str, default: float = 0.5) -> float:
        """Capability score for a domain, defaulting to neutral 0.5 if untracked."""
        return self._model.capabilities.get(domain.lower(), default)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_capability(self, domain: str, score: float) -> None:
        domain = domain.lower().strip()
        score = max(0.0, min(1.0, score))
        self._model.capabilities[domain] = score
        self._sync_derived_lists(domain, score)
        self._model.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

    # ------------------------------------------------------------------
    # v0.3.13 — Live knowledge gap query (no new persistence)
    # ------------------------------------------------------------------

    def knowledge_gaps(self, knowledge_gap_tracker=None):
        """
        v0.3.13 live property: returns current knowledge gaps by querying
        ``KnowledgeGapTracker`` (if provided) and cross-referencing with
        SelfModel weaknesses. No new persistence layer — KnowledgeGapTracker
        is the single source of truth.

        When called without a tracker, falls back to the SelfModel's own
        ``weaknesses`` list as a rough proxy.
        """
        if knowledge_gap_tracker is not None:
            return knowledge_gap_tracker.gaps()
        # Fallback: map weaknesses to minimal KnowledgeGap-like dicts
        from knowledge.knowledge_gap_tracker import GapSeverity, KnowledgeGap
        return [
            KnowledgeGap(domain=w, severity=GapSeverity.MEDIUM, uncertainty=0.6,
                         gap_reason="Listed in SelfModel weaknesses")
            for w in self._model.weaknesses
        ]

    def _sync_derived_lists(self, domain: str, score: float) -> None:
        """Keep weaknesses/strengths lists roughly in sync with crossed thresholds."""
        if score < _WEAKNESS_THRESHOLD:
            if domain not in self._model.weaknesses:
                self._model.weaknesses.append(domain)
            if domain in self._model.strengths:
                self._model.strengths.remove(domain)
        elif score >= _STRENGTH_THRESHOLD:
            if domain not in self._model.strengths:
                self._model.strengths.append(domain)
            if domain in self._model.weaknesses:
                self._model.weaknesses.remove(domain)
        else:
            # Mid-range — no longer a flagged weakness or strength.
            if domain in self._model.weaknesses:
                self._model.weaknesses.remove(domain)
            if domain in self._model.strengths:
                self._model.strengths.remove(domain)

    def add_known_limit(self, statement: str) -> None:
        if statement not in self._model.known_limits:
            self._model.known_limits.append(statement)
            self._save()

    def set_preference(self, key: str, value: str) -> None:
        self._model.preferences[key] = value
        self._save()

    def flag_weakness(self, domain: str) -> None:
        domain = domain.lower().strip()
        if domain not in self._model.weaknesses:
            self._model.weaknesses.append(domain)
            self._save()

    def flag_strength(self, domain: str) -> None:
        domain = domain.lower().strip()
        if domain not in self._model.strengths:
            self._model.strengths.append(domain)
            self._save()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def is_weak_in(self, domain: str) -> bool:
        domain = domain.lower().strip()
        return domain in self._model.weaknesses or self.capability(domain) < _WEAKNESS_THRESHOLD

    def is_strong_in(self, domain: str) -> bool:
        domain = domain.lower().strip()
        return domain in self._model.strengths or self.capability(domain) >= _STRENGTH_THRESHOLD

    def low_capability_domains(self, threshold: float = _WEAKNESS_THRESHOLD) -> list[str]:
        return sorted(d for d, s in self._model.capabilities.items() if s < threshold)

    def high_capability_domains(self, threshold: float = _STRENGTH_THRESHOLD) -> list[str]:
        return sorted(d for d, s in self._model.capabilities.items() if s >= threshold)

    def summary(self) -> str:
        if not self._model.capabilities:
            return "No tracked capabilities yet."
        ranked = sorted(self._model.capabilities.items(), key=lambda kv: -kv[1])
        parts = [f"{domain}={score:.2f}" for domain, score in ranked]
        return "Capabilities: " + ", ".join(parts)
