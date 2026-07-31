"""
Knowledge Gap Tracker — Blix v0.3.13  (New module 4, "Curiosity + Active Experimentation")

Answers the three questions the spec names explicitly:

    "What don't I know?"
    "Where am I weak?"
    "What needs exploration?"

by maintaining a persistent store of ``KnowledgeGap`` records:

    KnowledgeGap(
        domain,
        severity,           # how urgently this gap needs filling
        uncertainty,        # how unsure Blix is about this domain
        evidence_count,     # how many observations exist for it
    )

Gaps are populated from three real sources:

  1. ``metacognition.self_model.SelfModelStore`` — low capability
     estimates signal a knowledge gap in that domain.
  2. ``agents.failure_memory.FailureMemory`` — recurring failures in a
     domain that have no matching principle signal a gap in understanding
     why they happen.
  3. ``causality.cause_graph.CauseGraph`` — domains mentioned in
     low-confidence or poorly-evidenced edges need more investigation.

This module feeds ``metacognition.self_model.SelfModelStore.knowledge_gaps``
(a live property query, v0.3.13) and ``curiosity.curiosity_engine.CuriosityEngine``
(the "unknown domains" trigger, v0.3.13).

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


class GapSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KnowledgeGap:
    """One domain where Blix's knowledge is identified as incomplete or unreliable."""

    domain: str
    severity: GapSeverity = GapSeverity.MEDIUM
    uncertainty: float = 0.5      # 0-1; high = very uncertain
    evidence_count: int = 0       # how many observations exist
    gap_reason: str = ""          # human-readable source of this gap
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "domain": self.domain, "severity": self.severity.value,
            "uncertainty": round(self.uncertainty, 4), "evidence_count": self.evidence_count,
            "gap_reason": self.gap_reason, "discovered_at": self.discovered_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeGap":
        return cls(
            domain=d["domain"], severity=GapSeverity(d.get("severity", "medium")),
            uncertainty=d.get("uncertainty", 0.5), evidence_count=d.get("evidence_count", 0),
            gap_reason=d.get("gap_reason", ""), discovered_at=d.get("discovered_at", ""),
            last_updated=d.get("last_updated", ""),
        )

    @property
    def needs_exploration(self) -> bool:
        return self.severity in (GapSeverity.HIGH, GapSeverity.CRITICAL) or self.uncertainty > 0.6


class KnowledgeGapTracker:
    """
    Discovers and persists knowledge gaps from SelfModel capability
    estimates, failure memory, and CauseGraph edge confidence.

    Parameters
    ----------
    gap_file:
        Path to ``knowledge_gaps.json``.
    low_capability_threshold:
        SelfModel capability score below this triggers a gap.
    low_confidence_edge_threshold:
        CauseGraph edge confidence below this flags an investigative gap.
    """

    def __init__(
        self,
        gap_file: Path,
        low_capability_threshold: float = 0.4,
        low_confidence_edge_threshold: float = 0.5,
    ) -> None:
        self._file = gap_file
        self._capability_threshold = low_capability_threshold
        self._edge_threshold = low_confidence_edge_threshold
        self._gaps: dict[str, KnowledgeGap] = {}
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
                g = KnowledgeGap.from_dict(item)
                self._gaps[g.domain] = g
        except Exception as exc:
            log.warning("KnowledgeGapTracker: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([g.to_dict() for g in self._gaps.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Manual gap registration
    # ------------------------------------------------------------------

    def register_gap(self, domain: str, severity: GapSeverity, uncertainty: float, evidence_count: int = 0, reason: str = "") -> KnowledgeGap:
        """Register or update one knowledge gap explicitly."""
        existing = self._gaps.get(domain)
        if existing:
            existing.severity = severity
            existing.uncertainty = max(0.0, min(1.0, uncertainty))
            existing.evidence_count = evidence_count
            existing.gap_reason = reason or existing.gap_reason
            existing.last_updated = datetime.now(timezone.utc).isoformat()
            self._save()
            return existing

        gap = KnowledgeGap(
            domain=domain, severity=severity, uncertainty=max(0.0, min(1.0, uncertainty)),
            evidence_count=evidence_count, gap_reason=reason,
        )
        self._gaps[domain] = gap
        self._save()
        return gap

    def resolve_gap(self, domain: str) -> Optional[KnowledgeGap]:
        """Remove a gap once it's been addressed by experimentation/observation."""
        gap = self._gaps.pop(domain, None)
        if gap:
            self._save()
        return gap

    # ------------------------------------------------------------------
    # Discovery from existing infrastructure
    # ------------------------------------------------------------------

    def discover_from_self_model(self, self_model_store) -> list[KnowledgeGap]:
        """Scan SelfModelStore capability scores — low scores signal knowledge gaps."""
        discovered: list[KnowledgeGap] = []
        capabilities: dict = self_model_store._model.capabilities
        for domain, score in capabilities.items():
            if score < self._capability_threshold:
                uncertainty = 1.0 - score
                severity = GapSeverity.CRITICAL if score < 0.2 else (GapSeverity.HIGH if score < 0.3 else GapSeverity.MEDIUM)
                gap = self.register_gap(
                    domain=domain, severity=severity, uncertainty=uncertainty,
                    reason=f"SelfModel capability score {score:.2f} below threshold {self._capability_threshold}",
                )
                discovered.append(gap)
        return discovered

    def discover_from_failure_memory(self, failure_memory) -> list[KnowledgeGap]:
        """Scan FailureMemory — recurring failures with no principle suggest a causal gap."""
        discovered: list[KnowledgeGap] = []
        records = failure_memory.most_common_failures(top_k=failure_memory.count)
        # Group by tool as a proxy for domain
        by_tool: dict[str, int] = {}
        for r in records:
            tool = r.tool or "unknown"
            by_tool[tool] = by_tool.get(tool, 0) + r.occurrences

        for tool, occurrences in by_tool.items():
            if occurrences >= 3:
                severity = GapSeverity.HIGH if occurrences >= 6 else GapSeverity.MEDIUM
                gap = self.register_gap(
                    domain=f"tool:{tool}", severity=severity, uncertainty=0.7,
                    evidence_count=occurrences,
                    reason=f"{occurrences} failures recorded for tool '{tool}' — cause not fully understood.",
                )
                discovered.append(gap)
        return discovered

    def discover_from_cause_graph(self, cause_graph) -> list[KnowledgeGap]:
        """Scan CauseGraph — low-confidence edges signal domains needing more evidence."""
        discovered: list[KnowledgeGap] = []
        for edge in cause_graph.all_edges():
            if edge.confidence < self._edge_threshold and edge.evidence_count < 3:
                domain = f"cause:{edge.trigger[:40]}"
                gap = self.register_gap(
                    domain=domain, severity=GapSeverity.MEDIUM,
                    uncertainty=1.0 - edge.confidence, evidence_count=edge.evidence_count,
                    reason=f"CauseGraph edge '{edge.trigger}->{edge.effect}' has low confidence ({edge.confidence:.2f}).",
                )
                discovered.append(gap)
        return discovered

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def gaps(self, min_severity: Optional[GapSeverity] = None) -> list[KnowledgeGap]:
        """All gaps, optionally filtered by minimum severity, highest uncertainty first."""
        _order = {GapSeverity.LOW: 0, GapSeverity.MEDIUM: 1, GapSeverity.HIGH: 2, GapSeverity.CRITICAL: 3}
        results = list(self._gaps.values())
        if min_severity is not None:
            results = [g for g in results if _order[g.severity] >= _order[min_severity]]
        return sorted(results, key=lambda g: -g.uncertainty)

    def needs_exploration(self) -> list[KnowledgeGap]:
        """Gaps that urgently need exploration (high severity or high uncertainty)."""
        return [g for g in self._gaps.values() if g.needs_exploration]

    def get(self, domain: str) -> Optional[KnowledgeGap]:
        return self._gaps.get(domain)

    @property
    def count(self) -> int:
        return len(self._gaps)
