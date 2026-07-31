"""
Temporal Graph — Blix v0.3.7  (New module 7)

The v0.3.1 ``MemoryGraph`` is static: an edge either exists or doesn't.
``TemporalGraph`` adds a time dimension to graph relations, so the
knowledge graph can represent evolution rather than only a snapshot:

    Sayan —uses→ Python      [valid_from=2024-01, valid_to=2025-01]
    Sayan —uses→ PyTorch     [valid_from=2025-01, valid_to=2026-01]
    Sayan —uses→ Rust        [valid_from=2026-01, valid_to=None]

This module wraps/extends ``core.memory_graph.MemoryGraph`` rather than
replacing it: a ``TemporalGraph`` holds its own list of
``TemporalEdge`` records (entity labels + relation + time window) and
can answer "what was true at time T" or "show me the full evolution",
while delegating plain current-state graph queries to the underlying
``MemoryGraph``/``GraphReasoner`` (v0.3.1/v0.3.4) when only the
currently-active edges matter.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Temporal edge
# ---------------------------------------------------------------------------


@dataclass
class TemporalEdge:
    """
    One relation between two entities, valid over a time window.

    Fields
    ------
    from_label / relation / to_label:
        Same semantics as ``core.memory_graph.GraphEdge`` — but here
        we store labels directly (not internal node ids) since
        ``TemporalGraph`` is a thin, self-contained temporal layer.
    valid_from / valid_to:
        ISO timestamps. ``valid_to=None`` means still active.
    confidence:
        0–1.
    """

    from_label: str
    relation: str
    to_label: str
    valid_from: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    valid_to: Optional[str] = None
    confidence: float = 0.7
    edge_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_memory_ids: list[int] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.valid_to is None

    def covers(self, timestamp: str) -> bool:
        if timestamp < self.valid_from:
            return False
        if self.valid_to is not None and timestamp >= self.valid_to:
            return False
        return True

    def close(self, valid_to: Optional[str] = None) -> None:
        self.valid_to = valid_to or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "from_label": self.from_label,
            "relation": self.relation,
            "to_label": self.to_label,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "is_active": self.is_active,
            "confidence": round(self.confidence, 3),
            "source_memory_ids": self.source_memory_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalEdge":
        return cls(
            from_label=d["from_label"], relation=d["relation"], to_label=d["to_label"],
            valid_from=d.get("valid_from", ""), valid_to=d.get("valid_to"),
            confidence=d.get("confidence", 0.7),
            edge_id=d.get("edge_id", uuid.uuid4().hex[:8]),
            source_memory_ids=d.get("source_memory_ids", []),
        )


# ---------------------------------------------------------------------------
# Temporal Graph
# ---------------------------------------------------------------------------


class TemporalGraph:
    """
    Stores and queries time-scoped graph relations.

    Parameters
    ----------
    temporal_graph_file:
        Path to ``temporal_graph.json``.
    """

    def __init__(self, temporal_graph_file: Path) -> None:
        self._file = temporal_graph_file
        self._edges: list[TemporalEdge] = []
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
            self._edges = [TemporalEdge.from_dict(e) for e in raw]
            log.info("TemporalGraph: loaded %d edge(s).", len(self._edges))
        except Exception as exc:
            log.warning("TemporalGraph: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._edges], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _norm(self, label: str) -> str:
        return label.strip().lower()

    def add_relation(
        self,
        from_label: str,
        relation: str,
        to_label: str,
        confidence: float = 0.7,
        source_memory_id: Optional[int] = None,
        timestamp: Optional[str] = None,
        close_previous: bool = True,
    ) -> TemporalEdge:
        """
        Add a new time-scoped relation. By default, closes any previously
        active edge with the SAME (from_label, relation) but a DIFFERENT
        to_label — i.e. this models an evolving single-valued relation
        (e.g. "current tech stack") rather than an accumulating
        multi-valued one. Pass ``close_previous=False`` for relations
        that are naturally multi-valued (e.g. "knows" — many people).
        """
        from_n, rel_n, to_n = self._norm(from_label), relation.strip().lower(), self._norm(to_label)

        if close_previous:
            for edge in self._edges:
                if (
                    edge.is_active
                    and self._norm(edge.from_label) == from_n
                    and edge.relation == rel_n
                    and self._norm(edge.to_label) != to_n
                ):
                    edge.close(valid_to=timestamp)

        new_edge = TemporalEdge(
            from_label=from_label, relation=rel_n, to_label=to_label,
            valid_from=timestamp or datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            source_memory_ids=[source_memory_id] if source_memory_id is not None else [],
        )
        self._edges.append(new_edge)
        self._save()
        log.info("TemporalGraph: %s --%s--> %s", from_label, rel_n, to_label)
        return new_edge

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def current_relations(self, from_label: str, relation: Optional[str] = None) -> list[TemporalEdge]:
        """All currently-active edges from an entity, optionally filtered by relation."""
        from_n = self._norm(from_label)
        matches = [e for e in self._edges if self._norm(e.from_label) == from_n and e.is_active]
        if relation:
            rel_n = relation.strip().lower()
            matches = [e for e in matches if e.relation == rel_n]
        return matches

    def relations_at_time(
        self, from_label: str, timestamp: str, relation: Optional[str] = None
    ) -> list[TemporalEdge]:
        """All edges from an entity that were active at a given point in time."""
        from_n = self._norm(from_label)
        matches = [e for e in self._edges if self._norm(e.from_label) == from_n and e.covers(timestamp)]
        if relation:
            rel_n = relation.strip().lower()
            matches = [e for e in matches if e.relation == rel_n]
        return matches

    def evolution(self, from_label: str, relation: str) -> list[TemporalEdge]:
        """Full chronological evolution of a (from_label, relation) pair, oldest first."""
        from_n, rel_n = self._norm(from_label), relation.strip().lower()
        matches = [
            e for e in self._edges
            if self._norm(e.from_label) == from_n and e.relation == rel_n
        ]
        return sorted(matches, key=lambda e: e.valid_from)

    def changes_since(self, since_timestamp: str, from_label: Optional[str] = None) -> list[TemporalEdge]:
        matches = [e for e in self._edges if e.valid_from >= since_timestamp]
        if from_label:
            from_n = self._norm(from_label)
            matches = [e for e in matches if self._norm(e.from_label) == from_n]
        return sorted(matches, key=lambda e: e.valid_from)

    def all_for_entity(self, from_label: str) -> list[TemporalEdge]:
        from_n = self._norm(from_label)
        return [e for e in self._edges if self._norm(e.from_label) == from_n]

    @property
    def count(self) -> int:
        return len(self._edges)

    @property
    def active_count(self) -> int:
        return sum(1 for e in self._edges if e.is_active)
