"""
Memory Graph — Blix v0.3  (Feature 3)

Implements an entity-relationship graph over memory.

Entity types:   Person, Project, Skill, Goal, Topic, Organization
Relation types: works_on, studies_at, interested_in, goal_is, uses,
                collaborates_with

Storage is a plain JSON file.  The API is designed to slot behind a
NetworkX or neo4j adapter in a future version without changing callers.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntityKind(str, Enum):
    PERSON = "person"
    PROJECT = "project"
    SKILL = "skill"
    GOAL = "goal"
    TOPIC = "topic"
    ORGANIZATION = "organization"


class RelationKind(str, Enum):
    WORKS_ON = "works_on"
    STUDIES_AT = "studies_at"
    INTERESTED_IN = "interested_in"
    GOAL_IS = "goal_is"
    USES = "uses"
    COLLABORATES_WITH = "collaborates_with"


# ---------------------------------------------------------------------------
# Node / Edge models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A named entity in the memory graph."""

    id: str = Field(..., description="Lowercase slug, e.g. 'sayan', 'blix'.")
    kind: EntityKind
    label: str = Field(..., description="Display name.")
    aliases: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class GraphEdge(BaseModel):
    """A directed relationship between two entities."""

    from_id: str
    relation: RelationKind
    to_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_memory_ids: list[int] = Field(default_factory=list)

    model_config = {"use_enum_values": True}

    @property
    def key(self) -> str:
        return f"{self.from_id}:{self.relation}:{self.to_id}"


# ---------------------------------------------------------------------------
# Graph storage
# ---------------------------------------------------------------------------


class MemoryGraph:
    """
    In-memory entity-relationship graph backed by a JSON file.

    Designed so a future version can swap the backend to networkx or
    neo4j by implementing the same interface.

    Parameters
    ----------
    graph_file:
        Path to ``graph.json`` persistence file.
    """

    def __init__(self, graph_file: Path) -> None:
        self._file = graph_file
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}  # key → edge
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for raw in data.get("nodes", []):
                n = GraphNode.model_validate(raw)
                self._nodes[n.id] = n
            for raw in data.get("edges", []):
                e = GraphEdge.model_validate(raw)
                self._edges[e.key] = e
            log.info(
                "MemoryGraph loaded: %d nodes, %d edges.",
                len(self._nodes),
                len(self._edges),
            )
        except Exception as exc:
            log.warning("MemoryGraph load failed (%s); starting empty.", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges.values()],
        }
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        log.debug("MemoryGraph saved: %d nodes, %d edges.", len(self._nodes), len(self._edges))

    # ------------------------------------------------------------------
    # Node API
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> GraphNode:
        """
        Add or update a node.  If a node with the same id exists, the
        new aliases and metadata are merged (entity deduplication).
        """
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
            log.debug("Graph: added node %r (%s)", node.id, node.kind)
        else:
            # Merge aliases
            merged_aliases = list(set(existing.aliases + node.aliases))
            merged_meta = {**existing.metadata, **node.metadata}
            self._nodes[node.id] = existing.model_copy(
                update={"aliases": merged_aliases, "metadata": merged_meta}
            )
            log.debug("Graph: merged node %r", node.id)
        self._save()
        return self._nodes[node.id]

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def find_node_by_label(self, label: str) -> Optional[GraphNode]:
        """Case-insensitive label + alias search (entity deduplication helper)."""
        lower = label.lower()
        for node in self._nodes.values():
            if node.label.lower() == lower or lower in [a.lower() for a in node.aliases]:
                return node
        return None

    def list_nodes(self, kind: Optional[EntityKind] = None) -> list[GraphNode]:
        nodes = list(self._nodes.values())
        if kind is not None:
            nodes = [n for n in nodes if n.kind == kind.value]
        return nodes

    # ------------------------------------------------------------------
    # Edge API
    # ------------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """
        Add or update an edge.  If the same (from, relation, to) triple
        already exists, the confidence and source_memory_ids are merged.
        """
        existing = self._edges.get(edge.key)
        if existing is None:
            self._edges[edge.key] = edge
            log.debug("Graph: added edge %s", edge.key)
        else:
            merged_ids = list(set(existing.source_memory_ids + edge.source_memory_ids))
            new_confidence = max(existing.confidence, edge.confidence)
            self._edges[edge.key] = existing.model_copy(
                update={"confidence": new_confidence, "source_memory_ids": merged_ids}
            )
        self._save()
        return self._edges[edge.key]

    def get_edges(
        self,
        from_id: Optional[str] = None,
        relation: Optional[RelationKind] = None,
        to_id: Optional[str] = None,
    ) -> list[GraphEdge]:
        """Filter edges by any combination of from_id, relation, to_id."""
        result = list(self._edges.values())
        if from_id is not None:
            result = [e for e in result if e.from_id == from_id]
        if relation is not None:
            result = [e for e in result if e.relation == relation.value]
        if to_id is not None:
            result = [e for e in result if e.to_id == to_id]
        return result

    def neighbours(self, node_id: str) -> list[tuple[RelationKind, GraphNode]]:
        """
        Return all (relation, target_node) pairs reachable from *node_id*.
        """
        out: list[tuple[RelationKind, GraphNode]] = []
        for edge in self.get_edges(from_id=node_id):
            target = self._nodes.get(edge.to_id)
            if target:
                out.append((RelationKind(edge.relation), target))
        return out

    # ------------------------------------------------------------------
    # Graph update pipeline
    # ------------------------------------------------------------------

    def upsert_relation(
        self,
        from_label: str,
        from_kind: EntityKind,
        relation: RelationKind,
        to_label: str,
        to_kind: EntityKind,
        confidence: float = 1.0,
        source_memory_id: Optional[int] = None,
    ) -> None:
        """
        High-level helper: ensure both nodes exist, then add the edge.

        Finds existing nodes by label (deduplication) before creating.
        """
        from_node = self.find_node_by_label(from_label) or self.add_node(
            GraphNode(
                id=_slug(from_label),
                kind=from_kind,
                label=from_label,
            )
        )
        to_node = self.find_node_by_label(to_label) or self.add_node(
            GraphNode(
                id=_slug(to_label),
                kind=to_kind,
                label=to_label,
            )
        )
        edge = GraphEdge(
            from_id=from_node.id,
            relation=relation,
            to_id=to_node.id,
            confidence=confidence,
            source_memory_ids=[source_memory_id] if source_memory_id else [],
        )
        self.add_edge(edge)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(label: str) -> str:
    """Convert a label to a lowercase slug id."""
    return label.lower().strip().replace(" ", "_").replace("-", "_")
