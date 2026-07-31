"""
MemoryCluster — a named group of semantically related MemoryNodes.

Clusters are the seeds of Concepts in the hierarchy.
When a cluster is stable and large enough, it is promoted to a Concept node.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryCluster:
    """
    A named semantic cluster of MemoryNodes.

    Parameters
    ----------
    cluster_id:
        Unique identifier.
    name:
        Human-readable name (auto-generated or user-assigned).
    node_ids:
        Set of MemoryNode IDs in this cluster.
    centroid_embedding_id:
        Embedding ID of the cluster centroid in VectorStore.
    concept_node_id:
        If this cluster has been promoted to a Concept, its node_id.
    coherence:
        Average pairwise similarity within the cluster [0,1].
    size:
        Cached count of member nodes.
    tags:
        Free-form tags.
    metadata:
        Arbitrary structured data.
    created_at / updated_at:
        UTC ISO timestamps.
    """
    cluster_id:           str            = field(default_factory=lambda: str(uuid.uuid4()))
    name:                 str            = ""
    node_ids:             list[str]      = field(default_factory=list)
    centroid_embedding_id: str | None    = None
    concept_node_id:      str | None     = None
    coherence:            float          = 0.0
    tags:                 list[str]      = field(default_factory=list)
    metadata:             dict[str, Any] = field(default_factory=dict)
    created_at:           str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:           str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def size(self) -> int:
        return len(self.node_ids)

    def add_node(self, node_id: str) -> None:
        if node_id not in self.node_ids:
            self.node_ids.append(node_id)
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def remove_node(self, node_id: str) -> None:
        if node_id in self.node_ids:
            self.node_ids.remove(node_id)
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def is_promoted(self) -> bool:
        return self.concept_node_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id":            self.cluster_id,
            "name":                  self.name,
            "node_ids":              self.node_ids,
            "centroid_embedding_id": self.centroid_embedding_id,
            "concept_node_id":       self.concept_node_id,
            "coherence":             self.coherence,
            "tags":                  self.tags,
            "metadata":              self.metadata,
            "created_at":            self.created_at,
            "updated_at":            self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryCluster":
        return cls(
            cluster_id=            d["cluster_id"],
            name=                  d["name"],
            node_ids=              d.get("node_ids", []),
            centroid_embedding_id= d.get("centroid_embedding_id"),
            concept_node_id=       d.get("concept_node_id"),
            coherence=             d.get("coherence", 0.0),
            tags=                  d.get("tags", []),
            metadata=              d.get("metadata", {}),
            created_at=            d["created_at"],
            updated_at=            d["updated_at"],
        )

    def __repr__(self) -> str:
        return f"MemoryCluster(id={self.cluster_id[:8]}…, name={self.name!r}, size={self.size})"
