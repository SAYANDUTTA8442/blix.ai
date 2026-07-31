"""
MemoryEdge — typed, weighted, timestamped relationship between two MemoryNodes.

Every edge has:
  - a source and target node_id
  - a typed relation (from EdgeRelation enum)
  - confidence and weight (both [0,1])
  - provenance (where the edge came from)
  - timestamps for creation and last update
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EdgeRelation(str, Enum):
    """All supported typed relationships in the HGSHM knowledge graph."""
    SUPPORTS      = "supports"       # A lends credence to B
    CONTRADICTS   = "contradicts"    # A conflicts with B
    CAUSES        = "causes"         # A causally produces B
    DEPENDS_ON    = "depends_on"     # A requires B to hold
    PART_OF       = "part_of"        # A is a component of B
    DERIVED_FROM  = "derived_from"   # A was inferred/synthesised from B
    SIMILAR_TO    = "similar_to"     # A and B are semantically close
    EXPLAINS      = "explains"       # A provides an account for B
    REFERENCES    = "references"     # A mentions or cites B
    PRECEDES      = "precedes"       # A temporally comes before B
    FOLLOWS       = "follows"        # A temporally comes after B
    BELONGS_TO    = "belongs_to"     # A is a member of concept/cluster B
    REQUIRES      = "requires"       # A needs B as a precondition
    RELATED_TO    = "related_to"     # generic weak association
    ENABLES       = "enables"        # A makes B possible
    BLOCKS        = "blocks"         # A prevents B
    SUMMARISES    = "summarises"     # A is a compressed form of B
    INSTANCE_OF   = "instance_of"    # A is an example of concept B
    EVOLVES_TO    = "evolves_to"     # A was superseded by B
    CO_OCCURS     = "co_occurs"      # A and B appeared together in context


# Relations that are semantically inverses
INVERSE_RELATIONS: dict[EdgeRelation, EdgeRelation] = {
    EdgeRelation.SUPPORTS:    EdgeRelation.SUPPORTS,    # symmetric
    EdgeRelation.CONTRADICTS: EdgeRelation.CONTRADICTS, # symmetric
    EdgeRelation.SIMILAR_TO:  EdgeRelation.SIMILAR_TO,  # symmetric
    EdgeRelation.RELATED_TO:  EdgeRelation.RELATED_TO,  # symmetric
    EdgeRelation.CO_OCCURS:   EdgeRelation.CO_OCCURS,   # symmetric
    EdgeRelation.PRECEDES:    EdgeRelation.FOLLOWS,
    EdgeRelation.FOLLOWS:     EdgeRelation.PRECEDES,
    EdgeRelation.CAUSES:      EdgeRelation.DEPENDS_ON,
    EdgeRelation.PART_OF:     EdgeRelation.BELONGS_TO,
    EdgeRelation.BELONGS_TO:  EdgeRelation.PART_OF,
    EdgeRelation.ENABLES:     EdgeRelation.REQUIRES,
    EdgeRelation.REQUIRES:    EdgeRelation.ENABLES,
}

# Relations whose direction implies constraint propagation
PROPAGATING_RELATIONS: frozenset[EdgeRelation] = frozenset({
    EdgeRelation.SUPPORTS,
    EdgeRelation.CONTRADICTS,
    EdgeRelation.CAUSES,
    EdgeRelation.DEPENDS_ON,
    EdgeRelation.DERIVED_FROM,
    EdgeRelation.ENABLES,
    EdgeRelation.BLOCKS,
})


@dataclass
class MemoryEdge:
    """
    A typed, weighted, timestamped edge in the HGSHM knowledge graph.

    Parameters
    ----------
    edge_id:
        Globally unique identifier.
    source_id:
        node_id of the source MemoryNode.
    target_id:
        node_id of the target MemoryNode.
    relation:
        Typed semantic relationship.
    confidence:
        How certain we are that this relationship holds [0, 1].
    weight:
        Relative strength of the relationship [0, 1].
    provenance:
        Free-form string describing where this edge came from.
    metadata:
        Arbitrary structured data.
    evidence_count:
        Number of independent observations supporting this edge.
    created_at / updated_at:
        UTC ISO timestamps.
    """
    edge_id:       str          = field(default_factory=lambda: str(uuid.uuid4()))
    source_id:     str          = ""
    target_id:     str          = ""
    relation:      EdgeRelation = EdgeRelation.RELATED_TO
    confidence:    float        = 0.7
    weight:        float        = 0.5
    provenance:    str          = "system"
    metadata:      dict[str, Any] = field(default_factory=dict)
    evidence_count: int         = 1
    created_at:    str          = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:    str          = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ----------------------------------------------------------------
    # Mutation helpers
    # ----------------------------------------------------------------

    def reinforce(self, confidence_delta: float = 0.05) -> None:
        """Strengthen this edge with new supporting evidence."""
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence + confidence_delta)
        self.weight = min(1.0, self.weight + confidence_delta * 0.5)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def weaken(self, confidence_delta: float = 0.05) -> None:
        """Weaken this edge (contradicting evidence)."""
        self.confidence = max(0.0, self.confidence - confidence_delta)
        self.weight = max(0.0, self.weight - confidence_delta * 0.5)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_symmetric(self) -> bool:
        inv = INVERSE_RELATIONS.get(self.relation)
        return inv == self.relation

    @property
    def inverse_relation(self) -> EdgeRelation:
        return INVERSE_RELATIONS.get(self.relation, EdgeRelation.RELATED_TO)

    # ----------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id":       self.edge_id,
            "source_id":     self.source_id,
            "target_id":     self.target_id,
            "relation":      self.relation.value,
            "confidence":    self.confidence,
            "weight":        self.weight,
            "provenance":    self.provenance,
            "metadata":      self.metadata,
            "evidence_count": self.evidence_count,
            "created_at":    self.created_at,
            "updated_at":    self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEdge":
        return cls(
            edge_id=       d["edge_id"],
            source_id=     d["source_id"],
            target_id=     d["target_id"],
            relation=      EdgeRelation(d["relation"]),
            confidence=    d["confidence"],
            weight=        d["weight"],
            provenance=    d.get("provenance", "system"),
            metadata=      d.get("metadata", {}),
            evidence_count= d.get("evidence_count", 1),
            created_at=    d["created_at"],
            updated_at=    d["updated_at"],
        )

    def __repr__(self) -> str:
        return (f"MemoryEdge({self.source_id[:8]}…—[{self.relation.value}]→"
                f"{self.target_id[:8]}…, conf={self.confidence:.2f})")
