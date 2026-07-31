"""
MemoryNode — the atomic unit of HGSHM.

Every piece of knowledge in Blix v0.3.15+ is a MemoryNode.
Old constructs (Belief, CauseEdge, Principle, etc.) are projections
onto the same underlying node/edge substrate.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Ontological type of a memory node."""
    RAW        = "raw"          # unprocessed observation
    EPISODE    = "episode"      # a single event / interaction turn
    BELIEF     = "belief"       # a held proposition with confidence
    FACT       = "fact"         # high-confidence, stable assertion
    PRINCIPLE  = "principle"    # generalised rule derived from experience
    CONCEPT    = "concept"      # abstract cluster / named idea
    CAUSE      = "cause"        # causal observation record
    HYPOTHESIS = "hypothesis"   # tentative claim under investigation
    PLAN       = "plan"         # intended sequence of actions
    GOAL       = "goal"         # desired future state
    SUMMARY    = "summary"      # compressed representation of lower nodes
    WORLD_MODEL = "world_model" # latent state snapshot
    GAP        = "gap"          # identified knowledge gap
    EXPERIMENT = "experiment"   # planned or completed experiment
    REFLECTION = "reflection"   # metacognitive note


class HierarchyLevel(int, Enum):
    """Abstraction level in the memory hierarchy (lower = more concrete)."""
    RAW         = 0
    EPISODE     = 1
    CONVERSATION = 2
    SESSION     = 3
    DAILY       = 4
    WEEKLY      = 5
    MONTHLY     = 6
    PROJECT     = 7
    CONCEPT     = 8
    PRINCIPLE   = 9
    KNOWLEDGE   = 10
    WORLD_MODEL = 11


class EpistemicStatus(str, Enum):
    OBSERVED      = "observed"
    DERIVED       = "derived"
    PREDICTED     = "predicted"
    COUNTERFACTUAL = "counterfactual"
    HYPOTHESIS    = "hypothesis"
    PRINCIPLE     = "principle"
    UNKNOWN       = "unknown"


@dataclass
class MemoryNode:
    """
    The fundamental unit of HGSHM.

    Parameters
    ----------
    node_id:
        Globally unique identifier (UUID4 by default).
    text:
        The natural-language content of this memory.
    memory_type:
        Ontological classification.
    hierarchy_level:
        Abstraction level within the memory hierarchy.
    confidence:
        Epistemic confidence [0, 1].
    importance:
        Dynamic importance score [0, 1]; updated by usage patterns.
    embedding_id:
        Foreign key into VectorStore (set after embedding).
    concept_id:
        Optional reference to a parent ConceptNode.
    metadata:
        Arbitrary structured data attached to this node.
    source:
        Origin identifier (agent_id, tool name, subsystem, etc.).
    epistemic_status:
        Epistemological classification of how this was derived.
    created_at:
        UTC timestamp of creation.
    updated_at:
        UTC timestamp of last modification.
    last_accessed_at:
        UTC timestamp of last retrieval.
    valid_from / valid_until:
        Temporal validity window (None = open-ended).
    access_count:
        Number of times this node has been retrieved.
    version:
        Monotonically increasing version counter (for history tracking).
    tags:
        Free-form string tags for filtering.
    """
    node_id:         str            = field(default_factory=lambda: str(uuid.uuid4()))
    text:            str            = ""
    memory_type:     MemoryType     = MemoryType.RAW
    hierarchy_level: HierarchyLevel = HierarchyLevel.RAW
    confidence:      float          = 0.7
    importance:      float          = 0.5
    embedding_id:    str | None     = None
    concept_id:      str | None     = None
    source:          str            = "system"
    epistemic_status: EpistemicStatus = EpistemicStatus.OBSERVED
    metadata:        dict[str, Any] = field(default_factory=dict)
    tags:            list[str]      = field(default_factory=list)
    created_at:      str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:      str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed_at: str | None    = None
    valid_from:      str | None     = None
    valid_until:     str | None     = None
    access_count:    int            = 0
    version:         int            = 1

    # ----------------------------------------------------------------
    # Derived properties
    # ----------------------------------------------------------------

    @property
    def created_dt(self) -> datetime:
        return datetime.fromisoformat(self.created_at)

    @property
    def updated_dt(self) -> datetime:
        return datetime.fromisoformat(self.updated_at)

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_dt).total_seconds()

    @property
    def recency_score(self) -> float:
        """Exponential decay score: 1.0 = just created, → 0 as time passes.
        Half-life ≈ 7 days.
        """
        import math
        half_life = 7 * 24 * 3600  # 7 days in seconds
        return math.exp(-0.693 * self.age_seconds / half_life)

    # ----------------------------------------------------------------
    # Mutation helpers
    # ----------------------------------------------------------------

    def touch(self) -> None:
        """Record an access."""
        self.last_accessed_at = datetime.now(timezone.utc).isoformat()
        self.access_count += 1

    def update_text(self, new_text: str) -> None:
        self.text = new_text
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.version += 1

    def update_confidence(self, delta: float) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.version += 1

    def update_importance(self, new_importance: float) -> None:
        self.importance = max(0.0, min(1.0, new_importance))
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ----------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id":          self.node_id,
            "text":             self.text,
            "memory_type":      self.memory_type.value,
            "hierarchy_level":  self.hierarchy_level.value,
            "confidence":       self.confidence,
            "importance":       self.importance,
            "embedding_id":     self.embedding_id,
            "concept_id":       self.concept_id,
            "source":           self.source,
            "epistemic_status": self.epistemic_status.value,
            "metadata":         self.metadata,
            "tags":             self.tags,
            "created_at":       self.created_at,
            "updated_at":       self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "valid_from":       self.valid_from,
            "valid_until":      self.valid_until,
            "access_count":     self.access_count,
            "version":          self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryNode":
        return cls(
            node_id=         d["node_id"],
            text=            d["text"],
            memory_type=     MemoryType(d["memory_type"]),
            hierarchy_level= HierarchyLevel(d["hierarchy_level"]),
            confidence=      d["confidence"],
            importance=      d["importance"],
            embedding_id=    d.get("embedding_id"),
            concept_id=      d.get("concept_id"),
            source=          d.get("source", "system"),
            epistemic_status= EpistemicStatus(d.get("epistemic_status", "observed")),
            metadata=        d.get("metadata", {}),
            tags=            d.get("tags", []),
            created_at=      d["created_at"],
            updated_at=      d["updated_at"],
            last_accessed_at= d.get("last_accessed_at"),
            valid_from=      d.get("valid_from"),
            valid_until=     d.get("valid_until"),
            access_count=    d.get("access_count", 0),
            version=         d.get("version", 1),
        )

    def __repr__(self) -> str:
        return (f"MemoryNode(id={self.node_id[:8]}…, type={self.memory_type.value}, "
                f"conf={self.confidence:.2f}, imp={self.importance:.2f}, "
                f"text={self.text[:40]!r})")
