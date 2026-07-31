"""
MemoryContext — the structured cognitive context produced by the ContextBuilder.

This replaces concatenated string prompts with a rich, typed object
that downstream subsystems (Planner, WorldModel, Reflection, etc.) can
query without re-parsing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from memory.hybrid.models.memory_node import MemoryNode
from memory.hybrid.models.memory_edge import MemoryEdge


@dataclass
class RetrievedMemory:
    """A single memory node returned by retrieval, annotated with its scores."""
    node:               MemoryNode
    semantic_score:     float = 0.0
    vector_score:       float = 0.0
    graph_score:        float = 0.0
    temporal_score:     float = 0.0
    importance_score:   float = 0.0
    final_score:        float = 0.0
    retrieval_path:     list[str] = field(default_factory=list)  # graph path taken

    def to_dict(self) -> dict[str, Any]:
        return {
            "node":             self.node.to_dict(),
            "semantic_score":   round(self.semantic_score, 6),
            "vector_score":     round(self.vector_score, 6),
            "graph_score":      round(self.graph_score, 6),
            "temporal_score":   round(self.temporal_score, 6),
            "importance_score": round(self.importance_score, 6),
            "final_score":      round(self.final_score, 6),
            "retrieval_path":   self.retrieval_path,
        }


@dataclass
class MemoryContext:
    """
    Structured cognitive context assembled by the ContextBuilder.

    Parameters
    ----------
    query:
        The original query that triggered context assembly.
    primary_memories:
        Top-ranked directly relevant memories.
    supporting_memories:
        Graph-expanded supporting context.
    temporal_memories:
        Time-relevant memories (recent, scheduled, historical patterns).
    concept_nodes:
        High-level concept nodes related to the query.
    principle_nodes:
        Applicable principles extracted from memory.
    belief_nodes:
        Beliefs relevant to the query (confidence-filtered).
    contradictions:
        Pairs of nodes with CONTRADICTS edges (to surface conflicts).
    causal_chains:
        Ordered node sequences connected by CAUSES/ENABLES/BLOCKS edges.
    knowledge_gaps:
        GAP-type nodes surfaced during retrieval.
    graph_neighbourhood:
        Edges connecting returned nodes (for graph visualisation / reasoning).
    total_nodes_searched:
        Diagnostic: how many nodes were evaluated.
    retrieval_latency_ms:
        Diagnostic: end-to-end retrieval time.
    metadata:
        Arbitrary context-level metadata.
    """
    query:                 str                   = ""
    primary_memories:      list[RetrievedMemory] = field(default_factory=list)
    supporting_memories:   list[RetrievedMemory] = field(default_factory=list)
    temporal_memories:     list[RetrievedMemory] = field(default_factory=list)
    concept_nodes:         list[MemoryNode]      = field(default_factory=list)
    principle_nodes:       list[MemoryNode]      = field(default_factory=list)
    belief_nodes:          list[MemoryNode]      = field(default_factory=list)
    contradictions:        list[tuple[MemoryNode, MemoryNode]] = field(default_factory=list)
    causal_chains:         list[list[MemoryNode]] = field(default_factory=list)
    knowledge_gaps:        list[MemoryNode]      = field(default_factory=list)
    graph_neighbourhood:   list[MemoryEdge]      = field(default_factory=list)
    total_nodes_searched:  int                   = 0
    retrieval_latency_ms:  float                 = 0.0
    metadata:              dict[str, Any]         = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Convenience accessors
    # ----------------------------------------------------------------

    @property
    def all_memories(self) -> list[RetrievedMemory]:
        """All retrieved memories ranked by final_score descending."""
        combined = self.primary_memories + self.supporting_memories + self.temporal_memories
        return sorted(combined, key=lambda r: r.final_score, reverse=True)

    @property
    def top_memory(self) -> MemoryNode | None:
        if self.primary_memories:
            return self.primary_memories[0].node
        return None

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def has_causal_chains(self) -> bool:
        return len(self.causal_chains) > 0

    @property
    def total_memories(self) -> int:
        return (len(self.primary_memories) +
                len(self.supporting_memories) +
                len(self.temporal_memories))

    def get_text_summary(self, max_nodes: int = 10) -> str:
        """Return a readable text summary of the most relevant memories."""
        lines = [f"Context for: {self.query!r}"]
        for i, rm in enumerate(self.all_memories[:max_nodes]):
            lines.append(f"  [{i+1}] ({rm.final_score:.3f}) {rm.node.text[:120]}")
        if self.principle_nodes:
            lines.append("Principles:")
            for p in self.principle_nodes[:3]:
                lines.append(f"  • {p.text[:100]}")
        if self.has_contradictions:
            lines.append(f"Conflicts: {len(self.contradictions)} contradiction(s) detected")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":                self.query,
            "primary_memories":     [r.to_dict() for r in self.primary_memories],
            "supporting_memories":  [r.to_dict() for r in self.supporting_memories],
            "temporal_memories":    [r.to_dict() for r in self.temporal_memories],
            "concept_nodes":        [n.to_dict() for n in self.concept_nodes],
            "principle_nodes":      [n.to_dict() for n in self.principle_nodes],
            "belief_nodes":         [n.to_dict() for n in self.belief_nodes],
            "knowledge_gaps":       [n.to_dict() for n in self.knowledge_gaps],
            "causal_chains":        [[n.to_dict() for n in chain] for chain in self.causal_chains],
            "contradictions":       [[a.to_dict(), b.to_dict()] for a, b in self.contradictions],
            "total_nodes_searched": self.total_nodes_searched,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "total_memories":       self.total_memories,
            "metadata":             self.metadata,
        }
