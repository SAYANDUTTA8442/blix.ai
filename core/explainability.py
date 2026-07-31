"""
Explainability Layer — Blix v0.3.4  (Feature 6)

For every answer Blix produces, the explainability layer can annotate it
with a structured evidence chain:

    Answer: "ChromaDB, Transformers, FastAPI"

    Evidence:
    - Memory #14: "I integrated ChromaDB as the embedding store."
    - Memory #42: "Switched to Transformers-based embeddings."
    - Graph path: Blix →[uses]→ ChromaDB (confidence=0.95)
    - Canonical Fact #3: "Blix uses FastAPI for its API layer." (conf=0.88)

    Reasoning path:
      Blix →[uses]→ ChromaDB
      Blix →[uses]→ Transformers
      Blix →[uses]→ FastAPI

This is the publishable-research tier of the Blix architecture — it makes
the system's reasoning transparent and auditable.

``ExplainedResponse`` wraps any string response with attached evidence.
``ExplainabilityEngine`` builds the evidence chain from multiple sources.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.cognitive_query_engine import ReasoningTrace


# ---------------------------------------------------------------------------
# Evidence items
# ---------------------------------------------------------------------------


@dataclass
class MemoryEvidence:
    """A supporting MemoryEntry."""

    memory_id: int
    excerpt: str              # short excerpt of the memory output
    relevance_score: float    # 0–1 retrieval/scorer score
    topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": "memory",
            "memory_id": self.memory_id,
            "excerpt": self.excerpt,
            "relevance_score": round(self.relevance_score, 3),
            "topics": self.topics,
        }

    def __str__(self) -> str:
        return f"Memory #{self.memory_id}: \"{self.excerpt[:80]}\" (rel={self.relevance_score:.2f})"


@dataclass
class FactEvidence:
    """A supporting canonical fact."""

    fact_id: str
    fact: str
    confidence: float
    evidence_count: int

    def to_dict(self) -> dict:
        return {
            "type": "canonical_fact",
            "fact_id": self.fact_id,
            "fact": self.fact,
            "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
        }

    def __str__(self) -> str:
        return f"Fact {self.fact_id}: \"{self.fact}\" (conf={self.confidence:.2f}, n={self.evidence_count})"


@dataclass
class GraphEvidence:
    """A supporting graph path or edge."""

    description: str          # human-readable, e.g. "Blix →[uses]→ FastAPI"
    path_nodes: list[str] = field(default_factory=list)
    path_relations: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "type": "graph_path",
            "description": self.description,
            "path_nodes": self.path_nodes,
            "path_relations": self.path_relations,
            "confidence": round(self.confidence, 3),
        }

    def __str__(self) -> str:
        return f"Graph path: {self.description} (conf={self.confidence:.2f})"


@dataclass
class InsightEvidence:
    """A supporting reflection insight."""

    insight: str
    confidence: float
    scope: str = ""

    def to_dict(self) -> dict:
        return {"type": "insight", "insight": self.insight,
                "confidence": round(self.confidence, 3), "scope": self.scope}

    def __str__(self) -> str:
        return f"Insight ({self.scope}): \"{self.insight}\" (conf={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Explained response
# ---------------------------------------------------------------------------


@dataclass
class ExplainedResponse:
    """
    A Blix response annotated with a full evidence chain.

    Attributes
    ----------
    answer:
        The primary natural-language answer.
    memory_evidence:
        Supporting memory entries.
    fact_evidence:
        Supporting canonical facts.
    graph_evidence:
        Supporting graph paths/edges.
    insight_evidence:
        Supporting reflection insights.
    reasoning_trace:
        Optional graph reasoning trace (from CognitiveQueryEngine).
    overall_confidence:
        Weighted average confidence across all evidence sources.
    """

    answer: str
    memory_evidence: list[MemoryEvidence] = field(default_factory=list)
    fact_evidence: list[FactEvidence] = field(default_factory=list)
    graph_evidence: list[GraphEvidence] = field(default_factory=list)
    insight_evidence: list[InsightEvidence] = field(default_factory=list)
    reasoning_trace: Optional[ReasoningTrace] = None

    @property
    def overall_confidence(self) -> float:
        """Weighted average confidence across all evidence."""
        scores: list[tuple[float, float]] = []   # (confidence, weight)
        for me in self.memory_evidence:
            scores.append((me.relevance_score, 1.0))
        for fe in self.fact_evidence:
            scores.append((fe.confidence, 1.5))   # facts weighted higher
        for ge in self.graph_evidence:
            scores.append((ge.confidence, 1.2))
        for ie in self.insight_evidence:
            scores.append((ie.confidence, 0.8))
        if not scores:
            return 0.0
        total_weight = sum(w for _, w in scores)
        return round(sum(c * w for c, w in scores) / total_weight, 3)

    @property
    def total_evidence_count(self) -> int:
        return (len(self.memory_evidence) + len(self.fact_evidence)
                + len(self.graph_evidence) + len(self.insight_evidence))

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "overall_confidence": self.overall_confidence,
            "total_evidence": self.total_evidence_count,
            "memory_evidence": [m.to_dict() for m in self.memory_evidence],
            "fact_evidence": [f.to_dict() for f in self.fact_evidence],
            "graph_evidence": [g.to_dict() for g in self.graph_evidence],
            "insight_evidence": [i.to_dict() for i in self.insight_evidence],
            "reasoning_trace": self.reasoning_trace.to_dict() if self.reasoning_trace else None,
        }

    def explain_str(self) -> str:
        """Return a human-readable explainability report."""
        lines = [
            f"Answer: {self.answer}",
            f"\nOverall confidence: {self.overall_confidence:.2f}",
            f"Total evidence sources: {self.total_evidence_count}",
        ]
        if self.memory_evidence:
            lines.append("\nMemory evidence:")
            for m in self.memory_evidence[:5]:
                lines.append(f"  - {m}")
        if self.fact_evidence:
            lines.append("\nCanonical facts:")
            for f in self.fact_evidence[:5]:
                lines.append(f"  - {f}")
        if self.graph_evidence:
            lines.append("\nGraph evidence:")
            for g in self.graph_evidence[:5]:
                lines.append(f"  - {g}")
        if self.insight_evidence:
            lines.append("\nInsight evidence:")
            for i in self.insight_evidence[:3]:
                lines.append(f"  - {i}")
        if self.reasoning_trace:
            lines.append(f"\n{self.reasoning_trace}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Explainability Engine
# ---------------------------------------------------------------------------


class ExplainabilityEngine:
    """
    Builds an ``ExplainedResponse`` by collecting evidence from every
    available Blix subsystem.

    Parameters
    ----------
    memory_manager, retriever, consolidation_engine, reflection_engine,
    graph, graph_reasoner, cognitive_query_engine:
        Optional Blix components. Missing components are skipped silently.
    max_memories, max_facts, max_graph_paths:
        Limits on how many evidence items to include per type.
    """

    def __init__(
        self,
        memory_manager: Optional[object] = None,
        retriever: Optional[object] = None,
        consolidation_engine: Optional[object] = None,
        reflection_engine: Optional[object] = None,
        graph: Optional[object] = None,
        graph_reasoner: Optional[object] = None,
        cognitive_query_engine: Optional[object] = None,
        max_memories: int = 5,
        max_facts: int = 5,
        max_graph_paths: int = 5,
    ) -> None:
        self._mm = memory_manager
        self._retriever = retriever
        self._facts = consolidation_engine
        self._reflection = reflection_engine
        self._graph = graph
        self._reasoner = graph_reasoner
        self._cqe = cognitive_query_engine
        self._max_mem = max_memories
        self._max_facts = max_facts
        self._max_graph = max_graph_paths

    def explain(
        self,
        answer: str,
        query: str,
        *,
        reasoning_trace: Optional[ReasoningTrace] = None,
        memory_ids: Optional[list[int]] = None,
        fact_ids: Optional[list[str]] = None,
    ) -> ExplainedResponse:
        """
        Build an ``ExplainedResponse`` for a given answer to ``query``.

        Parameters
        ----------
        answer:
            The text answer to explain.
        query:
            The original user query (for semantic retrieval of supporting evidence).
        reasoning_trace:
            Optional ``ReasoningTrace`` from a ``CognitiveQueryEngine`` call.
        memory_ids:
            Optional explicit list of memory ids to include as evidence
            (in addition to semantic retrieval).
        fact_ids:
            Optional explicit list of canonical fact ids to include.
        """
        mem_evidence = self._collect_memory_evidence(query, memory_ids)
        fact_evidence = self._collect_fact_evidence(query, answer, fact_ids)
        graph_evidence = self._collect_graph_evidence(query, reasoning_trace)
        insight_evidence = self._collect_insight_evidence(query)

        return ExplainedResponse(
            answer=answer,
            memory_evidence=mem_evidence,
            fact_evidence=fact_evidence,
            graph_evidence=graph_evidence,
            insight_evidence=insight_evidence,
            reasoning_trace=reasoning_trace,
        )

    # ------------------------------------------------------------------
    # Evidence collectors
    # ------------------------------------------------------------------

    def _collect_memory_evidence(
        self,
        query: str,
        explicit_ids: Optional[list[int]],
    ) -> list[MemoryEvidence]:
        if self._mm is None:
            return []

        all_memories = self._mm.get_all_memories()  # type: ignore[union-attr]
        id_set = set(explicit_ids or [])
        scored: list[tuple[object, float]] = []

        # Semantic retrieval
        if self._retriever is not None:
            retrieved = self._retriever.retrieve(all_memories, query)[:self._max_mem]  # type: ignore[union-attr]
            for m in retrieved:
                scored.append((m, 0.7))  # default relevance if scorer not available

        # Explicit ids override
        if id_set:
            id_map = {m.id: m for m in all_memories}
            for mid in id_set:
                if mid in id_map:
                    scored.append((id_map[mid], 0.9))

        # Deduplicate by id, keep highest relevance
        seen: dict[int, tuple[object, float]] = {}
        for m, score in scored:
            if m.id not in seen or score > seen[m.id][1]:
                seen[m.id] = (m, score)

        result = []
        for m, score in list(seen.values())[:self._max_mem]:
            result.append(MemoryEvidence(
                memory_id=m.id,
                excerpt=(getattr(m, "output", "") or "")[:120].replace("\n", " "),
                relevance_score=score,
                topics=list(getattr(m, "topics", [])),
            ))
        return result

    def _collect_fact_evidence(
        self,
        query: str,
        answer: str,
        explicit_ids: Optional[list[str]],
    ) -> list[FactEvidence]:
        if self._facts is None:
            return []

        # Semantic match: facts whose text overlaps with query or answer
        all_facts = self._facts.list_facts()  # type: ignore[union-attr]
        query_words = set(re.findall(r"[a-z]+", query.lower()))
        answer_words = set(re.findall(r"[a-z]+", answer.lower()))
        search_words = query_words | answer_words

        matched: list[object] = []
        explicit_set = set(explicit_ids or [])

        for cf in all_facts:
            if cf.fact_id in explicit_set:
                matched.append(cf)
                continue
            fact_words = set(re.findall(r"[a-z]+", cf.fact.lower()))
            overlap = len(search_words & fact_words)
            if overlap >= 2:
                matched.append(cf)

        matched = matched[:self._max_facts]
        return [
            FactEvidence(
                fact_id=cf.fact_id,
                fact=cf.fact,
                confidence=cf.confidence,
                evidence_count=cf.evidence_count,
            )
            for cf in matched
        ]

    def _collect_graph_evidence(
        self,
        query: str,
        trace: Optional[ReasoningTrace],
    ) -> list[GraphEvidence]:
        if self._graph is None:
            return []

        result: list[GraphEvidence] = []

        # If we already have a reasoning trace, convert its steps to GraphEvidence
        if trace is not None and trace.steps:
            for step in trace.steps[:self._max_graph]:
                desc = f"{step.from_label} →[{step.relation}]→ {step.to_label}"
                result.append(GraphEvidence(
                    description=desc,
                    path_nodes=[step.from_label, step.to_label],
                    path_relations=[step.relation],
                    confidence=step.confidence,
                ))
            return result

        # Otherwise: find entities mentioned in query and get their edges
        if self._reasoner is None:
            return []

        words = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,}", query)
        for word in words[:3]:
            node = self._graph.find_node_by_label(word)  # type: ignore[union-attr]
            if node is None:
                continue
            for edge in self._graph.get_edges(from_id=node.id)[:3]:  # type: ignore[union-attr]
                tgt = self._graph.get_node(edge.to_id)  # type: ignore[union-attr]
                if tgt:
                    desc = f"{node.label} →[{edge.relation}]→ {tgt.label}"
                    result.append(GraphEvidence(
                        description=desc,
                        path_nodes=[node.label, tgt.label],
                        path_relations=[edge.relation],
                        confidence=edge.confidence,
                    ))
            if len(result) >= self._max_graph:
                break

        return result[:self._max_graph]

    def _collect_insight_evidence(self, query: str) -> list[InsightEvidence]:
        if self._reflection is None:
            return []

        insights = self._reflection.get_recent_insights(limit=20)  # type: ignore[union-attr]
        query_words = set(re.findall(r"[a-z]+", query.lower()))
        matched: list[InsightEvidence] = []

        for ins in insights:
            text = getattr(ins, "insight", "")
            ins_words = set(re.findall(r"[a-z]+", text.lower()))
            if len(query_words & ins_words) >= 2:
                matched.append(InsightEvidence(
                    insight=text[:200],
                    confidence=getattr(ins, "confidence", 0.5),
                    scope=str(getattr(ins, "scope", "")),
                ))
        return matched[:3]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

import re
