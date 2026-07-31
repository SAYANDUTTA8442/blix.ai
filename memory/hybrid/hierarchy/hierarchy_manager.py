"""
Hierarchy subsystem — automatic layered memory compression.

Raw memories → Episodes → Conversations → Sessions → Daily → Weekly →
Monthly → Project → Concept → Principle → Knowledge → World Model

Rules
-----
- RAW nodes accumulate until a threshold, then get compressed into an EPISODE.
- EPISODE nodes accumulate until a threshold, then get compressed into a
  CONVERSATION summary.
- Higher levels follow the same pattern with larger thresholds.
- No information is permanently lost: SUMMARISES edges link summaries to sources.
- Compression is triggered automatically by HierarchyManager.consolidate().
"""
from __future__ import annotations
import logging
from typing import Any

from memory.hybrid.models.memory_node import (
    MemoryNode, MemoryType, HierarchyLevel, EpistemicStatus
)
from memory.hybrid.models.memory_edge import EdgeRelation
from memory.hybrid.graph.graph_store import GraphStore
from memory.hybrid.graph.graph_builder import GraphBuilder

log = logging.getLogger(__name__)

# ─── Compression thresholds (nodes at level N → trigger compression) ────────
_THRESHOLDS: dict[HierarchyLevel, int] = {
    HierarchyLevel.RAW:          10,
    HierarchyLevel.EPISODE:      8,
    HierarchyLevel.CONVERSATION: 6,
    HierarchyLevel.SESSION:      5,
    HierarchyLevel.DAILY:        4,
    HierarchyLevel.WEEKLY:       4,
    HierarchyLevel.MONTHLY:      3,
    HierarchyLevel.PROJECT:      3,
    HierarchyLevel.CONCEPT:      2,
    HierarchyLevel.PRINCIPLE:    2,
}

_LEVEL_UP: dict[HierarchyLevel, HierarchyLevel] = {
    HierarchyLevel.RAW:          HierarchyLevel.EPISODE,
    HierarchyLevel.EPISODE:      HierarchyLevel.CONVERSATION,
    HierarchyLevel.CONVERSATION: HierarchyLevel.SESSION,
    HierarchyLevel.SESSION:      HierarchyLevel.DAILY,
    HierarchyLevel.DAILY:        HierarchyLevel.WEEKLY,
    HierarchyLevel.WEEKLY:       HierarchyLevel.MONTHLY,
    HierarchyLevel.MONTHLY:      HierarchyLevel.PROJECT,
    HierarchyLevel.PROJECT:      HierarchyLevel.CONCEPT,
    HierarchyLevel.CONCEPT:      HierarchyLevel.PRINCIPLE,
    HierarchyLevel.PRINCIPLE:    HierarchyLevel.KNOWLEDGE,
    HierarchyLevel.KNOWLEDGE:    HierarchyLevel.WORLD_MODEL,
}


class Summarizer:
    """
    Produces concise natural-language summaries of a set of MemoryNodes.

    Without an LLM, we use a symbolic extractive approach:
      - Take the most important nodes by importance score
      - Concatenate their text with separators
      - Prepend a summary label

    With an LLM backend (settable via set_llm), generate abstractive summaries.
    """

    def __init__(self) -> None:
        self._llm: Any = None

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def summarise(self, nodes: list[MemoryNode], level: HierarchyLevel,
                  max_length: int = 300) -> str:
        if not nodes:
            return "(empty)"
        if self._llm is not None:
            return self._llm_summarise(nodes, level, max_length)
        return self._extractive_summarise(nodes, level, max_length)

    def _extractive_summarise(self, nodes: list[MemoryNode],
                               level: HierarchyLevel, max_length: int) -> str:
        sorted_nodes = sorted(nodes, key=lambda n: n.importance, reverse=True)
        level_label = level.name.replace("_", " ").title()
        parts = [f"[{level_label} Summary]"]
        chars = len(parts[0])
        for node in sorted_nodes:
            snippet = node.text[:100].strip()
            if chars + len(snippet) + 2 > max_length:
                break
            parts.append(f"• {snippet}")
            chars += len(snippet) + 2
        return "\n".join(parts)

    def _llm_summarise(self, nodes: list[MemoryNode],
                        level: HierarchyLevel, max_length: int) -> str:
        texts = "\n".join(f"- {n.text[:120]}" for n in nodes[:10])
        prompt = (f"Summarise the following {level.name} memories in "
                  f"under {max_length} characters:\n{texts}")
        try:
            return self._llm(prompt)
        except Exception:
            return self._extractive_summarise(nodes, level, max_length)


class AbstractionEngine:
    """
    Promotes clusters of related nodes to higher-level abstractions.

    Detects when a set of SIMILAR_TO-linked nodes can be abstracted into
    a single CONCEPT node, or when multiple CAUSE nodes can yield a PRINCIPLE.
    """

    def __init__(self, graph_store: GraphStore,
                 graph_builder: GraphBuilder,
                 min_cluster_size: int = 3) -> None:
        self._graph = graph_store
        self._builder = graph_builder
        self._min_size = min_cluster_size

    def abstract_causes_to_principle(
        self, cause_node_ids: list[str],
    ) -> MemoryNode | None:
        """
        If enough CAUSE nodes share a common trigger pattern,
        synthesise a PRINCIPLE node.
        """
        if len(cause_node_ids) < self._min_size:
            return None
        nodes = [self._graph.get_node(nid) for nid in cause_node_ids]
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return None

        # Extractive: find the most common content fragment
        all_words: dict[str, int] = {}
        for node in nodes:
            for word in node.text.lower().split():
                all_words[word] = all_words.get(word, 0) + 1

        top_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = " ".join(w for w, _ in top_words)
        statement = f"Pattern observed: {keywords} (from {len(nodes)} cause observations)"

        principle = self._builder.add_principle(
            statement=statement,
            confidence=min(0.9, 0.5 + len(nodes) * 0.05),
            derived_from_ids=cause_node_ids,
        )
        return principle

    def abstract_beliefs_to_concept(
        self, belief_node_ids: list[str], concept_name: str = "",
    ) -> MemoryNode | None:
        """
        Promote a cluster of related beliefs into a Concept node.
        """
        if len(belief_node_ids) < self._min_size:
            return None
        nodes = [self._graph.get_node(nid) for nid in belief_node_ids]
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return None
        if not concept_name:
            words: dict[str, int] = {}
            for n in nodes:
                for w in n.text.lower().split():
                    if len(w) > 3:
                        words[w] = words.get(w, 0) + 1
            concept_name = max(words, key=words.get) if words else "concept"

        return self._builder.add_concept(
            name=concept_name,
            description=f"Abstracted from {len(nodes)} belief nodes",
            member_node_ids=belief_node_ids,
        )


class HierarchyManager:
    """
    Manages automatic compression of the memory hierarchy.

    Call consolidate() periodically (e.g., after each conversation turn,
    or on a background timer) to compress nodes upward through the hierarchy.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        graph_builder: GraphBuilder,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._graph   = graph_store
        self._builder = graph_builder
        self._summarizer = summarizer or Summarizer()
        self._abstraction = AbstractionEngine(graph_store, graph_builder)

    def consolidate(self) -> dict[str, int]:
        """
        Scan all hierarchy levels and compress where thresholds are exceeded.
        Returns a dict of {level_name: summaries_created}.
        """
        created: dict[str, int] = {}
        for level, threshold in _THRESHOLDS.items():
            count = self._compress_level(level, threshold)
            if count > 0:
                created[level.name] = count
                log.info("HierarchyManager: compressed %d nodes at level %s",
                         count, level.name)
        return created

    def _compress_level(
        self, level: HierarchyLevel, threshold: int
    ) -> int:
        """Compress nodes at `level` if count exceeds `threshold`."""
        nodes = self._graph.all_nodes(hierarchy_level=level, limit=threshold * 10)
        if len(nodes) < threshold:
            return 0

        upper_level = _LEVEL_UP.get(level)
        if upper_level is None:
            return 0

        # Group into chunks of `threshold` and summarise each
        summaries_created = 0
        for i in range(0, len(nodes), threshold):
            chunk = nodes[i:i + threshold]
            if len(chunk) < threshold // 2:
                break  # too small to compress
            summary_text = self._summarizer.summarise(chunk, upper_level)
            avg_confidence = sum(n.confidence for n in chunk) / len(chunk)
            avg_importance  = sum(n.importance for n in chunk)  / len(chunk)

            summary = MemoryNode(
                text=summary_text,
                memory_type=MemoryType.SUMMARY,
                hierarchy_level=upper_level,
                confidence=avg_confidence,
                importance=avg_importance,
                source="hierarchy_manager",
                epistemic_status=EpistemicStatus.DERIVED,
            )
            self._graph.add_node(summary)
            for node in chunk:
                from memory.hybrid.models.memory_edge import MemoryEdge
                self._graph.add_edge(MemoryEdge(
                    source_id=summary.node_id,
                    target_id=node.node_id,
                    relation=EdgeRelation.SUMMARISES,
                    confidence=0.9,
                    weight=0.9,
                    provenance="hierarchy_manager",
                ))
            summaries_created += 1

        return summaries_created

    def get_hierarchy_stats(self) -> dict[str, int]:
        """Count nodes at each hierarchy level."""
        stats = {}
        for level in HierarchyLevel:
            count = self._graph.count_nodes()  # filtered below
            nodes = self._graph.all_nodes(hierarchy_level=level, limit=1)
            # Just get count from all_nodes with limit check
            all_at_level = self._graph.all_nodes(hierarchy_level=level, limit=100_000)
            stats[level.name] = len(all_at_level)
        return stats
