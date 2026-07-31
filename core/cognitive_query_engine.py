"""
Cognitive Query Engine — Blix v0.3.4  (Features 1 & 2)

Enables Blix to REASON over its knowledge graph instead of merely
retrieving from it.

Feature 1 — Graph Reasoning Queries
-------------------------------------
Natural-language queries are parsed into structured graph traversals:

    Query: "What technologies does Blix use?"
    → Traverse outgoing `uses` edges from node "blix"
    → Answer: ChromaDB, Transformers, FastAPI

    Query: "What does Sayan work on?"
    → Traverse outgoing `works_on` / `develops` edges from "sayan"
    → Answer: Blix, ECOT

Feature 2 — Multi-Hop Inference
---------------------------------
Explicit transitive reasoning across the graph:

    A → B, B → C  ⟹  infer A → C (with lower confidence)

    Example:
    Sayan → works_on → Blix
    Blix  → uses     → Reflection Engine

    Question: "What systems has Sayan worked on that use Reflection?"
    → 2-hop path: Sayan → works_on → Blix → uses → Reflection Engine
    → Answer: Blix (the intermediate node that satisfies both conditions)

Both features include an Explainability Layer (Feature 6):
Every ``QueryResult`` carries a full ``ReasoningTrace`` with the graph
path, confidence chain, and source memory ids.

Python 3.10 compatible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from core.memory_graph import EntityKind, MemoryGraph, RelationKind
from core.graph_reasoner import GraphReasoner, GraphPath
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reasoning trace (Explainability Layer — Feature 6)
# ---------------------------------------------------------------------------


@dataclass
class ReasoningStep:
    """One hop in a reasoning chain."""

    from_label: str
    relation: str
    to_label: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.from_label} →[{self.relation}]→ {self.to_label}"


@dataclass
class ReasoningTrace:
    """
    Full provenance of a cognitive query answer.

    Attributes
    ----------
    steps:
        Ordered reasoning steps (graph path hops).
    source_memory_ids:
        MemoryEntry ids that supported the relevant graph edges.
    confidence:
        Product of per-step confidences (propagated).
    explanation:
        Human-readable explanation of the reasoning chain.
    """

    steps: list[ReasoningStep] = field(default_factory=list)
    source_memory_ids: list[int] = field(default_factory=list)
    confidence: float = 1.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "steps": [
                {"from": s.from_label, "relation": s.relation,
                 "to": s.to_label, "confidence": round(s.confidence, 3)}
                for s in self.steps
            ],
            "source_memory_ids": self.source_memory_ids,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
        }

    def __str__(self) -> str:
        if not self.steps:
            return "No reasoning path."
        chain = "\n  → ".join(str(s) for s in self.steps)
        return f"Reasoning path:\n  {chain}\n  (confidence={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Query result
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """The answer to one cognitive query."""

    query: str
    answer: list[str]                   # human-readable answer entities/facts
    trace: ReasoningTrace = field(default_factory=ReasoningTrace)
    raw_nodes: list[dict] = field(default_factory=list)  # graph node details

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "trace": self.trace.to_dict(),
            "raw_nodes": self.raw_nodes,
        }

    def is_empty(self) -> bool:
        return not self.answer


# ---------------------------------------------------------------------------
# Query pattern library
# ---------------------------------------------------------------------------

# Patterns for extracting (subject, relation_hint) from natural-language queries.
# Each entry: (compiled_regex, relation_hint_or_none)

_QUERY_PATTERNS: list[tuple[re.Pattern, Optional[str]]] = [
    # "What does X use?" / "What technologies does X use?"
    (re.compile(r"what (?:does\s+)?(.+?)\s+use\??$", re.I), "uses"),
    # "What is X used for?" — invert: find nodes that X is a target of
    (re.compile(r"what (?:is|are)\s+(.+?)\s+used (?:for|in|by)\??$", re.I), "uses_inverse"),
    # "What does X work on?" / "What projects does X work on?"
    (re.compile(r"what (?:does\s+)?(.+?)\s+work(?:s)?\s+on\??$", re.I), "works_on"),
    # "What is X interested in?"
    (re.compile(r"what (?:is|are)\s+(.+?)\s+interested in\??$", re.I), "interested_in"),
    # "Who works on X?" / "Who uses X?"
    (re.compile(r"who\s+(works? on|uses?|developed?|studies?)\s+(.+?)\??$", re.I), None),
    # "What does X study?" / "Where does X study?"
    (re.compile(r"(?:what|where)\s+does\s+(.+?)\s+stud(?:y|ies)\??$", re.I), "studies_at"),
    # "What skills does X have?" / "What is X skilled in?"
    (re.compile(r"what\s+(?:skills?|capabilities?)?\s*does\s+(.+?)\s+(?:have|use|know)\??$", re.I), "uses"),
    # "What does X goal?" / "What are X's goals?"
    (re.compile(r"what (?:are|is)\s+(.+?)(?:\'s)?\s+goals?\??$", re.I), "goal_is"),
    # "Who collaborates with X?"
    (re.compile(r"who\s+collaborates?\s+with\s+(.+?)\??$", re.I), "collaborates_with"),
    # Fallback: "What does X ...?" — generic outgoing traversal
    (re.compile(r"what\s+(?:does\s+)?(.+?)\??$", re.I), None),
]

_RELATION_VERB_MAP: dict[str, RelationKind] = {
    "works on": RelationKind.WORKS_ON,
    "work on": RelationKind.WORKS_ON,
    "developed": RelationKind.WORKS_ON,
    "develops": RelationKind.WORKS_ON,
    "uses": RelationKind.USES,
    "use": RelationKind.USES,
    "studies": RelationKind.STUDIES_AT,
    "study": RelationKind.STUDIES_AT,
}


# ---------------------------------------------------------------------------
# Cognitive Query Engine
# ---------------------------------------------------------------------------


class CognitiveQueryEngine:
    """
    Enables natural-language reasoning over the Blix knowledge graph.

    Parameters
    ----------
    graph:
        ``MemoryGraph`` instance to reason over.
    reasoner:
        ``GraphReasoner`` — provides path search and centrality.
        If ``None``, one is created from ``graph``.
    max_depth:
        Maximum traversal depth for multi-hop queries.
    min_confidence:
        Minimum edge confidence to follow during traversal.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        reasoner: Optional[GraphReasoner] = None,
        max_depth: int = 4,
        min_confidence: float = 0.0,
    ) -> None:
        self._graph = graph
        self._reasoner = reasoner or GraphReasoner(graph)
        self._max_depth = max_depth
        self._min_conf = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, natural_language_query: str) -> QueryResult:
        """
        Answer a natural-language query by reasoning over the graph.

        Returns a ``QueryResult`` with the answer list and full
        ``ReasoningTrace`` for explainability.
        """
        parsed = self._parse_query(natural_language_query)
        if parsed is None:
            return QueryResult(
                query=natural_language_query,
                answer=[],
                trace=ReasoningTrace(explanation="Query not understood."),
            )

        subject_label, relation_hint, inverse = parsed
        log.debug(
            "CognitiveQuery: subject=%r relation=%r inverse=%s",
            subject_label, relation_hint, inverse,
        )

        # Resolve subject label to a node id
        subject_node = self._graph.find_node_by_label(subject_label)
        if subject_node is None:
            return QueryResult(
                query=natural_language_query,
                answer=[],
                trace=ReasoningTrace(
                    explanation=f"Entity '{subject_label}' not found in the knowledge graph."
                ),
            )

        if inverse:
            return self._answer_inverse(natural_language_query, subject_node, relation_hint)

        return self._answer_outgoing(natural_language_query, subject_node, relation_hint)

    def multi_hop_query(
        self,
        start_label: str,
        end_label: str,
        intermediate_relation: Optional[str] = None,
    ) -> QueryResult:
        """
        Two-hop inference: given start and end entities, find intermediate
        nodes that connect them (e.g. "What does Sayan work on that uses X?").

        Parameters
        ----------
        start_label:
            The source entity (e.g. "Sayan").
        end_label:
            The target entity the intermediate must connect to (e.g. "Reflection Engine").
        intermediate_relation:
            Optional: the relation between the intermediate and ``end_label``
            (e.g. "uses"). If None, accepts any relation.
        """
        start = self._graph.find_node_by_label(start_label)
        end = self._graph.find_node_by_label(end_label)

        if start is None or end is None:
            missing = start_label if start is None else end_label
            return QueryResult(
                query=f"multi-hop: {start_label} → ? → {end_label}",
                answer=[],
                trace=ReasoningTrace(explanation=f"Entity '{missing}' not found."),
            )

        # Find all paths from start to end within max_depth
        paths = self._reasoner.find_paths(start.id, end.id, max_depth=self._max_depth)
        if not paths:
            return QueryResult(
                query=f"multi-hop: {start_label} → ? → {end_label}",
                answer=[],
                trace=ReasoningTrace(
                    explanation=f"No path found from '{start_label}' to '{end_label}'."
                ),
            )

        # Filter by intermediate relation if requested
        if intermediate_relation:
            try:
                target_rel = RelationKind(intermediate_relation).value
                paths = [
                    p for p in paths
                    if len(p.relations) >= 2 and p.relations[-1] == target_rel
                ]
            except ValueError:
                pass

        # Collect intermediate nodes (all nodes except start and end)
        intermediates: list[str] = []
        traces: list[ReasoningStep] = []
        for path in paths[:5]:
            for i, node_id in enumerate(path.nodes[1:-1], start=1):
                node = self._graph.get_node(node_id)
                if node and node.label not in intermediates:
                    intermediates.append(node.label)
            # Build trace from best path
            if not traces:
                traces = self._path_to_steps(paths[0])

        confidence = paths[0].total_confidence if paths else 0.0
        explanation = (
            f"Found {len(paths)} path(s) from '{start_label}' to '{end_label}'. "
            f"Intermediate entities: {', '.join(intermediates) or 'none'}."
        )
        mem_ids = self._collect_memory_ids(start.id, end.id)

        return QueryResult(
            query=f"multi-hop: {start_label} → ? → {end_label}",
            answer=intermediates,
            trace=ReasoningTrace(
                steps=traces,
                source_memory_ids=mem_ids,
                confidence=confidence,
                explanation=explanation,
            ),
            raw_nodes=[{"id": start.id, "label": start.label},
                       {"id": end.id, "label": end.label}],
        )

    def infer_transitive(
        self,
        entity_label: str,
        relation: str,
        depth: int = 2,
    ) -> QueryResult:
        """
        Collect all entities reachable from ``entity_label`` via ``relation``
        up to ``depth`` hops, inferring transitive closure.

        A → uses → B, B → uses → C  ⟹  A → uses (transitively) → C

        Useful for: "What does Blix transitively depend on?"
        """
        node = self._graph.find_node_by_label(entity_label)
        if node is None:
            return QueryResult(
                query=f"transitive({entity_label}, {relation})",
                answer=[],
                trace=ReasoningTrace(explanation=f"Entity '{entity_label}' not found."),
            )

        try:
            rel = RelationKind(relation)
        except ValueError:
            return QueryResult(
                query=f"transitive({entity_label}, {relation})",
                answer=[],
                trace=ReasoningTrace(explanation=f"Unknown relation: '{relation}'."),
            )

        # BFS, following only edges with the target relation
        visited: dict[str, int] = {node.id: 0}
        queue: list[tuple[str, list[ReasoningStep], float]] = [(node.id, [], 1.0)]
        result_labels: list[str] = []
        all_steps: list[ReasoningStep] = []
        mem_ids: set[int] = set()

        while queue:
            cur_id, path_steps, cum_conf = queue.pop(0)
            if visited.get(cur_id, 0) >= depth and cur_id != node.id:
                continue
            for edge in self._graph.get_edges(from_id=cur_id, relation=rel):
                if edge.confidence < self._min_conf:
                    continue
                tgt = self._graph.get_node(edge.to_id)
                if tgt is None or edge.to_id in visited:
                    continue
                visited[edge.to_id] = visited[cur_id] + 1
                new_conf = cum_conf * edge.confidence
                step = ReasoningStep(
                    from_label=self._graph.get_node(cur_id).label,
                    relation=rel.value,
                    to_label=tgt.label,
                    confidence=edge.confidence,
                )
                if tgt.label not in result_labels:
                    result_labels.append(tgt.label)
                    all_steps.append(step)
                mem_ids.update(edge.source_memory_ids)
                if visited[edge.to_id] < depth:
                    queue.append((edge.to_id, path_steps + [step], new_conf))

        explanation = (
            f"Transitive closure of '{entity_label}' via '{relation}' "
            f"(depth={depth}): {len(result_labels)} entit(ies) found."
        )
        return QueryResult(
            query=f"transitive({entity_label}, {relation}, depth={depth})",
            answer=result_labels,
            trace=ReasoningTrace(
                steps=all_steps[:10],
                source_memory_ids=list(mem_ids),
                confidence=1.0 if result_labels else 0.0,
                explanation=explanation,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_query(
        self, query: str
    ) -> Optional[tuple[str, Optional[str], bool]]:
        """
        Parse a natural-language query into (subject_label, relation_hint, inverse).

        Returns ``None`` if the query doesn't match any pattern.
        """
        q = query.strip()
        for pattern, rel_hint in _QUERY_PATTERNS:
            m = pattern.match(q)
            if m is None:
                continue

            if rel_hint == "uses_inverse":
                subject = m.group(1).strip()
                return subject, "uses", True

            # "Who X Y?" patterns — subject is group 2, relation from group 1
            if rel_hint is None and len(m.groups()) == 2:
                verb = m.group(1).strip().lower()
                subject = m.group(2).strip()
                rel_kind = _RELATION_VERB_MAP.get(verb)
                rel_str = rel_kind.value if rel_kind else None
                return subject, rel_str, True

            subject = m.group(1).strip()
            return subject, rel_hint, False

        return None

    def _answer_outgoing(
        self,
        query: str,
        subject_node: object,
        relation_hint: Optional[str],
    ) -> QueryResult:
        """Return all entities reachable from subject_node via the hinted relation."""
        subj_id = subject_node.id  # type: ignore[union-attr]
        subj_label = subject_node.label  # type: ignore[union-attr]

        if relation_hint is not None:
            try:
                rel = RelationKind(relation_hint)
                edges = self._graph.get_edges(from_id=subj_id, relation=rel)
            except ValueError:
                edges = self._graph.get_edges(from_id=subj_id)
        else:
            edges = self._graph.get_edges(from_id=subj_id)

        edges = [e for e in edges if e.confidence >= self._min_conf]

        answers: list[str] = []
        steps: list[ReasoningStep] = []
        raw_nodes: list[dict] = []
        mem_ids: set[int] = set()

        for edge in edges:
            tgt = self._graph.get_node(edge.to_id)
            if tgt is None:
                continue
            answers.append(tgt.label)
            steps.append(ReasoningStep(
                from_label=subj_label,
                relation=edge.relation,
                to_label=tgt.label,
                confidence=edge.confidence,
            ))
            raw_nodes.append({"id": tgt.id, "label": tgt.label, "kind": tgt.kind})
            mem_ids.update(edge.source_memory_ids)

        confidence = min((e.confidence for e in edges), default=0.0) if edges else 0.0
        explanation = (
            f"Traversed outgoing '{relation_hint or 'all'}' edges from '{subj_label}'. "
            f"Found {len(answers)} result(s)."
        )

        return QueryResult(
            query=query,
            answer=answers,
            trace=ReasoningTrace(
                steps=steps,
                source_memory_ids=list(mem_ids),
                confidence=confidence,
                explanation=explanation,
            ),
            raw_nodes=raw_nodes,
        )

    def _answer_inverse(
        self,
        query: str,
        subject_node: object,
        relation_hint: Optional[str],
    ) -> QueryResult:
        """Find entities that point TO subject_node (inverse traversal)."""
        subj_id = subject_node.id  # type: ignore[union-attr]
        subj_label = subject_node.label  # type: ignore[union-attr]

        if relation_hint and relation_hint != "uses_inverse":
            try:
                rel = RelationKind(relation_hint)
                edges = self._graph.get_edges(to_id=subj_id, relation=rel)
            except ValueError:
                edges = self._graph.get_edges(to_id=subj_id)
        else:
            edges = self._graph.get_edges(to_id=subj_id)

        edges = [e for e in edges if e.confidence >= self._min_conf]

        answers: list[str] = []
        steps: list[ReasoningStep] = []
        mem_ids: set[int] = set()

        for edge in edges:
            src = self._graph.get_node(edge.from_id)
            if src is None:
                continue
            answers.append(src.label)
            steps.append(ReasoningStep(
                from_label=src.label,
                relation=edge.relation,
                to_label=subj_label,
                confidence=edge.confidence,
            ))
            mem_ids.update(edge.source_memory_ids)

        explanation = (
            f"Inverse traversal: entities pointing to '{subj_label}' "
            f"via '{relation_hint or 'any'}'. Found {len(answers)} result(s)."
        )
        return QueryResult(
            query=query,
            answer=answers,
            trace=ReasoningTrace(
                steps=steps,
                source_memory_ids=list(mem_ids),
                confidence=1.0 if answers else 0.0,
                explanation=explanation,
            ),
        )

    def _path_to_steps(self, path: GraphPath) -> list[ReasoningStep]:
        steps = []
        for i, (nid, rel) in enumerate(zip(path.nodes, path.relations)):
            tgt_id = path.nodes[i + 1]
            src = self._graph.get_node(nid)
            tgt = self._graph.get_node(tgt_id)
            if src and tgt:
                edges = self._graph.get_edges(from_id=nid, to_id=tgt_id)
                conf = edges[0].confidence if edges else 1.0
                steps.append(ReasoningStep(
                    from_label=src.label, relation=rel,
                    to_label=tgt.label, confidence=conf,
                ))
        return steps

    def _collect_memory_ids(self, from_id: str, to_id: str) -> list[int]:
        ids: set[int] = set()
        for edge in self._graph.get_edges(from_id=from_id):
            ids.update(edge.source_memory_ids)
        for edge in self._graph.get_edges(to_id=to_id):
            ids.update(edge.source_memory_ids)
        return list(ids)
