"""
MQL v2 — Expression-style Memory Query Language — Blix v0.3.4  (Feature 3)

Extends the v0.3.2 MQL (keyword "show X" commands) with EXPRESSION-style
queries that support filter predicates and field comparisons:

    memories where topics contains "transformers"
    memories where project = "Blix"
    facts about "transformers"
    facts min_confidence = 0.8
    insights last_30_days
    insights category = "trend"
    goals status = active
    goals priority <= 2
    graph neighbours "Sayan"
    graph path "Sayan" to "FastAPI"
    query "What does Sayan work on?"           ← cognitive query (Feature 1)
    infer "Sayan" via "works_on" depth 3       ← transitive inference (Feature 2)
    multihop "Sayan" to "FastAPI"              ← multi-hop path (Feature 2)

All expression queries are additive to the existing "show ..." commands:
``MQLv2Engine`` first tries expression patterns, then falls back to
the v0.3.2 ``MQLEngine``.

Python 3.10 compatible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MQLv2Result:
    """Result of an MQL v2 expression query."""

    command: str
    matched: bool
    text: str
    data: list = field(default_factory=list)
    trace: Optional[dict] = None       # ReasoningTrace.to_dict() when available

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Expression pattern definitions
# ---------------------------------------------------------------------------

# Each pattern: (compiled_regex, handler_name)
_EXPRESSION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # memories where topics contains "X"
    (re.compile(r'^memories\s+where\s+topics?\s+contains?\s+"([^"]+)"$', re.I), "mem_topics_contains"),
    # memories where project = "X"
    (re.compile(r'^memories\s+where\s+project\s*=\s*"([^"]+)"$', re.I), "mem_project_eq"),
    # memories where importance >= N
    (re.compile(r'^memories\s+where\s+importance\s*(>=|<=|>|<|=)\s*([\d.]+)$', re.I), "mem_importance_cmp"),
    # facts about "X"
    (re.compile(r'^facts\s+about\s+"([^"]+)"$', re.I), "facts_about"),
    # facts min_confidence = N
    (re.compile(r'^facts\s+min_confidence\s*=\s*([\d.]+)$', re.I), "facts_min_conf"),
    # facts topic = "X"
    (re.compile(r'^facts\s+topic\s*=\s*"([^"]+)"$', re.I), "facts_topic_eq"),
    # insights last_N_days
    (re.compile(r'^insights\s+last_(\d+)_days?$', re.I), "insights_last_days"),
    # insights category = "X"
    (re.compile(r'^insights\s+category\s*=\s*"?(\w+)"?$', re.I), "insights_category"),
    # goals status = X
    (re.compile(r'^goals\s+status\s*=\s*(\w+)$', re.I), "goals_status"),
    # goals priority <= N
    (re.compile(r'^goals\s+priority\s*(>=|<=|>|<|=)\s*(\d+)$', re.I), "goals_priority_cmp"),
    # goals project = "X"
    (re.compile(r'^goals\s+project\s*=\s*"([^"]+)"$', re.I), "goals_project"),
    # graph neighbours "X"
    (re.compile(r'^graph\s+neighbours?\s+"([^"]+)"$', re.I), "graph_neighbours"),
    # graph path "X" to "Y"
    (re.compile(r'^graph\s+path\s+"([^"]+)"\s+to\s+"([^"]+)"$', re.I), "graph_path"),
    # query "..."  ← natural-language cognitive query
    (re.compile(r'^query\s+"([^"]+)"$', re.I), "cognitive_query"),
    # infer "X" via "rel" depth N
    (re.compile(r'^infer\s+"([^"]+)"\s+via\s+"([^"]+)"(?:\s+depth\s+(\d+))?$', re.I), "infer_transitive"),
    # multihop "X" to "Y"
    (re.compile(r'^multihop\s+"([^"]+)"\s+to\s+"([^"]+)"$', re.I), "multihop"),
]


# ---------------------------------------------------------------------------
# MQLv2 Parser
# ---------------------------------------------------------------------------


class MQLv2Parser:
    """Parses expression-style MQL v2 commands."""

    def parse(self, command: str) -> Optional[tuple[str, dict]]:
        """
        Returns ``(handler_name, args)`` or ``None`` if no pattern matches.
        """
        text = command.strip()
        for pattern, handler in _EXPRESSION_PATTERNS:
            m = pattern.match(text)
            if m is None:
                continue
            groups = m.groups()
            return handler, {"groups": groups, "raw": text}
        return None


# ---------------------------------------------------------------------------
# MQLv2 Executor
# ---------------------------------------------------------------------------


class MQLv2Executor:
    """
    Executes MQL v2 expression queries.

    All components are optional — missing components yield a graceful
    "not available" result.

    Parameters
    ----------
    memory_manager, retriever, consolidation_engine, reflection_engine,
    insight_engine, goal_tracker, graph, graph_reasoner, cognitive_query_engine:
        Optional Blix components.
    """

    def __init__(
        self,
        memory_manager: Optional[object] = None,
        retriever: Optional[object] = None,
        consolidation_engine: Optional[object] = None,
        reflection_engine: Optional[object] = None,
        insight_engine: Optional[object] = None,
        goal_tracker: Optional[object] = None,
        graph: Optional[object] = None,
        graph_reasoner: Optional[object] = None,
        cognitive_query_engine: Optional[object] = None,
    ) -> None:
        self._mm = memory_manager
        self._retriever = retriever
        self._facts = consolidation_engine
        self._reflection = reflection_engine
        self._insights = insight_engine
        self._goals = goal_tracker
        self._graph = graph
        self._reasoner = graph_reasoner
        self._cqe = cognitive_query_engine

    def execute(self, handler: str, args: dict) -> MQLv2Result:
        groups = args.get("groups", ())
        raw = args.get("raw", handler)
        method = getattr(self, f"_h_{handler}", None)
        if method is None:
            return MQLv2Result(command=raw, matched=False, text=f"No handler for '{handler}'.")
        return method(groups, raw)

    # ------------------------------------------------------------------
    # Memory handlers
    # ------------------------------------------------------------------

    def _h_mem_topics_contains(self, groups: tuple, raw: str) -> MQLv2Result:
        topic = groups[0].lower()
        if self._mm is None:
            return _unavail(raw, "MemoryManager")
        memories = self._mm.get_all_memories()  # type: ignore[union-attr]
        results = [m for m in memories if topic in [t.lower() for t in getattr(m, "topics", [])]]
        return _format_memories(raw, results, f"memories with topic='{topic}'")

    def _h_mem_project_eq(self, groups: tuple, raw: str) -> MQLv2Result:
        project = groups[0].lower()
        if self._mm is None:
            return _unavail(raw, "MemoryManager")
        # Heuristic: memories whose input or output mentions the project
        memories = self._mm.get_all_memories()  # type: ignore[union-attr]
        results = [
            m for m in memories
            if project in getattr(m, "input", "").lower()
            or project in getattr(m, "output", "").lower()
        ]
        return _format_memories(raw, results, f"memories mentioning project='{project}'")

    def _h_mem_importance_cmp(self, groups: tuple, raw: str) -> MQLv2Result:
        op, val_str = groups[0], groups[1]
        threshold = float(val_str)
        if self._mm is None:
            return _unavail(raw, "MemoryManager")
        memories = self._mm.get_all_memories()  # type: ignore[union-attr]
        results = [m for m in memories if _cmp(getattr(m, "importance", None) or 0.0, op, threshold)]
        return _format_memories(raw, results, f"memories where importance {op} {threshold}")

    # ------------------------------------------------------------------
    # Facts handlers
    # ------------------------------------------------------------------

    def _h_facts_about(self, groups: tuple, raw: str) -> MQLv2Result:
        query = groups[0].lower()
        if self._facts is None:
            return _unavail(raw, "ConsolidationEngine")
        facts = self._facts.list_facts()  # type: ignore[union-attr]
        results = [f for f in facts if query in f.fact.lower()]
        return _format_facts(raw, results, f"facts about '{query}'")

    def _h_facts_min_conf(self, groups: tuple, raw: str) -> MQLv2Result:
        threshold = float(groups[0])
        if self._facts is None:
            return _unavail(raw, "ConsolidationEngine")
        results = self._facts.list_facts(min_confidence=threshold)  # type: ignore[union-attr]
        return _format_facts(raw, results, f"facts with confidence >= {threshold}")

    def _h_facts_topic_eq(self, groups: tuple, raw: str) -> MQLv2Result:
        topic = groups[0]
        if self._facts is None:
            return _unavail(raw, "ConsolidationEngine")
        results = self._facts.list_facts(topic=topic)  # type: ignore[union-attr]
        return _format_facts(raw, results, f"facts with topic='{topic}'")

    # ------------------------------------------------------------------
    # Insight handlers
    # ------------------------------------------------------------------

    def _h_insights_last_days(self, groups: tuple, raw: str) -> MQLv2Result:
        days = int(groups[0])
        if self._reflection is None:
            return _unavail(raw, "ReflectionEngine")
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        insights = self._reflection.get_insights_since(since)  # type: ignore[union-attr]
        return _format_insights(raw, insights, f"insights from last {days} day(s)")

    def _h_insights_category(self, groups: tuple, raw: str) -> MQLv2Result:
        category = groups[0].lower()
        if self._insights is None:
            return _unavail(raw, "InsightGenerationEngine")
        from reflection.insight_engine import InsightCategory
        try:
            cat = InsightCategory(category)
        except ValueError:
            return MQLv2Result(command=raw, matched=True,
                               text=f"Unknown category '{category}'. Try: trend, bottleneck, research_interest, project_pattern")
        results = self._insights.list_insights(category=cat)  # type: ignore[union-attr]
        return _format_actionable_insights(raw, results, f"insights with category='{category}'")

    # ------------------------------------------------------------------
    # Goals handlers
    # ------------------------------------------------------------------

    def _h_goals_status(self, groups: tuple, raw: str) -> MQLv2Result:
        status_str = groups[0].lower()
        if self._goals is None:
            return _unavail(raw, "GoalTracker")
        from reflection.goal_tracker import GoalStatus
        try:
            status = GoalStatus(status_str)
        except ValueError:
            return MQLv2Result(command=raw, matched=True,
                               text=f"Unknown status '{status_str}'. Try: active, paused, completed, abandoned")
        goals = self._goals.list_goals(status=status)  # type: ignore[union-attr]
        return _format_goals(raw, goals, f"goals with status='{status_str}'")

    def _h_goals_priority_cmp(self, groups: tuple, raw: str) -> MQLv2Result:
        op, val_str = groups[0], groups[1]
        threshold = int(val_str)
        if self._goals is None:
            return _unavail(raw, "GoalTracker")
        all_goals = self._goals.list_goals()  # type: ignore[union-attr]
        results = [g for g in all_goals if _cmp(float(g.priority), op, float(threshold))]
        return _format_goals(raw, results, f"goals where priority {op} {threshold}")

    def _h_goals_project(self, groups: tuple, raw: str) -> MQLv2Result:
        project = groups[0].lower()
        if self._goals is None:
            return _unavail(raw, "GoalTracker")
        all_goals = self._goals.list_goals()  # type: ignore[union-attr]
        results = [g for g in all_goals if g.related_project.lower() == project]
        return _format_goals(raw, results, f"goals linked to project='{project}'")

    # ------------------------------------------------------------------
    # Graph handlers
    # ------------------------------------------------------------------

    def _h_graph_neighbours(self, groups: tuple, raw: str) -> MQLv2Result:
        label = groups[0]
        if self._graph is None:
            return _unavail(raw, "MemoryGraph")
        node = self._graph.find_node_by_label(label)  # type: ignore[union-attr]
        if node is None:
            return MQLv2Result(command=raw, matched=True,
                               text=f"Entity '{label}' not found in graph.", data=[])
        if self._reasoner is None:
            return _unavail(raw, "GraphReasoner")
        neighbours = self._reasoner.related_entities(node.id, depth=1)  # type: ignore[union-attr]
        lines = [f"Neighbours of '{label}':"]
        data = []
        for nb_node, hop in neighbours:
            lines.append(f"  - {nb_node.label} ({nb_node.kind})")
            data.append({"id": nb_node.id, "label": nb_node.label, "kind": nb_node.kind, "hop": hop})
        return MQLv2Result(command=raw, matched=True, text="\n".join(lines) or "No neighbours.", data=data)

    def _h_graph_path(self, groups: tuple, raw: str) -> MQLv2Result:
        from_label, to_label = groups[0], groups[1]
        if self._reasoner is None or self._graph is None:
            return _unavail(raw, "GraphReasoner + MemoryGraph")
        fn = self._graph.find_node_by_label(from_label)  # type: ignore[union-attr]
        tn = self._graph.find_node_by_label(to_label)  # type: ignore[union-attr]
        if fn is None or tn is None:
            missing = from_label if fn is None else to_label
            return MQLv2Result(command=raw, matched=True, text=f"Entity '{missing}' not found.", data=[])
        path = self._reasoner.shortest_path(fn.id, tn.id)  # type: ignore[union-attr]
        if path is None:
            return MQLv2Result(command=raw, matched=True,
                               text=f"No path found from '{from_label}' to '{to_label}'.", data=[])
        text = (
            f"Shortest path ({len(path.nodes) - 1} hop(s), confidence={path.total_confidence:.2f}):\n"
            f"  {str(path)}"
        )
        return MQLv2Result(command=raw, matched=True, text=text,
                           data={"nodes": path.nodes, "relations": path.relations,
                                 "confidence": path.total_confidence})

    # ------------------------------------------------------------------
    # Cognitive query handlers
    # ------------------------------------------------------------------

    def _h_cognitive_query(self, groups: tuple, raw: str) -> MQLv2Result:
        nl_query = groups[0]
        if self._cqe is None:
            return _unavail(raw, "CognitiveQueryEngine")
        result = self._cqe.query(nl_query)  # type: ignore[union-attr]
        lines = [f'Query: "{nl_query}"']
        if result.answer:
            lines.append(f"Answer: {', '.join(result.answer)}")
        else:
            lines.append("Answer: (no results found)")
        lines.append(f"\n{result.trace}")
        return MQLv2Result(
            command=raw, matched=True, text="\n".join(lines),
            data=result.raw_nodes, trace=result.trace.to_dict()
        )

    def _h_infer_transitive(self, groups: tuple, raw: str) -> MQLv2Result:
        entity, relation = groups[0], groups[1]
        depth = int(groups[2]) if len(groups) > 2 and groups[2] else 2
        if self._cqe is None:
            return _unavail(raw, "CognitiveQueryEngine")
        result = self._cqe.infer_transitive(entity, relation, depth=depth)  # type: ignore[union-attr]
        lines = [f'Transitive closure: "{entity}" via "{relation}" (depth={depth})']
        if result.answer:
            lines.append(f"Found: {', '.join(result.answer)}")
        else:
            lines.append("Found: (no results)")
        lines.append(result.trace.explanation)
        return MQLv2Result(
            command=raw, matched=True, text="\n".join(lines),
            data=result.answer, trace=result.trace.to_dict()
        )

    def _h_multihop(self, groups: tuple, raw: str) -> MQLv2Result:
        start, end = groups[0], groups[1]
        if self._cqe is None:
            return _unavail(raw, "CognitiveQueryEngine")
        result = self._cqe.multi_hop_query(start, end)  # type: ignore[union-attr]
        lines = [f'Multi-hop: "{start}" → ? → "{end}"']
        if result.answer:
            lines.append(f"Intermediates: {', '.join(result.answer)}")
        else:
            lines.append("No connecting path found.")
        lines.append(result.trace.explanation)
        return MQLv2Result(
            command=raw, matched=True, text="\n".join(lines),
            data=result.answer, trace=result.trace.to_dict()
        )


# ---------------------------------------------------------------------------
# Combined engine: MQLv2 expressions + fallback to v0.3.2 "show X" commands
# ---------------------------------------------------------------------------


class MQLv2Engine:
    """
    Unified MQL engine combining v0.3.4 expression syntax with the
    v0.3.2 ``MQLEngine`` "show ..." commands.

    Usage
    -----
        engine = MQLv2Engine(cognitive_query_engine=cqe, ...)
        result = engine.run('query "What does Sayan work on?"')
        result = engine.run('memories where topics contains "nlp"')
        result = engine.run('show active goals')   # falls back to v0.3.2
    """

    def __init__(
        self,
        *,
        memory_manager: Optional[object] = None,
        retriever: Optional[object] = None,
        consolidation_engine: Optional[object] = None,
        reflection_engine: Optional[object] = None,
        insight_engine: Optional[object] = None,
        goal_tracker: Optional[object] = None,
        graph: Optional[object] = None,
        graph_reasoner: Optional[object] = None,
        cognitive_query_engine: Optional[object] = None,
        # v0.3.2 MQLEngine components (passed through for fallback)
        project_manager: Optional[object] = None,
        project_intelligence: Optional[object] = None,
        semantic_cluster_index: Optional[object] = None,
        contradiction_detector: Optional[object] = None,
        lifecycle_manager: Optional[object] = None,
    ) -> None:
        self._parser = MQLv2Parser()
        self._executor = MQLv2Executor(
            memory_manager=memory_manager,
            retriever=retriever,
            consolidation_engine=consolidation_engine,
            reflection_engine=reflection_engine,
            insight_engine=insight_engine,
            goal_tracker=goal_tracker,
            graph=graph,
            graph_reasoner=graph_reasoner,
            cognitive_query_engine=cognitive_query_engine,
        )
        # v0.3.2 fallback
        from reflection.mql import MQLEngine as _MQLv1
        self._v1 = _MQLv1(
            goal_tracker=goal_tracker,
            project_manager=project_manager,
            project_intelligence=project_intelligence,
            memory_manager=memory_manager,
            retriever=retriever,
            reflection_engine=reflection_engine,
            consolidation_engine=consolidation_engine,
            semantic_cluster_index=semantic_cluster_index,
            contradiction_detector=contradiction_detector,
            lifecycle_manager=lifecycle_manager,
        )

    def run(self, command: str) -> MQLv2Result:
        """
        Execute an MQL command (expression syntax or "show ..." fallback).
        """
        # 1. Try v0.3.4 expression parser
        parsed = self._parser.parse(command)
        if parsed is not None:
            handler, args = parsed
            return self._executor.execute(handler, args)

        # 2. Fall back to v0.3.2 "show ..." MQLEngine
        v1_result = self._v1.run(command)
        return MQLv2Result(
            command=command,
            matched=v1_result.matched,
            text=v1_result.text,
            data=v1_result.data if isinstance(v1_result.data, list) else [],
        )

    def is_mql_command(self, text: str) -> bool:
        """Quick check: looks like an MQL command."""
        t = text.strip().lower()
        return (
            t.startswith("show ")
            or t.startswith("memories ")
            or t.startswith("facts ")
            or t.startswith("insights ")
            or t.startswith("goals ")
            or t.startswith("graph ")
            or t.startswith("query ")
            or t.startswith("infer ")
            or t.startswith("multihop ")
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unavail(raw: str, component: str) -> MQLv2Result:
    return MQLv2Result(command=raw, matched=True,
                       text=f"This query requires {component}, which is not configured.")


def _cmp(value: float, op: str, threshold: float) -> bool:
    return {">": value > threshold, ">=": value >= threshold,
            "<": value < threshold, "<=": value <= threshold,
            "=": abs(value - threshold) < 1e-9}.get(op, False)


def _format_memories(raw: str, memories: list, label: str) -> MQLv2Result:
    if not memories:
        return MQLv2Result(command=raw, matched=True, text=f"No {label}.", data=[])
    lines = [f"{label} ({len(memories)}):"]
    for m in memories[:10]:
        preview = getattr(m, "output", "")[:80].replace("\n", " ")
        lines.append(f"  [{getattr(m, 'id')}] {preview}")
    data = [{"id": m.id, "input": m.input[:60], "topics": list(m.topics)} for m in memories]
    return MQLv2Result(command=raw, matched=True, text="\n".join(lines), data=data)


def _format_facts(raw: str, facts: list, label: str) -> MQLv2Result:
    if not facts:
        return MQLv2Result(command=raw, matched=True, text=f"No {label}.", data=[])
    lines = [f"{label} ({len(facts)}):"]
    for f in facts[:10]:
        lines.append(f"  [{f.fact_id}] {f.fact} (conf={f.confidence:.2f}, n={f.evidence_count})")
    return MQLv2Result(command=raw, matched=True, text="\n".join(lines), data=[f.to_dict() for f in facts])


def _format_insights(raw: str, insights: list, label: str) -> MQLv2Result:
    if not insights:
        return MQLv2Result(command=raw, matched=True, text=f"No {label}.", data=[])
    lines = [f"{label} ({len(insights)}):"]
    for i in insights:
        lines.append(f"  ({getattr(i, 'confidence', 0):.2f}) {getattr(i, 'insight', str(i))}")
    return MQLv2Result(command=raw, matched=True, text="\n".join(lines), data=[])


def _format_actionable_insights(raw: str, insights: list, label: str) -> MQLv2Result:
    if not insights:
        return MQLv2Result(command=raw, matched=True, text=f"No {label}.", data=[])
    lines = [f"{label} ({len(insights)}):"]
    for i in insights:
        lines.append(f"  [{i.category.value}] ({i.confidence:.2f}) {i.insight}")
        if i.recommendation:
            lines.append(f"    → {i.recommendation}")
    return MQLv2Result(command=raw, matched=True, text="\n".join(lines), data=[i.to_dict() for i in insights])


def _format_goals(raw: str, goals: list, label: str) -> MQLv2Result:
    if not goals:
        return MQLv2Result(command=raw, matched=True, text=f"No {label}.", data=[])
    lines = [f"{label} ({len(goals)}):"]
    for g in goals:
        lines.append(f"  [{g.status.value}] {g.title} ({g.progress}%, priority={g.priority})")
    return MQLv2Result(command=raw, matched=True, text="\n".join(lines),
                       data=[g.to_summary_dict() for g in goals])
