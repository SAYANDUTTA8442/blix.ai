"""
Memory Query Language (MQL) — Blix v0.3.2  (Feature 10)

Lets users inspect Blix's memory directly with simple natural-language
commands, e.g.:

    show active goals
    show project blix
    show memories about transformers
    show reflections this week
    show strongest skills
    show contradictions
    show project risks

Design
------
``MQLParser`` matches a small set of regex patterns against the input
and dispatches to ``MQLExecutor``, which calls into the relevant v0.3 /
v0.3.1 / v0.3.2 components (all optional — missing components yield a
helpful "not available" message rather than an error).

Results are returned as ``MQLResult`` (structured data + a rendered
text block) so the CLI can display them directly.

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
class MQLResult:
    """Result of executing one MQL command."""

    command: str
    matched: bool
    text: str
    data: list = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Command patterns
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("active_goals", re.compile(r"^show\s+active\s+goals$", re.I)),
    ("all_goals", re.compile(r"^show\s+(?:all\s+)?goals$", re.I)),
    ("project", re.compile(r"^show\s+project\s+(.+)$", re.I)),
    ("project_risks", re.compile(r"^show\s+project\s+risks$", re.I)),
    ("memories_about", re.compile(r"^show\s+memories\s+about\s+(.+)$", re.I)),
    ("reflections_period", re.compile(r"^show\s+reflections\s+(this\s+week|this\s+month|today)$", re.I)),
    ("reflections_all", re.compile(r"^show\s+reflections$", re.I)),
    ("strongest_skills", re.compile(r"^show\s+strongest\s+skills$", re.I)),
    ("strongest_facts", re.compile(r"^show\s+(?:strongest\s+)?facts$", re.I)),
    ("contradictions", re.compile(r"^show\s+contradictions$", re.I)),
    ("clusters", re.compile(r"^show\s+(?:topic\s+)?clusters$", re.I)),
    ("lifecycle", re.compile(r"^show\s+(?:memory\s+)?lifecycle$", re.I)),
]

# Note: more specific patterns (e.g. "show project risks") must be checked
# before more general ones (e.g. "show project <name>"); ordering above
# handles this since "risks" won't match the capture-group pattern's
# greedy "(.+)" due to checking project_risks first in _PATTERNS order
# during matching in MQLParser.parse().


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class MQLParser:
    """Parses raw MQL command strings into (command_type, args)."""

    # Order matters: more specific patterns first
    _ORDER = [
        "project_risks", "active_goals", "all_goals", "project",
        "reflections_period", "reflections_all",
        "memories_about", "strongest_skills", "strongest_facts",
        "contradictions", "clusters", "lifecycle",
    ]

    def parse(self, command: str) -> Optional[tuple[str, dict]]:
        """
        Parse a command string.

        Returns ``(command_type, args)`` or ``None`` if no pattern matches.
        """
        text = command.strip()
        pattern_map = dict(_PATTERNS)
        for name in self._ORDER:
            pattern = pattern_map[name]
            m = pattern.match(text)
            if m:
                args = {}
                if name == "project" and m.groups():
                    args["project_name"] = m.group(1).strip()
                elif name == "memories_about" and m.groups():
                    args["topic"] = m.group(1).strip()
                elif name == "reflections_period" and m.groups():
                    args["period"] = m.group(1).strip().lower()
                return name, args
        return None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MQLExecutor:
    """
    Executes parsed MQL commands against Blix's components.

    All components are optional — if a component required by a command
    is not provided, the result explains what's missing rather than
    raising.

    Parameters
    ----------
    goal_tracker, project_manager, project_intelligence, memory_manager,
    retriever, reflection_engine, consolidation_engine, semantic_cluster_index,
    contradiction_detector, lifecycle_manager:
        Optional component instances. Pass whichever are available.
    """

    def __init__(
        self,
        goal_tracker: Optional[object] = None,
        project_manager: Optional[object] = None,
        project_intelligence: Optional[object] = None,
        memory_manager: Optional[object] = None,
        retriever: Optional[object] = None,
        reflection_engine: Optional[object] = None,
        consolidation_engine: Optional[object] = None,
        semantic_cluster_index: Optional[object] = None,
        contradiction_detector: Optional[object] = None,
        lifecycle_manager: Optional[object] = None,
    ) -> None:
        self._goals = goal_tracker
        self._pm = project_manager
        self._pi = project_intelligence
        self._mm = memory_manager
        self._retriever = retriever
        self._reflection = reflection_engine
        self._consolidation = consolidation_engine
        self._clusters = semantic_cluster_index
        self._contradictions = contradiction_detector
        self._lifecycle = lifecycle_manager

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, command_type: str, args: dict) -> MQLResult:
        handler = getattr(self, f"_cmd_{command_type}", None)
        if handler is None:
            return MQLResult(command=command_type, matched=False, text=f"Unknown command: {command_type}")
        return handler(args)

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

    def _cmd_active_goals(self, args: dict) -> MQLResult:
        if self._goals is None:
            return _unavailable("active_goals", "GoalTracker")
        from reflection.goal_tracker import GoalStatus
        goals = self._goals.list_goals(status=GoalStatus.ACTIVE)  # type: ignore[union-attr]
        if not goals:
            return MQLResult("active_goals", True, "No active goals.", [])
        lines = ["Active goals:"]
        for g in goals:
            blockers = ", ".join(b.description for b in g.active_blockers) or "none"
            lines.append(f"  - {g.title} ({g.progress}%) — blockers: {blockers}")
        return MQLResult("active_goals", True, "\n".join(lines), [g.to_summary_dict() for g in goals])

    def _cmd_all_goals(self, args: dict) -> MQLResult:
        if self._goals is None:
            return _unavailable("all_goals", "GoalTracker")
        goals = self._goals.list_goals()  # type: ignore[union-attr]
        if not goals:
            return MQLResult("all_goals", True, "No goals tracked.", [])
        lines = ["All goals:"]
        for g in goals:
            lines.append(f"  - [{g.status.value:10s}] {g.title} ({g.progress}%)")
        return MQLResult("all_goals", True, "\n".join(lines), [g.to_summary_dict() for g in goals])

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def _cmd_project(self, args: dict) -> MQLResult:
        name = args.get("project_name", "")
        if self._pi is not None:
            report = self._pi.project_report(name)  # type: ignore[union-attr]
            lines = [f"Project: {report.get('project', name)}"]
            if "status" in report:
                lines.append(f"  Status: {report['status']}")
            lines.append(f"  Focus: {report.get('focus', '—')}")
            lines.append(f"  Progress: {report.get('progress', 0)}%")
            lines.append(f"  Risk level: {report.get('risk_level', 'low')}")
            if report.get("risks"):
                lines.append(f"  Risks: {', '.join(report['risks'])}")
            if report.get("next_steps"):
                lines.append(f"  Next steps: {', '.join(report['next_steps'])}")
            if report.get("goals"):
                lines.append(f"  Goals: {', '.join(report['goals'])}")
            return MQLResult("project", True, "\n".join(lines), [report])

        if self._pm is not None:
            summary = self._pm.get(name)  # type: ignore[union-attr]
            if summary is None:
                return MQLResult("project", True, f"No project named '{name}'.", [])
            lines = [
                f"Project: {summary.project_name}",
                f"  Status: {summary.current_status}",
                f"  Goals: {', '.join(summary.goals) or '—'}",
                f"  Next actions: {', '.join(summary.next_actions) or '—'}",
            ]
            return MQLResult("project", True, "\n".join(lines), [summary.to_summary_dict() if hasattr(summary, "to_summary_dict") else {}])

        return _unavailable("project", "ProjectManager / ProjectIntelligenceEngine")

    def _cmd_project_risks(self, args: dict) -> MQLResult:
        if self._pi is None:
            return _unavailable("project_risks", "ProjectIntelligenceEngine")
        at_risk = self._pi.at_risk_projects()  # type: ignore[union-attr]
        if not at_risk:
            return MQLResult("project_risks", True, "No projects currently at risk.", [])
        lines = ["Projects at risk:"]
        for p in at_risk:
            lines.append(f"  - {p.project_name} [{p.risk_level.value}]: {', '.join(p.risks)}")
        return MQLResult("project_risks", True, "\n".join(lines), [p.to_summary_dict() for p in at_risk])

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def _cmd_memories_about(self, args: dict) -> MQLResult:
        topic = args.get("topic", "")
        if self._retriever is None or self._mm is None:
            return _unavailable("memories_about", "SemanticRetriever + MemoryManager")
        memories = self._retriever.retrieve(self._mm.get_all_memories(), topic)  # type: ignore[union-attr]
        if not memories:
            return MQLResult("memories_about", True, f"No memories found about '{topic}'.", [])
        lines = [f"Memories about '{topic}':"]
        for m in memories[:10]:
            preview = (getattr(m, "output", "")[:80]).replace("\n", " ")
            lines.append(f"  - [{getattr(m, 'id')}] {preview}")
        return MQLResult("memories_about", True, "\n".join(lines), [m.id for m in memories])

    # ------------------------------------------------------------------
    # Reflections
    # ------------------------------------------------------------------

    def _cmd_reflections_period(self, args: dict) -> MQLResult:
        if self._reflection is None:
            return _unavailable("reflections_period", "ReflectionEngine")
        period = args.get("period", "this week")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if period == "today":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "this month":
            since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # "this week"
            since = now - timedelta(days=now.weekday())
            since = since.replace(hour=0, minute=0, second=0, microsecond=0)

        insights = self._reflection.get_insights_since(since)  # type: ignore[union-attr]
        if not insights:
            return MQLResult("reflections_period", True, f"No reflections {period}.", [])
        lines = [f"Reflections {period}:"]
        for ins in insights:
            lines.append(f"  - ({ins.confidence:.2f}) {ins.insight}")
        return MQLResult("reflections_period", True, "\n".join(lines), [i.to_dict() for i in insights])

    def _cmd_reflections_all(self, args: dict) -> MQLResult:
        if self._reflection is None:
            return _unavailable("reflections_all", "ReflectionEngine")
        insights = self._reflection.get_recent_insights(limit=10)  # type: ignore[union-attr]
        if not insights:
            return MQLResult("reflections_all", True, "No reflections recorded yet.", [])
        lines = ["Recent reflections:"]
        for ins in insights:
            lines.append(f"  - [{ins.scope.value}] ({ins.confidence:.2f}) {ins.insight}")
        return MQLResult("reflections_all", True, "\n".join(lines), [i.to_dict() for i in insights])

    # ------------------------------------------------------------------
    # Skills / facts
    # ------------------------------------------------------------------

    def _cmd_strongest_skills(self, args: dict) -> MQLResult:
        if self._consolidation is None:
            return _unavailable("strongest_skills", "ConsolidationEngine")
        facts = [f for f in self._consolidation.strongest_facts(20)  # type: ignore[union-attr]
                 if "skill" in f.topic.lower() or "prefer" in f.fact.lower()]
        if not facts:
            facts = self._consolidation.strongest_facts(5)  # type: ignore[union-attr]
        if not facts:
            return MQLResult("strongest_skills", True, "No consolidated facts yet.", [])
        lines = ["Strongest skills/facts:"]
        for f in facts[:5]:
            lines.append(f"  - {f.fact} (confidence={f.confidence:.2f}, n={f.evidence_count})")
        return MQLResult("strongest_skills", True, "\n".join(lines), [f.to_dict() for f in facts[:5]])

    def _cmd_strongest_facts(self, args: dict) -> MQLResult:
        if self._consolidation is None:
            return _unavailable("strongest_facts", "ConsolidationEngine")
        facts = self._consolidation.strongest_facts(10)  # type: ignore[union-attr]
        if not facts:
            return MQLResult("strongest_facts", True, "No consolidated facts yet.", [])
        lines = ["Strongest facts:"]
        for f in facts:
            lines.append(f"  - {f.fact} (confidence={f.confidence:.2f}, n={f.evidence_count})")
        return MQLResult("strongest_facts", True, "\n".join(lines), [f.to_dict() for f in facts])

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def _cmd_contradictions(self, args: dict) -> MQLResult:
        if self._contradictions is None:
            return _unavailable("contradictions", "ContradictionDetector")
        unresolved = self._contradictions.get_contradictions(resolved=False)  # type: ignore[union-attr]
        if not unresolved:
            return MQLResult("contradictions", True, "No unresolved contradictions.", [])
        lines = ["Unresolved contradictions:"]
        for c in unresolved:
            lines.append(f"  - [{c.field}] memory {c.memory_a_id} vs {c.memory_b_id} (severity={c.severity:.2f})")
        return MQLResult("contradictions", True, "\n".join(lines), unresolved)

    # ------------------------------------------------------------------
    # Clusters / lifecycle
    # ------------------------------------------------------------------

    def _cmd_clusters(self, args: dict) -> MQLResult:
        if self._clusters is None:
            return _unavailable("clusters", "SemanticClusterIndex")
        clusters = self._clusters.list_clusters()  # type: ignore[union-attr]
        if not clusters:
            return MQLResult("clusters", True, "No topic clusters yet.", [])
        lines = ["Topic clusters:"]
        for c in clusters[:10]:
            lines.append(f"  - {c.label} ({len(c.member_ids)} memories)")
        return MQLResult("clusters", True, "\n".join(lines), [c.to_dict() for c in clusters])

    def _cmd_lifecycle(self, args: dict) -> MQLResult:
        if self._lifecycle is None:
            return _unavailable("lifecycle", "MemoryLifecycleManager")
        counts = self._lifecycle.state_counts()  # type: ignore[union-attr]
        lines = ["Memory lifecycle state:"]
        for state, n in counts.items():
            lines.append(f"  - {state}: {n}")
        return MQLResult("lifecycle", True, "\n".join(lines), [counts])


# ---------------------------------------------------------------------------
# Combined facade
# ---------------------------------------------------------------------------


class MQLEngine:
    """
    Combines ``MQLParser`` + ``MQLExecutor`` into a single ``run()`` call.

    Usage
    -----
        engine = MQLEngine(goal_tracker=gt, project_intelligence=pi, ...)
        result = engine.run("show active goals")
        print(result.text)
    """

    def __init__(self, **components: object) -> None:
        self._parser = MQLParser()
        self._executor = MQLExecutor(**components)

    def run(self, command: str) -> MQLResult:
        parsed = self._parser.parse(command)
        if parsed is None:
            return MQLResult(
                command=command, matched=False,
                text=(
                    f"Unrecognised MQL command: {command!r}\n"
                    "Try: show active goals | show project <name> | "
                    "show memories about <topic> | show reflections this week | "
                    "show strongest skills | show contradictions | show project risks"
                ),
            )
        command_type, args = parsed
        return self._executor.execute(command_type, args)

    def is_mql_command(self, text: str) -> bool:
        """Quick check: does this look like an MQL command (starts with 'show')?"""
        return text.strip().lower().startswith("show ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unavailable(command: str, component: str) -> MQLResult:
    return MQLResult(
        command=command, matched=True,
        text=f"'{command}' requires {component}, which is not configured.",
    )
