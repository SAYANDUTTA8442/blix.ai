"""
Meta-Reflection — Blix v0.3.8  (New module 9)

Extends v0.3.2's ``reflection.reflection_engine.ReflectionEngine``
(Experience → Reflection) and v0.3.6's
``agents.plan_reflection.PlanReflection`` (single-plan post-mortems)
with one more layer: looking ACROSS many reflections/plan-reflections
to spot recurring patterns in HOW Blix operates, not just what
happened in one run.

    Experience → Reflection → Strategy Analysis → Behavior Change

Example (from the spec):

    Frequent replanning observed.
    Insight: Current planning strategy is too shallow.

``MetaReflectionEngine`` reads recent run-level summaries (dicts shaped
like ``AgentRunResult.to_dict()``) plus pattern-detection heuristics,
looks for recurring process issues (frequent replanning, frequently
low confidence, repeated tool bottlenecks), and emits
``BehaviorChangeInsight`` records — persisted into the v0.3.2
``ReflectionEngine`` under ``ReflectionScope.BEHAVIOR`` via its normal
``reflect()`` entry point, so behavior-change insights show up
alongside every other kind of reflective insight rather than living in
a separate, disconnected store.

Python 3.10 compatible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from reflection.reflection_engine import ReflectionEngine, ReflectionScope
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class BehaviorChangeInsight:
    """One pattern-level insight about how Blix's own process is behaving."""

    pattern: str               # e.g. "frequent_replanning"
    observation: str
    suggested_change: str
    occurrence_count: int = 1
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "observation": self.observation,
            "suggested_change": self.suggested_change,
            "occurrence_count": self.occurrence_count,
            "generated_at": self.generated_at,
        }

    def as_material(self) -> str:
        """Render as a short text blob suitable for feeding ReflectionEngine.reflect()."""
        return f"{self.observation} {self.suggested_change}"


# ---------------------------------------------------------------------------
# Meta-Reflection Engine
# ---------------------------------------------------------------------------

# Thresholds for pattern detection — deliberately simple/heuristic,
# matching the rest of the v0.3.x reflection modules' style.
_FREQUENT_REPLAN_THRESHOLD = 2          # mean replans/run at or above this is "frequent"
_LOW_CONFIDENCE_THRESHOLD = 0.5
_LOW_CONFIDENCE_RATE_THRESHOLD = 0.4    # fraction of runs below threshold to flag as a pattern
_REPEATED_TOOL_SWITCH_THRESHOLD = 3      # same tool appearing as a bottleneck this many times


class MetaReflectionEngine:
    """
    Looks across multiple run-level reflections for recurring process
    patterns and proposes behavior changes.

    Parameters
    ----------
    reflection_engine:
        v0.3.2 ``ReflectionEngine`` — behavior-change insights are
        persisted here under ``ReflectionScope.BEHAVIOR`` so they're
        visible through the same reflection API as everything else.
    """

    def __init__(self, reflection_engine: Optional[ReflectionEngine] = None) -> None:
        self._reflection_engine = reflection_engine

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def analyze_runs(
        self, run_summaries: list[dict], scope_ref: str = "agent_runs",
    ) -> list[BehaviorChangeInsight]:
        """
        Analyze a batch of run-level summaries (dicts with keys like
        ``replan_count``, ``agent_state`` (containing ``confidence``),
        ``plan_reflection`` (containing ``bottleneck_tool``)) for
        recurring process patterns.

        Each recognised pattern becomes one ``BehaviorChangeInsight``,
        persisted to the underlying ``ReflectionEngine`` if configured.
        """
        if not run_summaries:
            return []

        insights: list[BehaviorChangeInsight] = []

        replan_insight = self._check_frequent_replanning(run_summaries)
        if replan_insight:
            insights.append(replan_insight)

        confidence_insight = self._check_low_confidence_pattern(run_summaries)
        if confidence_insight:
            insights.append(confidence_insight)

        bottleneck_insight = self._check_repeated_tool_bottleneck(run_summaries)
        if bottleneck_insight:
            insights.append(bottleneck_insight)

        for insight in insights:
            self._persist(insight, scope_ref)

        return insights

    def _check_frequent_replanning(self, runs: list[dict]) -> Optional[BehaviorChangeInsight]:
        replan_counts = [r.get("replan_count", 0) for r in runs]
        if not replan_counts:
            return None
        mean_replans = sum(replan_counts) / len(replan_counts)
        if mean_replans < _FREQUENT_REPLAN_THRESHOLD:
            return None
        return BehaviorChangeInsight(
            pattern="frequent_replanning",
            observation=(
                f"Frequent replanning observed (mean {mean_replans:.1f} replans/run "
                f"across {len(runs)} run(s))."
            ),
            suggested_change=(
                "Current planning strategy is too shallow — consider decomposing plans "
                "further upfront or invoking the critic earlier."
            ),
            occurrence_count=sum(1 for c in replan_counts if c > 0),
        )

    def _check_low_confidence_pattern(self, runs: list[dict]) -> Optional[BehaviorChangeInsight]:
        confidences = []
        for r in runs:
            state = r.get("agent_state") or {}
            if "confidence" in state:
                confidences.append(state["confidence"])
        if not confidences:
            return None
        low_count = sum(1 for c in confidences if c < _LOW_CONFIDENCE_THRESHOLD)
        rate = low_count / len(confidences)
        if rate < _LOW_CONFIDENCE_RATE_THRESHOLD:
            return None
        return BehaviorChangeInsight(
            pattern="frequent_low_confidence",
            observation=(
                f"{low_count}/{len(confidences)} run(s) ended with confidence below "
                f"{_LOW_CONFIDENCE_THRESHOLD:.1f}."
            ),
            suggested_change=(
                "Confidence is consistently low — consider invoking the critic earlier "
                "in the loop or gathering more evidence before committing to a plan."
            ),
            occurrence_count=low_count,
        )

    def _check_repeated_tool_bottleneck(self, runs: list[dict]) -> Optional[BehaviorChangeInsight]:
        bottlenecks = []
        for r in runs:
            pr = r.get("plan_reflection") or {}
            tool = pr.get("bottleneck_tool")
            if tool:
                bottlenecks.append(tool)
        if not bottlenecks:
            return None
        counts = Counter(bottlenecks)
        tool, count = counts.most_common(1)[0]
        if count < _REPEATED_TOOL_SWITCH_THRESHOLD:
            return None
        return BehaviorChangeInsight(
            pattern="repeated_tool_bottleneck",
            observation=f"Tool '{tool}' was the reported bottleneck in {count} run(s).",
            suggested_change=(
                f"Consider deprioritising '{tool}' as a default tool choice, or addressing "
                "its underlying reliability issue directly."
            ),
            occurrence_count=count,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, insight: BehaviorChangeInsight, scope_ref: str) -> None:
        if self._reflection_engine is None:
            return
        self._reflection_engine.reflect(
            scope=ReflectionScope.BEHAVIOR,
            scope_ref=scope_ref,
            material=insight.as_material(),
        )
