"""
ADMA Memory Domains — v0.3.16

SystemMemory stores how Blix operates:
  reasoning strategies, planner configs, benchmark history,
  successful workflows, failure patterns, API knowledge,
  architecture documentation, algorithms.

UserMemory stores how Blix behaves for one user:
  preferences, projects, learning progress, coding style,
  interaction history, corrections, goals, accepted/rejected suggestions.

Both are thin wrappers over HGSHM that add domain-specific
convenience methods and tag all nodes with their domain.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_node import (
    MemoryNode, MemoryType, HierarchyLevel, EpistemicStatus
)
from memory.hybrid.models.memory_context import MemoryContext

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# SystemMemory
# ────────────────────────────────────────────────────────────────────

class SystemMemory:
    """
    Operational knowledge store for Blix itself.

    All nodes are tagged with ["system_memory"] and stored at
    HierarchyLevel.SESSION or above to distinguish them from
    raw user-interaction memories.

    Examples of what lives here:
      • Successful reasoning workflows
      • Benchmark result history
      • Planner performance traces
      • Known failure patterns and their resolutions
      • API and algorithm reference knowledge
      • Architecture documentation summaries
    """

    DOMAIN_TAG = "system_memory"

    def __init__(self, hgshm: HGSHM) -> None:
        self._hgshm = hgshm
        log.debug("SystemMemory initialised")

    # ── Storage ───────────────────────────────────────────────────────

    def store_workflow(
        self,
        description: str,
        success: bool,
        latency_ms: float = 0.0,
        subsystems: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record a successful or failed reasoning workflow."""
        status = "SUCCESS" if success else "FAILURE"
        text = f"[{status}] {description}"
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.EPISODE,
            hierarchy_level=HierarchyLevel.SESSION,
            confidence=0.9 if success else 0.4,
            importance=0.7 if success else 0.5,
            source="system",
            tags=[self.DOMAIN_TAG, "workflow", status.lower()] + (subsystems or []),
            metadata={
                "success": success,
                "latency_ms": latency_ms,
                "subsystems": subsystems or [],
                **(metadata or {}),
            },
        )

    def store_benchmark_result(
        self,
        benchmark_name: str,
        score: float,
        n_cases: int,
        version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record a benchmark run result."""
        text = (f"Benchmark {benchmark_name}: score={score:.4f}, "
                f"cases={n_cases}" + (f", v{version}" if version else ""))
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.FACT,
            hierarchy_level=HierarchyLevel.PROJECT,
            confidence=1.0,
            importance=0.6,
            source="benchmark_runner",
            tags=[self.DOMAIN_TAG, "benchmark", benchmark_name],
            metadata={"score": score, "n_cases": n_cases,
                      "version": version, **(metadata or {})},
        )

    def store_failure_pattern(
        self,
        pattern: str,
        resolution: str = "",
        frequency: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record a failure pattern and its resolution."""
        text = f"Failure pattern: {pattern}"
        if resolution:
            text += f" → Resolution: {resolution}"
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.CAUSE,
            hierarchy_level=HierarchyLevel.SESSION,
            confidence=0.8,
            importance=min(1.0, 0.5 + frequency * 0.1),
            source="failure_memory",
            tags=[self.DOMAIN_TAG, "failure_pattern"],
            metadata={"pattern": pattern, "resolution": resolution,
                      "frequency": frequency, **(metadata or {})},
        )

    def store_principle(
        self,
        statement: str,
        confidence: float = 0.85,
        derived_from: str = "",
    ) -> MemoryNode:
        """Store an operational principle (how Blix should behave)."""
        return self._hgshm.remember(
            text=statement,
            memory_type=MemoryType.PRINCIPLE,
            hierarchy_level=HierarchyLevel.KNOWLEDGE,
            confidence=confidence,
            importance=0.85,
            source="system",
            tags=[self.DOMAIN_TAG, "principle"],
            metadata={"derived_from": derived_from},
        )

    def store_api_knowledge(
        self, topic: str, content: str,
        source: str = "documentation",
    ) -> MemoryNode:
        """Store API or algorithm reference knowledge."""
        return self._hgshm.remember(
            text=f"[API] {topic}: {content}",
            memory_type=MemoryType.FACT,
            hierarchy_level=HierarchyLevel.KNOWLEDGE,
            confidence=0.95,
            importance=0.7,
            source=source,
            tags=[self.DOMAIN_TAG, "api", topic],
        )

    # ── Retrieval ────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 10) -> MemoryContext:
        """Retrieve system knowledge relevant to query."""
        # Filter to system_memory tagged nodes
        ctx = self._hgshm.recall(query, top_k=top_k, context_hint="system operational")
        return ctx

    def recent_failures(self, top_k: int = 10) -> list[MemoryNode]:
        """Return the most recent failure pattern nodes — indexed tag query."""
        nodes = self._hgshm.nodes_by_tags(
            required_tags=[self.DOMAIN_TAG, "failure_pattern"],
            memory_type=MemoryType.CAUSE,
            limit=top_k * 2,
            order_by="updated_at",
        )
        return nodes[:top_k]

    def benchmark_history(
        self, benchmark_name: str | None = None, limit: int = 20
    ) -> list[MemoryNode]:
        """Return benchmark result nodes — indexed tag query."""
        required = [self.DOMAIN_TAG, "benchmark"]
        if benchmark_name:
            required.append(benchmark_name)
        nodes = self._hgshm.nodes_by_tags(
            required_tags=required,
            memory_type=MemoryType.FACT,
            limit=limit,
            order_by="updated_at",
        )
        return nodes[:limit]

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """
        Return {memory_type: count} for system_memory nodes.

        Uses a single GROUP BY query — O(1) instead of O(n) full table
        scan followed by Python-side filtering.  (ISSUE-008)
        """
        by_type = self._hgshm.stats_by_tag(self.DOMAIN_TAG)
        total = sum(by_type.values())
        return {"total": total, **by_type}


# ────────────────────────────────────────────────────────────────────
# UserMemory
# ────────────────────────────────────────────────────────────────────
