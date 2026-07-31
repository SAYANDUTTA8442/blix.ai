"""
MemoryManager — unified routing across System, User, and HGSHM domains.

Routes queries to the appropriate memory domain(s), merges results,
removes duplicates, and applies policy-driven ranking.

This is the single entry point for all memory access in v0.3.16.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_context import MemoryContext, RetrievedMemory
from memory.hybrid.models.memory_node import MemoryNode, MemoryType
from memory.system.system_memory import SystemMemory
from memory.user.user_memory import UserMemory

log = logging.getLogger(__name__)


@dataclass
class RoutedContext:
    """
    The merged output of MemoryManager.query().

    Combines system + user + general HGSHM results with routing metadata.
    """
    query:              str                   = ""
    system_context:     MemoryContext | None  = None
    user_context:       MemoryContext | None  = None
    general_context:    MemoryContext | None  = None
    merged_memories:    list[RetrievedMemory] = field(default_factory=list)
    routing_latency_ms: float                 = 0.0
    domains_queried:    list[str]             = field(default_factory=list)
    metadata:           dict[str, Any]        = field(default_factory=dict)

    @property
    def total_memories(self) -> int:
        return len(self.merged_memories)

    @property
    def top_memory(self) -> MemoryNode | None:
        if self.merged_memories:
            return self.merged_memories[0].node
        return None

    def to_memory_context(self) -> MemoryContext:
        """Convert to a flat MemoryContext for backward compatibility."""
        ctx = MemoryContext(query=self.query)
        ctx.primary_memories = self.merged_memories[:10]
        ctx.supporting_memories = self.merged_memories[10:20]
        if self.system_context:
            ctx.principle_nodes = self.system_context.principle_nodes
        if self.user_context:
            ctx.belief_nodes = self.user_context.belief_nodes
            ctx.knowledge_gaps = self.user_context.knowledge_gaps
        ctx.retrieval_latency_ms = self.routing_latency_ms
        return ctx


class MemoryManager:
    """
    Unified memory routing and merging for ADMA.

    Parameters
    ----------
    hgshm : HGSHM
        The shared HGSHM substrate.
    system_memory : SystemMemory
        Operational knowledge store.
    user_memory_cache : dict
        Cache of UserMemory instances by user_id.
    policy_selector : PolicySelector | None
        If set, uses policy-driven routing weights.
    """

    def __init__(
        self,
        hgshm: HGSHM,
        system_memory: SystemMemory,
        policy_selector: Any = None,  # policy.compiler.PolicySelector
    ) -> None:
        self._hgshm   = hgshm
        self._system  = system_memory
        self._users:  dict[str, UserMemory] = {}
        self._selector = policy_selector
        log.debug("MemoryManager initialised")

    def get_user_memory(self, user_id: str) -> UserMemory:
        """Get or create a UserMemory for the given user."""
        if user_id not in self._users:
            self._users[user_id] = UserMemory(self._hgshm, user_id)
        return self._users[user_id]

    # ── Primary query API ─────────────────────────────────────────────

    def query(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 10,
        include_system: bool = True,
        include_user: bool = True,
        include_general: bool = True,
        context: dict[str, Any] | None = None,
    ) -> RoutedContext:
        """
        Route a query across memory domains and return merged results.

        Parameters
        ----------
        query : str
            Natural-language query.
        user_id : str
            Current user (determines which UserMemory to query).
        top_k : int
            Results per domain.
        include_system : bool
            Whether to query SystemMemory.
        include_user : bool
            Whether to query UserMemory.
        include_general : bool
            Whether to query the general HGSHM (cross-domain).
        context : dict
            Additional routing context.
        """
        t0 = time.perf_counter()
        routed = RoutedContext(query=query)

        # ── Domain queries ────────────────────────────────────────────
        if include_system:
            try:
                routed.system_context = self._system.recall(query, top_k=top_k // 2)
                routed.domains_queried.append("system")
            except Exception as exc:
                log.warning("MemoryManager: system query failed: %s", exc)

        if include_user:
            try:
                user_mem = self.get_user_memory(user_id)
                routed.user_context = user_mem.recall(query, top_k=top_k // 2)
                routed.domains_queried.append("user")
            except Exception as exc:
                log.warning("MemoryManager: user query failed: %s", exc)

        if include_general:
            try:
                routed.general_context = self._hgshm.recall(query, top_k=top_k)
                routed.domains_queried.append("general")
            except Exception as exc:
                log.warning("MemoryManager: general query failed: %s", exc)

        # ── Merge and deduplicate ──────────────────────────────────────
        routed.merged_memories = self._merge(
            routed.system_context,
            routed.user_context,
            routed.general_context,
            top_k=top_k * 2,
        )

        routed.routing_latency_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "MemoryManager: query=%r domains=%s memories=%d latency=%.1fms",
            query[:40], routed.domains_queried,
            routed.total_memories, routed.routing_latency_ms)
        return routed

    def _merge(
        self,
        *contexts: MemoryContext | None,
        top_k: int = 20,
    ) -> list[RetrievedMemory]:
        """Merge RetrievedMemory lists, dedup by node_id, rank by final_score."""
        seen: dict[str, RetrievedMemory] = {}
        for ctx in contexts:
            if ctx is None:
                continue
            for rm in ctx.all_memories:
                nid = rm.node.node_id
                if nid not in seen or rm.final_score > seen[nid].final_score:
                    seen[nid] = rm
        merged = sorted(seen.values(), key=lambda r: r.final_score, reverse=True)
        return merged[:top_k]

    # ── Convenience writes ────────────────────────────────────────────

    def store_system(self, text: str, **kwargs) -> MemoryNode:
        return self._system.store_workflow(text, **kwargs)

    def store_user(self, user_id: str, text: str, **kwargs) -> MemoryNode:
        return self.get_user_memory(user_id).store_preference(
            category=kwargs.pop("category", "general"),
            preference=text,
            **kwargs)

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self, user_id: str = "default") -> dict[str, Any]:
        return {
            "hgshm":  self._hgshm.stats(),
            "system": self._system.stats(),
            "user":   self.get_user_memory(user_id).stats(),
        }
