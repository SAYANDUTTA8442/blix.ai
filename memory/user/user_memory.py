"""
UserMemory — personalisation store for one user (v0.3.16).

All nodes are tagged with ["user_memory", user_id].
"""
from __future__ import annotations
import logging
from typing import Any

from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_node import (
    MemoryNode, MemoryType, HierarchyLevel, EpistemicStatus
)
from memory.hybrid.models.memory_context import MemoryContext

log = logging.getLogger(__name__)


class UserMemory:
    """
    Personalisation store for one user.

    All nodes are tagged with ["user_memory", user_id] to ensure
    complete isolation between users.

    Examples of what lives here:
      • User preferences (coding language, verbosity, topics)
      • Project context (current project, goals, progress)
      • Interaction history (what was asked, accepted, corrected)
      • Learning progress (what has been understood, what is still unclear)
      • Goals (long-term objectives, milestones)
    """

    DOMAIN_TAG = "user_memory"

    def __init__(self, hgshm: HGSHM, user_id: str) -> None:
        self._hgshm = hgshm
        self.user_id = user_id
        self._user_tag = f"user:{user_id}"
        log.debug("UserMemory initialised for user=%s", user_id)

    def _user_tags(self, extra: list[str] | None = None) -> list[str]:
        return [self.DOMAIN_TAG, self._user_tag] + (extra or [])

    # ── Storage ───────────────────────────────────────────────────────

    def store_preference(
        self,
        category: str,
        preference: str,
        strength: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record a user preference."""
        text = f"User {self.user_id} prefers [{category}]: {preference}"
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.BELIEF,
            hierarchy_level=HierarchyLevel.SESSION,
            confidence=strength,
            importance=0.8,
            source="user_interaction",
            tags=self._user_tags(["preference", category]),
            metadata={"category": category, "preference": preference,
                      "user_id": self.user_id, **(metadata or {})},
        )

    def store_interaction(
        self,
        query: str,
        response_accepted: bool,
        correction: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record one interaction turn."""
        status = "accepted" if response_accepted else "corrected"
        text = f"[{status.upper()}] Query: {query[:100]}"
        if correction:
            text += f" | Correction: {correction[:80]}"
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.EPISODE,
            hierarchy_level=HierarchyLevel.EPISODE,
            confidence=0.9,
            importance=0.6 if response_accepted else 0.9,  # corrections are important
            source="user_interaction",
            tags=self._user_tags(["interaction", status]),
            metadata={"accepted": response_accepted, "correction": correction,
                      "user_id": self.user_id, **(metadata or {})},
        )

    def store_goal(
        self,
        goal: str,
        priority: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record a long-term user goal."""
        return self._hgshm.remember(
            text=f"User goal: {goal}",
            memory_type=MemoryType.GOAL,
            hierarchy_level=HierarchyLevel.PROJECT,
            confidence=1.0,
            importance=priority,
            source="user_interaction",
            tags=self._user_tags(["goal"]),
            metadata={"goal": goal, "priority": priority,
                      "user_id": self.user_id, **(metadata or {})},
        )

    def store_learning_progress(
        self,
        topic: str,
        understood: bool,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record what the user has understood or still finds unclear."""
        status = "understood" if understood else "still unclear"
        text = f"User {self.user_id} [{status}]: {topic}"
        if notes:
            text += f" — {notes}"
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.BELIEF,
            hierarchy_level=HierarchyLevel.SESSION,
            confidence=0.85,
            importance=0.6 if understood else 0.9,
            source="learning_tracker",
            tags=self._user_tags(["learning", status.replace(" ", "_")]),
            metadata={"topic": topic, "understood": understood,
                      "user_id": self.user_id, **(metadata or {})},
        )

    def store_project(
        self,
        name: str,
        description: str,
        stack: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record a user project with its tech stack."""
        text = f"Project: {name} — {description}"
        if stack:
            text += f" | Stack: {', '.join(stack)}"
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.FACT,
            hierarchy_level=HierarchyLevel.PROJECT,
            confidence=1.0,
            importance=0.85,
            source="user_interaction",
            tags=self._user_tags(["project"] + (stack or [])),
            metadata={"name": name, "description": description,
                      "stack": stack or [], "user_id": self.user_id,
                      **(metadata or {})},
        )

    def record_correction(
        self,
        original: str,
        correction: str,
        severity: float = 0.5,
    ) -> MemoryNode:
        """Record that the user corrected Blix's response."""
        text = f"Correction by {self.user_id}: {correction[:100]} (was: {original[:60]})"
        return self._hgshm.remember(
            text=text,
            memory_type=MemoryType.EPISODE,
            hierarchy_level=HierarchyLevel.EPISODE,
            confidence=0.95,
            importance=min(1.0, 0.6 + severity * 0.4),
            source="correction_tracker",
            tags=self._user_tags(["correction"]),
            metadata={"original": original, "correction": correction,
                      "severity": severity, "user_id": self.user_id},
        )

    # ── Retrieval ────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 10) -> MemoryContext:
        """Retrieve personalisation context relevant to query."""
        ctx = self._hgshm.recall(
            query, top_k=top_k, context_hint=f"user {self.user_id} personalisation")
        return ctx

    def preferences(self, category: str | None = None) -> list[MemoryNode]:
        """Return preference nodes for this user — indexed tag query."""
        required = [self.DOMAIN_TAG, self._user_tag, "preference"]
        if category:
            required.append(category)
        return self._hgshm.nodes_by_tags(
            required_tags=required,
            memory_type=MemoryType.BELIEF,
            limit=200,
            order_by="importance",
        )

    def goals(self) -> list[MemoryNode]:
        """Return goal nodes for this user — indexed tag query."""
        return self._hgshm.nodes_by_tags(
            required_tags=[self.DOMAIN_TAG, self._user_tag],
            memory_type=MemoryType.GOAL,
            limit=100,
            order_by="importance",
        )

    def corrections(self, limit: int = 20) -> list[MemoryNode]:
        """Return most recent correction nodes — indexed tag query."""
        return self._hgshm.nodes_by_tags(
            required_tags=[self.DOMAIN_TAG, self._user_tag, "correction"],
            memory_type=MemoryType.EPISODE,
            limit=limit,
            order_by="updated_at",
        )

    def cold_start_profile(self) -> dict[str, Any]:
        """
        Return what's known about this user.
        Empty dict = cold start (no prior knowledge).
        """
        prefs = self.preferences()
        goals_list = self.goals()
        corr = self.corrections(limit=5)
        return {
            "user_id": self.user_id,
            "n_preferences": len(prefs),
            "n_goals": len(goals_list),
            "n_corrections": len(corr),
            "is_cold_start": len(prefs) == 0 and len(goals_list) == 0,
            "top_preferences": [
                {"category": n.metadata.get("category", ""),
                 "preference": n.metadata.get("preference", "")}
                for n in prefs[:3]
            ],
            "top_goals": [n.text for n in goals_list[:3]],
        }

    def stats(self) -> dict[str, int]:
        """
        Return {memory_type: count} for this user's nodes.

        Uses a single GROUP BY query — O(1) instead of O(n) full table
        scan followed by Python-side filtering.  (ISSUE-008)
        """
        by_type = self._hgshm.stats_by_tag(self._user_tag)
        total = sum(by_type.values())
        return {"total": total, "user_id": self.user_id, **by_type}
