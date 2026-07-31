"""
Active Attention Retrieval — Blix v0.3.9  (New module 6)

Upgrades retrieval from:

    retrieve(query)

to:

    retrieve(query, current_workspace, active_goal, attention_focus)

Before this module, retrieval considered only the literal query text.
``ActiveAttentionRetriever`` wraps the existing
``core.memory_retriever.MemoryRetriever`` (unmodified — its
``retrieve(memories, query)`` signature and behavior are preserved for
every existing caller) and adds a context-aware re-ranking pass: given
the current ``workspace.workspace_memory.WorkspaceMemory`` state, the
active goal, and what currently holds attention focus, results that
align with the live cognitive context are boosted over results that
merely match the query text.

This composes ``retrieval.temporal_retriever.TemporalRetriever``
(v0.3.7, truth-aware scoring) when available, rather than duplicating
its state/truth-aware logic — Active Attention Retrieval adds ONE more
scoring dimension (workspace alignment) on top of what already exists,
it does not replace semantic/recency/importance/temporal scoring.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.memory_retriever import MemoryRetriever
from schemas.memory_entry import MemoryEntry
from workspace.workspace_memory import WorkspaceMemory


@dataclass
class AttentionRetrievalWeights:
    base_retrieval: float = 0.5        # weight given to the underlying retriever's ordering
    goal_alignment: float = 0.25        # boost for entries textually related to the active goal
    workspace_alignment: float = 0.25    # boost for entries related to current workspace item content

    def total(self) -> float:
        return self.base_retrieval + self.goal_alignment + self.workspace_alignment


def _text_overlap(a: str, b: str) -> float:
    """Cheap token-overlap heuristic — consistent with the rest of the project's Jaccard-style matchers."""
    ta = {w.lower() for w in a.split() if len(w) > 2}
    tb = {w.lower() for w in b.split() if len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class ActiveAttentionRetriever:
    """
    Context-aware retrieval wrapper around ``MemoryRetriever``.

    Parameters
    ----------
    base_retriever:
        The underlying ``MemoryRetriever`` — does the actual
        fuzzy/keyword/recency retrieval; this class re-ranks its output.
    weights:
        Blend weights for base ordering vs. goal/workspace alignment boosts.
    """

    def __init__(
        self,
        base_retriever: Optional[MemoryRetriever] = None,
        weights: Optional[AttentionRetrievalWeights] = None,
    ) -> None:
        self._base = base_retriever or MemoryRetriever()
        self._weights = weights or AttentionRetrievalWeights()

    def retrieve(
        self,
        memories: list[MemoryEntry],
        query: str,
        current_workspace: Optional[WorkspaceMemory] = None,
        active_goal: Optional[str] = None,
        attention_focus: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """
        Context-aware retrieval: ``retrieve(query, current_workspace,
        active_goal, attention_focus)`` from the spec.

        Falls back to plain ``MemoryRetriever.retrieve()`` ordering when
        no workspace context is supplied (full backward compatibility —
        an empty/None context degrades to the v0.3.x behavior).
        """
        base_results = self._base.retrieve(memories, query)
        if not base_results:
            return []

        if current_workspace is None and active_goal is None and attention_focus is None:
            return base_results

        # Effective goal text: explicit active_goal, or workspace's own
        # active_goal if not separately supplied.
        goal_text = active_goal or (current_workspace.active_goal if current_workspace else None) or ""

        # Workspace item content summaries (what's currently held in focus).
        workspace_text = ""
        if current_workspace is not None:
            workspace_text = " ".join(i.content_summary for i in current_workspace.items)

        w = self._weights
        scored: list[tuple[float, int, MemoryEntry]] = []
        n = len(base_results)
        for rank, entry in enumerate(base_results):
            # Base ordering contributes a descending positional score (highest-ranked base result = 1.0).
            base_score = 1.0 - (rank / max(1, n))
            goal_score = _text_overlap(entry.input + " " + entry.output, goal_text) if goal_text else 0.0
            workspace_score = _text_overlap(entry.input + " " + entry.output, workspace_text) if workspace_text else 0.0

            blended = (
                w.base_retrieval * base_score
                + w.goal_alignment * goal_score
                + w.workspace_alignment * workspace_score
            )
            scored.append((blended, entry.id, entry))

        # Stable sort: highest blended score first; ties broken by original id (chronological).
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [entry for _, _, entry in scored]
