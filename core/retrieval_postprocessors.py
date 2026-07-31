"""
Project-Biased & Diversity-Aware Retrieval — Blix v0.3.1  (Issues 8 & 13)

Addresses:
  Issue 8:  "Project memory is isolated — doesn't influence retrieval."
  Issue 13: "Retrieval has no diversity objective — top-K may be redundant."

Two retrieval post-processors:

1. ``ProjectBiasedRetriever``
   Boosts the retrieval score of memories linked to the active project.
   Integrates with ``ProjectManager`` and ``GraphReasoner``:
   - Memories directly linked to the active project via session ids → +bias
   - Memories linked via graph proximity → scaled +bias
   - All others → unchanged

2. ``MMRReranker`` (Maximal Marginal Relevance)
   Diversifies a candidate set to avoid redundant results.
   Classic MMR:
       MMR(d) = λ · relevance(d,q) − (1-λ) · max_{d'∈S} sim(d, d')
   where S is the already-selected set.

Both return a re-ordered list of MemoryEntry objects.
Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ===========================================================================
# Issue 8 — Project-Biased Retrieval
# ===========================================================================


class ProjectBiasedRetriever:
    """
    Boosts retrieval scores for memories related to the active project.

    Parameters
    ----------
    project_bias:
        Additive score bonus for directly linked memories (0–1).
    graph_bias_decay:
        Multiplier applied to graph-proximity-based bias per hop.
    max_graph_depth:
        Maximum graph hops to traverse for proximity bias.
    """

    def __init__(
        self,
        project_bias: float = 0.25,
        graph_bias_decay: float = 0.5,
        max_graph_depth: int = 2,
    ) -> None:
        self._bias = project_bias
        self._decay = graph_bias_decay
        self._max_depth = max_graph_depth

    def rerank(
        self,
        memories: list,                     # list[MemoryEntry]
        scores: dict[int, float],           # memory_id → current score
        active_project_name: Optional[str],
        project_manager: Optional[object],  # ProjectManager
        graph_reasoner: Optional[object],   # GraphReasoner
        graph: Optional[object] = None,     # MemoryGraph (for edge lookup)
    ) -> list:
        """
        Re-rank memories with a project-context bias.

        Returns the same memories sorted by boosted score.
        """
        if not active_project_name or project_manager is None:
            return sorted(memories, key=lambda m: -scores.get(m.id, 0.0))

        project = project_manager.get(active_project_name)  # type: ignore[union-attr]
        if project is None:
            return sorted(memories, key=lambda m: -scores.get(m.id, 0.0))

        linked_session_ids = set(project.related_session_ids)
        boosted = dict(scores)

        # Direct link: memories whose session_id is linked to the project
        for mem in memories:
            session_hint = getattr(mem, "session_id", None)
            if session_hint and session_hint in linked_session_ids:
                boosted[mem.id] = min(1.0, boosted.get(mem.id, 0.0) + self._bias)
                log.debug("ProjectBias: memory %d +%.2f (direct)", mem.id, self._bias)

        # Graph proximity: memories referenced by project graph node
        if graph_reasoner is not None and graph is not None:
            project_slug = active_project_name.lower().replace(" ", "_")
            mem_ids = [m.id for m in memories]
            ranked = graph_reasoner.rank_memories_by_graph(  # type: ignore[union-attr]
                mem_ids, project_slug, graph,
                decay_per_hop=self._decay,
                max_depth=self._max_depth,
            )
            for mid, graph_score in ranked:
                if graph_score > 0.0:
                    boosted[mid] = min(1.0, boosted.get(mid, 0.0) + self._bias * graph_score)

        return sorted(memories, key=lambda m: -boosted.get(m.id, 0.0))


# ===========================================================================
# Issue 13 — MMR Diversity Reranker
# ===========================================================================


class MMRReranker:
    """
    Maximal Marginal Relevance (MMR) diversification of retrieval results.

    Selects the next memory that maximises:
        λ · relevance(d, query) − (1-λ) · max_{d'∈selected} sim(d, d')

    Parameters
    ----------
    lambda_mmr:
        Trade-off between relevance (1.0) and diversity (0.0).
        Default 0.5 balances both equally.
    top_k:
        Maximum number of memories to return after diversification.
    """

    def __init__(self, lambda_mmr: float = 0.5, top_k: int = 5) -> None:
        if not 0.0 <= lambda_mmr <= 1.0:
            raise ValueError(f"lambda_mmr must be in [0,1], got {lambda_mmr}")
        self._lambda = lambda_mmr
        self._top_k = top_k

    def rerank(
        self,
        memories: list,                      # list[MemoryEntry]
        query_scores: dict[int, float],      # memory_id → relevance score (0–1)
        embeddings: Optional[dict[int, np.ndarray]] = None,  # memory_id → embedding
    ) -> list:
        """
        Return top_k memories selected by MMR.

        Parameters
        ----------
        memories:
            Candidate memories (pre-retrieved).
        query_scores:
            Relevance scores from the scoring pipeline.
        embeddings:
            Optional dict of embeddings for similarity computation.
            If None, falls back to pure relevance ranking (MMR degrades to greedy).
        """
        if not memories:
            return []

        candidates = {getattr(m, "id"): m for m in memories}
        remaining_ids = set(candidates.keys())
        selected_ids: list[int] = []
        selected_embeddings: list[np.ndarray] = []

        for _ in range(min(self._top_k, len(memories))):
            if not remaining_ids:
                break

            best_id: Optional[int] = None
            best_mmr = float("-inf")

            for mid in remaining_ids:
                rel = query_scores.get(mid, 0.0)

                # Redundancy: max similarity to any already-selected memory
                if selected_embeddings and embeddings and mid in embeddings:
                    emb = _norm(embeddings[mid])
                    redundancy = max(
                        float(np.dot(emb, _norm(sel_emb)))
                        for sel_emb in selected_embeddings
                    )
                else:
                    redundancy = 0.0

                mmr_score = self._lambda * rel - (1 - self._lambda) * redundancy

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_id = mid

            if best_id is None:
                break

            selected_ids.append(best_id)
            remaining_ids.discard(best_id)
            if embeddings and best_id in embeddings:
                selected_embeddings.append(embeddings[best_id])

        return [candidates[mid] for mid in selected_ids if mid in candidates]

    def set_lambda(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"lambda_mmr must be in [0,1], got {value}")
        self._lambda = value

    @property
    def lambda_mmr(self) -> float:
        return self._lambda


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v
