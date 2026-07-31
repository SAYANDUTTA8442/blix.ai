"""
SemanticRetriever — embedding-based memory retrieval.  v0.2

Wraps ``EmbeddingStore`` and the legacy ``MemoryRetriever`` into a
unified retrieval interface.

Retrieval pipeline
------------------
1. **Semantic pass** (``EmbeddingStore.search``)
   Cosine-similarity search over sentence embeddings.  Most accurate
   for paraphrase/meaning matches.

2. **Fuzzy + keyword pass** (legacy ``MemoryRetriever``)
   Surface entries missed by the semantic pass, especially short
   queries or exact technical terms not well-handled by the embed model.

3. **Recent pass** (always included)
   Last *k* turns for conversational continuity.

All three passes are merged, deduplicated, and sorted by entry id.

Python 3.10 compatible.
"""
# DEPRECATED — core.semantic_retriever (ISSUE-009)
#
# This module is superseded by memory.hybrid.retrieval.hybrid_retriever.
# The class ``SemanticRetriever`` here is the v0.3.x implementation;
# ``memory.hybrid.retrieval.hybrid_retriever.SemanticRetriever`` is the v0.3.15+ HGSHM implementation.
#
# These are different classes with different APIs. Callers that need
# the v0.3.15+ version must update their imports:
#
#     # Old (this file — legacy):
#     from core.semantic_retriever import SemanticRetriever
#
#     # New (HGSHM-backed):
#     from memory.hybrid.retrieval.hybrid_retriever import SemanticRetriever
#
# This file will be removed in v0.4. Do not add new callers.
# Issue: https://github.com/blix/blix/issues/9
#


from __future__ import annotations

from typing import Optional

from core.embedding_store import EmbeddingStore
from core.memory_retriever import MemoryRetriever
from schemas.memory_entry import MemoryEntry
from utils.logger import get_logger

log = get_logger(__name__)


class SemanticRetriever:
    """
    Unified retriever combining semantic + fuzzy + recent strategies.

    Parameters
    ----------
    embedding_store:
        The persistent embedding index.
    legacy_retriever:
        The v0.1 ``MemoryRetriever`` (fuzzy + keyword + recent).
    semantic_top_k:
        Override for the embedding search top-k.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        legacy_retriever: MemoryRetriever,
        semantic_top_k: Optional[int] = None,
    ) -> None:
        self._store = embedding_store
        self._legacy = legacy_retriever
        self._semantic_top_k = semantic_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        memories: list[MemoryEntry],
        query: str,
    ) -> list[MemoryEntry]:
        """
        Run all retrieval passes and return a merged, deduplicated,
        chronologically-ordered list of relevant entries.

        Parameters
        ----------
        memories:
            Full in-memory list of ``MemoryEntry`` objects (all turns).
        query:
            The user's current input string.

        Returns
        -------
        list[MemoryEntry]
            Relevant entries, oldest-first.
        """
        if not memories:
            return []

        # Build a fast id→entry lookup
        by_id: dict[int, MemoryEntry] = {m.id: m for m in memories}

        seen: set[int] = set()
        merged: list[MemoryEntry] = []

        def _add_ids(ids: list[int]) -> None:
            for eid in ids:
                if eid not in seen and eid in by_id:
                    seen.add(eid)
                    merged.append(by_id[eid])

        # 1. Semantic pass
        semantic_hits = self._store.search(query, top_k=self._semantic_top_k)
        semantic_ids = [eid for eid, _score in semantic_hits]
        log.debug("SemanticRetriever: semantic pass → %d hits", len(semantic_ids))
        _add_ids(semantic_ids)

        # 2. Fuzzy + keyword + recent (legacy)
        legacy_hits = self._legacy.retrieve(memories, query)
        _add_ids([m.id for m in legacy_hits])
        log.debug("SemanticRetriever: legacy pass → %d new hits", len(merged) - len(semantic_ids))

        # Sort chronologically
        merged.sort(key=lambda e: e.id)
        log.info(
            "SemanticRetriever.retrieve: %d total entries for query %r",
            len(merged),
            query[:40],
        )
        return merged

    def index_entry(self, entry: MemoryEntry) -> None:
        """
        Add a newly saved ``MemoryEntry`` to the embedding index.

        Called by ``TutorAgent.save_interaction`` immediately after
        the entry is persisted.

        Parameters
        ----------
        entry:
            The entry to embed and index.
        """
        text = entry.input + " " + entry.output
        row_idx = self._store.add(entry.id, text)
        if row_idx is not None:
            log.debug("Indexed entry id=%d at embedding row=%d", entry.id, row_idx)

    def rebuild_index(self, memories: list[MemoryEntry]) -> None:
        """
        Rebuild the entire embedding index from *memories*.

        Use this after switching embedding models or on first startup
        when pre-existing memories have no embeddings yet.

        Parameters
        ----------
        memories:
            All stored ``MemoryEntry`` objects.
        """
        pairs = [(m.id, m.input + " " + m.output) for m in memories]
        self._store.rebuild(pairs)
        log.info("SemanticRetriever.rebuild_index: %d entries indexed.", len(pairs))

    @property
    def index_size(self) -> int:
        """Number of entries currently in the embedding index."""
        return self._store.size
