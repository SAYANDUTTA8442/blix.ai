"""
MemoryRetriever — surfaces the most relevant past interactions.

Three retrieval strategies, composited in ``retrieve()``:

1. **Recent** — always include the last *k* turns for conversational
   continuity regardless of relevance.
2. **Keyword** — case-insensitive substring match across
   ``input + output`` text.
3. **Fuzzy** — RapidFuzz ``token_set_ratio`` search for semantic
   closeness when exact keywords are absent.

Python 3.10 compatibility
--------------------------
``from __future__ import annotations`` defers evaluation, so
``list[MemoryEntry]`` and ``int | None`` syntax is safe.

Design note
-----------
The interface is intentionally coarse-grained: v2 can replace the
internals with embedding similarity or importance scoring without
any callers changing.
"""

from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz

from schemas.memory_entry import MemoryEntry
from utils.logger import get_logger

log = get_logger(__name__)


class MemoryRetriever:
    """
    Retrieves relevant memories from a list of ``MemoryEntry`` objects.

    Parameters
    ----------
    recent_k:
        Number of most-recent entries always included in ``retrieve()``.
    fuzzy_top_k:
        Maximum results from the fuzzy pass.
    fuzzy_threshold:
        Minimum RapidFuzz token-set-ratio score (0–100) to qualify.
    keyword_top_k:
        Maximum results from the keyword pass.
    """

    def __init__(
        self,
        recent_k: int = 5,
        fuzzy_top_k: int = 5,
        fuzzy_threshold: float = 60.0,
        keyword_top_k: int = 5,
    ) -> None:
        self._recent_k = recent_k
        self._fuzzy_top_k = fuzzy_top_k
        self._fuzzy_threshold = fuzzy_threshold
        self._keyword_top_k = keyword_top_k

    # ------------------------------------------------------------------
    # Individual strategies
    # ------------------------------------------------------------------

    def recent(
        self,
        memories: list[MemoryEntry],
        k: Optional[int] = None,
    ) -> list[MemoryEntry]:
        """
        Return the *k* most-recent memory entries.

        Parameters
        ----------
        memories:
            Full memory list, assumed newest-last.
        k:
            Override the instance default if provided.

        Returns
        -------
        list[MemoryEntry]
        """
        limit = k if k is not None else self._recent_k
        return memories[-limit:] if memories else []

    def keyword_search(
        self,
        memories: list[MemoryEntry],
        query: str,
        top_k: Optional[int] = None,
    ) -> list[MemoryEntry]:
        """
        Return entries whose ``input`` or ``output`` contains *query* as a
        case-insensitive substring.

        Results are returned newest-first so the most recent matches
        are preferred when the list is later truncated.

        Parameters
        ----------
        memories:
            Pool to search.
        query:
            The user's current input string.
        top_k:
            Override the instance default if provided.

        Returns
        -------
        list[MemoryEntry]
        """
        limit = top_k if top_k is not None else self._keyword_top_k
        needle = query.lower()
        hits: list[MemoryEntry] = []
        for entry in reversed(memories):
            haystack = (entry.input + " " + entry.output).lower()
            if needle in haystack:
                hits.append(entry)
            if len(hits) >= limit:
                break
        log.debug("keyword_search(%r): %d hits", query[:40], len(hits))
        return hits

    def fuzzy_search(
        self,
        memories: list[MemoryEntry],
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[MemoryEntry]:
        """
        Return the top-k entries most similar to *query* by RapidFuzz
        ``token_set_ratio``.

        ``token_set_ratio`` is chosen over simple ratio because it handles
        word-order variation well (e.g. "gradient descent SGD" matches
        "SGD and gradient descent").

        Parameters
        ----------
        memories:
            Pool to search.
        query:
            The user's current input string.
        top_k:
            Override the instance default if provided.
        threshold:
            Minimum score (0–100) override.

        Returns
        -------
        list[MemoryEntry]
        """
        limit = top_k if top_k is not None else self._fuzzy_top_k
        cutoff = threshold if threshold is not None else self._fuzzy_threshold

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in memories:
            score = fuzz.token_set_ratio(query, entry.input)
            if score >= cutoff:
                scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        results = [e for _, e in scored[:limit]]
        log.debug(
            "fuzzy_search(%r): %d results above threshold=%.1f",
            query[:40],
            len(results),
            cutoff,
        )
        return results

    # ------------------------------------------------------------------
    # Composite retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        memories: list[MemoryEntry],
        query: str,
    ) -> list[MemoryEntry]:
        """
        Combine all three strategies, deduplicate, and return a merged
        list ordered by ``id`` (oldest first for chronological reading).

        Merge priority: fuzzy → keyword → recent.  This means semantically
        relevant entries appear in the prompt before sheer recency.

        Parameters
        ----------
        memories:
            Full memory list.
        query:
            The user's current input.

        Returns
        -------
        list[MemoryEntry]
            Deduplicated, chronologically-ordered relevant entries.
        """
        if not memories:
            return []

        seen_ids: set[int] = set()
        merged: list[MemoryEntry] = []

        def _add(entries: list[MemoryEntry]) -> None:
            for e in entries:
                if e.id not in seen_ids:
                    seen_ids.add(e.id)
                    merged.append(e)

        _add(self.fuzzy_search(memories, query))
        _add(self.keyword_search(memories, query))
        _add(self.recent(memories))

        # Sort by id ascending so the prompt reads chronologically
        merged.sort(key=lambda e: e.id)
        log.info(
            "retrieve: returning %d relevant memories for query %r",
            len(merged),
            query[:40],
        )
        return merged
