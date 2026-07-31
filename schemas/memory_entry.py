"""
Pydantic schema for a single memory entry — v0.2.

New in v0.2
-----------
* ``embedding_id`` — links this entry to its vector in the embedding store.
* ``extracted_facts`` — auto-extracted facts from CoT memory extractor.
* ``topics`` — topic tags assigned by the memory extractor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


class MemoryEntry(BaseModel):
    """
    One complete user ↔ Blix exchange stored in persistent memory.

    Attributes
    ----------
    id:
        Auto-incrementing integer primary key (sequential, never reused).
    input:
        Raw text typed by the user.
    output:
        Blix's response to that input.
    timestamp:
        UTC time at which the interaction was saved.
    tags:
        Free-form labels (legacy; superseded by ``topics`` in v0.2).
    importance:
        Float importance score assigned by the memory extractor (0–1).
    embedding_id:
        Index into the embedding matrix stored in ``embeddings.npy``.
        ``None`` until the semantic indexer has processed this entry.
    extracted_facts:
        Short factual sentences extracted by the CoT memory extractor.
    topics:
        Topic labels inferred from the conversation turn.
    """

    id: int = Field(..., description="Unique sequential identifier.")
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    # --- v1 extension points (kept for backwards compat) ---
    tags: list[str] = Field(default_factory=list)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # --- v0.2 new fields ---
    embedding_id: Optional[int] = Field(
        default=None,
        description="Row index in embeddings.npy for this entry.",
    )
    extracted_facts: list[str] = Field(
        default_factory=list,
        description="CoT-extracted factual sentences from this turn.",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Topic labels inferred by the memory extractor.",
    )

    @field_serializer("timestamp")
    def _ser_timestamp(self, v: datetime) -> str:
        return v.isoformat()
