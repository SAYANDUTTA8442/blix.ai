"""
/memory router — Blix v0.3.3

Endpoints
---------
GET  /memory                 — paginated list (most recent first)
GET  /memory/{id}            — single memory by id
GET  /memory/search          — semantic search over memory
GET  /memory/lifecycle       — lifecycle state summary
POST /memory/{id}/compress   — manually compress a memory
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.context import BlixContext
from api.deps import get_context
from api.models import (
    MemoryItem, MemoryListResponse, MemorySearchRequest, MemorySearchResponse,
)

router = APIRouter(prefix="/memory", tags=["Memory"])


def _to_item(m: object, lifecycle_state: str = "active") -> MemoryItem:
    return MemoryItem(
        id=getattr(m, "id"),
        input=getattr(m, "input", ""),
        output=getattr(m, "output", ""),
        timestamp=getattr(m, "timestamp"),
        topics=list(getattr(m, "topics", [])),
        importance=getattr(m, "importance", None),
        extracted_facts=list(getattr(m, "extracted_facts", [])),
        lifecycle_state=lifecycle_state,
    )


@router.get("", response_model=MemoryListResponse, summary="List memories")
async def list_memories(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    ctx: BlixContext = Depends(get_context),
) -> MemoryListResponse:
    """Return memories newest-first, paginated."""
    all_memories = list(reversed(ctx.memory_manager.get_all_memories()))
    total = len(all_memories)
    start = (page - 1) * page_size
    page_memories = all_memories[start : start + page_size]
    items = [
        _to_item(m, ctx.lifecycle.get_state(m.id).value)
        for m in page_memories
    ]
    return MemoryListResponse(memories=items, total=total, page=page, page_size=page_size)


@router.get("/search", response_model=MemorySearchResponse, summary="Semantic memory search")
async def search_memories(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50),
    ctx: BlixContext = Depends(get_context),
) -> MemorySearchResponse:
    """Retrieve and score memories semantically matching the query."""
    all_memories = ctx.memory_manager.get_all_memories()
    results = ctx.retriever.retrieve(all_memories, q)[:top_k]
    items = [_to_item(m, ctx.lifecycle.get_state(m.id).value) for m in results]
    return MemorySearchResponse(query=q, results=items)


@router.get("/lifecycle", summary="Memory lifecycle state counts")
async def lifecycle_stats(ctx: BlixContext = Depends(get_context)) -> dict:
    """Return counts of memories in each lifecycle state."""
    return ctx.lifecycle.state_counts()


@router.get("/{memory_id}", response_model=MemoryItem, summary="Get memory by id")
async def get_memory(memory_id: int, ctx: BlixContext = Depends(get_context)) -> MemoryItem:
    m = ctx.memory_manager.get_memory_by_id(memory_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")
    return _to_item(m, ctx.lifecycle.get_state(memory_id).value)


@router.post("/{memory_id}/compress", summary="Manually compress a memory")
async def compress_memory(
    memory_id: int,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Transition a memory to COMPRESSED state, replacing raw text with its
    extracted_facts summary. The memory remains searchable but is deprioritised.
    """
    m = ctx.memory_manager.get_memory_by_id(memory_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")
    summary = " ".join(m.extracted_facts[:3]) or m.output[:200]
    ctx.lifecycle.compress(memory_id, summary)
    return {"memory_id": memory_id, "state": "compressed", "summary": summary}
