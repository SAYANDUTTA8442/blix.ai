"""
v0.3.4 API routers:

  /reason                — cognitive graph queries + multi-hop + transitive
  /research              — research assistant mode (paper → structured notes)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from api.context import BlixContext
from api.deps import get_context

# ---------------------------------------------------------------------------
# /reason router
# ---------------------------------------------------------------------------

reason_router = APIRouter(prefix="/reason", tags=["Reasoning"])


class CognitiveQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural-language graph query.")
    explain: bool = Field(default=True, description="Include explainability trace.")


class MultiHopRequest(BaseModel):
    start: str = Field(..., description="Start entity label.")
    end: str = Field(..., description="End entity label.")
    intermediate_relation: Optional[str] = Field(default=None, description="Optional relation filter.")


class TransitiveRequest(BaseModel):
    entity: str = Field(..., description="Source entity label.")
    relation: str = Field(..., description="Relation to follow transitively.")
    depth: int = Field(default=2, ge=1, le=6)


@reason_router.post("/query", summary="Cognitive graph query")
async def cognitive_query(
    req: CognitiveQueryRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Answer a natural-language query by reasoning over the knowledge graph.

    Example queries:
    - "What does Sayan work on?"
    - "What technologies does Blix use?"
    - "Who collaborates with Alice?"
    """
    result = ctx.cognitive_query_engine.query(req.query)
    response: dict = {
        "query": result.query,
        "answer": result.answer,
        "is_empty": result.is_empty(),
    }
    if req.explain:
        response["trace"] = result.trace.to_dict()
        # Build full explainability
        explained = ctx.explainability_engine.explain(
            answer=", ".join(result.answer) if result.answer else "(no answer)",
            query=req.query,
            reasoning_trace=result.trace,
        )
        response["explanation"] = explained.to_dict()
    return response


@reason_router.post("/multihop", summary="Multi-hop graph reasoning")
async def multihop_query(
    req: MultiHopRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Find intermediate entities connecting start → ? → end in the knowledge graph.

    Example: start="Sayan", end="FastAPI"
    → Finds: Blix (Sayan works_on Blix, Blix uses FastAPI)
    """
    result = ctx.cognitive_query_engine.multi_hop_query(
        start_label=req.start,
        end_label=req.end,
        intermediate_relation=req.intermediate_relation,
    )
    return result.to_dict()


@reason_router.post("/infer", summary="Transitive inference")
async def transitive_infer(
    req: TransitiveRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Compute the transitive closure of an entity via a relation.

    Example: entity="Blix", relation="uses", depth=2
    → All things Blix uses, and all things those things use.
    """
    result = ctx.cognitive_query_engine.infer_transitive(
        entity_label=req.entity,
        relation=req.relation,
        depth=req.depth,
    )
    return result.to_dict()


@reason_router.get("/explain", summary="Explain a query with full evidence chain")
async def explain_query(
    q: str = Query(..., description="The query to explain."),
    answer: str = Query(default="", description="Known answer to annotate (optional)."),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """
    Build a full evidence chain for a query: memory, facts, graph, insights.
    """
    explained = ctx.explainability_engine.explain(
        answer=answer or "(not provided)",
        query=q,
    )
    return explained.to_dict()


# ---------------------------------------------------------------------------
# /research router
# ---------------------------------------------------------------------------

research_router = APIRouter(prefix="/research", tags=["Research Assistant"])


@research_router.get("", summary="List research notes")
async def list_research_notes(
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """List all processed research notes."""
    notes = ctx.research_assistant.list_all()
    return {
        "total": len(notes),
        "notes": [
            {
                "doc_id": n.doc_id,
                "title": n.title,
                "summary": n.summary[:200],
                "methodology": n.methodology[:150],
                "findings_count": len(n.key_findings),
                "limitations_count": len(n.limitations),
                "future_work_count": len(n.future_work),
                "related_topics": n.related_topics,
                "confidence": n.confidence,
                "created_at": n.created_at,
            }
            for n in notes
        ],
    }


@research_router.get("/search", summary="Search research notes")
async def search_research_notes(
    q: str = Query(..., min_length=1, description="Search query."),
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Search research notes by title, summary, or concept."""
    notes = ctx.research_assistant.search(q)
    return {
        "query": q,
        "total": len(notes),
        "notes": [n.to_dict() for n in notes],
    }


@research_router.get("/{doc_id}", summary="Get research notes by doc_id")
async def get_research_notes(
    doc_id: str,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Retrieve full structured research notes for a document."""
    notes = ctx.research_assistant.get(doc_id)
    if notes is None:
        raise HTTPException(status_code=404, detail=f"Research notes for '{doc_id}' not found.")
    return notes.to_dict()
