"""
/knowledge router — Blix v0.3.3

Endpoints
---------
GET  /knowledge/facts              — list canonical facts
GET  /knowledge/facts/strongest    — top facts by confidence
POST /knowledge/synthesize         — generate a knowledge report
GET  /knowledge/reports            — list knowledge reports
GET  /knowledge/reports/{id}       — single report
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.context import BlixContext
from api.deps import get_context
from api.models import (
    CanonicalFactItem, CanonicalFactsResponse, KnowledgeReportItem,
    KnowledgeReportsResponse, SynthesisRequest,
)
from knowledge.synthesis import KnowledgeSynthesisEngine, SynthesisSource

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])


def _fact_item(cf: object) -> CanonicalFactItem:
    return CanonicalFactItem(
        fact_id=getattr(cf, "fact_id"),
        fact=getattr(cf, "fact"),
        confidence=getattr(cf, "confidence"),
        evidence_count=getattr(cf, "evidence_count"),
        topic=getattr(cf, "topic", ""),
        variants=list(getattr(cf, "variants", [])),
    )


def _report_item(r: object) -> KnowledgeReportItem:
    return KnowledgeReportItem(
        report_id=getattr(r, "report_id"),
        title=getattr(r, "title"),
        narrative=getattr(r, "narrative"),
        key_points=list(getattr(r, "key_points", [])),
        topics=list(getattr(r, "topics", [])),
        created_at=getattr(r, "created_at", ""),
    )


@router.get("/facts", response_model=CanonicalFactsResponse, summary="List canonical facts")
async def list_facts(
    topic: str | None = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    min_evidence: int = Query(default=1, ge=1),
    ctx: BlixContext = Depends(get_context),
) -> CanonicalFactsResponse:
    """Return canonical facts, optionally filtered by topic and confidence."""
    facts = ctx.consolidation.list_facts(
        topic=topic, min_confidence=min_confidence, min_evidence=min_evidence
    )
    return CanonicalFactsResponse(facts=[_fact_item(f) for f in facts], total=len(facts))


@router.get("/facts/strongest", response_model=CanonicalFactsResponse, summary="Strongest canonical facts")
async def strongest_facts(
    top_k: int = Query(default=10, ge=1, le=50),
    ctx: BlixContext = Depends(get_context),
) -> CanonicalFactsResponse:
    """Return the top-k canonical facts by confidence."""
    facts = ctx.consolidation.strongest_facts(top_k)
    return CanonicalFactsResponse(facts=[_fact_item(f) for f in facts], total=len(facts))


@router.post("/synthesize", response_model=KnowledgeReportItem, summary="Generate knowledge report")
async def synthesize(
    req: SynthesisRequest,
    ctx: BlixContext = Depends(get_context),
) -> KnowledgeReportItem:
    """
    Synthesise a knowledge report from memories, projects, and canonical facts.

    The caller specifies which memory ids to include (or leave empty to
    include recent ones) and whether to incorporate project/fact context.
    """
    memories = ctx.memory_manager.get_all_memories()
    if req.source_memory_ids:
        memories = [m for m in memories if m.id in set(req.source_memory_ids)]
    else:
        memories = memories[-20:]  # default: last 20

    sources: list[SynthesisSource] = KnowledgeSynthesisEngine.from_memories(memories)

    if req.include_projects:
        project_states = ctx.project_intelligence.list_all()
        sources.extend(KnowledgeSynthesisEngine.from_projects(project_states))

    if req.include_facts:
        facts = ctx.consolidation.strongest_facts(10)
        for f in facts:
            sources.append(SynthesisSource(
                kind="canonical_fact", ref_id=f.fact_id, text=f.fact, topics=[f.topic],
            ))

    report = ctx.synthesis.synthesize(sources)
    return _report_item(report)


@router.get("/reports", response_model=KnowledgeReportsResponse, summary="List knowledge reports")
async def list_reports(ctx: BlixContext = Depends(get_context)) -> KnowledgeReportsResponse:
    reports = ctx.synthesis.list_all()
    return KnowledgeReportsResponse(reports=[_report_item(r) for r in reports], total=len(reports))


@router.get("/reports/{report_id}", response_model=KnowledgeReportItem, summary="Get knowledge report")
async def get_report(report_id: str, ctx: BlixContext = Depends(get_context)) -> KnowledgeReportItem:
    report = ctx.synthesis.get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return _report_item(report)
