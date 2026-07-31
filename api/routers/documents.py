"""
/documents router — Blix v0.3.3

Endpoints
---------
POST /documents/upload   — upload and process a document (PDF/TXT/MD/DOCX/HTML)
GET  /documents          — list processed documents (from knowledge reports)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.context import BlixContext
from api.deps import get_context
from api.models import DocumentUploadResponse
from knowledge.document_processor import detect_format, DocumentFormat

router = APIRouter(prefix="/documents", tags=["Documents"])

_ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}


@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload and process a document")
async def upload_document(
    file: UploadFile = File(..., description="Document file: PDF, TXT, MD, DOCX, or HTML."),
    ctx: BlixContext = Depends(get_context),
) -> DocumentUploadResponse:
    """
    Upload a document, process it through the DocumentProcessor pipeline,
    and store the results in the knowledge base.

    Extracted facts are consolidated into ``ConsolidationEngine``; entities
    are upserted into the memory graph; topic clusters are updated.
    """
    suffix = Path(file.filename or "file.txt").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix!r}. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )

    # Save to a temp file (DocumentProcessor expects a Path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        doc = ctx.document_processor.process_file(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # --- Downstream integration ---

    # 1. Consolidate extracted facts
    for concept in doc.concepts:
        ctx.consolidation.consolidate(concept, source_memory_id=-1, topic=doc.related_topics[0] if doc.related_topics else "")

    # 2. Upsert entities into the memory graph
    from core.memory_graph import EntityKind, RelationKind
    for entity_label, entity_kind_str in doc.entities:
        try:
            ekind = EntityKind(entity_kind_str.lower())
        except ValueError:
            ekind = EntityKind.TOPIC
        ctx.graph.upsert_relation(
            from_label=doc.title, from_kind=EntityKind.TOPIC,
            relation=RelationKind.USES,
            to_label=entity_label, to_kind=ekind,
            confidence=0.8,
        )

    # 3. Update semantic cluster index with document topics
    # (embeddings for chunks would be computed here in a full pipeline;
    #  we use a zero-vector stub so the index stays consistent)
    import numpy as np
    stub_emb = np.zeros(384, dtype=np.float32)  # 384-dim zero (no-op for clustering)
    if doc.related_topics:
        ctx.cluster_index.add_memory(-len(ctx.cluster_index.list_clusters()) - 1, stub_emb, doc.related_topics)

    return DocumentUploadResponse(
        doc_id=doc.doc_id,
        title=doc.title,
        format=doc.format.value,
        summary=doc.summary,
        key_findings=doc.key_findings,
        concepts=doc.concepts,
        related_topics=doc.related_topics,
        chunk_count=len(doc.chunks),
        raw_text_length=doc.raw_text_length,
    )


@router.get("", summary="List processed documents")
async def list_documents(ctx: BlixContext = Depends(get_context)) -> dict:
    """
    Return a summary of document-derived knowledge reports
    (documents produce knowledge reports on upload).
    """
    reports = ctx.synthesis.list_all()
    doc_reports = [r for r in reports if any(s.kind == "document" for s in r.sources)]
    return {
        "total_documents": len(doc_reports),
        "documents": [{"report_id": r.report_id, "title": r.title, "created_at": r.created_at}
                      for r in doc_reports[:50]],
    }
