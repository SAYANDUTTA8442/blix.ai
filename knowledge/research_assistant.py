"""
Research Assistant Mode — Blix v0.3.4  (Feature 5)

Processes an academic paper (or any technical document) into structured
research notes aligned with a research workflow:

    Paper
      ↓ DocumentProcessor (v0.3.2 — text extraction + chunking)
      ↓ ResearchAssistant
      ↓ ResearchNotes{summary, methodology, findings, limitations,
                       future_work, concepts, entities, related_topics}
      ↓ MemoryGraph (entities upserted)
      ↓ ConsolidationEngine (key findings consolidated as facts)
      ↓ KnowledgeSynthesisEngine (cross-paper synthesis available)

Designed for the IIT Patna research workflow: upload NPTEL notes,
research papers, or course materials and get structured knowledge
immediately available for cognitive queries.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm.base import LLMProvider
from knowledge.document_processor import ProcessedDocument
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Research notes model
# ---------------------------------------------------------------------------


@dataclass
class ResearchNotes:
    """
    Structured research notes extracted from one document.

    Fields map directly to the v0.3.4 spec:
        summary, methodology, limitations, future_work, related_concepts

    Additional fields:
        key_findings    — (from v0.3.2 ``ProcessedDocument.key_findings``)
        entities        — extracted named entities (author, institution, tool, …)
        related_topics  — topic labels for graph/cluster integration
        confidence      — overall confidence in the extraction (0–1)
    """

    doc_id: str
    title: str
    summary: str = ""
    methodology: str = ""
    key_findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    future_work: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)
    entities: list[tuple[str, str]] = field(default_factory=list)  # (label, kind)
    related_topics: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "methodology": self.methodology,
            "key_findings": self.key_findings,
            "limitations": self.limitations,
            "future_work": self.future_work,
            "related_concepts": self.related_concepts,
            "entities": [list(e) for e in self.entities],
            "related_topics": self.related_topics,
            "confidence": round(self.confidence, 3),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchNotes":
        return cls(
            doc_id=d["doc_id"],
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            methodology=d.get("methodology", ""),
            key_findings=d.get("key_findings", []),
            limitations=d.get("limitations", []),
            future_work=d.get("future_work", []),
            related_concepts=d.get("related_concepts", []),
            entities=[tuple(e) for e in d.get("entities", [])],
            related_topics=d.get("related_topics", []),
            confidence=d.get("confidence", 0.5),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RESEARCH_PROMPT = """\
You are a research assistant. Analyse the following document excerpt and
extract structured research notes.

Respond with ONLY a JSON object (no prose, no code fences):

{{
  "summary": "2-3 sentence high-level summary",
  "methodology": "brief description of the method/approach used (or 'N/A' if not a paper)",
  "key_findings": ["finding 1", "finding 2"],
  "limitations": ["limitation 1", "limitation 2"],
  "future_work": ["future direction 1", "future direction 2"],
  "related_concepts": ["concept 1", "concept 2"],
  "entities": [["entity name", "type"]],
  "related_topics": ["topic 1", "topic 2"],
  "confidence": 0.85
}}

Entity types: person, project, skill, goal, topic, organization, tool, dataset.

Document excerpt ({title}):
{text}
"""


# ---------------------------------------------------------------------------
# Section heuristics (offline fallback)
# ---------------------------------------------------------------------------

_SECTION_HEADERS = {
    "abstract":     re.compile(r"\b(abstract)\b", re.I),
    "introduction": re.compile(r"\b(introduction|background)\b", re.I),
    "methodology":  re.compile(r"\b(method|methodology|approach|system|design)\b", re.I),
    "results":      re.compile(r"\b(results?|experiments?|findings?|evaluation)\b", re.I),
    "limitations":  re.compile(r"\b(limitations?|weaknesses?|constraints?)\b", re.I),
    "future":       re.compile(r"\b(future\s+work|future\s+directions?|conclusion)\b", re.I),
}


def _split_by_sections(text: str) -> dict[str, str]:
    """Split document text into labelled sections using header heuristics."""
    sections: dict[str, list[str]] = {k: [] for k in _SECTION_HEADERS}
    current = "introduction"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = False
        for label, pattern in _SECTION_HEADERS.items():
            if pattern.search(stripped) and len(stripped) < 80:
                current = label
                matched = True
                break
        if not matched:
            sections.setdefault(current, []).append(stripped)
    return {k: " ".join(v)[:2000] for k, v in sections.items() if v}


def _heuristic_research_notes(doc: ProcessedDocument) -> ResearchNotes:
    """
    Offline fallback: derive research notes from DocumentProcessor output
    and section splitting.
    """
    full_text = " ".join(c.text for c in doc.chunks)
    sections = _split_by_sections(full_text)

    summary = doc.summary or sections.get("abstract", sections.get("introduction", ""))[:500]
    methodology = sections.get("methodology", "")[:400] or "Not identified."
    key_findings = doc.key_findings or []
    if not key_findings and sections.get("results"):
        # Extract first 3 sentences from results section
        sents = re.split(r"(?<=[.!?])\s+", sections["results"])
        key_findings = sents[:3]

    limitations: list[str] = []
    if sections.get("limitations"):
        sents = re.split(r"(?<=[.!?])\s+", sections["limitations"])
        limitations = [s for s in sents[:3] if len(s) > 20]

    future_work: list[str] = []
    if sections.get("future"):
        sents = re.split(r"(?<=[.!?])\s+", sections["future"])
        future_work = [s for s in sents[:3] if len(s) > 20]

    return ResearchNotes(
        doc_id=doc.doc_id,
        title=doc.title,
        summary=summary[:400],
        methodology=methodology[:300],
        key_findings=key_findings[:5],
        limitations=limitations,
        future_work=future_work,
        related_concepts=doc.concepts[:8],
        entities=list(doc.entities)[:10],
        related_topics=doc.related_topics[:5],
        confidence=0.4,
    )


# ---------------------------------------------------------------------------
# Research Assistant
# ---------------------------------------------------------------------------


class ResearchAssistant:
    """
    Converts a ``ProcessedDocument`` into structured ``ResearchNotes`` and
    integrates the knowledge downstream into memory, graph, and facts.

    Parameters
    ----------
    notes_file:
        Path to ``research_notes.json`` (persistence).
    llm:
        Optional LLM for structured extraction. Falls back to heuristics.
    consolidation_engine:
        Optional ``ConsolidationEngine`` — key findings are consolidated
        as canonical facts.
    graph:
        Optional ``MemoryGraph`` — entities are upserted as nodes/edges.
    synthesis_engine:
        Optional ``KnowledgeSynthesisEngine`` — adds notes as a synthesis source.
    """

    def __init__(
        self,
        notes_file: Path,
        llm: Optional[LLMProvider] = None,
        consolidation_engine: Optional[object] = None,
        graph: Optional[object] = None,
        synthesis_engine: Optional[object] = None,
    ) -> None:
        self._file = notes_file
        self._llm = llm
        self._consolidation = consolidation_engine
        self._graph = graph
        self._synthesis = synthesis_engine
        self._notes: dict[str, ResearchNotes] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                n = ResearchNotes.from_dict(item)
                self._notes[n.doc_id] = n
            log.info("ResearchAssistant: loaded %d note(s).", len(self._notes))
        except Exception as exc:
            log.warning("ResearchAssistant: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([n.to_dict() for n in self._notes.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, doc: ProcessedDocument) -> ResearchNotes:
        """
        Convert a ``ProcessedDocument`` into ``ResearchNotes``.

        Runs LLM extraction if available, else heuristics. Then
        integrates results into graph, facts, and synthesis engine.
        """
        notes = self._extract(doc)
        self._notes[notes.doc_id] = notes
        self._save()
        self._integrate(notes)
        log.info(
            "ResearchAssistant: processed '%s' (conf=%.2f, findings=%d)",
            notes.title, notes.confidence, len(notes.key_findings),
        )
        return notes

    def _extract(self, doc: ProcessedDocument) -> ResearchNotes:
        """Extract structured notes from the document."""
        if self._llm is not None:
            return self._llm_extract(doc)
        return _heuristic_research_notes(doc)

    def _llm_extract(self, doc: ProcessedDocument) -> ResearchNotes:
        """Use LLM to extract structured research notes."""
        # Use first 6000 chars of full text as the excerpt
        full_text = " ".join(c.text for c in doc.chunks)[:6000]
        prompt = _RESEARCH_PROMPT.format(title=doc.title, text=full_text)
        try:
            raw = self._llm.generate(prompt).strip()  # type: ignore[union-attr]
            raw = _strip_fence(raw)
            data = json.loads(raw)
            return ResearchNotes(
                doc_id=doc.doc_id,
                title=doc.title,
                summary=str(data.get("summary", doc.summary))[:500],
                methodology=str(data.get("methodology", ""))[:400],
                key_findings=list(data.get("key_findings", doc.key_findings))[:8],
                limitations=list(data.get("limitations", []))[:5],
                future_work=list(data.get("future_work", []))[:5],
                related_concepts=list(data.get("related_concepts", doc.concepts))[:8],
                entities=[tuple(e) for e in data.get("entities", doc.entities) if len(e) == 2][:10],
                related_topics=list(data.get("related_topics", doc.related_topics))[:5],
                confidence=float(data.get("confidence", 0.75)),
            )
        except Exception as exc:
            log.warning("ResearchAssistant: LLM extraction failed (%s); using heuristic.", exc)
            return _heuristic_research_notes(doc)

    def _integrate(self, notes: ResearchNotes) -> None:
        """Downstream integration: facts, graph, synthesis."""
        # 1. Consolidate key findings as canonical facts
        if self._consolidation is not None:
            for finding in notes.key_findings:
                try:
                    topic = notes.related_topics[0] if notes.related_topics else ""
                    self._consolidation.consolidate(  # type: ignore[union-attr]
                        finding, source_memory_id=-hash(notes.doc_id) % 10000, topic=topic
                    )
                except Exception as exc:
                    log.warning("ResearchAssistant: fact consolidation failed (%s)", exc)

        # 2. Upsert entities into the memory graph
        if self._graph is not None:
            from core.memory_graph import EntityKind, RelationKind
            for entity_label, entity_kind_str in notes.entities:
                try:
                    ekind = EntityKind(entity_kind_str.lower())
                except ValueError:
                    ekind = EntityKind.TOPIC
                try:
                    self._graph.upsert_relation(  # type: ignore[union-attr]
                        from_label=notes.title, from_kind=EntityKind.TOPIC,
                        relation=RelationKind.USES,
                        to_label=entity_label, to_kind=ekind,
                        confidence=notes.confidence,
                    )
                except Exception as exc:
                    log.warning("ResearchAssistant: graph upsert failed (%s)", exc)

        # 3. Add as synthesis source
        if self._synthesis is not None:
            from knowledge.synthesis import KnowledgeSynthesisEngine, SynthesisSource
            src = SynthesisSource(
                kind="research_paper",
                ref_id=notes.doc_id,
                text=notes.summary,
                topics=notes.related_topics,
            )
            try:
                self._synthesis.synthesize([src])  # type: ignore[union-attr]
            except Exception as exc:
                log.warning("ResearchAssistant: synthesis failed (%s)", exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, doc_id: str) -> Optional[ResearchNotes]:
        return self._notes.get(doc_id)

    def list_all(self) -> list[ResearchNotes]:
        return sorted(self._notes.values(), key=lambda n: n.created_at, reverse=True)

    def search(self, query: str) -> list[ResearchNotes]:
        """Simple text-based search over titles, summaries, and concepts."""
        q = query.lower()
        return [
            n for n in self._notes.values()
            if q in n.title.lower()
            or q in n.summary.lower()
            or any(q in c.lower() for c in n.related_concepts)
        ]

    @property
    def count(self) -> int:
        return len(self._notes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
