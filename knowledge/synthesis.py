"""
Knowledge Synthesis Engine — Blix v0.3.2  (Feature 8)

Generates higher-level knowledge reports by combining multiple sources:

    Memories + Projects + Documents + Media + Graph
                    ↓
              Knowledge Report

A ``KnowledgeReport`` is a structured digest: a synthesised narrative plus
linked concepts, facts, and provenance — suitable for display or for
seeding new ``Insight`` / ``CanonicalFact`` objects.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm.base import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Source bundle
# ---------------------------------------------------------------------------


@dataclass
class SynthesisSource:
    """One input item to be synthesised (memory, project, doc, media, or graph fact)."""

    kind: str          # "memory" | "project" | "document" | "media" | "graph"
    ref_id: str        # source identifier (memory id, doc_id, project name, etc.)
    text: str          # the content/summary to synthesise from
    topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ref_id": self.ref_id, "text": self.text, "topics": self.topics}


# ---------------------------------------------------------------------------
# Knowledge report
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeReport:
    """
    A synthesised knowledge digest combining multiple sources.

    Fields
    ------
    report_id:
        Stable id.
    title:
        Short title (often the dominant topic).
    narrative:
        Synthesised prose summary tying sources together.
    key_points:
        Bullet-style synthesised points.
    topics:
        Aggregated topic labels across all sources.
    sources:
        The ``SynthesisSource`` objects used.
    """

    report_id: str
    title: str
    narrative: str = ""
    key_points: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sources: list[SynthesisSource] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "narrative": self.narrative,
            "key_points": self.key_points,
            "topics": self.topics,
            "sources": [s.to_dict() for s in self.sources],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeReport":
        return cls(
            report_id=d["report_id"],
            title=d.get("title", ""),
            narrative=d.get("narrative", ""),
            key_points=d.get("key_points", []),
            topics=d.get("topics", []),
            sources=[SynthesisSource(**s) for s in d.get("sources", [])],
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are Blix's knowledge synthesis module. Combine the following sources
(memories, project state, documents, media transcripts, graph facts) into
a single coherent knowledge report.

Produce:
1. title: a short title for this report (the dominant topic/theme)
2. narrative: 2-4 sentence synthesis connecting the sources
3. key_points: 3-6 bullet-style synthesised points (each a full sentence)

Respond with ONLY a JSON object:
{{"title": "...", "narrative": "...", "key_points": ["..."]}}

Sources:
{sources}
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class KnowledgeSynthesisEngine:
    """
    Synthesises a ``KnowledgeReport`` from a list of ``SynthesisSource``.

    Parameters
    ----------
    reports_file:
        Path to ``knowledge_reports.json``.
    llm:
        Optional LLM for narrative synthesis. Falls back to a heuristic
        concatenation + topic-frequency summary if ``None``.
    """

    def __init__(self, reports_file: Path, llm: Optional[LLMProvider] = None) -> None:
        self._file = reports_file
        self._llm = llm
        self._reports: dict[str, KnowledgeReport] = {}
        self._next_id = 0
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
                r = KnowledgeReport.from_dict(item)
                self._reports[r.report_id] = r
            if self._reports:
                self._next_id = max(
                    int(rid.replace("report_", "")) for rid in self._reports
                ) + 1
            log.info("KnowledgeSynthesisEngine: loaded %d report(s).", len(self._reports))
        except Exception as exc:
            log.warning("KnowledgeSynthesisEngine: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._reports.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, sources: list[SynthesisSource]) -> KnowledgeReport:
        """
        Generate and persist a ``KnowledgeReport`` from the given sources.

        Returns an empty-narrative report if ``sources`` is empty.
        """
        rid = f"report_{self._next_id}"
        self._next_id += 1

        if not sources:
            report = KnowledgeReport(report_id=rid, title="Empty report")
            self._reports[rid] = report
            self._save()
            return report

        if self._llm is not None:
            data = self._llm_synthesize(sources)
        else:
            data = self._heuristic_synthesize(sources)

        topics = list({t for s in sources for t in s.topics})

        report = KnowledgeReport(
            report_id=rid,
            title=data["title"],
            narrative=data["narrative"],
            key_points=data["key_points"],
            topics=topics,
            sources=sources,
        )
        self._reports[rid] = report
        self._save()
        log.info(
            "KnowledgeSynthesisEngine: created %s from %d source(s) → %r",
            rid, len(sources), report.title,
        )
        return report

    def _llm_synthesize(self, sources: list[SynthesisSource]) -> dict:
        rendered = "\n\n".join(
            f"[{s.kind}:{s.ref_id}] {s.text[:600]}" for s in sources[:20]
        )
        prompt = _SYNTHESIS_PROMPT.format(sources=rendered[:6000])
        try:
            raw = _strip_code_fence(self._llm.generate(prompt).strip())  # type: ignore[union-attr]
            data = json.loads(raw)
            return {
                "title": str(data.get("title", "Knowledge Report")),
                "narrative": str(data.get("narrative", "")),
                "key_points": list(data.get("key_points", [])),
            }
        except Exception as exc:
            log.warning("KnowledgeSynthesisEngine: LLM synthesis failed (%s); using heuristic.", exc)
            return self._heuristic_synthesize(sources)

    def _heuristic_synthesize(self, sources: list[SynthesisSource]) -> dict:
        topic_freq: dict[str, int] = {}
        for s in sources:
            for t in s.topics:
                topic_freq[t] = topic_freq.get(t, 0) + 1
        top_topics = sorted(topic_freq.items(), key=lambda kv: -kv[1])[:3]
        title = ", ".join(t for t, _ in top_topics) if top_topics else "Knowledge Report"

        by_kind: dict[str, int] = {}
        for s in sources:
            by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        kinds_desc = ", ".join(f"{v} {k}(s)" for k, v in by_kind.items())
        narrative = f"This report synthesises {len(sources)} source(s) ({kinds_desc})."
        if top_topics:
            narrative += f" Dominant topics: {title}."

        key_points = [
            f"Source [{s.kind}:{s.ref_id}]: {s.text[:140].strip()}"
            for s in sources[:6]
        ]

        return {"title": title, "narrative": narrative, "key_points": key_points}

    # ------------------------------------------------------------------
    # Convenience source builders
    # ------------------------------------------------------------------

    @staticmethod
    def from_memories(memories: list) -> list[SynthesisSource]:
        return [
            SynthesisSource(
                kind="memory", ref_id=str(getattr(m, "id")),
                text=getattr(m, "output", ""), topics=list(getattr(m, "topics", [])),
            )
            for m in memories
        ]

    @staticmethod
    def from_projects(project_states: list) -> list[SynthesisSource]:
        out = []
        for p in project_states:
            text = f"Focus: {getattr(p, 'focus', '')}. Risks: {', '.join(getattr(p, 'risks', []))}"
            out.append(SynthesisSource(
                kind="project", ref_id=getattr(p, "project_name", "?"), text=text,
            ))
        return out

    @staticmethod
    def from_documents(docs: list) -> list[SynthesisSource]:
        return [
            SynthesisSource(
                kind="document", ref_id=getattr(d, "doc_id", "?"),
                text=getattr(d, "summary", ""), topics=list(getattr(d, "related_topics", [])),
            )
            for d in docs
        ]

    @staticmethod
    def from_media(media: list) -> list[SynthesisSource]:
        return [
            SynthesisSource(
                kind="media", ref_id=getattr(m, "media_id", "?"),
                text=getattr(m, "summary", ""), topics=list(getattr(m, "topics", [])),
            )
            for m in media
        ]

    @staticmethod
    def from_graph_facts(facts: list[tuple[str, str, str]]) -> list[SynthesisSource]:
        """facts: list of (from_label, relation, to_label)."""
        return [
            SynthesisSource(
                kind="graph", ref_id=f"{f}-{r}-{t}",
                text=f"{f} {r} {t}",
            )
            for f, r, t in facts
        ]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, report_id: str) -> Optional[KnowledgeReport]:
        return self._reports.get(report_id)

    def list_all(self) -> list[KnowledgeReport]:
        return sorted(self._reports.values(), key=lambda r: r.created_at, reverse=True)

    @property
    def count(self) -> int:
        return len(self._reports)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
