"""
TutorAgent — top-level orchestrator for the Blix chat pipeline.  v0.3

v0.3 changes vs v0.2
---------------------
* Background processing: memory extraction, profile update, graph update,
  and summary generation are ALL dispatched to a BackgroundProcessor
  thread.  Chat latency is no longer gated on extraction.
* MemoryScorer replaces pure similarity ranking.
* HierarchyManager, MemoryGraph, ProjectManager, and ProfileEvolver
  are injected via constructor (dependency injection).
* All new dependencies are optional: passing None silently degrades
  to v0.2 behaviour, preserving full backwards compatibility.

Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from llm.base import LLMProvider
from core.memory_manager import MemoryManager
from core.semantic_retriever import SemanticRetriever
from core.prompt_builder import PromptBuilder
from core.memory_extractor import MemoryExtractor, ExtractionResult
from core.memory_scorer import MemoryScorer
from core.background_processor import BackgroundProcessor, ProcessorJob
from schemas.memory_entry import MemoryEntry
from utils.logger import get_logger

if TYPE_CHECKING:
    from core.hierarchy_manager import HierarchyManager
    from core.memory_graph import MemoryGraph
    from core.project_manager import ProjectManager
    from core.profile_evolver import ProfileEvolver

log = get_logger(__name__)


class TutorAgent:
    """
    Orchestrates the full v0.3 chat pipeline.

    All v0.3 dependencies are Optional — passing None provides full v0.2
    backwards compatibility.
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory_manager: MemoryManager,
        retriever: SemanticRetriever,
        prompt_builder: PromptBuilder,
        extractor: Optional[MemoryExtractor] = None,
        scorer: Optional[MemoryScorer] = None,
        background_processor: Optional[BackgroundProcessor] = None,
        hierarchy_manager: Optional["HierarchyManager"] = None,
        memory_graph: Optional["MemoryGraph"] = None,
        project_manager: Optional["ProjectManager"] = None,
        profile_evolver: Optional["ProfileEvolver"] = None,
    ) -> None:
        self._llm = llm
        self._mm = memory_manager
        self._retriever = retriever
        self._builder = prompt_builder
        self._extractor = extractor
        self._scorer = scorer or MemoryScorer()
        self._bg = background_processor
        self._hierarchy = hierarchy_manager
        self._graph = memory_graph
        self._projects = project_manager
        self._evolver = profile_evolver

        self._session_memories: list[MemoryEntry] = []
        self._session_index: int = 0

        if self._bg is not None:
            self._register_bg_handlers()
            self._bg.start()

        log.info(
            "TutorAgent v0.3 ready — model=%s  memories=%d  index=%d  extraction=%s  bg=%s",
            llm.model_name(),
            memory_manager.memory_count(),
            retriever.index_size,
            "on" if extractor is not None else "off",
            "on" if background_processor is not None else "off",
        )

    # ------------------------------------------------------------------
    # Background handler registration
    # ------------------------------------------------------------------

    def _register_bg_handlers(self) -> None:
        assert self._bg is not None
        self._bg.register(ProcessorJob.EXTRACT_AND_UPDATE, self._bg_extract_and_update)
        self._bg.register(ProcessorJob.UPDATE_GRAPH, self._bg_update_graph)
        self._bg.register(ProcessorJob.REGENERATE_SUMMARY, self._bg_regenerate_summary)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """
        Process user_input through the v0.3 pipeline and return Blix's reply.

        Memory extraction is offloaded to background — this method returns
        as soon as the LLM response is ready.
        """
        log.info("chat: %r", user_input[:60])

        relevant = self.retrieve_memory(user_input)
        prompt = self.build_prompt(user_input, relevant)
        response = self.generate_response(prompt)
        entry = self.save_interaction(user_input, response)
        self._index_entry(entry)
        self._session_memories.append(entry)

        if self._bg is not None and self._extractor is not None:
            self._bg.submit(ProcessorJob.EXTRACT_AND_UPDATE, {
                "entry_id": entry.id,
                "user_input": user_input,
                "response": response,
            })
        else:
            self._extract_and_apply_sync(entry, user_input, response)

        return response

    # ------------------------------------------------------------------
    # Pipeline steps (public for unit testing)
    # ------------------------------------------------------------------

    def retrieve_memory(self, query: str) -> list[MemoryEntry]:
        """Retrieve + re-rank memories using the composite scorer."""
        candidates = self._retriever.retrieve(self._mm.get_all_memories(), query)
        if not candidates:
            return candidates
        return self._rerank(query, candidates)

    def _rerank(self, query: str, memories: list[MemoryEntry]) -> list[MemoryEntry]:
        """Re-rank retrieved memories using MemoryScorer."""
        sim_map: dict[int, float] = {}
        if hasattr(self._retriever, "_store"):
            hits = self._retriever._store.search(query, top_k=len(memories))
            sim_map = {eid: score for eid, score in hits}

        max_access = max(
            (getattr(m, "access_count", 1) for m in memories), default=1
        ) or 1

        entries_for_scoring = [
            {
                "id": m.id,
                "relevance": sim_map.get(m.id, 0.3),
                "importance": m.importance if m.importance is not None else 0.5,
                "timestamp": m.timestamp,
                "access_count": getattr(m, "access_count", 1),
                "max_access_count": max_access,
            }
            for m in memories
        ]
        scored = self._scorer.score_batch(entries_for_scoring)
        id_to_score = {s.memory_id: s.final_score for s in scored}
        return sorted(memories, key=lambda m: id_to_score.get(m.id, 0.0), reverse=True)

    def build_prompt(self, user_input: str, relevant_memories: list[MemoryEntry]) -> str:
        """Assemble the complete LLM prompt, injecting hierarchy context if available."""
        base = self._builder.build(
            user_input=user_input,
            profile=self._effective_profile(),
            learning_state=self._mm.learning_state,
            relevant_memories=relevant_memories,
        )
        if self._hierarchy is not None:
            ctx = self._hierarchy.get_hierarchy_context()
            if ctx:
                base = base + "\n\n" + ctx
        return base

    def generate_response(self, prompt: str) -> str:
        return self._llm.generate(prompt)

    def save_interaction(self, user_input: str, response: str) -> MemoryEntry:
        return self._mm.add_memory(user_input, response)

    # ------------------------------------------------------------------
    # Sync extraction (v0.2 fallback when bg=None)
    # ------------------------------------------------------------------

    def _extract_and_apply_sync(
        self,
        entry: MemoryEntry,
        user_input: str,
        response: str,
    ) -> None:
        if self._extractor is None:
            return
        try:
            result: ExtractionResult = self._extractor.extract(user_input, response)
            self._apply_extraction(entry.id, result)
        except Exception as exc:
            log.warning("Sync extraction error: %s", exc)

    # ------------------------------------------------------------------
    # Background handlers
    # ------------------------------------------------------------------

    def _bg_extract_and_update(self, payload: dict) -> None:
        entry = self._mm.get_memory_by_id(payload["entry_id"])
        if entry is None or self._extractor is None:
            return
        result = self._extractor.extract(payload["user_input"], payload["response"])
        self._apply_extraction(entry.id, result)
        if self._graph is not None:
            self._bg_update_graph_from_result(result, entry.id)
        if self._projects is not None:
            for project_name in result.profile_new_projects:
                self._projects.get_or_create(project_name)

    def _apply_extraction(self, entry_id: int, result: ExtractionResult) -> None:
        self._mm.update_memory(entry_id, **{
            "extracted_facts": result.facts,
            "topics": result.topics,
            "importance": result.importance if result.importance > 0.0 else None,
        })
        new_state = self._extractor.apply_to_learning_state(  # type: ignore[union-attr]
            self._mm.learning_state, result
        )
        self._mm.learning_state = new_state

        if self._evolver is not None:
            self._evolver.update(
                name=result.profile_name,
                education=result.profile_education,
                new_interests=result.profile_new_interests,
                new_projects=result.profile_new_projects,
                new_goals=result.profile_new_goals,
                confidence=result.importance or 0.5,
                source="extraction",
            )
        else:
            new_profile = self._extractor.apply_to_profile(  # type: ignore[union-attr]
                self._mm.profile, result
            )
            if new_profile is not self._mm.profile:
                self._mm.profile = new_profile

    def _bg_update_graph(self, payload: dict) -> None:
        if self._graph is None:
            return
        from core.memory_graph import EntityKind, RelationKind
        self._graph.upsert_relation(
            from_label=payload["from_label"],
            from_kind=EntityKind(payload["from_kind"]),
            relation=RelationKind(payload["relation"]),
            to_label=payload["to_label"],
            to_kind=EntityKind(payload["to_kind"]),
            confidence=payload.get("confidence", 1.0),
            source_memory_id=payload.get("source_memory_id"),
        )

    def _bg_update_graph_from_result(
        self, result: ExtractionResult, memory_id: int
    ) -> None:
        from core.memory_graph import EntityKind, RelationKind
        profile = self._effective_profile()
        person_label = profile.name or "user"
        for topic in result.topics:
            self._graph.upsert_relation(  # type: ignore[union-attr]
                from_label=person_label,
                from_kind=EntityKind.PERSON,
                relation=RelationKind.INTERESTED_IN,
                to_label=topic,
                to_kind=EntityKind.TOPIC,
                confidence=result.importance or 0.5,
                source_memory_id=memory_id,
            )
        for project in result.profile_new_projects:
            self._graph.upsert_relation(  # type: ignore[union-attr]
                from_label=person_label,
                from_kind=EntityKind.PERSON,
                relation=RelationKind.WORKS_ON,
                to_label=project,
                to_kind=EntityKind.PROJECT,
                confidence=result.importance or 0.5,
                source_memory_id=memory_id,
            )

    def _bg_regenerate_summary(self, payload: dict) -> None:
        if self._hierarchy is None or not self._session_memories:
            return
        self._session_index += 1
        self._hierarchy.create_session_summary(
            self._session_index, list(self._session_memories)
        )
        if self._session_memories:
            date_str = self._session_memories[-1].timestamp.strftime("%Y-%m-%d")
            self._hierarchy.roll_up_daily(date_str)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def flush_session(self) -> None:
        """Trigger session summary generation for current session."""
        if not self._session_memories:
            return
        if self._bg is not None:
            self._bg.submit(ProcessorJob.REGENERATE_SUMMARY, {
                "session_index": self._session_index + 1,
            })
        elif self._hierarchy is not None:
            self._session_index += 1
            self._hierarchy.create_session_summary(
                self._session_index, list(self._session_memories)
            )

    def new_session(self) -> None:
        """Mark the start of a new session."""
        self.flush_session()
        self._session_memories = []

    def shutdown(self) -> None:
        """Gracefully flush the session and stop background workers."""
        self.flush_session()
        if self._bg is not None:
            self._bg.stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_profile(self):
        if self._evolver is not None:
            return self._evolver.profile
        return self._mm.profile

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def rebuild_index(self) -> None:
        self._retriever.rebuild_index(self._mm.get_all_memories())
        log.info("Embedding index rebuilt (%d entries).", self.index_size)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def memory_manager(self) -> MemoryManager:
        return self._mm

    @property
    def index_size(self) -> int:
        return self._retriever.index_size

    @property
    def bg_stats(self) -> dict:
        if self._bg is None:
            return {}
        return self._bg.stats

    def _index_entry(self, entry: MemoryEntry) -> None:
        try:
            self._retriever.index_entry(entry)
        except Exception as exc:
            log.warning("Embedding index update failed (id=%d): %s", entry.id, exc)
