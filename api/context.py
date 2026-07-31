"""
BlixContext — unified dependency container — Blix v0.3.3  (Feature 1 foundation)

Wires together every component from v0.3 / v0.3.1 / v0.3.2 into a single
object that can be:

* constructed once at process startup (CLI ``app.py`` or FastAPI ``api/server.py``)
* passed to ``TutorAgent`` for chat
* passed to ``MQLEngine`` for inspection
* exposed directly via REST endpoints in ``api/routers/*``

This is purely a wiring/composition module — no new behaviour. It exists
so the FastAPI layer and the CLI share exactly one construction path,
avoiding drift between the two entry points.

Python 3.10 compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.settings import settings
from core.background_processor import BackgroundProcessor
from core.embedding_store import EmbeddingStore
from core.fact_verifier import ConfidencePropagator, FactVerifier
from core.graph_reasoner import ContradictionDetector, GraphReasoner
from core.hierarchy_manager import HierarchyManager
from core.memory_extractor import MemoryExtractor
from core.memory_graph import MemoryGraph
from core.memory_lifecycle import MemoryLifecycleManager
from core.memory_manager import MemoryManager
from core.memory_retriever import MemoryRetriever
from core.memory_scorer import MemoryScorer, ScoringWeights
from core.profile_evolver import ProfileEvolver
from core.project_manager import ProjectManager
from core.prompt_builder import PromptBuilder
from core.retrieval_postprocessors import MMRReranker, ProjectBiasedRetriever
from core.semantic_clusters import SemanticClusterIndex
from core.semantic_retriever import SemanticRetriever
from core.tutor_agent import TutorAgent
from core.memory_types import TypeAwareRetriever
from core.cognitive_query_engine import CognitiveQueryEngine
from core.explainability import ExplainabilityEngine
from evaluation.agent_eval import AgentEvaluator
from evaluation.cognitive import CognitiveEvaluator
from knowledge.document_processor import DocumentProcessor
from knowledge.media_processor import MediaProcessor
from knowledge.research_assistant import ResearchAssistant
from knowledge.synthesis import KnowledgeSynthesisEngine
from llm.base import LLMProvider
from llm.provider_factory import build_provider
from reflection.consolidation_engine import ConsolidationEngine
from reflection.goal_tracker import GoalTracker
from reflection.insight_engine import InsightGenerationEngine
from reflection.mql import MQLEngine
from reflection.project_intelligence import ProjectIntelligenceEngine
from reflection.reflection_engine import ReflectionEngine
from reflection.scheduler import ReflectionScheduler
from utils.logger import get_logger

log = get_logger(__name__)


class BlixContext:
    """
    Holds one fully-wired instance of every Blix component.

    Construct via ``BlixContext.build()`` (the standard path) or directly
    for tests with a custom ``memory_dir``.

    Attributes are intentionally public — this is a composition root, not
    an encapsulated service. Both the CLI (``app.py``) and the API
    (``api/server.py``) read these attributes directly.
    """

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        memory_dir.mkdir(parents=True, exist_ok=True)

        mem_cfg = settings.memory
        embed_cfg = settings.embed
        llm_cfg = settings.llm

        # ------------------------------------------------------------
        # v0.2 / v0.3 core
        # ------------------------------------------------------------
        self.memory_manager = MemoryManager(
            conversations_file=mem_cfg.conversations_file,
            profile_file=mem_cfg.profile_file,
            learning_state_file=mem_cfg.learning_state_file,
        )
        self.embedding_store = EmbeddingStore(
            embed_model_name=embed_cfg.model,
            embeddings_file=embed_cfg.embeddings_file,
            ids_file=embed_cfg.embedding_ids_file,
            threshold=embed_cfg.threshold,
            top_k=embed_cfg.top_k,
        )
        legacy_retriever = MemoryRetriever(
            recent_k=mem_cfg.recent_k,
            fuzzy_top_k=mem_cfg.fuzzy_top_k,
            fuzzy_threshold=mem_cfg.fuzzy_threshold,
            keyword_top_k=mem_cfg.keyword_top_k,
        )
        self.retriever = SemanticRetriever(
            embedding_store=self.embedding_store,
            legacy_retriever=legacy_retriever,
            semantic_top_k=embed_cfg.top_k,
        )
        self.prompt_builder = PromptBuilder()
        self.llm: LLMProvider = build_provider(llm_cfg)
        self.extractor: Optional[MemoryExtractor] = (
            MemoryExtractor(llm=self.llm, enabled=True) if mem_cfg.auto_extract else None
        )

        # Re-index any memories missing from the embedding store
        unindexed = [
            m for m in self.memory_manager.get_all_memories()
            if m.id not in self.embedding_store.indexed_ids
        ]
        if unindexed:
            log.info("BlixContext: indexing %d unindexed memory(ies).", len(unindexed))
            self.retriever.rebuild_index(self.memory_manager.get_all_memories())

        # ------------------------------------------------------------
        # v0.3 — hierarchy, scoring, graph, projects, profile, background
        # ------------------------------------------------------------
        self.scorer = MemoryScorer(
            weights=ScoringWeights(relevance=0.4, importance=0.3, recency=0.2, frequency=0.1)
        )
        self.hierarchy = HierarchyManager(hierarchy_dir=memory_dir / "hierarchy", llm=self.llm)
        self.graph = MemoryGraph(graph_file=memory_dir / "graph.json")
        self.project_manager = ProjectManager(projects_file=memory_dir / "projects.json")
        self.profile_evolver = ProfileEvolver(versioned_profile_file=memory_dir / "versioned_profile.json")
        self.background = BackgroundProcessor(
            max_queue_size=100,
            worker_count=1,
            overflow_file=memory_dir / "bg_overflow.jsonl",
        )
        self.background.drain_overflow()

        # ------------------------------------------------------------
        # v0.3.1 — lifecycle, clusters, reasoning, contradictions, fact verification, retrieval post-processors
        # ------------------------------------------------------------
        self.lifecycle = MemoryLifecycleManager(lifecycle_file=memory_dir / "lifecycle.json")
        self.cluster_index = SemanticClusterIndex(clusters_file=memory_dir / "semantic_clusters.json")
        self.graph_reasoner = GraphReasoner(self.graph)
        self.contradiction_detector = ContradictionDetector(lifecycle_manager=self.lifecycle)
        self.fact_verifier = FactVerifier()
        self.confidence_propagator = ConfidencePropagator()
        self.project_biased_retriever = ProjectBiasedRetriever()
        self.mmr_reranker = MMRReranker(lambda_mmr=0.5, top_k=embed_cfg.top_k)
        self.type_aware_retriever = TypeAwareRetriever()

        # ------------------------------------------------------------
        # v0.3.2 — reflection, consolidation, goals, project intelligence,
        #          documents, media, synthesis, scheduler
        # ------------------------------------------------------------
        self.reflection = ReflectionEngine(reflections_file=memory_dir / "reflections.json", llm=self.llm)
        self.consolidation = ConsolidationEngine(facts_file=memory_dir / "canonical_facts.json")
        self.goals = GoalTracker(goals_file=memory_dir / "goals.json")
        self.project_intelligence = ProjectIntelligenceEngine(
            states_file=memory_dir / "project_intelligence.json",
            project_manager=self.project_manager,
        )
        self.document_processor = DocumentProcessor(llm=self.llm)
        self.media_processor = MediaProcessor(llm=self.llm)
        self.synthesis = KnowledgeSynthesisEngine(reports_file=memory_dir / "knowledge_reports.json", llm=self.llm)
        self.scheduler = ReflectionScheduler(schedule_file=memory_dir / "reflection_schedule.json")
        self.insight_engine = InsightGenerationEngine(insights_file=memory_dir / "actionable_insights.json", llm=self.llm)

        # ------------------------------------------------------------
        # v0.3.4 — cognitive query, explainability, research assistant
        # ------------------------------------------------------------
        self.cognitive_query_engine = CognitiveQueryEngine(
            graph=self.graph,
            reasoner=self.graph_reasoner,
        )
        self.explainability_engine = ExplainabilityEngine(
            memory_manager=self.memory_manager,
            retriever=self.retriever,
            consolidation_engine=self.consolidation,
            reflection_engine=self.reflection,
            graph=self.graph,
            graph_reasoner=self.graph_reasoner,
            cognitive_query_engine=self.cognitive_query_engine,
        )
        self.research_assistant = ResearchAssistant(
            notes_file=memory_dir / "research_notes.json",
            llm=self.llm,
            consolidation_engine=self.consolidation,
            graph=self.graph,
            synthesis_engine=self.synthesis,
        )

        # ------------------------------------------------------------
        # v0.3.5 — Agent Execution Framework
        # ------------------------------------------------------------
        from agents.executor import AgentExecutor, AgentSession, ExecutorConfig
        from agents.observation import ObservationLayer
        from agents.reflection_loop import ReflectionLoop
        from agents.working_memory import WorkingMemory
        from planning.planner import MilestoneTracker, Planner
        from tools.registry import (
            FileTool, LLMTool, MemorySearchTool, MemoryWriteTool,
            PythonTool, ReasoningTool, SynthesisTool, ToolRegistry, WebSearchTool,
        )

        self.agent_workspace = memory_dir / "agent_workspace"
        self.agent_workspace.mkdir(parents=True, exist_ok=True)

        self.tool_registry = ToolRegistry([
            MemorySearchTool(self.memory_manager, self.retriever),
            MemoryWriteTool(self.memory_manager),
            WebSearchTool(),
            FileTool(self.agent_workspace),
            PythonTool(),
            SynthesisTool(self.synthesis),
            ReasoningTool(self.cognitive_query_engine),
            LLMTool(self.llm),
        ])

        self.planner = Planner(llm=self.llm)
        self.milestone_tracker = MilestoneTracker(self.goals)

        self.agent_working_memory = WorkingMemory(max_entries=50, default_ttl=20)
        self.observation_layer = ObservationLayer(llm=self.llm)
        self.reflection_loop = ReflectionLoop(
            history_file=memory_dir / "execution_history.json",
            reflection_engine=self.reflection,
            memory_manager=self.memory_manager,
            llm=self.llm,
        )

        # ------------------------------------------------------------
        # v0.3.6 — Adaptive Planning & Verification Engine
        # ------------------------------------------------------------
        from agents.failure_memory import FailureMemory
        from agents.plan_reflection import PlanReflection
        from agents.tool_reliability import ToolReliabilityRegistry
        from planning.critic import PlanCritic
        from planning.replanner import Replanner
        from verification.verifier import VerificationEngine

        self.failure_memory = FailureMemory(memory_dir / "failure_memory.json")
        self.tool_reliability_registry = ToolReliabilityRegistry(memory_dir / "tool_reliability.json")
        self.plan_critic = PlanCritic(
            tool_registry=self.tool_registry,
            tool_reliability=self.tool_reliability_registry,
            failure_memory=self.failure_memory,
        )
        self.verification_engine = VerificationEngine()
        self.replanner = Replanner(
            tool_registry=self.tool_registry,
            failure_memory=self.failure_memory,
            tool_reliability=self.tool_reliability_registry,
        )
        self.plan_reflection = PlanReflection(
            failure_memory=self.failure_memory,
            reflection_engine=self.reflection,
        )

        self.agent_executor = AgentExecutor(
            tool_registry=self.tool_registry,
            working_memory=self.agent_working_memory,
            observation_layer=self.observation_layer,
            reflection_loop=self.reflection_loop,
            milestone_tracker=self.milestone_tracker,
            config=ExecutorConfig(max_steps=50, max_task_retries=2),
            plan_critic=self.plan_critic,
            verification_engine=self.verification_engine,
            replanner=self.replanner,
            plan_reflection=self.plan_reflection,
        )
        self.agent_session = AgentSession(
            planner=self.planner,
            executor=self.agent_executor,
            goal_tracker=self.goals,
        )
        from evaluation.agent_benchmark import AdaptiveAgentEvaluator
        self.agent_evaluator = AdaptiveAgentEvaluator()

        # ------------------------------------------------------------
        # v0.3.7 — Temporal State Tracking & Truth Maintenance
        # ------------------------------------------------------------
        from core.contradiction_resolver import ContradictionResolver
        from core.state_tracker import StateTracker
        from core.state_transition import StateTransitionEngine
        from core.truth_manager import TruthManager
        from evaluation.state_metrics import StateMetrics
        from graph.temporal_graph import TemporalGraph
        from memory.beliefs import BeliefStore
        from reasoning.temporal_query import TemporalQueryEngine
        from reflection.state_reflection import StateReflectionEngine
        from retrieval.temporal_retriever import TemporalRetriever

        self.state_tracker = StateTracker(memory_dir / "state_snapshots.json")
        self.state_transitions = StateTransitionEngine(
            tracker=self.state_tracker,
            transitions_file=memory_dir / "state_transitions.json",
        )
        self.truth_manager = TruthManager(memory_dir / "truth_records.json")
        self.belief_store = BeliefStore(memory_dir / "beliefs.json")
        self.contradiction_resolver = ContradictionResolver(
            truth_manager=self.truth_manager,
            belief_store=self.belief_store,
        )
        self.temporal_graph = TemporalGraph(memory_dir / "temporal_graph.json")
        self.temporal_retriever = TemporalRetriever(
            state_tracker=self.state_tracker,
            truth_manager=self.truth_manager,
            belief_store=self.belief_store,
        )
        self.temporal_query_engine = TemporalQueryEngine(
            state_tracker=self.state_tracker,
            transition_engine=self.state_transitions,
            temporal_graph=self.temporal_graph,
        )
        self.state_reflection = StateReflectionEngine(
            transition_engine=self.state_transitions,
        )
        self.state_metrics = StateMetrics()

        # ------------------------------------------------------------
        # v0.3.8 — Meta-Cognitive Layer
        # ------------------------------------------------------------
        from agents.execution_feedback import ExecutionFeedbackLoop
        from evaluation.metacognition_metrics import MetacognitionMetrics
        from memory.procedural_memory import ProceduralMemory
        from metacognition.capability_tracker import CapabilityTracker
        from metacognition.confidence_manager import ConfidenceManager
        from metacognition.controller import MetaCognitiveController
        from metacognition.self_model import SelfModelStore
        from metacognition.strategy_manager import StrategyManager
        from planning.plan_evaluator import PlanQualityEvaluator
        from reasoning.confidence_reasoner import ConfidenceReasoner
        from reflection.meta_reflection import MetaReflectionEngine

        self.self_model = SelfModelStore(memory_dir / "self_model.json")
        self.confidence_manager = ConfidenceManager(memory_dir / "confidence_records.json")
        self.confidence_reasoner = ConfidenceReasoner(tool_reliability=self.tool_reliability_registry)
        self.strategy_manager = StrategyManager()
        self.capability_tracker = CapabilityTracker(memory_dir / "capability_tracker.json")
        self.procedural_memory = ProceduralMemory(memory_dir / "procedural_memory.json")
        self.plan_evaluator = PlanQualityEvaluator(
            tool_reliability=self.tool_reliability_registry,
            confidence_reasoner=self.confidence_reasoner,
        )
        self.execution_feedback = ExecutionFeedbackLoop(
            feedback_file=memory_dir / "execution_feedback.json",
            failure_memory=self.failure_memory,
            capability_tracker=self.capability_tracker,
        )
        self.meta_reflection = MetaReflectionEngine(reflection_engine=self.reflection)
        self.meta_controller = MetaCognitiveController(
            plan_evaluator=self.plan_evaluator,
            confidence_reasoner=self.confidence_reasoner,
            strategy_manager=self.strategy_manager,
        )
        self.metacognition_metrics = MetacognitionMetrics()

        # ------------------------------------------------------------
        # v0.3.9 — Global Workspace
        # ------------------------------------------------------------
        from events.event_bus import EventBus
        from events.event_store import EventStore
        from events.event_types import EventType
        from workspace.attention_manager import AttentionManager
        from workspace.broadcast_bus import BroadcastBus
        from workspace.global_workspace import GlobalWorkspace
        from workspace.inner_dialogue import InnerDialogue, critic_voice, planner_voice, reflection_voice, self_model_voice
        from workspace.snapshot import WorkspaceSnapshotStore
        from workspace.workspace_memory import WorkspaceMemory
        from specialists.consensus import SpecialistConsensus
        from specialists.memory_specialist import MemorySpecialist
        from specialists.planning_specialist import PlanningSpecialist
        from specialists.reflection_specialist import ReflectionSpecialist
        from specialists.verification_specialist import VerificationSpecialist
        from retrieval.active_attention_retriever import ActiveAttentionRetriever
        from evaluation.workspace_metrics import WorkspaceMetrics
        from evaluation.attention_metrics import AttentionMetrics
        from evaluation.coordination_metrics import CoordinationMetrics

        self.event_store = EventStore(memory_dir / "event_log.json")
        self.event_bus = EventBus(event_store=self.event_store)
        self.attention_manager = AttentionManager()
        self.workspace_memory = WorkspaceMemory()
        self.broadcast_bus = BroadcastBus(event_bus=self.event_bus)
        self.global_workspace = GlobalWorkspace(
            attention_manager=self.attention_manager,
            workspace_memory=self.workspace_memory,
            broadcast_bus=self.broadcast_bus,
        )
        self.workspace_snapshots = WorkspaceSnapshotStore(memory_dir / "workspace_snapshots.json")

        self.active_attention_retriever = ActiveAttentionRetriever(base_retriever=MemoryRetriever())

        self.memory_specialist = MemorySpecialist(
            lookup_fn=lambda topic: ([b] if (b := self.belief_store.find_similar(topic)) else [])
        )
        self.planning_specialist = PlanningSpecialist(plan_evaluator=self.plan_evaluator)
        self.reflection_specialist = ReflectionSpecialist(failure_memory=self.failure_memory)
        self.verification_specialist = VerificationSpecialist(verification_engine=self.verification_engine)
        self.specialist_consensus = SpecialistConsensus([
            self.memory_specialist, self.planning_specialist,
            self.reflection_specialist, self.verification_specialist,
        ])

        self.inner_dialogue = InnerDialogue()
        self.inner_dialogue.register_voice("Self Model", self_model_voice(self.self_model, "general"))
        self.inner_dialogue.register_voice("Reflection", reflection_voice(self.failure_memory))

        self.workspace_metrics = WorkspaceMetrics()
        self.attention_metrics = AttentionMetrics()
        self.coordination_metrics = CoordinationMetrics()

        # ------------------------------------------------------------
        # v0.3.10 — Hybrid Symbolic + ML
        # ------------------------------------------------------------
        # Note: CrossEncoderReranker is constructed with attempt_model_load=False
        # by default (no network path to model hosts in most deployments of this
        # sandbox; avoids a ~5s network-timeout cost on every BlixContext startup).
        # Deployments with real model access can re-construct with
        # attempt_model_load=True, or call ctx.cross_encoder_reranker._model = ...
        from agents.tool_success_predictor import ToolSuccessPredictor
        from learning.continual_adapter import ContinualLearningAdapter
        from learning.failure_clusterer import FailureClusterer
        from memory.future_memory import FutureMemoryStore
        from memory.importance_model import MemoryImportancePredictor
        from memory.semantic_compressor import SemanticCompressor
        from metacognition.strategy_selector import StrategySelectorNetwork
        from procedural.skill_discovery import SkillDiscoveryEngine
        from reasoning.confidence_model import ConfidenceModel
        from retrieval.cross_encoder_reranker import CrossEncoderReranker
        from workspace.neural_attention import NeuralAttentionScorer
        from world_model.latent_world_model import LatentWorldModel
        from world_model.scenario_ranker import ScenarioRanker
        from world_model.value_network import ValueNetwork

        self.latent_world_model = LatentWorldModel(memory_dir / "world_model_examples.json")
        self.value_network = ValueNetwork(memory_dir / "value_network_examples.json")
        self.scenario_ranker = ScenarioRanker(value_network=self.value_network)
        self.cross_encoder_reranker = CrossEncoderReranker(attempt_model_load=False)
        self.confidence_model = ConfidenceModel(
            memory_dir / "confidence_model_examples.json", confidence_reasoner=self.confidence_reasoner,
        )
        self.tool_success_predictor = ToolSuccessPredictor(
            memory_dir / "tool_success_examples.json", tool_reliability=self.tool_reliability_registry,
        )
        self.neural_attention_scorer = NeuralAttentionScorer(self.attention_manager, memory_dir / "neural_attention_examples.json")
        self.strategy_selector = StrategySelectorNetwork(self.strategy_manager, memory_dir / "strategy_selector_examples.json")
        self.failure_clusterer = FailureClusterer(self.failure_memory)
        self.skill_discovery = SkillDiscoveryEngine(self.procedural_memory)
        self.future_memory = FutureMemoryStore(memory_dir / "future_memory.json")
        self.semantic_compressor = SemanticCompressor(llm=self.llm)
        self.memory_importance_predictor = MemoryImportancePredictor(memory_dir / "importance_model_examples.json")
        self.continual_learning = ContinualLearningAdapter(
            self_model=self.self_model, capability_tracker=self.capability_tracker,
            procedural_memory=self.procedural_memory, tool_success_predictor=self.tool_success_predictor,
            confidence_model=self.confidence_model, memory_importance_predictor=self.memory_importance_predictor,
        )

        # ------------------------------------------------------------
        # v0.3.11 — Causal Cognition
        # ------------------------------------------------------------
        from causality.belief_dependency_graph import BeliefDependencyGraph
        from causality.causal_memory import CausalMemoryStore
        from causality.causal_reflection import CausalReflection
        from causality.cause_graph import CauseGraph
        from causality.counterfactual_engine import CounterfactualScenarioEngine
        from causality.meta_causal_reflection import MetaCausalReflection
        from causality.principle import PrincipleStore
        from causality.principle_graph import PrincipleGraph
        from causality.principle_synthesizer import PrincipleSynthesizer
        from metacognition.strategy_evolution import StrategyEvolution

        # Phase 1 — foundations
        self.cause_graph = CauseGraph(memory_dir / "cause_graph.json")
        self.belief_dependency_graph = BeliefDependencyGraph(memory_dir / "belief_dependency_graph.json", self.belief_store)
        self.causal_memory = CausalMemoryStore(memory_dir / "causal_memory.json")

        # Phase 2 — synthesis
        self.principle_store = PrincipleStore(memory_dir / "principles.json")
        self.principle_synthesizer = PrincipleSynthesizer(
            self.principle_store, self.cause_graph, failure_clusterer=self.failure_clusterer, llm=self.llm,
        )
        self.principle_graph = PrincipleGraph(memory_dir / "principle_graph.json", self.principle_store)

        # Phase 3 — reflection & strategy, operating over principles
        self.causal_reflection = CausalReflection(
            reflection_engine=self.reflection, principle_store=self.principle_store, value_network=self.value_network,
        )
        self.meta_causal_reflection = MetaCausalReflection(self.cause_graph, principle_store=self.principle_store)
        self.strategy_evolution = StrategyEvolution(
            self.cause_graph, principle_store=self.principle_store, strategy_selector=self.strategy_selector,
        )

        # Phase 4 — lightweight counterfactuals
        self.counterfactual_engine = CounterfactualScenarioEngine(self.value_network, scenario_ranker=self.scenario_ranker)

        # ------------------------------------------------------------
        # v0.3.12 — Imagination + Search
        # ------------------------------------------------------------
        from planning.beam_search import BeamSearchPlanner
        from planning.search_critic import SearchCritic
        from evaluation.prediction_evaluator import PredictionEvaluator
        from simulation.trajectory_graph import TrajectoryGraph

        self.trajectory_graph = TrajectoryGraph()
        self.beam_search_planner = BeamSearchPlanner(self.value_network)
        self.search_critic = SearchCritic(value_network=self.value_network)
        self.prediction_evaluator = PredictionEvaluator(self.future_memory)

        # ------------------------------------------------------------
        # v0.3.13 — Curiosity + Active Experimentation
        # ------------------------------------------------------------
        from curiosity.curiosity_engine import CuriosityEngine
        from hypothesis.hypothesis_manager import HypothesisManager
        from experiments.experiment_planner import ExperimentPlanner
        from knowledge.knowledge_gap_tracker import KnowledgeGapTracker

        self.knowledge_gap_tracker = KnowledgeGapTracker(memory_dir / "knowledge_gaps.json")
        self.curiosity_engine = CuriosityEngine(
            self.belief_store, failure_memory=self.failure_memory,
            cause_graph=self.cause_graph, knowledge_gap_tracker=self.knowledge_gap_tracker,
        )
        self.hypothesis_manager = HypothesisManager(
            memory_dir / "hypotheses.json", belief_store=self.belief_store,
        )
        self.experiment_planner = ExperimentPlanner(
            memory_dir / "experiments.json", hypothesis_manager=self.hypothesis_manager,
        )

        # ------------------------------------------------------------
        # EventBus subscriptions (gap fix: wire handlers so events fire)
        # FAILURE → FailureMemory records it
        # CONFIDENCE_CHANGED → CuriosityEngine generates signals when confidence is low
        # BELIEF_UPDATED → BeliefDependencyGraph propagates if needed
        # ------------------------------------------------------------
        from events.event_types import EventType

        def _on_failure(event):
            payload = event.payload or {}
            self.failure_memory.record(
                task_title=payload.get("task_title", event.source),
                tool=payload.get("tool", "unknown"),
                failure=payload.get("failure", "unspecified failure"),
            )

        def _on_confidence_changed(event):
            payload = event.payload or {}
            confidence = payload.get("confidence", 1.0)
            if confidence < 0.4:
                # Surface low-confidence signal without blocking the event loop
                try:
                    self.curiosity_engine.generate_signals(top_k=3)
                except Exception:
                    pass

        self.event_bus.subscribe(EventType.FAILURE, _on_failure)
        self.event_bus.subscribe(EventType.CONFIDENCE_CHANGED, _on_confidence_changed)

        # ------------------------------------------------------------
        # v0.3.3 — evaluation + MQL + chat agent
        # ------------------------------------------------------------
        self.evaluator = CognitiveEvaluator()

        # v0.3.4 — upgrade MQL to MQLv2
        from reflection.mql_v2 import MQLv2Engine
        self.mql = MQLv2Engine(
            memory_manager=self.memory_manager,
            retriever=self.retriever,
            consolidation_engine=self.consolidation,
            reflection_engine=self.reflection,
            insight_engine=self.insight_engine,
            goal_tracker=self.goals,
            graph=self.graph,
            graph_reasoner=self.graph_reasoner,
            cognitive_query_engine=self.cognitive_query_engine,
            project_manager=self.project_manager,
            project_intelligence=self.project_intelligence,
            semantic_cluster_index=self.cluster_index,
            contradiction_detector=self.contradiction_detector,
            lifecycle_manager=self.lifecycle,
        )

        self.agent = TutorAgent(
            llm=self.llm,
            memory_manager=self.memory_manager,
            retriever=self.retriever,
            prompt_builder=self.prompt_builder,
            extractor=self.extractor,
            scorer=self.scorer,
            background_processor=self.background,
            hierarchy_manager=self.hierarchy,
            memory_graph=self.graph,
            project_manager=self.project_manager,
            profile_evolver=self.profile_evolver,
        )

        log.info(
            "BlixContext ready — memories=%d index=%d graph=%dN/%dE goals=%d projects=%d",
            self.memory_manager.memory_count(),
            self.retriever.index_size,
            self.graph.node_count, self.graph.edge_count,
            self.goals.count, self.project_manager.count,
        )

    # ----------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------

    @classmethod
    def build(cls, memory_dir: Optional[Path] = None) -> "BlixContext":
        """
        Build the standard context using ``memory/`` relative to the
        project root (matching v0.3 ``app.py`` layout), or a custom dir.
        """
        if memory_dir is None:
            memory_dir = Path(__file__).resolve().parent.parent / "memory"
        return cls(memory_dir)

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush session, persist all stores, and stop background workers."""
        # Persist in-memory state so nothing is lost
        try:
            self.belief_store.persist()
        except Exception:
            pass
        try:
            self.hypothesis_manager.expire_stale(max_age_days=30)
        except Exception:
            pass
        # Stop background workers
        try:
            self.agent.shutdown()
        except Exception:
            pass
        try:
            self.background_processor.shutdown()
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Dashboard stats (Feature 7)
    # ----------------------------------------------------------------

    def dashboard_stats(self) -> dict:
        """
        Aggregate counters across every subsystem for the
        ``/stats`` endpoint and CLI ``/stats`` command.
        """
        memories = self.memory_manager.get_all_memories()
        return {
            "memory_count": len(memories),
            "embedding_index_size": self.retriever.index_size,
            "knowledge_facts": self.consolidation.fact_count,
            "projects": self.project_manager.count,
            "projects_at_risk": len(self.project_intelligence.at_risk_projects()),
            "graph_nodes": self.graph.node_count,
            "graph_edges": self.graph.edge_count,
            "goals": self.goals.count,
            "active_goals": len(self.goals.list_goals(status=__import__(
                "reflection.goal_tracker", fromlist=["GoalStatus"]
            ).GoalStatus.ACTIVE)),
            "insights": self.reflection.insight_count,
            "reflection_records": self.reflection.record_count,
            "knowledge_reports": self.synthesis.count,
            "research_notes": self.research_assistant.count,
            "actionable_insights": self.insight_engine.count,
            "agent_sessions": self.agent_session.session_count,
            "execution_history_count": self.reflection_loop.history_count,
            "agent_success_rate": round(self.reflection_loop.success_rate(), 3),
            "failure_memory_count": self.failure_memory.count,
            "tool_reliability_tracked_count": self.tool_reliability_registry.tracked_tool_count,
            "state_snapshots_tracked": self.state_tracker.count,
            "state_transitions_recorded": self.state_transitions.count,
            "beliefs_tracked": self.belief_store.count,
            "temporal_graph_edges": self.temporal_graph.count,
            "tracked_capabilities": self.capability_tracker.tracked_domain_count,
            "learned_skills": self.procedural_memory.count,
            "confidence_records_tracked": self.confidence_manager.count,
            "execution_feedback_entries": self.execution_feedback.count,
            "workspace_cycle_count": self.global_workspace.cycle_count,
            "workspace_snapshots_stored": self.workspace_snapshots.count,
            "cognitive_events_logged": self.event_store.count,
            "broadcasts_sent": self.broadcast_bus.broadcast_count,
            "world_model_trained": self.latent_world_model.is_trained,
            "value_network_trained": self.value_network.is_trained,
            "tool_success_predictor_trained": self.tool_success_predictor.is_trained,
            "confidence_model_trained": self.confidence_model.is_trained,
            "future_predictions_tracked": self.future_memory.count,
            "continual_learning_events": self.continual_learning.event_count,
            "cause_graph_edges": self.cause_graph.count,
            "belief_dependency_edges": self.belief_dependency_graph.count,
            "causal_memories": self.causal_memory.count,
            "principles_synthesized": self.principle_store.count,
            "principle_graph_edges": self.principle_graph.count,
            "active_trajectories": self.trajectory_graph.count,
            "resolved_predictions": len(self.future_memory.resolved()),
            "knowledge_gaps": self.knowledge_gap_tracker.count,
            "pending_hypotheses": len(self.hypothesis_manager.pending()),
            "experiments_planned": self.experiment_planner.count,
            # --- v0.3.13 gap fix: additional operational metrics ---
            "truth_manager_beliefs": self.truth_manager.count if hasattr(self.truth_manager, "count") else None,
            "principles_synthesized": self.principle_store.count,
            "principle_graph_edges": self.principle_graph.count,
            "self_model_capabilities": len(self.self_model._model.capabilities),
            "self_model_weaknesses": len(self.self_model._model.weaknesses),
            "failure_clusters": len(self.failure_clusterer.recurring_clusters()) if self.failure_clusterer else 0,
            "workspace_broadcast_count": self.event_bus.broadcast_count if hasattr(self.event_bus, "broadcast_count") else 0,
            "semantic_clusters": self.cluster_index.cluster_count,
            "lifecycle_state_counts": self.lifecycle.state_counts(),
            "contradictions_unresolved": self.contradiction_detector.unresolved_count,
            "session_count": self.hierarchy.session_count,
            "daily_summaries": self.hierarchy.daily_count,
            "weekly_summaries": self.hierarchy.weekly_count,
            "background": self.background.stats,
        }
