"""
blix_eval — Advanced Evaluation Framework deliverable  (Blix v0.3.2, Feature 5)

This package is the spec-mandated ``blix_eval/`` deliverable. It re-exports
the full evaluation stack (v0.3 → v0.3.2) under a single stable namespace:

    from blix_eval import (
        MemoryEvaluator,        # v0.3 — precision/recall/F1/fact/profile/graph/summary
        ExtendedMemoryEvaluator,# v0.3.1 — retention, forgetting curve, drift, hypotheses
        CognitiveEvaluator,     # v0.3.2 — Recall@K, MRR, project/milestone/insight metrics
        EvalCase, EvalDataset, EvalReport,
        HypothesisRegistry, ResearchHypothesis, HypothesisStatus,
    )

    cog = CognitiveEvaluator()
    cog.print_report(cog.evaluate(dataset))
    cog.evaluate_cognitive(retrieval_results=..., lifecycle_manager=lm, ...)

Implementation lives in ``evaluation/`` (this package's parent); this
module is a thin re-export layer matching the deliverable name from the
v0.3.2 spec ("Deliverable: blix_eval/").
"""

from __future__ import annotations

from evaluation import EvalCase, EvalDataset, EvalReport, MemoryEvaluator, MetricResult
from evaluation.agent_benchmark import AdaptiveAgentEvaluator, AgentBenchmarkCase
from evaluation.agent_eval import AgentEvalCase, AgentEvaluator
from evaluation.cognitive import CognitiveEvaluator
from evaluation.reasoning import ReasoningEvaluator, ReasoningCase
from evaluation.research import (
    ExtendedMemoryEvaluator,
    HypothesisRegistry,
    HypothesisStatus,
    ResearchHypothesis,
)
from evaluation.state_metrics import StateAccuracyCase, StateMetrics, TransitionAccuracyCase
from evaluation.confidence_metrics import CalibrationCase, CalibrationBucketResult, ConfidenceMetrics
from evaluation.capability_metrics import CapabilityMetrics, SelfAwarenessGap
from evaluation.metacognition_metrics import AdaptationCase, MetacognitionMetrics
from evaluation.workspace_metrics import WorkspaceCycleStats, WorkspaceMetrics
from evaluation.attention_metrics import AttentionGroundTruthCase, AttentionMetrics
from evaluation.coordination_metrics import CoordinationMetrics, SubsystemParticipation

__all__ = [
    "MemoryEvaluator",
    "ExtendedMemoryEvaluator",
    "CognitiveEvaluator",
    "ReasoningEvaluator",
    "ReasoningCase",
    "AgentEvaluator",
    "AgentEvalCase",
    "AdaptiveAgentEvaluator",
    "AgentBenchmarkCase",
    "StateMetrics",
    "StateAccuracyCase",
    "TransitionAccuracyCase",
    "ConfidenceMetrics",
    "CalibrationCase",
    "CalibrationBucketResult",
    "CapabilityMetrics",
    "SelfAwarenessGap",
    "MetacognitionMetrics",
    "AdaptationCase",
    "WorkspaceMetrics",
    "WorkspaceCycleStats",
    "AttentionMetrics",
    "AttentionGroundTruthCase",
    "CoordinationMetrics",
    "SubsystemParticipation",
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "MetricResult",
    "HypothesisRegistry",
    "ResearchHypothesis",
    "HypothesisStatus",
]
