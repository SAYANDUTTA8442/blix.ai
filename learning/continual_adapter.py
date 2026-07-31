"""
Continual Learning — Blix v0.3.10  (New module 12)

The orchestration layer that closes the loop on everything v0.3.10
adds: as Blix runs, every outcome (success, failure, feedback) should
propagate into the modules that hold Blix's self-knowledge —
``metacognition.self_model.SelfModelStore``,
``metacognition.capability_tracker.CapabilityTracker``, and
``memory.procedural_memory.ProceduralMemory`` — AND into the v0.3.10
learned models (``ToolSuccessPredictor``, ``ConfidenceModel``,
``MemoryImportancePredictor``), so the system actually improves over
many runs rather than re-deriving the same cold-start defaults every
time.

This module does not duplicate any of those stores' own update logic —
it is purely the fan-out layer that ensures ONE outcome observation
reaches every module that should learn from it, mirroring (and
extending into ML territory) the role
``agents.execution_feedback.ExecutionFeedbackLoop`` already plays for
the v0.3.8 symbolic trackers.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agents.tool_success_predictor import ToolSuccessPredictor
from memory.importance_model import MemoryImportancePredictor
from memory.procedural_memory import ProceduralMemory
from metacognition.capability_tracker import CapabilityTracker
from metacognition.self_model import SelfModelStore
from reasoning.confidence_model import ConfidenceModel
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ContinualLearningEvent:
    """One outcome fanned out across all learning targets, logged for audit."""

    domain: str
    success: bool
    tool: str = ""
    updated_targets: list[str] = field(default_factory=list)
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "domain": self.domain, "success": self.success, "tool": self.tool,
            "updated_targets": self.updated_targets, "recorded_at": self.recorded_at,
        }


class ContinualLearningAdapter:
    """
    Fans out a single outcome observation to every self-knowledge and
    learned-model target that should update from it.

    Parameters
    ----------
    self_model:
        ``SelfModelStore`` — re-synced from ``capability_tracker``
        whenever a domain crosses into confidently-measured territory.
    capability_tracker:
        ``CapabilityTracker`` — records the raw per-domain outcome.
    procedural_memory:
        Optional ``ProceduralMemory`` — successful multi-step sequences
        can be learned as skills.
    tool_success_predictor:
        Optional ``ToolSuccessPredictor`` — learns from tool outcomes.
    confidence_model:
        Optional ``ConfidenceModel`` — learns from answer-correctness outcomes.
    memory_importance_predictor:
        Optional ``MemoryImportancePredictor`` — learns from observed
        memory importance signals.
    """

    def __init__(
        self,
        self_model: SelfModelStore,
        capability_tracker: CapabilityTracker,
        procedural_memory: Optional[ProceduralMemory] = None,
        tool_success_predictor: Optional[ToolSuccessPredictor] = None,
        confidence_model: Optional[ConfidenceModel] = None,
        memory_importance_predictor: Optional[MemoryImportancePredictor] = None,
    ) -> None:
        self._self_model = self_model
        self._capability_tracker = capability_tracker
        self._procedural_memory = procedural_memory
        self._tool_success_predictor = tool_success_predictor
        self._confidence_model = confidence_model
        self._memory_importance_predictor = memory_importance_predictor
        self._event_log: list[ContinualLearningEvent] = []

    # ------------------------------------------------------------------
    # Core fan-out
    # ------------------------------------------------------------------

    def observe_task_outcome(
        self,
        domain: str,
        success: bool,
        tool: str = "",
        task_complexity_hint: float = 0.5,
        goal: str = "",
        steps: Optional[list[str]] = None,
        skill_name: Optional[str] = None,
    ) -> ContinualLearningEvent:
        """
        Fan out one task outcome to capability tracking, self-model
        sync, procedural-skill learning (on success with a step
        sequence), and the tool success predictor (if a tool was used).
        """
        updated: list[str] = []

        self._capability_tracker.record_outcome(domain, success)
        updated.append("capability_tracker")

        if self._capability_tracker.is_confident(domain):
            self._self_model.set_capability(domain, self._capability_tracker.accuracy(domain))
            updated.append("self_model")

        if success and self._procedural_memory is not None and steps:
            self._procedural_memory.learn_from_success(goal or domain, steps, name=skill_name)
            updated.append("procedural_memory")

        if tool and self._tool_success_predictor is not None:
            self._tool_success_predictor.observe_outcome(
                tool, success=success, task_complexity_hint=task_complexity_hint,
            )
            updated.append("tool_success_predictor")

        event = ContinualLearningEvent(domain=domain, success=success, tool=tool, updated_targets=updated)
        self._event_log.append(event)
        return event

    def observe_answer_outcome(
        self,
        was_correct: bool,
        evidence_count: int = 0,
        source_count: int = 0,
        contradicting_evidence_count: int = 0,
        verification_passed: Optional[bool] = None,
    ) -> Optional[ContinualLearningEvent]:
        """Fan out an answer-correctness outcome to the confidence model."""
        if self._confidence_model is None:
            return None
        self._confidence_model.observe_outcome(
            was_correct=was_correct, evidence_count=evidence_count, source_count=source_count,
            contradicting_evidence_count=contradicting_evidence_count, verification_passed=verification_passed,
        )
        event = ContinualLearningEvent(domain="answer_correctness", success=was_correct, updated_targets=["confidence_model"])
        self._event_log.append(event)
        return event

    def observe_memory_importance(
        self, observed_importance: float, heuristic_importance: float,
        input_text: str, output_text: str, retrieval_count: int = 0,
    ) -> Optional[ContinualLearningEvent]:
        """Fan out an observed-importance signal to the memory importance predictor."""
        if self._memory_importance_predictor is None:
            return None
        self._memory_importance_predictor.observe_true_importance(
            observed_importance=observed_importance, heuristic_importance=heuristic_importance,
            input_text=input_text, output_text=output_text, retrieval_count=retrieval_count,
        )
        event = ContinualLearningEvent(
            domain="memory_importance", success=observed_importance >= 0.5,
            updated_targets=["memory_importance_predictor"],
        )
        self._event_log.append(event)
        return event

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def recent_events(self, limit: int = 20) -> list[ContinualLearningEvent]:
        return self._event_log[-limit:]

    @property
    def event_count(self) -> int:
        return len(self._event_log)

    def learning_status(self) -> dict:
        """Summary of which learned models are trained vs. still cold-starting."""
        status = {}
        if self._tool_success_predictor is not None:
            status["tool_success_predictor"] = {
                "is_trained": self._tool_success_predictor.is_trained,
                "sample_count": self._tool_success_predictor.sample_count,
            }
        if self._confidence_model is not None:
            status["confidence_model"] = {
                "is_trained": self._confidence_model.is_trained,
                "sample_count": self._confidence_model.sample_count,
            }
        if self._memory_importance_predictor is not None:
            status["memory_importance_predictor"] = {
                "is_trained": self._memory_importance_predictor.is_trained,
                "sample_count": self._memory_importance_predictor.sample_count,
            }
        return status
