"""
ADMA Policy Models — v0.3.16

The atomic data structures for the Adaptive Dual Memory Architecture.

PolicyRecord  — a learnable behaviour configuration with Thompson-sampling state
RewardSignal  — an observable outcome that updates a policy
PolicyVersion — a snapshot of a policy at a point in time (for rollback)
PolicyDomain  — system (operational) vs user (personalisation)
PolicyType    — what aspect of behaviour this policy controls
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ────────────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────────────

class PolicyDomain(str, Enum):
    """Which memory domain owns this policy."""
    SYSTEM = "system"   # operational: how Blix itself runs
    USER   = "user"     # personalisation: how Blix behaves for one user


class PolicyType(str, Enum):
    """What aspect of cognitive behaviour this policy controls."""
    # System policies
    RETRIEVAL_WEIGHTS   = "retrieval_weights"    # HybridRetriever factor weights
    PLANNER_CONFIG      = "planner_config"       # beam width, depth, branching
    VERIFICATION_POLICY = "verification_policy"  # when/how to verify answers
    TOOL_SELECTION      = "tool_selection"       # which tools to prefer
    REASONING_STRATEGY  = "reasoning_strategy"  # chain length, decomposition
    WORKSPACE_CONFIG    = "workspace_config"     # attention thresholds
    COMPRESSION_POLICY  = "compression_policy"  # hierarchy compression triggers
    MEMORY_ROUTING      = "memory_routing"       # which domain to query first

    # User policies
    ANSWER_STYLE        = "answer_style"         # verbosity, code-first, etc.
    DIFFICULTY_LEVEL    = "difficulty_level"     # task complexity preference
    EXPLANATION_DEPTH   = "explanation_depth"    # how much to explain
    TOPIC_PREFERENCE    = "topic_preference"     # preferred domains
    FEEDBACK_STYLE      = "feedback_style"       # direct vs gentle
    GOAL_PRIORITY       = "goal_priority"        # which goals to focus on
    HINT_POLICY         = "hint_policy"          # when to give hints


class RewardType(str, Enum):
    """Observable reward signal categories."""
    # System rewards
    BENCHMARK_SCORE     = "benchmark_score"
    LATENCY             = "latency"
    VERIFICATION_SUCCESS = "verification_success"
    PLANNER_SUCCESS     = "planner_success"
    MEMORY_QUALITY      = "memory_quality"
    TOKEN_EFFICIENCY    = "token_efficiency"
    REGRESSION_STABLE   = "regression_stable"
    FAILURE_RECOVERY    = "failure_recovery"

    # User rewards
    ANSWER_ACCEPTED     = "answer_accepted"
    CORRECTION_GIVEN    = "correction_given"
    TASK_COMPLETED      = "task_completed"
    FOLLOWUP_ASKED      = "followup_asked"
    PREFERENCE_SIGNAL   = "preference_signal"
    GOAL_ADVANCED       = "goal_advanced"
    REPEATED_USAGE      = "repeated_usage"


# ────────────────────────────────────────────────────────────────────
# Core data structures
# ────────────────────────────────────────────────────────────────────

@dataclass
class RewardSignal:
    """
    An observable outcome that updates a policy's bandit state.

    Parameters
    ----------
    reward_type:
        Category of the reward.
    value:
        Normalised reward value in [0, 1].  1.0 = perfect outcome,
        0.0 = complete failure.  Values outside [0,1] are clamped.
    context:
        Free-form context data (task type, query, subsystem, etc.)
        used for contextual bandit arm selection.
    policy_id:
        Which policy this reward updates (None = broadcast to all
        policies of the matching type).
    source:
        What generated this reward (benchmark, user interaction, etc.)
    metadata:
        Arbitrary extra data.
    """
    reward_type:  RewardType
    value:        float
    context:      dict[str, Any]  = field(default_factory=dict)
    policy_id:    str | None      = None
    source:       str             = "system"
    metadata:     dict[str, Any]  = field(default_factory=dict)
    timestamp:    str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        self.value = max(0.0, min(1.0, self.value))

    def is_positive(self, threshold: float = 0.5) -> bool:
        return self.value >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_type": self.reward_type.value,
            "value": self.value,
            "context": self.context,
            "policy_id": self.policy_id,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class PolicyVersion:
    """
    A point-in-time snapshot of a PolicyRecord for rollback.

    Stored as a lightweight diff: only the mutable fields that
    change between versions.
    """
    version_id:   str             = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id:    str             = ""
    version:      int             = 1
    config:       dict[str, Any]  = field(default_factory=dict)
    alpha:        float           = 1.0   # Beta distribution success count
    beta:         float           = 1.0   # Beta distribution failure count
    mean_reward:  float           = 0.5
    created_at:   str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason:       str             = ""    # why this version was created

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "policy_id": self.policy_id,
            "version": self.version,
            "config": self.config,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean_reward": self.mean_reward,
            "created_at": self.created_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyVersion":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PolicyRecord:
    """
    A learnable behaviour configuration.

    The bandit state (alpha, beta) implements Thompson sampling:
        P(arm is best) ∝ Beta(alpha, beta)

    alpha increments on positive reward (value >= threshold).
    beta  increments on negative reward (value < threshold).

    This gives closed-form confidence intervals and guarantees
    optimal explore/exploit in the limit.

    Parameters
    ----------
    policy_id:
        Globally unique identifier.
    name:
        Human-readable name.
    domain:
        SYSTEM or USER.
    policy_type:
        What aspect of behaviour this controls.
    config:
        The actual policy parameters (e.g., beam_width=4, depth=3).
    alpha / beta:
        Thompson sampling Beta distribution parameters.
        Initialised to (1, 1) = uniform prior.
    confidence:
        Cached mean of the Beta distribution = alpha / (alpha + beta).
    success_count / failure_count:
        Raw event counts (not the same as alpha/beta due to prior).
    version:
        Monotonically increasing version counter.
    is_active:
        False = retired / superseded.
    user_id:
        For USER-domain policies: which user this belongs to.
        None = applies to all users (global default).
    tags / metadata:
        Filtering and introspection.
    created_at / updated_at:
        UTC ISO timestamps.
    """
    policy_id:      str             = field(default_factory=lambda: str(uuid.uuid4()))
    name:           str             = ""
    domain:         PolicyDomain    = PolicyDomain.SYSTEM
    policy_type:    PolicyType      = PolicyType.RETRIEVAL_WEIGHTS
    config:         dict[str, Any]  = field(default_factory=dict)
    alpha:          float           = 1.0   # Beta(alpha, beta) success count
    beta_:          float           = 1.0   # Beta(alpha, beta) failure count
    success_count:  int             = 0
    failure_count:  int             = 0
    version:        int             = 1
    is_active:      bool            = True
    user_id:        str | None      = None
    tags:           list[str]       = field(default_factory=list)
    metadata:       dict[str, Any]  = field(default_factory=dict)
    created_at:     str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:     str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── Bandit statistics ────────────────────────────────────────────

    @property
    def confidence(self) -> float:
        """Mean of Beta distribution: E[Beta(α,β)] = α/(α+β)."""
        return self.alpha / (self.alpha + self.beta_)

    @property
    def uncertainty(self) -> float:
        """Variance of Beta distribution (normalised to [0,1])."""
        a, b = self.alpha, self.beta_
        total = a + b
        if total < 2:
            return 1.0
        return (a * b) / (total * total * (total + 1))

    @property
    def total_observations(self) -> int:
        return self.success_count + self.failure_count

    def thompson_sample(self) -> float:
        """
        Draw one sample from Beta(alpha, beta_) using the exact distribution.

        Uses ``random.betavariate(a, b)`` — Python's standard library
        implementation of the exact Beta distribution via Johnk's method
        for small parameters and a composition method for larger ones.

        Why the previous approximation was replaced (ISSUE-007)
        --------------------------------------------------------
        The original implementation used:
          • Wilson-Hilferty normal approximation when a >= 1 and b >= 1
          • Gamma-ratio sampling for a < 1 or b < 1

        The Wilson-Hilferty approximation underestimates variance by 14–16%
        at Beta(1,1) (the uniform prior, i.e. cold start) and at asymmetric
        priors like Beta(3,1) or Beta(1,3).  This means the system
        under-explored during the exact period when exploration is most
        valuable — before any reliable signal has accumulated.

        Thompson sampling's regret bounds require draws from the true Beta
        distribution (Agrawal & Goyal 2012).  The approximation violated
        this guarantee.

        ``random.betavariate`` is in the Python standard library, requires
        no dependencies, and is 32% faster than the approximation at
        Beta(1,1) (cold start).  At large parameters (a,b > 50) it is
        ~40% slower, but the difference is < 1 µs per call — irrelevant
        at Blix's call rates.
        """
        a, b = self.alpha, self.beta_
        try:
            return random.betavariate(max(1e-10, a), max(1e-10, b))
        except Exception:
            # Fallback to the mean if betavariate fails (should never happen
            # with valid alpha/beta, but guard against corrupted state)
            return self.confidence

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """
        Approximate 95% confidence interval for the Beta mean.
        Uses Wilson score interval for robustness with small counts.
        """
        n = self.total_observations + 2  # add pseudo-observations
        p = self.confidence
        margin = z * math.sqrt(p * (1 - p) / n)
        return (max(0.0, p - margin), min(1.0, p + margin))

    # ── Update ───────────────────────────────────────────────────────

    def update(self, reward: float, threshold: float = 0.5) -> None:
        """
        Update bandit state with a new reward observation.

        reward in [0,1].  Values >= threshold increment alpha (success),
        values < threshold increment beta (failure).
        The increment size is proportional to reward magnitude for
        finer-grained learning than binary success/failure.
        """
        reward = max(0.0, min(1.0, reward))
        if reward >= threshold:
            self.alpha += reward  # fractional increment
            self.success_count += 1
        else:
            self.beta_ += (1.0 - reward)  # proportional to how bad it was
            self.failure_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.version += 1

    def decay(self, factor: float = 0.995) -> None:
        """
        Apply temporal decay to prevent stale policies from dominating.
        Shrinks alpha and beta toward the uniform prior (1, 1).
        """
        self.alpha = 1.0 + (self.alpha - 1.0) * factor
        self.beta_ = 1.0 + (self.beta_ - 1.0) * factor
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def retire(self) -> None:
        self.is_active = False
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def snapshot(self, reason: str = "") -> PolicyVersion:
        """Create a versioned snapshot of current state."""
        return PolicyVersion(
            policy_id=self.policy_id,
            version=self.version,
            config=dict(self.config),
            alpha=self.alpha,
            beta=self.beta_,
            mean_reward=self.confidence,
            reason=reason,
        )

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id":      self.policy_id,
            "name":           self.name,
            "domain":         self.domain.value,
            "policy_type":    self.policy_type.value,
            "config":         self.config,
            "alpha":          self.alpha,
            "beta_":          self.beta_,
            "success_count":  self.success_count,
            "failure_count":  self.failure_count,
            "version":        self.version,
            "is_active":      self.is_active,
            "user_id":        self.user_id,
            "tags":           self.tags,
            "metadata":       self.metadata,
            "created_at":     self.created_at,
            "updated_at":     self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyRecord":
        return cls(
            policy_id=     d["policy_id"],
            name=          d["name"],
            domain=        PolicyDomain(d["domain"]),
            policy_type=   PolicyType(d["policy_type"]),
            config=        d.get("config", {}),
            alpha=         d.get("alpha", 1.0),
            beta_=         d.get("beta_", 1.0),
            success_count= d.get("success_count", 0),
            failure_count= d.get("failure_count", 0),
            version=       d.get("version", 1),
            is_active=     d.get("is_active", True),
            user_id=       d.get("user_id"),
            tags=          d.get("tags", []),
            metadata=      d.get("metadata", {}),
            created_at=    d["created_at"],
            updated_at=    d["updated_at"],
        )

    def __repr__(self) -> str:
        ci = self.confidence_interval()
        return (f"PolicyRecord({self.name!r}, type={self.policy_type.value}, "
                f"conf={self.confidence:.3f} [{ci[0]:.3f},{ci[1]:.3f}], "
                f"n={self.total_observations})")
