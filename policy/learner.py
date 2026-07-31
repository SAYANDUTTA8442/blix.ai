"""
PolicyLearner — contextual bandit policy learning via Thompson sampling.

For each (policy_type, context_key) pair, we maintain a set of
PolicyRecord "arms". On each selection, we draw from Beta(α, β) for
each arm and pick the highest sample (Thompson sampling). On each
observation, we update the chosen arm's α or β.

Context features are hashed into discrete bins so the bandit is
tractable without a learned feature model.

This is the core learning algorithm for ADMA. It is:
  • Provably optimal (Thompson sampling has sublinear regret)
  • Interpretable (confidence = α/(α+β), viewable at any time)
  • Fast (pure Python, no ML library)
  • Incremental (updates happen one reward at a time)
"""
from __future__ import annotations
import logging
import time
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType, RewardSignal, RewardType
)
from policy.store import PolicyStore

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Context hashing
# ────────────────────────────────────────────────────────────────────

def _context_key(context: dict[str, Any], features: list[str] | None = None) -> str:
    """
    Hash relevant context features into a discrete string key.

    We extract only the features that are semantically meaningful
    for policy selection (task_type, domain, user_id, query_length_bucket).
    This keeps the context space tractable.
    """
    if features is None:
        features = ["task_type", "domain", "user_id", "topic"]
    parts = []
    for f in features:
        v = context.get(f)
        if v is not None:
            parts.append(f"{f}={str(v)[:20]}")
    return "|".join(parts) if parts else "global"


def _query_length_bucket(query: str) -> str:
    n = len(query.split())
    if n <= 5:   return "short"
    if n <= 20:  return "medium"
    return "long"


# ────────────────────────────────────────────────────────────────────
# PolicyLearner
# ────────────────────────────────────────────────────────────────────

class PolicyLearner:
    """
    Contextual bandit learner for ADMA policy optimisation.

    Maintains Thompson-sampling state per policy via PolicyStore.
    Supports multi-arm selection, reward observation, and decay.

    Parameters
    ----------
    policy_store : PolicyStore
        Persistent store for PolicyRecord and version history.
    decay_factor : float
        Per-observation temporal decay applied to all non-selected arms
        to prevent stale dominance. Default 0.995 ≈ half-life of 139 obs.
    reward_threshold : float
        Reward values >= this are treated as successes (increment alpha).
        Values < this are failures (increment beta). Default 0.5.
    snapshot_every : int
        Save a PolicyVersion snapshot every N updates per policy.
    """

    def __init__(
        self,
        policy_store: PolicyStore,
        decay_factor: float = 0.995,
        reward_threshold: float = 0.5,
        snapshot_every: int = 20,
        decay_persist_every: int = 50,
        cache_max_size: int = 1000,
    ) -> None:
        self._store = policy_store
        self._decay_factor = decay_factor
        self._threshold = reward_threshold
        self._snapshot_every = snapshot_every
        self._decay_persist_every = decay_persist_every

        # ── Bounded LRU cache (ISSUE-004) ────────────────────────────
        # The old plain dict grew without bound: in a multi-user deployment
        # with 10,000 users × 15 policies each, the cache would hold
        # 150,000 PolicyRecord objects (~300 MB) indefinitely.
        #
        # We use an OrderedDict as a simple LRU:
        #   - On access (hit or miss→insert): move key to the end (MRU).
        #   - On insert when full: evict the front entry (LRU).
        #
        # cache_max_size=1000 fits comfortably in memory (< 2 MB) while
        # covering all active policies in typical single-tenant deployments
        # (default install has 15; large installs rarely exceed a few hundred).
        #
        # The cache is NOT thread-safe by itself — PolicyLearner is assumed
        # to be used from a single thread (PolicyStore has its own lock for
        # concurrent DB writes; the cache is per-learner in-process state).
        self._cache_max_size = cache_max_size
        self._cache: OrderedDict[str, PolicyRecord] = OrderedDict()

        # Track which context_key was last used for each policy (for update routing)
        self._last_selected: dict[str, str] = {}  # context_key → policy_id

        # ── Batched decay tracking (ISSUE-001) ──────────────────────
        # Instead of writing all N policies to SQLite on every observation,
        # we accumulate decay as a single epoch counter and flush to DB
        # only every `decay_persist_every` observations.
        #
        # For any arm not yet flushed, the pending decay is applied
        # in-memory when the arm is read from cache or DB. The
        # effective alpha/beta of an un-flushed arm is:
        #   effective_alpha = 1.0 + (stored_alpha - 1.0) * _decay_epoch
        #   effective_beta_ = 1.0 + (stored_beta_ - 1.0) * _decay_epoch
        #
        # _decay_observation_count: observations since last DB flush
        # _decay_epoch: cumulative decay factor = decay_factor^count
        #   (1.0 = no pending decay; < 1.0 = pending shrinkage)
        self._decay_observation_count: int = 0
        self._decay_epoch: float = 1.0

    # ── Policy registration ──────────────────────────────────────────

    def register(self, policy: PolicyRecord, overwrite: bool = False) -> PolicyRecord:
        """
        Register a new policy arm. If a policy with the same name,
        domain, and type already exists and overwrite=False, return
        the existing one.
        """
        existing = self._store.all_active(
            domain=policy.domain, policy_type=policy.policy_type)
        for e in existing:
            if e.name == policy.name and e.user_id == policy.user_id:
                if not overwrite:
                    self._cache_put(e.policy_id, e)
                    return e
                # overwrite: save current state as version, then update config
                self._store.save_version(e.snapshot(reason="overwrite"))
                e.config = policy.config
                e.version += 1
                self._store.save(e)
                self._cache_put(e.policy_id, e)
                return e
        self._store.save(policy)
        self._cache_put(policy.policy_id, policy)
        log.debug("PolicyLearner: registered %s (id=%s)", policy.name, policy.policy_id[:8])
        return policy

    def register_defaults(self) -> None:
        """Register the default policy set for a fresh Blix installation."""
        defaults = _default_policies()
        for p in defaults:
            self.register(p)
        log.info("PolicyLearner: %d default policies registered", len(defaults))

    # ── Arm selection (Thompson sampling) ────────────────────────────

    def select(
        self,
        policy_type: PolicyType,
        domain: PolicyDomain = PolicyDomain.SYSTEM,
        context: dict[str, Any] | None = None,
        user_id: str | None = None,
        top_k: int = 1,
    ) -> list[PolicyRecord]:
        """
        Select the best policy arm(s) via Thompson sampling.

        For each active policy of the given type, draw a sample from
        Beta(α, β) and rank by the drawn value. Return the top_k.

        Parameters
        ----------
        policy_type : PolicyType
            What kind of policy to select.
        domain : PolicyDomain
            System or user domain.
        context : dict
            Current context features for contextual routing.
        user_id : str | None
            For user-domain policies: which user's policies to prefer.
        top_k : int
            How many policies to return (default 1 = pure exploitation).
        """
        context = context or {}
        arms = self._store.all_active(
            domain=domain, policy_type=policy_type, user_id=user_id)

        if not arms:
            return []

        # Refresh cache
        for arm in arms:
            self._cache_put(arm.policy_id, arm)

        # Thompson sampling: draw from Beta(α, β) for each arm
        scored = [(arm, arm.thompson_sample()) for arm in arms]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [arm for arm, _ in scored[:top_k]]
        ctx_key = _context_key(context)
        for arm in selected:
            self._last_selected[ctx_key] = arm.policy_id
            arm.touch_time = time.time()  # temporary attribute for diagnostics

        log.debug(
            "PolicyLearner.select: type=%s ctx=%s → %s (conf=%.3f)",
            policy_type.value, ctx_key,
            selected[0].name if selected else "none",
            selected[0].confidence if selected else 0.0,
        )
        return selected

    def select_one(
        self,
        policy_type: PolicyType,
        domain: PolicyDomain = PolicyDomain.SYSTEM,
        context: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> PolicyRecord | None:
        results = self.select(policy_type, domain, context, user_id, top_k=1)
        return results[0] if results else None

    # ── Reward observation ───────────────────────────────────────────

    def observe(self, reward: RewardSignal) -> list[PolicyRecord]:
        """
        Update policy arms with a new reward signal.

        If reward.policy_id is set, update only that arm.
        Otherwise, update all arms of the matching reward_type→policy_type
        mapping that were recently selected.

        Returns the list of updated PolicyRecords.
        """
        self._store.log_reward(reward)
        updated = []

        if reward.policy_id:
            policy = self._get_cached(reward.policy_id)
            if policy and policy.is_active:
                self._update_arm(policy, reward.value)
                updated.append(policy)
        else:
            # Broadcast: update all arms of inferred policy type
            inferred_types = _reward_to_policy_types(reward.reward_type)
            for pt in inferred_types:
                domain = _reward_domain(reward.reward_type)
                arms = self._store.all_active(policy_type=pt, domain=domain)
                for arm in arms:
                    self._update_arm(arm, reward.value)
                    updated.append(arm)

        # Decay non-updated arms of the same types
        if updated:
            self._apply_decay(exclude_ids={p.policy_id for p in updated})

        return updated

    def observe_batch(self, rewards: list[RewardSignal]) -> int:
        """Process multiple reward signals. Returns count of updated policies."""
        total = 0
        for r in rewards:
            total += len(self.observe(r))
        return total

    # ── Internal update ──────────────────────────────────────────────

    def _update_arm(self, policy: PolicyRecord, reward_value: float) -> None:
        """Apply reward to a single arm and persist."""
        old_version = policy.version
        policy.update(reward_value, threshold=self._threshold)
        self._cache_put(policy.policy_id, policy)

        # Snapshot every N updates
        if policy.version % self._snapshot_every == 0:
            self._store.save_version(
                policy.snapshot(reason=f"auto-snapshot at v{policy.version}"))

        self._store.save(policy)
        log.debug(
            "PolicyLearner: updated %s → conf=%.3f (n=%d)",
            policy.name, policy.confidence, policy.total_observations)

    # ── Batched decay (ISSUE-001) ─────────────────────────────────────

    def _apply_decay(self, exclude_ids: set[str]) -> None:
        """
        Record one observation's worth of decay without writing to DB.

        Each call advances the epoch by one factor. Excluded (just-updated)
        arms are written immediately with their rewards; their stored alpha/beta
        already include the reward update so we do NOT apply the epoch to them.
        The epoch is flushed to DB every `decay_persist_every` observations.
        """
        if not exclude_ids:
            return  # nothing was updated, no decay needed

        # Advance the epoch: multiply accumulated factor by decay_factor
        self._decay_epoch *= self._decay_factor
        self._decay_observation_count += 1

        # Apply epoch in-memory to all cached arms that are NOT the updated ones
        for pid, arm in self._cache.items():
            if pid not in exclude_ids:
                self._apply_epoch_to_policy(arm)

        # Periodically flush to DB so persistent state stays accurate
        if self._decay_observation_count >= self._decay_persist_every:
            self.flush_decay(exclude_ids=exclude_ids)

    def _apply_epoch_to_policy(self, policy: "PolicyRecord") -> None:
        """
        Apply the pending decay epoch in-memory (no DB write).

        Mathematically equivalent to calling policy.decay(factor) once
        per observation since the last flush. Uses the accumulated
        multiplicative epoch factor for efficiency.
        """
        epoch = self._decay_epoch
        if abs(epoch - 1.0) < 1e-10:
            return  # no pending decay
        policy.alpha = 1.0 + (policy.alpha - 1.0) * epoch
        policy.beta_ = 1.0 + (policy.beta_ - 1.0) * epoch

    def flush_decay(self, exclude_ids: set[str] | None = None) -> int:
        """
        Write accumulated decay for all non-excluded active policies to DB.

        Called automatically every `decay_persist_every` observations.
        Can also be called explicitly (e.g., before shutdown) to ensure
        persistent state is current.

        Returns number of policies written.
        """
        if abs(self._decay_epoch - 1.0) < 1e-10:
            # No pending decay — reset counter and return
            self._decay_observation_count = 0
            return 0

        exclude_ids = exclude_ids or set()
        all_arms = self._store.all_active(limit=500)
        written = 0
        for arm in all_arms:
            if arm.policy_id in exclude_ids:
                continue
            # Use cached version (has epoch applied in-memory) if available
            cached = self._cache_get(arm.policy_id)
            if cached is not None:
                arm = cached  # epoch already applied; write current state
            else:
                # Arm was not in cache — apply epoch now before writing
                self._apply_epoch_to_policy(arm)
                self._cache_put(arm.policy_id, arm)
            self._store.save(arm)
            written += 1

        # Reset epoch: decay is now persisted
        self._decay_epoch = 1.0
        self._decay_observation_count = 0
        log.debug("PolicyLearner: flushed decay for %d policies", written)
        return written

    def _cache_put(self, policy_id: str, policy: "PolicyRecord") -> None:
        """
        Insert or update an entry in the LRU cache.

        Moves the entry to the MRU (right) end.  If the cache is at
        capacity, evicts the LRU (left) entry before inserting.

        O(1) for both hit and miss — OrderedDict.move_to_end is O(1).
        """
        if policy_id in self._cache:
            self._cache.move_to_end(policy_id)
        else:
            if len(self._cache) >= self._cache_max_size:
                self._cache.popitem(last=False)  # evict LRU (leftmost)
        self._cache[policy_id] = policy

    def _cache_get(self, policy_id: str) -> "PolicyRecord | None":
        """
        Retrieve an entry and promote it to MRU.
        Returns None on miss (caller should fall through to DB).
        """
        policy = self._cache.get(policy_id)
        if policy is not None:
            self._cache.move_to_end(policy_id)
        return policy

    def _get_cached(self, policy_id: str) -> PolicyRecord | None:
        p = self._cache_get(policy_id)
        if p is not None:
            return p
        p = self._store.get(policy_id)
        if p:
            # Apply any pending epoch so the in-memory view is current
            self._apply_epoch_to_policy(p)
            self._cache_put(policy_id, p)
        return p

    # ── Diagnostics ──────────────────────────────────────────────────

    def policy_summary(
        self,
        domain: PolicyDomain | None = None,
        policy_type: PolicyType | None = None,
    ) -> list[dict[str, Any]]:
        """Return a summary table of current policy states."""
        arms = self._store.all_active(domain=domain, policy_type=policy_type)
        rows = []
        for arm in arms:
            ci = arm.confidence_interval()
            rows.append({
                "name":           arm.name,
                "domain":         arm.domain.value,
                "type":           arm.policy_type.value,
                "confidence":     round(arm.confidence, 4),
                "ci_lower":       round(ci[0], 4),
                "ci_upper":       round(ci[1], 4),
                "uncertainty":    round(arm.uncertainty, 4),
                "successes":      arm.success_count,
                "failures":       arm.failure_count,
                "total_obs":      arm.total_observations,
                "version":        arm.version,
                "config":         arm.config,
            })
        return sorted(rows, key=lambda r: r["confidence"], reverse=True)

    def learning_curve(self, policy_id: str) -> list[dict[str, Any]]:
        """Return confidence at each version checkpoint."""
        history = self._store.get_history(policy_id)
        return [{"version": v.version, "confidence": v.mean_reward,
                 "alpha": v.alpha, "beta": v.beta} for v in history]

    def rollback(self, policy_id: str, to_version: int) -> PolicyRecord | None:
        result = self._store.rollback(policy_id, to_version)
        if result:
            self._cache_put(policy_id, result)
        return result


# ────────────────────────────────────────────────────────────────────
# Reward → PolicyType mapping
# ────────────────────────────────────────────────────────────────────

def _reward_to_policy_types(reward_type: RewardType) -> list[PolicyType]:
    mapping = {
        RewardType.BENCHMARK_SCORE:      [PolicyType.RETRIEVAL_WEIGHTS, PolicyType.PLANNER_CONFIG],
        RewardType.LATENCY:              [PolicyType.RETRIEVAL_WEIGHTS, PolicyType.PLANNER_CONFIG],
        RewardType.PLANNER_SUCCESS:      [PolicyType.PLANNER_CONFIG],
        RewardType.MEMORY_QUALITY:       [PolicyType.RETRIEVAL_WEIGHTS],
        RewardType.VERIFICATION_SUCCESS: [PolicyType.VERIFICATION_POLICY],
        RewardType.TOKEN_EFFICIENCY:     [PolicyType.REASONING_STRATEGY],
        RewardType.REGRESSION_STABLE:    [PolicyType.PLANNER_CONFIG],
        RewardType.FAILURE_RECOVERY:     [PolicyType.REASONING_STRATEGY],
        RewardType.ANSWER_ACCEPTED:      [PolicyType.ANSWER_STYLE],
        RewardType.CORRECTION_GIVEN:     [PolicyType.ANSWER_STYLE, PolicyType.EXPLANATION_DEPTH],
        RewardType.TASK_COMPLETED:       [PolicyType.DIFFICULTY_LEVEL],
        RewardType.FOLLOWUP_ASKED:       [PolicyType.EXPLANATION_DEPTH],
        RewardType.PREFERENCE_SIGNAL:    [PolicyType.ANSWER_STYLE, PolicyType.TOPIC_PREFERENCE],
        RewardType.GOAL_ADVANCED:        [PolicyType.GOAL_PRIORITY],
        RewardType.REPEATED_USAGE:       [PolicyType.ANSWER_STYLE],
    }
    return mapping.get(reward_type, [])


def _reward_domain(reward_type: RewardType) -> PolicyDomain:
    user_rewards = {
        RewardType.ANSWER_ACCEPTED, RewardType.CORRECTION_GIVEN,
        RewardType.TASK_COMPLETED, RewardType.FOLLOWUP_ASKED,
        RewardType.PREFERENCE_SIGNAL, RewardType.GOAL_ADVANCED,
        RewardType.REPEATED_USAGE,
    }
    return PolicyDomain.USER if reward_type in user_rewards else PolicyDomain.SYSTEM


# ────────────────────────────────────────────────────────────────────
# Default policy initialisation
# ────────────────────────────────────────────────────────────────────

def _default_policies() -> list[PolicyRecord]:
    """
    The default policy set installed on a fresh Blix system.

    Each policy starts with a uniform prior Beta(1, 1) = 50% confidence.
    The bandit will learn the correct values from experience.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    defaults = []

    # ── System: Retrieval weight policies (one per strategy) ─────────
    retrieval_arms = [
        ("retrieval_balanced",
         {**{k: 0.091 for k in ["semantic","vector","graph_distance","importance",
             "confidence","recency","hierarchy","context_similarity","attention",
             "belief_confidence","planning_relevance"]}}),
        ("retrieval_semantic_heavy",
         {"semantic": 0.35, "vector": 0.30, "graph_distance": 0.05,
          "importance": 0.10, "confidence": 0.08, "recency": 0.05,
          "hierarchy": 0.03, "context_similarity": 0.02, "attention": 0.01,
          "belief_confidence": 0.005, "planning_relevance": 0.005}),
        ("retrieval_graph_heavy",
         {"semantic": 0.15, "vector": 0.10, "graph_distance": 0.25,
          "importance": 0.20, "confidence": 0.12, "recency": 0.06,
          "hierarchy": 0.06, "context_similarity": 0.03, "attention": 0.02,
          "belief_confidence": 0.005, "planning_relevance": 0.005}),
    ]
    for name, config in retrieval_arms:
        defaults.append(PolicyRecord(
            name=name, domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS, config=config,
            created_at=now, updated_at=now))

    # ── System: Planner config policies ─────────────────────────────
    planner_arms = [
        ("planner_conservative", {"beam_width": 3, "max_depth": 2, "branching": 3}),
        ("planner_balanced",     {"beam_width": 5, "max_depth": 3, "branching": 4}),
        ("planner_aggressive",   {"beam_width": 8, "max_depth": 4, "branching": 6}),
    ]
    for name, config in planner_arms:
        defaults.append(PolicyRecord(
            name=name, domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG, config=config,
            created_at=now, updated_at=now))

    # ── System: Reasoning strategy policies ─────────────────────────
    reasoning_arms = [
        ("reasoning_direct",     {"chain_length": 1, "decompose": False, "verify": False}),
        ("reasoning_stepwise",   {"chain_length": 3, "decompose": True,  "verify": True}),
        ("reasoning_exhaustive", {"chain_length": 5, "decompose": True,  "verify": True}),
    ]
    for name, config in reasoning_arms:
        defaults.append(PolicyRecord(
            name=name, domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.REASONING_STRATEGY, config=config,
            created_at=now, updated_at=now))

    # ── User: Answer style policies ──────────────────────────────────
    style_arms = [
        ("style_concise",  {"verbosity": "low",  "code_first": False, "examples": False}),
        ("style_balanced", {"verbosity": "med",  "code_first": True,  "examples": True}),
        ("style_verbose",  {"verbosity": "high", "code_first": False, "examples": True}),
    ]
    for name, config in style_arms:
        defaults.append(PolicyRecord(
            name=name, domain=PolicyDomain.USER,
            policy_type=PolicyType.ANSWER_STYLE, config=config,
            created_at=now, updated_at=now))

    # ── User: Difficulty level policies ──────────────────────────────
    diff_arms = [
        ("difficulty_easy",   {"level": 1, "hints": True,  "scaffolding": "full"}),
        ("difficulty_medium", {"level": 3, "hints": True,  "scaffolding": "partial"}),
        ("difficulty_hard",   {"level": 5, "hints": False, "scaffolding": "none"}),
    ]
    for name, config in diff_arms:
        defaults.append(PolicyRecord(
            name=name, domain=PolicyDomain.USER,
            policy_type=PolicyType.DIFFICULTY_LEVEL, config=config,
            created_at=now, updated_at=now))

    return defaults
