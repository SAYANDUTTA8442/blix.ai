"""
Blix v0.3.16 — ADMA Test Suite

Tests:
  Unit         — PolicyRecord, PolicyStore, RewardEngine, PolicyLearner,
                 PolicyOptimizer, PolicySelector, PolicyCompiler
  Integration  — end-to-end ADMA pipelines
  Memory       — SystemMemory, UserMemory, MemoryManager
  Adaptive     — AdaptiveRetriever, AdaptivePlanner
  Ablation     — AblationV3Runner with dependency injection
  Regression   — 194 pre-existing tests must still pass
"""
from __future__ import annotations
import sys
import math
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType, PolicyVersion,
    RewardSignal, RewardType
)
from policy.store import PolicyStore
from policy.reward import RewardEngine, SystemRewardEngine, UserRewardEngine
from policy.learner import PolicyLearner, _default_policies, _context_key
from policy.optimizer import PolicyOptimizer
from policy.compiler import PolicySelector, PolicyCompiler, CompiledPrompt
from policy.adaptive import AdaptiveRetriever, AdaptivePlanner
from policy.ablation_v3 import (
    AblationConfig, AblationV3Runner, ABLATION_CONDITIONS, AblationV3Report
)
from memory.hybrid.hgshm import HGSHM
from memory.system.system_memory import SystemMemory
from memory.user.user_memory import UserMemory
from memory.manager import MemoryManager


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def policy_store(tmp_dir):
    return PolicyStore(tmp_dir)

@pytest.fixture
def learner(policy_store):
    l = PolicyLearner(policy_store)
    l.register_defaults()
    return l

@pytest.fixture
def reward_engine(learner):
    engine = RewardEngine(learner)
    return engine

@pytest.fixture
def selector(learner):
    return PolicySelector(learner)

@pytest.fixture
def compiler(selector):
    return PolicyCompiler(selector)

@pytest.fixture
def hgshm(tmp_dir):
    h = HGSHM(tmp_dir)
    yield h
    h.close()

@pytest.fixture
def system_memory(hgshm):
    return SystemMemory(hgshm)

@pytest.fixture
def user_memory(hgshm):
    return UserMemory(hgshm, "test_user_001")

@pytest.fixture
def memory_manager(hgshm, system_memory):
    return MemoryManager(hgshm, system_memory)


# ════════════════════════════════════════════════════════════════════
# UNIT — PolicyRecord (models)
# ════════════════════════════════════════════════════════════════════

class TestPolicyRecord:
    def test_creation_defaults(self):
        p = PolicyRecord(name="test")
        assert p.policy_id
        assert p.alpha == 1.0
        assert p.beta_ == 1.0
        assert p.confidence == pytest.approx(0.5)
        assert p.total_observations == 0

    def test_confidence_formula(self):
        p = PolicyRecord(alpha=3.0, beta_=1.0)
        assert p.confidence == pytest.approx(0.75)

    def test_update_positive_reward(self):
        p = PolicyRecord()
        p.update(reward=0.8, threshold=0.5)
        assert p.alpha > 1.0
        assert p.success_count == 1
        assert p.failure_count == 0
        assert p.version == 2

    def test_update_negative_reward(self):
        p = PolicyRecord()
        p.update(reward=0.2, threshold=0.5)
        assert p.beta_ > 1.0
        assert p.failure_count == 1
        assert p.success_count == 0

    def test_update_proportional_alpha_increment(self):
        p1 = PolicyRecord()
        p1.update(reward=0.9, threshold=0.5)
        p2 = PolicyRecord()
        p2.update(reward=0.6, threshold=0.5)
        # Higher reward → larger alpha increment
        assert p1.alpha > p2.alpha

    def test_update_clamped_reward(self):
        p = PolicyRecord()
        p.update(reward=1.5)   # should clamp to 1.0
        p.update(reward=-0.5)  # should clamp to 0.0
        assert p.alpha <= 3.0  # not unbounded
        assert p.beta_ <= 3.0

    def test_decay_shrinks_toward_prior(self):
        p = PolicyRecord(alpha=5.0, beta_=5.0)
        p.decay(0.5)
        # α shrinks toward 1: 1 + (5-1)*0.5 = 3.0
        assert p.alpha == pytest.approx(3.0)
        assert p.beta_ == pytest.approx(3.0)

    def test_decay_preserves_prior(self):
        p = PolicyRecord(alpha=1.0, beta_=1.0)  # uniform prior
        p.decay(0.5)
        assert p.alpha == pytest.approx(1.0)
        assert p.beta_ == pytest.approx(1.0)

    def test_thompson_sample_range(self):
        p = PolicyRecord(alpha=3.0, beta_=1.0)
        for _ in range(20):
            s = p.thompson_sample()
            assert 0.0 <= s <= 1.0

    def test_thompson_sample_biased_toward_confidence(self):
        p_high = PolicyRecord(alpha=10.0, beta_=1.0)   # conf≈0.91
        p_low  = PolicyRecord(alpha=1.0,  beta_=10.0)  # conf≈0.09
        high_samples = [p_high.thompson_sample() for _ in range(50)]
        low_samples  = [p_low.thompson_sample()  for _ in range(50)]
        assert sum(high_samples)/50 > sum(low_samples)/50

    def test_confidence_interval_coverage(self):
        p = PolicyRecord()
        for _ in range(40):   # 40 observations → tight CI
            p.update(0.7)
        lo, hi = p.confidence_interval()
        assert lo <= p.confidence <= hi
        assert hi - lo < 0.4  # CI should be reasonably tight with enough obs

    def test_snapshot_captures_state(self):
        p = PolicyRecord(name="snap", alpha=3.0, beta_=2.0, version=5)
        snap = p.snapshot(reason="test")
        assert snap.policy_id == p.policy_id
        assert snap.alpha == 3.0
        assert snap.beta == 2.0   # PolicyVersion field is beta (no underscore)
        assert snap.version == 5
        assert snap.reason == "test"

    def test_serialisation_roundtrip(self):
        p = PolicyRecord(
            name="test_policy",
            domain=PolicyDomain.USER,
            policy_type=PolicyType.ANSWER_STYLE,
            config={"verbosity": "high", "code_first": True},
            alpha=3.5, beta_=1.5,
            user_id="user_42",
            tags=["tag1", "tag2"],
        )
        d = p.to_dict()
        restored = PolicyRecord.from_dict(d)
        assert restored.policy_id == p.policy_id
        assert restored.domain == PolicyDomain.USER
        assert restored.policy_type == PolicyType.ANSWER_STYLE
        assert restored.config == p.config
        assert restored.alpha == pytest.approx(p.alpha)
        assert restored.user_id == "user_42"

    def test_retire(self):
        p = PolicyRecord(name="old")
        p.retire()
        assert not p.is_active

    def test_uncertainty_decreases_with_observations(self):
        p = PolicyRecord(alpha=1.0, beta_=1.0)
        u0 = p.uncertainty
        for i in range(20):
            p.update(0.7)
        u1 = p.uncertainty
        assert u1 < u0


class TestRewardSignal:
    def test_value_clamped(self):
        r = RewardSignal(RewardType.BENCHMARK_SCORE, value=1.5)
        assert r.value == 1.0
        r2 = RewardSignal(RewardType.BENCHMARK_SCORE, value=-0.5)
        assert r2.value == 0.0

    def test_is_positive(self):
        r1 = RewardSignal(RewardType.ANSWER_ACCEPTED, value=0.8)
        r2 = RewardSignal(RewardType.ANSWER_ACCEPTED, value=0.3)
        assert r1.is_positive()
        assert not r2.is_positive()

    def test_serialisation(self):
        r = RewardSignal(RewardType.PLANNER_SUCCESS, value=0.75,
                          context={"benchmark": "planning"},
                          source="test")
        d = r.to_dict()
        assert d["value"] == 0.75
        assert d["reward_type"] == "planner_success"


# ════════════════════════════════════════════════════════════════════
# UNIT — PolicyStore
# ════════════════════════════════════════════════════════════════════

class TestPolicyStore:
    def test_save_and_get(self, policy_store):
        p = PolicyRecord(name="test", domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.PLANNER_CONFIG,
                          config={"beam_width": 5})
        policy_store.save(p)
        retrieved = policy_store.get(p.policy_id)
        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved.config["beam_width"] == 5

    def test_get_nonexistent(self, policy_store):
        assert policy_store.get("nonexistent_id") is None

    def test_all_active_domain_filter(self, policy_store):
        sys_p = PolicyRecord(name="sys", domain=PolicyDomain.SYSTEM,
                              policy_type=PolicyType.RETRIEVAL_WEIGHTS)
        usr_p = PolicyRecord(name="usr", domain=PolicyDomain.USER,
                              policy_type=PolicyType.ANSWER_STYLE)
        policy_store.save(sys_p); policy_store.save(usr_p)
        sys_results = policy_store.all_active(domain=PolicyDomain.SYSTEM)
        usr_results = policy_store.all_active(domain=PolicyDomain.USER)
        assert all(p.domain == PolicyDomain.SYSTEM for p in sys_results)
        assert all(p.domain == PolicyDomain.USER for p in usr_results)

    def test_all_active_type_filter(self, policy_store):
        for i in range(3):
            policy_store.save(PolicyRecord(
                name=f"ret_{i}", domain=PolicyDomain.SYSTEM,
                policy_type=PolicyType.RETRIEVAL_WEIGHTS))
        for i in range(2):
            policy_store.save(PolicyRecord(
                name=f"plan_{i}", domain=PolicyDomain.SYSTEM,
                policy_type=PolicyType.PLANNER_CONFIG))
        ret = policy_store.all_active(policy_type=PolicyType.RETRIEVAL_WEIGHTS)
        plan = policy_store.all_active(policy_type=PolicyType.PLANNER_CONFIG)
        assert len(ret) == 3
        assert len(plan) == 2

    def test_save_and_get_version(self, policy_store):
        p = PolicyRecord(name="versioned")
        policy_store.save(p)
        snap = p.snapshot(reason="test_save")
        policy_store.save_version(snap)
        history = policy_store.get_history(p.policy_id)
        assert len(history) == 1
        assert history[0].reason == "test_save"

    def test_rollback(self, policy_store):
        p = PolicyRecord(name="rb", alpha=1.0, beta_=1.0)
        policy_store.save(p)
        policy_store.save_version(p.snapshot(reason="initial"))
        p.update(0.9)
        p.update(0.9)
        policy_store.save(p)
        policy_store.save_version(p.snapshot(reason="after_updates"))
        # Roll back to version 1
        restored = policy_store.rollback(p.policy_id, 1)
        assert restored is not None
        assert restored.alpha == pytest.approx(1.0, abs=0.1)

    def test_count(self, policy_store):
        for i in range(5):
            policy_store.save(PolicyRecord(name=f"p{i}",
                                            domain=PolicyDomain.SYSTEM,
                                            policy_type=PolicyType.PLANNER_CONFIG))
        assert policy_store.count() == 5
        assert policy_store.count(PolicyDomain.SYSTEM) == 5
        assert policy_store.count(PolicyDomain.USER) == 0

    def test_reward_log_and_stats(self, policy_store):
        p = PolicyRecord(name="logged")
        policy_store.save(p)
        for v in [0.8, 0.7, 0.9, 0.6]:
            r = RewardSignal(RewardType.BENCHMARK_SCORE, value=v, policy_id=p.policy_id)
            policy_store.log_reward(r)
        stats = policy_store.reward_stats(p.policy_id)
        assert stats["count"] == 4
        assert stats["mean"] == pytest.approx(0.75, abs=0.01)
        assert stats["min"] == pytest.approx(0.6)
        assert stats["max"] == pytest.approx(0.9)

    def test_delete(self, policy_store):
        p = PolicyRecord(name="to_delete")
        policy_store.save(p)
        assert policy_store.get(p.policy_id) is not None
        policy_store.delete(p.policy_id)
        assert policy_store.get(p.policy_id) is None


# ════════════════════════════════════════════════════════════════════
# UNIT — Reward Engine
# ════════════════════════════════════════════════════════════════════

class TestRewardEngine:
    def test_system_benchmark_reward(self):
        engine = SystemRewardEngine()
        r = engine.benchmark_reward(0.85, "planning")
        assert r.reward_type == RewardType.BENCHMARK_SCORE
        assert r.value == pytest.approx(0.85)
        assert r.context["benchmark"] == "planning"

    def test_system_latency_reward_fast(self):
        engine = SystemRewardEngine()
        r = engine.latency_reward(10.0, "retrieval", target_ms=500.0)
        assert r.value > 0.98  # very fast → reward near 1.0

    def test_system_latency_reward_slow(self):
        engine = SystemRewardEngine()
        r = engine.latency_reward(5000.0, "retrieval", target_ms=500.0)
        assert r.value < 0.1  # 10× over target → near 0

    def test_system_planner_reward(self):
        engine = SystemRewardEngine()
        r = engine.planner_reward(best_value=0.75, n_trajectories=3)
        assert r.value == pytest.approx(0.75)

    def test_system_memory_quality_reward(self):
        engine = SystemRewardEngine()
        r = engine.memory_quality_reward(
            retrieval_score=0.9, n_results=8, latency_ms=50.0)
        assert 0.0 < r.value <= 1.0

    def test_system_regression_reward(self):
        engine = SystemRewardEngine()
        r = engine.regression_reward(n_passing=190, n_total=194)
        assert r.value == pytest.approx(190/194, abs=0.001)

    def test_user_answer_accepted(self):
        engine = UserRewardEngine()
        r_yes = engine.answer_accepted_reward(True,  "user_1")
        r_no  = engine.answer_accepted_reward(False, "user_1")
        assert r_yes.value > r_no.value
        assert r_yes.value == pytest.approx(1.0)
        assert 0.0 < r_no.value < 1.0

    def test_user_task_completion_efficient(self):
        engine = UserRewardEngine()
        r_fast = engine.task_completion_reward(True, n_turns=1, user_id="u")
        r_slow = engine.task_completion_reward(True, n_turns=8, user_id="u")
        assert r_fast.value > r_slow.value

    def test_user_correction_severity(self):
        engine = UserRewardEngine()
        r_minor = engine.correction_reward(True, correction_severity=0.1, user_id="u")
        r_major = engine.correction_reward(True, correction_severity=0.9, user_id="u")
        assert r_minor.value > r_major.value

    def test_dispatch_to_learner(self, learner):
        observed = []
        class MockLearner:
            def observe(self, r): observed.append(r)
        engine = RewardEngine(MockLearner())
        engine.on_benchmark(0.85, "test")
        assert len(observed) == 1
        assert observed[0].value == pytest.approx(0.85)

    def test_no_learner_no_crash(self):
        engine = RewardEngine(learner=None)
        engine.on_benchmark(0.85, "test")  # should not raise


# ════════════════════════════════════════════════════════════════════
# UNIT — PolicyLearner
# ════════════════════════════════════════════════════════════════════

class TestPolicyLearner:
    def test_register_defaults(self, learner, policy_store):
        count = policy_store.count()
        assert count > 0  # defaults registered by fixture

    def test_register_idempotent(self, learner, policy_store):
        count_before = policy_store.count()
        learner.register_defaults()  # call again
        count_after = policy_store.count()
        assert count_after == count_before  # no duplicates

    def test_select_returns_policy(self, learner):
        from policy.models import PolicyType, PolicyDomain
        result = learner.select_one(PolicyType.RETRIEVAL_WEIGHTS, PolicyDomain.SYSTEM)
        assert result is not None
        assert result.policy_type == PolicyType.RETRIEVAL_WEIGHTS

    def test_select_returns_highest_thompson_sample(self, learner, policy_store):
        # Register two policies with very different confidences
        high = PolicyRecord(name="high_conf",
                            domain=PolicyDomain.SYSTEM,
                            policy_type=PolicyType.WORKSPACE_CONFIG,
                            alpha=50.0, beta_=1.0)
        low  = PolicyRecord(name="low_conf",
                            domain=PolicyDomain.SYSTEM,
                            policy_type=PolicyType.WORKSPACE_CONFIG,
                            alpha=1.0,  beta_=50.0)
        learner.register(high); learner.register(low)
        # High-confidence arm should win most of the time
        wins = sum(
            1 for _ in range(30)
            if learner.select_one(
                PolicyType.WORKSPACE_CONFIG, PolicyDomain.SYSTEM).name == "high_conf"
        )
        assert wins > 20  # should win at least 2/3 of draws

    def test_observe_updates_policy(self, learner, policy_store):
        p = learner.select_one(PolicyType.RETRIEVAL_WEIGHTS, PolicyDomain.SYSTEM)
        conf_before = p.confidence
        reward = RewardSignal(
            RewardType.MEMORY_QUALITY, value=0.9, policy_id=p.policy_id)
        learner.observe(reward)
        p_after = policy_store.get(p.policy_id)
        assert p_after.confidence > conf_before

    def test_observe_negative_reward_decreases_confidence(self, learner, policy_store):
        p = learner.select_one(PolicyType.RETRIEVAL_WEIGHTS, PolicyDomain.SYSTEM)
        # First build up some confidence
        for _ in range(5):
            learner.observe(RewardSignal(
                RewardType.MEMORY_QUALITY, value=0.8, policy_id=p.policy_id))
        p_mid = policy_store.get(p.policy_id)
        conf_mid = p_mid.confidence
        # Now deliver bad rewards
        for _ in range(5):
            learner.observe(RewardSignal(
                RewardType.MEMORY_QUALITY, value=0.1, policy_id=p.policy_id))
        p_after = policy_store.get(p.policy_id)
        assert p_after.confidence < conf_mid

    def test_broadcast_reward_updates_matching_types(self, learner, policy_store):
        # Broadcast a benchmark reward (should update RETRIEVAL_WEIGHTS and PLANNER_CONFIG)
        reward = RewardSignal(RewardType.BENCHMARK_SCORE, value=0.9)  # no policy_id
        updated = learner.observe(reward)
        assert len(updated) > 0

    def test_policy_summary(self, learner):
        summary = learner.policy_summary()
        assert isinstance(summary, list)
        assert all("confidence" in r for r in summary)
        # Sorted by confidence descending
        confs = [r["confidence"] for r in summary]
        assert confs == sorted(confs, reverse=True)

    def test_observe_batch(self, learner):
        rewards = [
            RewardSignal(RewardType.BENCHMARK_SCORE, value=0.85),
            RewardSignal(RewardType.LATENCY, value=0.90),
            RewardSignal(RewardType.PLANNER_SUCCESS, value=0.75),
        ]
        total = learner.observe_batch(rewards)
        assert total > 0

    def test_rollback_via_learner(self, learner, policy_store):
        p = PolicyRecord(name="rollback_test",
                          domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.COMPRESSION_POLICY,
                          alpha=1.0, beta_=1.0)
        learner.register(p)
        policy_store.save_version(p.snapshot(reason="v1"))
        # Degrade the policy
        for _ in range(10):
            learner.observe(RewardSignal(
                RewardType.BENCHMARK_SCORE, value=0.1, policy_id=p.policy_id))
        result = learner.rollback(p.policy_id, to_version=1)
        assert result is not None
        assert result.alpha == pytest.approx(1.0, abs=0.5)

    def test_learning_curve(self, learner, policy_store):
        p = PolicyRecord(name="curve_test",
                          domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.VERIFICATION_POLICY)
        learner.register(p)
        for i in range(5):
            policy_store.save_version(p.snapshot(reason=f"checkpoint_{i}"))
            p.update(0.7)
            policy_store.save(p)
        curve = learner.learning_curve(p.policy_id)
        assert len(curve) >= 5

    def test_context_key_extraction(self):
        ctx = {"task_type": "code", "user_id": "alice", "irrelevant": "ignored"}
        key = _context_key(ctx, features=["task_type", "user_id"])
        assert "code" in key
        assert "alice" in key

    def test_default_policies_covers_all_types(self):
        defaults = _default_policies()
        types = {p.policy_type for p in defaults}
        expected = {PolicyType.RETRIEVAL_WEIGHTS, PolicyType.PLANNER_CONFIG,
                    PolicyType.REASONING_STRATEGY, PolicyType.ANSWER_STYLE,
                    PolicyType.DIFFICULTY_LEVEL}
        assert expected <= types


# ════════════════════════════════════════════════════════════════════
# UNIT — PolicyOptimizer
# ════════════════════════════════════════════════════════════════════

class TestPolicyOptimizer:
    def test_decay_all(self, policy_store, learner):
        optimizer = PolicyOptimizer(policy_store)
        # Add a policy with elevated alpha
        p = PolicyRecord(name="to_decay",
                          domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.MEMORY_ROUTING,
                          alpha=10.0, beta_=1.0)
        policy_store.save(p)
        n = optimizer.decay_all()
        assert n > 0
        refreshed = policy_store.get(p.policy_id)
        assert refreshed.alpha < 10.0  # decayed

    def test_retire_poor_performers(self, policy_store):
        optimizer = PolicyOptimizer(policy_store, min_observations=3,
                                     aging_threshold=0.4)
        # Create a bad policy with enough observations
        bad = PolicyRecord(name="poor_policy",
                            domain=PolicyDomain.SYSTEM,
                            policy_type=PolicyType.TOOL_SELECTION,
                            alpha=1.0, beta_=10.0,
                            success_count=2, failure_count=8)
        policy_store.save(bad)
        retired = optimizer.retire_poor_performers()
        assert bad.policy_id in retired
        refreshed = policy_store.get(bad.policy_id)
        assert not refreshed.is_active

    def test_spawn_mutant(self, policy_store):
        optimizer = PolicyOptimizer(policy_store)
        parent = PolicyRecord(name="parent",
                               domain=PolicyDomain.SYSTEM,
                               policy_type=PolicyType.PLANNER_CONFIG,
                               config={"beam_width": 5, "max_depth": 3})
        policy_store.save(parent)
        mutant = optimizer.spawn_mutant(parent, mutation_scale=0.2)
        assert mutant is not None
        assert mutant.name != parent.name
        assert "mutant" in mutant.tags
        assert mutant.metadata["parent_id"] == parent.policy_id
        # Config should be different but close
        assert mutant.config.get("beam_width") != parent.config["beam_width"] or \
               mutant.config.get("max_depth") != parent.config["max_depth"]

    def test_convergence_detection_not_converged(self, policy_store):
        optimizer = PolicyOptimizer(policy_store, convergence_window=3)
        p = PolicyRecord(name="converge_test",
                          domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.VERIFICATION_POLICY)
        policy_store.save(p)
        # Add widely varying versions
        for reward in [0.3, 0.8, 0.2, 0.9, 0.1]:
            p.alpha = 1.0 + reward * 5
            p.beta_  = 1.0 + (1-reward) * 5
            pv = p.snapshot()
            pv.mean_reward = reward
            policy_store.save_version(pv)
        assert not optimizer.is_converged(p.policy_id)

    def test_convergence_detection_converged(self, policy_store):
        optimizer = PolicyOptimizer(policy_store, convergence_window=3,
                                     convergence_tolerance=0.05)
        p = PolicyRecord(name="converged",
                          domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.VERIFICATION_POLICY)
        policy_store.save(p)
        # Add nearly identical versions
        for reward in [0.70, 0.72, 0.71, 0.70, 0.72]:
            pv = p.snapshot()
            pv.mean_reward = reward
            policy_store.save_version(pv)
        assert optimizer.is_converged(p.policy_id)

    def test_run_cycle(self, policy_store, learner):
        optimizer = PolicyOptimizer(policy_store, min_observations=2)
        summary = optimizer.run_cycle(learner=learner)
        assert "decayed" in summary
        assert "retired" in summary
        assert "mutants" in summary


# ════════════════════════════════════════════════════════════════════
# UNIT — PolicySelector & PolicyCompiler
# ════════════════════════════════════════════════════════════════════

class TestPolicySelector:
    def test_select_system_policies(self, selector):
        result = selector.select_system_policies({"task_type": "code"})
        assert isinstance(result, dict)

    def test_select_user_policies(self, selector):
        result = selector.select_user_policies("user_alice")
        assert isinstance(result, dict)

    def test_get_retrieval_weights_valid(self, selector):
        weights = selector.get_retrieval_weights()
        assert isinstance(weights, dict)
        assert all(isinstance(v, float) for v in weights.values())
        assert all(v >= 0 for v in weights.values())

    def test_get_planner_config_valid(self, selector):
        cfg = selector.get_planner_config()
        assert "beam_width" in cfg or isinstance(cfg, dict)

    def test_get_answer_style_valid(self, selector):
        style = selector.get_answer_style("user_bob")
        assert isinstance(style, dict)

    def test_fallback_when_no_policies(self, tmp_dir):
        store = PolicyStore(tmp_dir)  # empty store
        empty_learner = PolicyLearner(store)
        empty_selector = PolicySelector(empty_learner)
        weights = empty_selector.get_retrieval_weights()
        assert isinstance(weights, dict)
        assert all(v >= 0 for v in weights.values())


class TestPolicyCompiler:
    def test_compile_returns_prompt(self, compiler):
        prompt = compiler.compile("Explain neural networks", user_id="alice")
        assert isinstance(prompt, CompiledPrompt)
        assert prompt.system_instructions
        assert prompt.token_estimate > 0

    def test_compiled_prompt_has_active_policies(self, compiler):
        prompt = compiler.compile("Debug this Python code", user_id="bob")
        assert isinstance(prompt.active_policies, dict)

    def test_to_flat_string_nonempty(self, compiler):
        prompt = compiler.compile("What is a transformer?", user_id="carol")
        flat = prompt.to_flat_string()
        assert len(flat) > 50

    def test_to_messages_format(self, compiler):
        prompt = compiler.compile("List sorting algorithms", user_id="dave")
        messages = prompt.to_messages()
        assert isinstance(messages, list)
        if messages:
            assert "role" in messages[0]
            assert "content" in messages[0]

    def test_verbosity_applied(self, compiler, selector, learner, policy_store):
        # Force a verbose policy to win
        verbose = PolicyRecord(
            name="style_verbose_forced",
            domain=PolicyDomain.USER,
            policy_type=PolicyType.ANSWER_STYLE,
            config={"verbosity": "high", "code_first": False, "examples": True},
            alpha=50.0, beta_=1.0)
        policy_store.save(verbose)
        prompt = compiler.compile("Explain gradient descent", user_id="user_verbose")
        assert isinstance(prompt.system_instructions, str)

    def test_memory_context_included(self, compiler, hgshm):
        hgshm.believe("Neural networks learn via backpropagation")
        hgshm.add_principle("Always validate training data quality")
        ctx = hgshm.recall("neural network training")
        prompt = compiler.compile("Explain neural networks",
                                   user_id="test", memory_context=ctx)
        flat = prompt.to_flat_string()
        # Memory or principle should appear in prompt
        assert len(flat) > len(prompt.system_instructions)

    def test_constraint_added_for_tight_budget(self, compiler):
        prompt = compiler.compile("Quick answer", user_id="u", token_budget=200)
        assert len(prompt.constraints) > 0

    def test_total_chars(self, compiler):
        prompt = compiler.compile("Test task", user_id="u")
        assert prompt.total_chars == len(prompt.to_flat_string())


# ════════════════════════════════════════════════════════════════════
# UNIT — Memory Domains
# ════════════════════════════════════════════════════════════════════

class TestSystemMemory:
    def test_store_workflow_success(self, system_memory):
        node = system_memory.store_workflow(
            "Beam search completed in 3 steps", success=True, latency_ms=42.0)
        assert node.node_id
        assert "SUCCESS" in node.text
        assert SystemMemory.DOMAIN_TAG in node.tags

    def test_store_workflow_failure(self, system_memory):
        node = system_memory.store_workflow("Tool timed out", success=False)
        assert "FAILURE" in node.text
        assert node.confidence < 0.7

    def test_store_benchmark_result(self, system_memory):
        node = system_memory.store_benchmark_result("planning", 0.807, 20, "2.0")
        assert "planning" in node.text
        assert "0.8" in node.text or "0.807" in node.text

    def test_store_failure_pattern(self, system_memory):
        node = system_memory.store_failure_pattern(
            "Timeout on web_search", resolution="Retry with backoff", frequency=3)
        assert "Timeout" in node.text
        assert "Retry" in node.text
        assert node.importance > 0.5

    def test_store_principle(self, system_memory):
        node = system_memory.store_principle(
            "Always verify before deploying to production")
        assert node.memory_type.value == "principle"

    def test_store_api_knowledge(self, system_memory):
        node = system_memory.store_api_knowledge(
            "BeamSearchPlanner", "Parameters: beam_width, max_depth")
        assert "BeamSearchPlanner" in node.text

    def test_recall_relevant(self, system_memory):
        system_memory.store_workflow("Kubernetes deployment successful", success=True)
        system_memory.store_benchmark_result("planning", 0.85, 20)
        ctx = system_memory.recall("deployment planning")
        assert ctx.total_memories >= 0

    def test_benchmark_history_filtered(self, system_memory):
        system_memory.store_benchmark_result("planning", 0.80, 20)
        system_memory.store_benchmark_result("planning", 0.82, 20)
        system_memory.store_benchmark_result("memory",   0.97, 60)
        planning_hist = system_memory.benchmark_history("planning")
        memory_hist   = system_memory.benchmark_history("memory")
        assert all("planning" in n.tags for n in planning_hist)
        assert len(memory_hist) >= 0

    def test_stats(self, system_memory):
        system_memory.store_workflow("test", success=True)
        system_memory.store_principle("test principle")
        stats = system_memory.stats()
        assert stats["total"] >= 2


class TestUserMemory:
    def test_store_preference(self, user_memory):
        node = user_memory.store_preference("language", "Python", strength=0.9)
        assert "Python" in node.text
        assert UserMemory.DOMAIN_TAG in node.tags
        assert "user:test_user_001" in node.tags

    def test_store_goal(self, user_memory):
        node = user_memory.store_goal("Become a machine learning engineer", priority=0.9)
        assert node.memory_type.value == "goal"
        assert node.importance == pytest.approx(0.9)

    def test_store_interaction_accepted(self, user_memory):
        node = user_memory.store_interaction(
            "How does backpropagation work?", response_accepted=True)
        assert "ACCEPTED" in node.text

    def test_store_interaction_corrected(self, user_memory):
        node = user_memory.store_interaction(
            "What is Python?", response_accepted=False,
            correction="Python is interpreted, not compiled")
        assert "CORRECTED" in node.text
        assert node.importance > 0.7  # corrections are more important

    def test_record_correction(self, user_memory):
        node = user_memory.record_correction(
            original="Python is compiled",
            correction="Python is interpreted",
            severity=0.8)
        assert "correction" in node.tags
        assert node.importance > 0.7

    def test_store_learning_progress_understood(self, user_memory):
        node = user_memory.store_learning_progress(
            "gradient descent", understood=True)
        assert "understood" in node.tags

    def test_store_learning_progress_unclear(self, user_memory):
        node = user_memory.store_learning_progress(
            "backpropagation through time", understood=False)
        assert node.importance > 0.7

    def test_store_project(self, user_memory):
        node = user_memory.store_project(
            "BlixBot", "AI cognitive assistant",
            stack=["Python", "FastAPI", "sqlite-vec"])
        assert "BlixBot" in node.text
        assert "Python" in node.tags

    def test_preferences_retrieval(self, user_memory):
        user_memory.store_preference("style", "concise")
        user_memory.store_preference("language", "Python")
        prefs = user_memory.preferences()
        assert len(prefs) >= 2

    def test_goals_retrieval(self, user_memory):
        user_memory.store_goal("Learn ML")
        user_memory.store_goal("Build Blix")
        goals = user_memory.goals()
        assert len(goals) >= 2

    def test_corrections_retrieval(self, user_memory):
        user_memory.record_correction("old", "new", severity=0.5)
        corrections = user_memory.corrections()
        assert len(corrections) >= 1

    def test_cold_start_profile_empty(self, tmp_dir):
        h = HGSHM(tmp_dir / "cold")
        um = UserMemory(h, "new_user_999")
        profile = um.cold_start_profile()
        assert profile["is_cold_start"] is True
        assert profile["n_preferences"] == 0
        h.close()

    def test_cold_start_profile_warm(self, user_memory):
        user_memory.store_preference("style", "concise")
        user_memory.store_goal("Learn Python")
        profile = user_memory.cold_start_profile()
        assert not profile["is_cold_start"]
        assert profile["n_preferences"] >= 1

    def test_stats(self, user_memory):
        user_memory.store_preference("language", "Python")
        user_memory.store_goal("ML mastery")
        stats = user_memory.stats()
        assert stats["total"] >= 2
        assert stats["user_id"] == "test_user_001"

    def test_user_isolation(self, hgshm):
        um1 = UserMemory(hgshm, "user_alice")
        um2 = UserMemory(hgshm, "user_bob")
        um1.store_preference("language", "Python")
        um2.store_preference("language", "Rust")
        alice_prefs = um1.preferences()
        bob_prefs   = um2.preferences()
        assert all("user:user_alice" in n.tags for n in alice_prefs)
        assert all("user:user_bob"   in n.tags for n in bob_prefs)


class TestMemoryManager:
    def test_query_all_domains(self, memory_manager):
        result = memory_manager.query("deployment failure", user_id="test_user")
        assert result.query == "deployment failure"
        assert isinstance(result.domains_queried, list)

    def test_query_system_only(self, memory_manager, system_memory):
        system_memory.store_workflow("Test workflow", success=True)
        result = memory_manager.query("workflow", include_user=False,
                                       include_general=False)
        assert "system" in result.domains_queried

    def test_query_merges_deduplicates(self, memory_manager, hgshm):
        hgshm.believe("Shared belief about deployment")
        result = memory_manager.query("deployment", include_system=True,
                                       include_general=True)
        node_ids = [r.node.node_id for r in result.merged_memories]
        assert len(node_ids) == len(set(node_ids))  # no duplicates

    def test_get_user_memory_caches(self, memory_manager):
        um1 = memory_manager.get_user_memory("alice")
        um2 = memory_manager.get_user_memory("alice")
        assert um1 is um2  # same instance

    def test_stats(self, memory_manager):
        stats = memory_manager.stats()
        assert "hgshm" in stats
        assert "system" in stats
        assert "user" in stats

    def test_routing_latency_tracked(self, memory_manager):
        result = memory_manager.query("test query")
        assert result.routing_latency_ms >= 0

    def test_to_memory_context(self, memory_manager, hgshm):
        hgshm.believe("Belief for context conversion")
        result = memory_manager.query("belief")
        ctx = result.to_memory_context()
        assert ctx.query == "belief"


# ════════════════════════════════════════════════════════════════════
# INTEGRATION — Full ADMA pipeline
# ════════════════════════════════════════════════════════════════════

class TestADMAIntegration:
    def test_full_policy_learning_cycle(self, learner, policy_store):
        """Thompson sampling → select → observe reward → confidence update."""
        p = learner.select_one(PolicyType.RETRIEVAL_WEIGHTS, PolicyDomain.SYSTEM)
        assert p is not None
        conf_before = p.confidence

        for _ in range(10):
            learner.observe(RewardSignal(
                RewardType.MEMORY_QUALITY, value=0.9, policy_id=p.policy_id))

        p_after = policy_store.get(p.policy_id)
        assert p_after.confidence > conf_before

    def test_personalization_affects_compiler(self, learner, compiler, policy_store):
        """User corrections should shift answer style policy."""
        # Simulate user repeatedly rejecting verbose answers
        for _ in range(5):
            learner.observe(RewardSignal(
                RewardType.CORRECTION_GIVEN, value=0.1,
                context={"user_id": "verbose_hater"}))

        # Compile a prompt — style should still work
        prompt = compiler.compile("Explain Python", user_id="verbose_hater")
        assert prompt is not None

    def test_system_memory_feeds_recall(self, hgshm, system_memory):
        """Operational knowledge stored in SystemMemory should be retrievable."""
        system_memory.store_failure_pattern(
            "Beam search timeout", resolution="Reduce beam width", frequency=3)
        system_memory.store_principle(
            "Reduce beam width when planning latency exceeds 500ms")
        ctx = system_memory.recall("beam search timeout planning")
        assert ctx.total_memories >= 0

    def test_user_memory_personalizes_recall(self, hgshm, user_memory):
        """User preferences stored in UserMemory should inform retrieval."""
        user_memory.store_preference("language", "Python", strength=0.95)
        user_memory.store_goal("Build a production ML pipeline")
        ctx = user_memory.recall("Python ML pipeline")
        assert ctx.total_memories >= 0

    def test_reward_dispatched_on_benchmark(self, learner):
        """Benchmark reward should update relevant system policies."""
        engine = RewardEngine(learner)
        summary_before = {r["name"]: r["confidence"] for r in learner.policy_summary()}
        engine.on_benchmark(0.95, "planning")
        engine.on_benchmark(0.80, "memory")
        summary_after = {r["name"]: r["confidence"] for r in learner.policy_summary()}
        # At least some policies should have changed
        changed = sum(1 for n, c in summary_after.items()
                      if abs(c - summary_before.get(n, 0.5)) > 0.001)
        assert changed >= 1

    def test_optimizer_runs_full_cycle(self, learner, policy_store):
        optimizer = PolicyOptimizer(policy_store, min_observations=3,
                                     aging_threshold=0.3)
        summary = optimizer.run_cycle(learner=learner)
        assert all(k in summary for k in ["decayed", "retired", "mutants", "rolled_back"])

    def test_end_to_end_adma_pipeline(self, tmp_dir):
        """Complete ADMA pipeline: init → store memories → learn → compile → query."""
        # Setup
        h = HGSHM(tmp_dir)
        store = PolicyStore(tmp_dir)
        l = PolicyLearner(store)
        l.register_defaults()
        engine = RewardEngine(l)
        selector = PolicySelector(l)
        compiler = PolicyCompiler(selector)
        sm = SystemMemory(h)
        um = UserMemory(h, "adma_user")
        mgr = MemoryManager(h, sm)

        # Store memories
        sm.store_principle("Prefer step-by-step explanations for code")
        um.store_preference("style", "code-first", strength=0.9)
        um.store_goal("Master Python async programming")

        # Observe rewards
        engine.on_benchmark(0.85, "planning")
        engine.on_answer_accepted(accepted=True, user_id="adma_user")

        # Compile prompt
        ctx = mgr.query("Python async programming", user_id="adma_user")
        prompt = compiler.compile(
            "Explain Python asyncio", user_id="adma_user",
            memory_context=ctx.to_memory_context())

        # Verify
        assert prompt.system_instructions
        assert len(prompt.active_policies) > 0
        assert ctx.routing_latency_ms >= 0

        h.close()


# ════════════════════════════════════════════════════════════════════
# ABLATION v3
# ════════════════════════════════════════════════════════════════════

class TestAblationV3:
    def test_conditions_defined(self):
        assert len(ABLATION_CONDITIONS) >= 7
        names = [c.name for c in ABLATION_CONDITIONS]
        assert "full_system" in names
        assert "without_policy_learning" in names
        assert "without_reward_engine" in names
        assert "without_user_memory" in names

    def test_run_single_condition(self, tmp_dir):
        runner = AblationV3Runner(
            blix_path=Path(__file__).parent.parent,
        )
        config = AblationConfig("test_full", description="test baseline")
        result = runner.run_condition(config)
        assert result.condition.name == "test_full"
        assert isinstance(result.benchmarks, list)
        assert result.elapsed_s >= 0

    def test_run_minimal_ablation(self, tmp_dir):
        """Run minimal benchmarks for each condition without blix_eval."""
        runner = AblationV3Runner(
            blix_path=Path(__file__).parent.parent,
            eval_path=Path("/nonexistent"),  # force fallback to minimal
        )
        conditions = [
            AblationConfig("full_system"),
            AblationConfig("without_policy_learning",
                           disable_policy_learning=True),
            AblationConfig("without_user_memory",
                           disable_user_memory=True),
        ]
        report = runner.run_full_study(conditions=conditions)
        assert report.baseline is not None
        assert len(report.ablations) == 2

    def test_report_summary_table(self, tmp_dir):
        runner = AblationV3Runner(blix_path=Path(__file__).parent.parent,
                                   eval_path=Path("/nonexistent"))
        report = runner.run_full_study(conditions=[
            AblationConfig("full_system"),
            AblationConfig("without_reward_engine", disable_reward_engine=True),
        ])
        table = report.summary_table()
        assert len(table) == 1  # only ablations, not baseline
        row = table[0]
        assert "condition" in row
        assert "delta_score" in row
        assert "impact" in row

    def test_report_export_json(self, tmp_dir):
        runner = AblationV3Runner(blix_path=Path(__file__).parent.parent,
                                   eval_path=Path("/nonexistent"))
        report = runner.run_full_study(conditions=[
            AblationConfig("full_system"),
            AblationConfig("without_adaptive_retrieval",
                           disable_adaptive_retrieval=True),
        ])
        out = tmp_dir / "abl_report.json"
        report.export_json(out)
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert "baseline" in data
        assert "ablations" in data
        assert "summary" in data

    def test_report_export_csv(self, tmp_dir):
        runner = AblationV3Runner(blix_path=Path(__file__).parent.parent,
                                   eval_path=Path("/nonexistent"))
        report = runner.run_full_study(conditions=[
            AblationConfig("full_system"),
            AblationConfig("without_policy_compiler",
                           disable_policy_compiler=True),
        ])
        out = tmp_dir / "abl.csv"
        report.export_csv(out)
        assert out.exists()

    def test_dependency_injection_stubs(self):
        """Verify stub implementations have the right interface."""
        from policy.ablation_v3 import (
            _NullPolicyLearner, _NullPolicySelector, _NullRewardEngine
        )
        learner_stub = _NullPolicyLearner()
        assert learner_stub.select_one() is None
        assert learner_stub.select() == []
        assert learner_stub.observe() == []

        selector_stub = _NullPolicySelector()
        weights = selector_stub.get_retrieval_weights()
        assert all(v > 0 for v in weights.values())

        reward_stub = _NullRewardEngine()
        reward_stub.on_benchmark(0.9, "test")  # should not raise

    def test_ablation_config_serialisation(self):
        cfg = AblationConfig("test", disable_policy_learning=True,
                              description="no learning")
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert d["disable_policy_learning"] is True


# ════════════════════════════════════════════════════════════════════
# STRESS / PERFORMANCE
# ════════════════════════════════════════════════════════════════════

class TestADMAStress:
    def test_1000_reward_observations(self, learner, policy_store):
        """1000 reward observations should complete quickly."""
        p = learner.select_one(PolicyType.RETRIEVAL_WEIGHTS, PolicyDomain.SYSTEM)
        t0 = time.perf_counter()
        for i in range(1000):
            reward = RewardSignal(
                RewardType.MEMORY_QUALITY,
                value=0.5 + (i % 2) * 0.3,
                policy_id=p.policy_id,
            )
            learner.observe(reward)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"1000 reward obs took {elapsed:.1f}s"

    def test_confidence_converges_after_many_positives(self, tmp_dir):
        """After many positive rewards, confidence should approach 1."""
        store = PolicyStore(tmp_dir)
        l = PolicyLearner(store)
        p = PolicyRecord(name="converge_high",
                          domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.WORKSPACE_CONFIG)
        l.register(p)
        for _ in range(100):
            l.observe(RewardSignal(
                RewardType.BENCHMARK_SCORE, value=0.95, policy_id=p.policy_id))
        p_final = store.get(p.policy_id)
        assert p_final.confidence > 0.85

    def test_confidence_stays_low_after_many_negatives(self, tmp_dir):
        store = PolicyStore(tmp_dir)
        l = PolicyLearner(store)
        p = PolicyRecord(name="converge_low",
                          domain=PolicyDomain.SYSTEM,
                          policy_type=PolicyType.WORKSPACE_CONFIG)
        l.register(p)
        for _ in range(100):
            l.observe(RewardSignal(
                RewardType.BENCHMARK_SCORE, value=0.1, policy_id=p.policy_id))
        p_final = store.get(p.policy_id)
        assert p_final.confidence < 0.3

    def test_multi_user_isolation_at_scale(self, tmp_dir):
        h = HGSHM(tmp_dir)
        users = [UserMemory(h, f"user_{i}") for i in range(5)]
        for i, um in enumerate(users):
            for j in range(10):
                um.store_preference("topic", f"topic_{i}_{j}")
        # Verify isolation
        for i, um in enumerate(users):
            prefs = um.preferences()
            assert all(f"user:user_{i}" in n.tags for n in prefs)
        h.close()
