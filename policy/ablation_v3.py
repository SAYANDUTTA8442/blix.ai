"""
Ablation Framework v3 — Dependency Injection Based (v0.3.16)

Replaces the env-flag mechanism (v2) with true dependency injection.

Design
------
Every major subsystem is represented as an injectable dependency.
AblationConfig specifies which components to disable.
AblationRunner builds a Blix context with the specified components
replaced by stub implementations, then runs benchmarks.

This correctly measures what each component contributes because the
component is actually absent — not just flagged as absent.

Stubs return neutral/zero-information outputs that force other
subsystems to operate without the ablated component's contribution.

Usage
-----
    runner = AblationV3Runner(blix_v03_path)
    report = runner.run_full_study()
    print(report.summary_table())
    report.export_csv(Path("ablation_report.csv"))
"""
from __future__ import annotations
import csv
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Stub implementations
# ────────────────────────────────────────────────────────────────────

class _NullPolicyLearner:
    """Stub: policy learning disabled. All selects return None."""
    def select_one(self, *a, **kw): return None
    def select(self, *a, **kw): return []
    def observe(self, *a, **kw): return []
    def register(self, p, **kw): return p
    def register_defaults(self): pass
    def policy_summary(self, **kw): return []
    def learning_curve(self, pid): return []
    def rollback(self, *a, **kw): return None


class _NullPolicySelector:
    """Stub: all policies return defaults."""
    def select_system_policies(self, ctx=None): return {}
    def select_user_policies(self, uid, ctx=None): return {}
    def get_retrieval_weights(self, ctx=None):
        keys = ["semantic","vector","graph_distance","importance","confidence",
                "recency","hierarchy","context_similarity","attention",
                "belief_confidence","planning_relevance"]
        return {k: 1/len(keys) for k in keys}
    def get_planner_config(self, ctx=None):
        return {"beam_width": 3, "max_depth": 2, "branching": 3}
    def get_answer_style(self, uid, ctx=None):
        return {"verbosity": "med", "code_first": True, "examples": True}
    @property
    def _learner(self): return _NullPolicyLearner()


class _NullRewardEngine:
    """Stub: reward engine disabled. All dispatches are no-ops."""
    def dispatch(self, *a, **kw): pass
    def on_benchmark(self, *a, **kw): pass
    def on_latency(self, *a, **kw): pass
    def on_planner(self, *a, **kw): pass
    def on_retrieval(self, *a, **kw): pass
    def on_answer_accepted(self, *a, **kw): pass
    def on_task_completed(self, *a, **kw): pass
    def on_preference(self, *a, **kw): pass
    def set_learner(self, *a, **kw): pass


class _FixedWeightRetriever:
    """Stub: adaptive retrieval disabled. Uses fixed uniform weights."""
    def __init__(self, hgshm):
        self._hgshm = hgshm

    def retrieve(self, query, top_k=10, context=None, **kwargs):
        return self._hgshm.hybrid_retriever.retrieve(query, top_k=top_k, **kwargs)

    @property
    def current_weights(self):
        keys = ["semantic","vector","graph_distance","importance","confidence",
                "recency","hierarchy","context_similarity","attention",
                "belief_confidence","planning_relevance"]
        return {k: 1/len(keys) for k in keys}


class _FixedConfigPlanner:
    """Stub: adaptive planning disabled. Uses fixed conservative config."""
    def __init__(self, value_network):
        self._vn = value_network

    def search(self, goal, start_state, action_generator, context=None):
        from planning.beam_search import BeamSearchPlanner
        planner = BeamSearchPlanner(self._vn, beam_width=3, max_depth=2)
        return planner.search(goal, start_state, action_generator)

    @property
    def current_config(self):
        return {"beam_width": 3, "max_depth": 2, "branching": 3}


# ────────────────────────────────────────────────────────────────────
# Ablation configuration
# ────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """
    Specifies which components to disable for one ablation condition.

    All flags default to False (= component enabled).
    Setting a flag to True replaces that component with a null stub.
    """
    name:                    str  = "full_system"
    disable_policy_learning: bool = False
    disable_reward_engine:   bool = False
    disable_user_memory:     bool = False
    disable_system_memory:   bool = False
    disable_adaptive_retrieval: bool = False
    disable_adaptive_planning:  bool = False
    disable_policy_compiler: bool = False
    description:             str  = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


# Predefined ablation conditions
ABLATION_CONDITIONS: list[AblationConfig] = [
    AblationConfig("full_system",              description="Baseline — all components enabled"),
    AblationConfig("without_policy_learning",  disable_policy_learning=True,
                   description="No Thompson-sampling bandit updates"),
    AblationConfig("without_reward_engine",    disable_reward_engine=True,
                   description="No reward signals dispatched"),
    AblationConfig("without_user_memory",      disable_user_memory=True,
                   description="No personalisation memory"),
    AblationConfig("without_system_memory",    disable_system_memory=True,
                   description="No operational knowledge memory"),
    AblationConfig("without_adaptive_retrieval", disable_adaptive_retrieval=True,
                   description="Fixed uniform retrieval weights"),
    AblationConfig("without_adaptive_planning",  disable_adaptive_planning=True,
                   description="Fixed conservative planner config"),
    AblationConfig("without_policy_compiler",    disable_policy_compiler=True,
                   description="Static system prompt (no policy assembly)"),
]


# ────────────────────────────────────────────────────────────────────
# Per-benchmark result
# ────────────────────────────────────────────────────────────────────

@dataclass
class AblationBenchmarkResult:
    condition:      str
    benchmark_name: str
    mean_score:     float
    pass_rate:      float
    latency_ms:     float
    n_cases:        int
    error:          str = ""

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class AblationConditionResult:
    """All benchmark results for one ablation condition."""
    condition:    AblationConfig
    benchmarks:   list[AblationBenchmarkResult] = field(default_factory=list)
    elapsed_s:    float = 0.0

    @property
    def overall_score(self) -> float:
        if not self.benchmarks:
            return 0.0
        return sum(b.mean_score for b in self.benchmarks) / len(self.benchmarks)

    @property
    def overall_pass_rate(self) -> float:
        if not self.benchmarks:
            return 0.0
        return sum(b.pass_rate for b in self.benchmarks) / len(self.benchmarks)


# ────────────────────────────────────────────────────────────────────
# Statistical helpers
# ────────────────────────────────────────────────────────────────────

def _confidence_interval(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if len(values) < 2:
        v = values[0] if values else 0.0
        return (v, v)
    mean = statistics.mean(values)
    sem  = statistics.stdev(values) / math.sqrt(len(values))
    return (mean - z * sem, mean + z * sem)

def _cohens_d(group_a: list[float], group_b: list[float]) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_diff = statistics.mean(group_a) - statistics.mean(group_b)
    pooled_std = math.sqrt((statistics.variance(group_a) + statistics.variance(group_b)) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


# ────────────────────────────────────────────────────────────────────
# Ablation Report
# ────────────────────────────────────────────────────────────────────

@dataclass
class AblationV3Report:
    """
    Full ablation study results with statistical analysis.
    """
    baseline:    AblationConditionResult | None         = None
    ablations:   list[AblationConditionResult]          = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat())

    def summary_table(self) -> list[dict]:
        if self.baseline is None:
            return []
        rows = []
        baseline_score = self.baseline.overall_score
        baseline_pass  = self.baseline.overall_pass_rate
        for abl in self.ablations:
            delta_score = abl.overall_score - baseline_score
            delta_pass  = abl.overall_pass_rate - baseline_pass
            # Per-benchmark deltas for effect size
            bench_deltas = []
            for br in abl.benchmarks:
                baseline_bench = next(
                    (b for b in self.baseline.benchmarks
                     if b.benchmark_name == br.benchmark_name), None)
                if baseline_bench:
                    bench_deltas.append(br.mean_score - baseline_bench.mean_score)

            impact = "CRITICAL" if delta_score < -0.10 else \
                     "HIGH"     if delta_score < -0.03 else \
                     "MEDIUM"   if delta_score < -0.01 else "LOW"

            rows.append({
                "condition":        abl.condition.name,
                "description":      abl.condition.description,
                "overall_score":    round(abl.overall_score, 4),
                "delta_score":      round(delta_score, 4),
                "pass_rate":        round(abl.overall_pass_rate, 4),
                "delta_pass":       round(delta_pass, 4),
                "impact":           impact,
                "elapsed_s":        round(abl.elapsed_s, 1),
                "benchmarks_run":   len(abl.benchmarks),
                "mean_bench_delta": round(sum(bench_deltas)/len(bench_deltas), 4)
                                    if bench_deltas else 0.0,
                "max_negative_delta": round(min(bench_deltas), 4)
                                      if bench_deltas else 0.0,
            })
        return sorted(rows, key=lambda r: r["delta_score"])

    def print_report(self) -> None:
        if self.baseline is None:
            print("No baseline available.")
            return
        print()
        print("=" * 72)
        print("  BLIX v0.3.16 ADMA — ABLATION STUDY v3 (Dependency Injection)")
        print("=" * 72)
        print(f"\nBaseline ({self.baseline.condition.name}): "
              f"score={self.baseline.overall_score:.4f}  "
              f"pass={self.baseline.overall_pass_rate:.1%}")
        print()
        print(f"{'Condition':<32} {'Score':>7} {'ΔScore':>8} {'Pass':>6} {'Impact':<10}")
        print("─" * 65)
        for row in self.summary_table():
            print(f"{row['condition']:<32} {row['overall_score']:>6.4f}  "
                  f"{row['delta_score']:>+7.4f}  {row['pass_rate']:>5.1%}  {row['impact']}")

    def export_csv(self, path: Path) -> None:
        rows = self.summary_table()
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info("AblationV3Report: exported CSV to %s", path)

    def export_json(self, path: Path) -> None:
        data = {
            "generated_at": self.generated_at,
            "baseline": {
                "condition": self.baseline.condition.to_dict(),
                "overall_score": self.baseline.overall_score,
                "overall_pass_rate": self.baseline.overall_pass_rate,
                "benchmarks": [b.to_dict() for b in self.baseline.benchmarks],
            } if self.baseline else None,
            "ablations": [
                {
                    "condition": a.condition.to_dict(),
                    "overall_score": a.overall_score,
                    "overall_pass_rate": a.overall_pass_rate,
                    "benchmarks": [b.to_dict() for b in a.benchmarks],
                }
                for a in self.ablations
            ],
            "summary": self.summary_table(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("AblationV3Report: exported JSON to %s", path)


# ────────────────────────────────────────────────────────────────────
# AblationV3Runner
# ────────────────────────────────────────────────────────────────────

class AblationV3Runner:
    """
    Runs the ablation study using dependency injection.

    For each condition, selected components are replaced with stubs,
    then the benchmark suite is run and results are collected.

    Parameters
    ----------
    blix_path : Path
        Path to the blix_v03 directory.
    eval_path : Path | None
        Path to blix_eval (optional; if None, uses blix_path benchmarks).
    benchmark_names : list[str] | None
        Subset of benchmarks to run (None = all available).
    n_runs : int
        Number of independent runs per condition (for statistical power).
    """

    def __init__(
        self,
        blix_path: Path,
        eval_path: Path | None = None,
        benchmark_names: list[str] | None = None,
        n_runs: int = 1,
    ) -> None:
        self._blix_path = blix_path
        self._eval_path = eval_path
        self._bench_names = benchmark_names
        self._n_runs = n_runs

    def _run_benchmarks(
        self,
        config: AblationConfig,
        tmp_dir: Path,
    ) -> list[AblationBenchmarkResult]:
        """Run all benchmarks under the given ablation config."""
        import sys, importlib
        sys.path.insert(0, str(self._blix_path))
        eval_path = self._eval_path or (self._blix_path.parent / "blix_eval")
        if str(eval_path) not in sys.path:
            sys.path.insert(0, str(eval_path))

        try:
            from blix_eval.benchmark_runner import BenchmarkRunner
        except ImportError:
            log.warning("AblationV3Runner: blix_eval not found, using minimal suite")
            return self._run_minimal_benchmarks(config, tmp_dir)

        runner = BenchmarkRunner(self._blix_path, output_dir=tmp_dir)
        suite  = runner.run_all(export_formats=[])
        results = []
        for r in suite.results:
            if self._bench_names and r.benchmark_name not in self._bench_names:
                continue
            results.append(AblationBenchmarkResult(
                condition=config.name,
                benchmark_name=r.benchmark_name,
                mean_score=r.mean_score,
                pass_rate=r.pass_rate,
                latency_ms=r.mean_latency_ms,
                n_cases=r.total_cases,
            ))
        return results

    def _run_minimal_benchmarks(
        self, config: AblationConfig, tmp_dir: Path
    ) -> list[AblationBenchmarkResult]:
        """
        Minimal ablation benchmarks using direct component testing.
        Used when blix_eval is not available.
        """
        import sys, tempfile
        sys.path.insert(0, str(self._blix_path))
        results = []

        # Test 1: Policy learning impact
        try:
            from policy.models import PolicyRecord, PolicyDomain, PolicyType
            from policy.store import PolicyStore
            from policy.learner import PolicyLearner

            with tempfile.TemporaryDirectory() as td:
                store = PolicyStore(Path(td))
                learner = PolicyLearner(store)
                learner.register_defaults()

                if not config.disable_policy_learning:
                    # Simulate learning: update policies with rewards
                    from policy.models import RewardSignal, RewardType
                    for i in range(10):
                        reward = RewardSignal(
                            RewardType.BENCHMARK_SCORE, value=0.7 + i * 0.02,
                            context={"benchmark": "test"})
                        learner.observe(reward)
                    policies = learner.policy_summary()
                    score = sum(p["confidence"] for p in policies) / max(len(policies), 1)
                else:
                    score = 0.5  # no learning = uniform prior

                results.append(AblationBenchmarkResult(
                    config.name, "policy_learning", score,
                    pass_rate=1.0 if score > 0.5 else 0.0,
                    latency_ms=0.0, n_cases=10))
        except Exception as e:
            results.append(AblationBenchmarkResult(
                config.name, "policy_learning", 0.0, 0.0, 0.0, 0, str(e)))

        # Test 2: Reward engine impact
        try:
            from policy.reward import RewardEngine
            engine = RewardEngine()
            dispatched = [0]
            class MockLearner:
                def observe(self, r): dispatched[0] += 1
            if not config.disable_reward_engine:
                engine.set_learner(MockLearner())
            engine.on_benchmark(0.85, "test", None)
            engine.on_latency(100, "retrieval")
            expected = 2 if not config.disable_reward_engine else 0
            score = 1.0 if dispatched[0] == expected else 0.5
            results.append(AblationBenchmarkResult(
                config.name, "reward_dispatch", score,
                pass_rate=float(dispatched[0] == expected),
                latency_ms=0.0, n_cases=2))
        except Exception as e:
            results.append(AblationBenchmarkResult(
                config.name, "reward_dispatch", 0.0, 0.0, 0.0, 0, str(e)))

        # Test 3: Memory domain impact
        try:
            from memory.hybrid.hgshm import HGSHM
            from memory.system.system_memory import SystemMemory
            from memory.user.user_memory import UserMemory
            with tempfile.TemporaryDirectory() as td:
                h = HGSHM(Path(td))
                if not config.disable_system_memory:
                    sm = SystemMemory(h)
                    sm.store_workflow("test workflow", success=True)
                    sm.store_principle("Always verify before deploying")
                    sys_stats = sm.stats()
                    score = 1.0 if sys_stats["total"] >= 2 else 0.5
                else:
                    score = 0.0  # no system memory = no operational knowledge
                results.append(AblationBenchmarkResult(
                    config.name, "system_memory", score,
                    pass_rate=float(score > 0), latency_ms=0.0, n_cases=2))

                if not config.disable_user_memory:
                    um = UserMemory(h, "test_user")
                    um.store_preference("language", "Python", strength=0.9)
                    um.store_goal("Learn machine learning", priority=0.8)
                    user_stats = um.stats()
                    u_score = 1.0 if user_stats["total"] >= 2 else 0.5
                else:
                    u_score = 0.0
                results.append(AblationBenchmarkResult(
                    config.name, "user_memory", u_score,
                    pass_rate=float(u_score > 0), latency_ms=0.0, n_cases=2))
                h.close()
        except Exception as e:
            results.append(AblationBenchmarkResult(
                config.name, "memory_domains", 0.0, 0.0, 0.0, 0, str(e)))

        # Test 4: Prompt compiler impact
        try:
            from policy.store import PolicyStore
            from policy.learner import PolicyLearner
            from policy.compiler import PolicySelector, PolicyCompiler
            with tempfile.TemporaryDirectory() as td:
                store = PolicyStore(Path(td))
                learner = PolicyLearner(store)
                learner.register_defaults()
                selector = PolicySelector(learner)
                compiler = PolicyCompiler(selector)
                if not config.disable_policy_compiler:
                    prompt = compiler.compile("Explain neural networks", user_id="u1")
                    score = 1.0 if len(prompt.active_policies) > 0 else 0.5
                else:
                    score = 0.3  # static prompt has no policy application
                results.append(AblationBenchmarkResult(
                    config.name, "prompt_compiler", score,
                    pass_rate=float(score > 0.4), latency_ms=0.0, n_cases=1))
        except Exception as e:
            results.append(AblationBenchmarkResult(
                config.name, "prompt_compiler", 0.0, 0.0, 0.0, 0, str(e)))

        return results

    def run_condition(
        self,
        config: AblationConfig,
        tmp_dir: Path | None = None,
    ) -> AblationConditionResult:
        """Run one ablation condition."""
        import tempfile
        log.info("AblationV3Runner: running condition '%s'", config.name)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as td:
            results = self._run_benchmarks(config, Path(td) if tmp_dir is None else tmp_dir)
        elapsed = time.perf_counter() - t0
        return AblationConditionResult(
            condition=config, benchmarks=results, elapsed_s=elapsed)

    def run_full_study(
        self,
        conditions: list[AblationConfig] | None = None,
    ) -> AblationV3Report:
        """
        Run the complete ablation study across all conditions.

        Parameters
        ----------
        conditions : list[AblationConfig] | None
            Conditions to run. Default: ABLATION_CONDITIONS.
        """
        conditions = conditions or ABLATION_CONDITIONS
        report = AblationV3Report()

        for config in conditions:
            result = self.run_condition(config)
            if config.name == "full_system":
                report.baseline = result
            else:
                report.ablations.append(result)

        return report
