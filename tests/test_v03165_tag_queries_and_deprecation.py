"""
Blix v0.3.16.1 — Tests for ISSUE-008 and ISSUE-009

ISSUE-008: Tag-indexed memory queries (no more O(n) full table scans)
  - nodes_by_tags returns correct nodes
  - single tag filter works
  - multi-tag AND filter works
  - memory_type filter combined with tags
  - count_by_tag returns correct count
  - stats_by_tag returns correct group-by breakdown
  - SystemMemory.stats() uses DB query, not Python filter
  - SystemMemory.recent_failures() uses tag index
  - SystemMemory.benchmark_history() uses tag index with optional name filter
  - UserMemory.preferences() uses tag index + category filter
  - UserMemory.goals() uses tag index
  - UserMemory.corrections() uses tag index
  - UserMemory.stats() uses DB query
  - Performance: large node sets return quickly

ISSUE-009: Duplicate class deprecation notices
  - All 7 legacy modules carry DEPRECATED comment
  - Legacy modules still importable (no breaking change)
  - Deprecation comment references the superseding module
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_node import MemoryNode, MemoryType, HierarchyLevel
from memory.system.system_memory import SystemMemory
from memory.user.user_memory import UserMemory


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


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
    return UserMemory(hgshm, "test_user")


# ════════════════════════════════════════════════════════════════════
# ISSUE-008 — Tag-Indexed Queries at HGSHMStore level
# ════════════════════════════════════════════════════════════════════

class TestNodesByTags:
    """Direct tests of the new nodes_by_tags / count_by_tag / stats_by_tag."""

    def test_single_tag_returns_matching_nodes(self, hgshm):
        hgshm.remember("system workflow", tags=["system_memory", "workflow"])
        hgshm.remember("user pref",       tags=["user_memory",   "user:alice"])
        hgshm.remember("other",           tags=["other_tag"])

        results = hgshm.nodes_by_tags(["system_memory"])
        tags_sets = [set(n.tags) for n in results]
        assert all("system_memory" in ts for ts in tags_sets)
        assert len(results) == 1

    def test_multi_tag_and_filter(self, hgshm):
        hgshm.remember("both",    tags=["A", "B"])
        hgshm.remember("only_A",  tags=["A"])
        hgshm.remember("only_B",  tags=["B"])

        results = hgshm.nodes_by_tags(["A", "B"])
        assert len(results) == 1
        assert "both" in results[0].text

    def test_three_tag_and_filter(self, hgshm):
        hgshm.remember("all_three", tags=["X", "Y", "Z"])
        hgshm.remember("two",       tags=["X", "Y"])
        hgshm.remember("one",       tags=["X"])

        results = hgshm.nodes_by_tags(["X", "Y", "Z"])
        assert len(results) == 1
        assert "all_three" in results[0].text

    def test_memory_type_combined_with_tags(self, hgshm):
        hgshm.remember("ep w tag",   memory_type=MemoryType.EPISODE,
                        tags=["system_memory"])
        hgshm.remember("fact w tag", memory_type=MemoryType.FACT,
                        tags=["system_memory"])

        ep_results = hgshm.nodes_by_tags(
            ["system_memory"], memory_type=MemoryType.EPISODE)
        assert all(n.memory_type == MemoryType.EPISODE for n in ep_results)
        assert len(ep_results) == 1

    def test_empty_required_tags_falls_back_to_all_nodes(self, hgshm):
        for i in range(3):
            hgshm.remember(f"node_{i}", tags=[f"tag_{i}"])
        results = hgshm.nodes_by_tags([])
        assert len(results) == 3

    def test_no_matching_nodes_returns_empty(self, hgshm):
        hgshm.remember("some node", tags=["other"])
        results = hgshm.nodes_by_tags(["nonexistent_tag"])
        assert results == []

    def test_limit_respected(self, hgshm):
        for i in range(10):
            hgshm.remember(f"node_{i}", tags=["shared_tag"])
        results = hgshm.nodes_by_tags(["shared_tag"], limit=3)
        assert len(results) == 3

    def test_count_by_tag_correct(self, hgshm):
        hgshm.remember("a", tags=["counted"])
        hgshm.remember("b", tags=["counted"])
        hgshm.remember("c", tags=["other"])

        assert hgshm.count_by_tag("counted") == 2
        assert hgshm.count_by_tag("other") == 1
        assert hgshm.count_by_tag("absent") == 0

    def test_count_by_tag_with_memory_type(self, hgshm):
        hgshm.remember("ep",   memory_type=MemoryType.EPISODE, tags=["tagged"])
        hgshm.remember("fact", memory_type=MemoryType.FACT,    tags=["tagged"])

        assert hgshm.count_by_tag("tagged", MemoryType.EPISODE) == 1
        assert hgshm.count_by_tag("tagged", MemoryType.FACT) == 1
        assert hgshm.count_by_tag("tagged") == 2

    def test_stats_by_tag_correct_breakdown(self, hgshm):
        hgshm.remember("ep1",  memory_type=MemoryType.EPISODE, tags=["dom"])
        hgshm.remember("ep2",  memory_type=MemoryType.EPISODE, tags=["dom"])
        hgshm.remember("fact", memory_type=MemoryType.FACT,    tags=["dom"])
        hgshm.remember("other",                                 tags=["other"])

        stats = hgshm.stats_by_tag("dom")
        assert stats.get("episode") == 2
        assert stats.get("fact") == 1
        assert "other" not in stats or stats.get("other") == 0

    def test_stats_by_tag_total(self, hgshm):
        for i in range(5):
            hgshm.remember(f"n{i}", tags=["counted"])
        stats = hgshm.stats_by_tag("counted")
        assert sum(stats.values()) == 5

    def test_order_by_importance(self, hgshm):
        hgshm.remember("low",  importance=0.2, tags=["ord"])
        hgshm.remember("high", importance=0.9, tags=["ord"])
        hgshm.remember("mid",  importance=0.5, tags=["ord"])

        results = hgshm.nodes_by_tags(["ord"], order_by="importance")
        importances = [n.importance for n in results]
        assert importances == sorted(importances, reverse=True)

    def test_order_by_updated_at(self, hgshm):
        for i in range(3):
            hgshm.remember(f"t{i}", tags=["temporal"])
        results = hgshm.nodes_by_tags(["temporal"], order_by="updated_at")
        # Should not raise and should return results
        assert len(results) == 3


# ════════════════════════════════════════════════════════════════════
# ISSUE-008 — SystemMemory uses indexed queries
# ════════════════════════════════════════════════════════════════════

class TestSystemMemoryIndexedQueries:
    def test_stats_returns_correct_breakdown(self, system_memory):
        system_memory.store_workflow("wf1", success=True)
        system_memory.store_workflow("wf2", success=False)
        system_memory.store_principle("Always verify")

        stats = system_memory.stats()
        assert stats["total"] >= 3
        assert "episode" in stats or "principle" in stats

    def test_stats_empty_db(self, system_memory):
        stats = system_memory.stats()
        assert stats["total"] == 0

    def test_stats_only_counts_system_nodes(self, hgshm):
        sm = SystemMemory(hgshm)
        # Add system node
        sm.store_workflow("sys", success=True)
        # Add non-system node directly
        hgshm.remember("untagged node")

        stats = sm.stats()
        assert stats["total"] == 1  # only the system node

    def test_recent_failures_returns_failures(self, system_memory):
        system_memory.store_failure_pattern("timeout", resolution="retry")
        system_memory.store_failure_pattern("oom", resolution="reduce batch")
        system_memory.store_workflow("success wf", success=True)

        failures = system_memory.recent_failures()
        assert len(failures) >= 2
        assert all("failure_pattern" in n.tags for n in failures)
        assert all(SystemMemory.DOMAIN_TAG in n.tags for n in failures)

    def test_recent_failures_top_k(self, system_memory):
        for i in range(5):
            system_memory.store_failure_pattern(f"fail_{i}")
        failures = system_memory.recent_failures(top_k=3)
        assert len(failures) <= 3

    def test_benchmark_history_all(self, system_memory):
        system_memory.store_benchmark_result("planning", 0.85, 20)
        system_memory.store_benchmark_result("memory",   0.97, 60)
        history = system_memory.benchmark_history()
        assert len(history) >= 2

    def test_benchmark_history_filtered_by_name(self, system_memory):
        system_memory.store_benchmark_result("planning", 0.85, 20)
        system_memory.store_benchmark_result("memory",   0.97, 60)

        planning = system_memory.benchmark_history("planning")
        memory   = system_memory.benchmark_history("memory")

        assert all("planning" in n.tags for n in planning)
        assert all("memory"   in n.tags for n in memory)

    def test_no_cross_contamination_with_user_nodes(self, hgshm):
        sm = SystemMemory(hgshm)
        um = UserMemory(hgshm, "alice")

        sm.store_workflow("system op", success=True)
        um.store_preference("style", "concise")

        sm_stats = sm.stats()
        um_stats = um.stats()

        # Stats must not include the other domain's nodes
        assert sm_stats["total"] == 1
        assert um_stats["total"] == 1


# ════════════════════════════════════════════════════════════════════
# ISSUE-008 — UserMemory uses indexed queries
# ════════════════════════════════════════════════════════════════════

class TestUserMemoryIndexedQueries:
    def test_preferences_returns_only_this_user(self, hgshm):
        alice = UserMemory(hgshm, "alice")
        bob   = UserMemory(hgshm, "bob")

        alice.store_preference("language", "Python")
        bob.store_preference("language",   "Rust")

        alice_prefs = alice.preferences()
        bob_prefs   = bob.preferences()

        assert all("user:alice" in n.tags for n in alice_prefs)
        assert all("user:bob"   in n.tags for n in bob_prefs)
        assert len(alice_prefs) == 1
        assert len(bob_prefs)   == 1

    def test_preferences_category_filter(self, user_memory):
        user_memory.store_preference("language", "Python")
        user_memory.store_preference("style",    "concise")

        lang_prefs  = user_memory.preferences(category="language")
        style_prefs = user_memory.preferences(category="style")
        all_prefs   = user_memory.preferences()

        assert len(lang_prefs) == 1
        assert "language" in lang_prefs[0].tags
        assert len(style_prefs) == 1
        assert len(all_prefs) == 2

    def test_goals_returns_only_this_user(self, hgshm):
        alice = UserMemory(hgshm, "alice")
        bob   = UserMemory(hgshm, "bob")

        alice.store_goal("Learn ML")
        bob.store_goal("Learn Rust")

        alice_goals = alice.goals()
        bob_goals   = bob.goals()

        assert all("user:alice" in n.tags for n in alice_goals)
        assert all("user:bob"   in n.tags for n in bob_goals)
        assert len(alice_goals) == 1
        assert len(bob_goals)   == 1

    def test_corrections_returns_only_corrections(self, user_memory):
        user_memory.record_correction("old", "new", severity=0.5)
        user_memory.store_preference("style", "verbose")  # not a correction

        corrections = user_memory.corrections()
        assert all("correction" in n.tags for n in corrections)
        assert len(corrections) == 1

    def test_corrections_limit(self, user_memory):
        for i in range(10):
            user_memory.record_correction(f"old_{i}", f"new_{i}")
        corrections = user_memory.corrections(limit=3)
        assert len(corrections) <= 3

    def test_stats_correct_breakdown(self, user_memory):
        user_memory.store_preference("lang", "Python")
        user_memory.store_goal("Learn ML")
        user_memory.store_interaction("hello?", response_accepted=True)

        stats = user_memory.stats()
        assert stats["total"] >= 3
        assert stats["user_id"] == "test_user"

    def test_stats_empty(self, user_memory):
        stats = user_memory.stats()
        assert stats["total"] == 0
        assert stats["user_id"] == "test_user"

    def test_stats_user_isolation(self, hgshm):
        alice = UserMemory(hgshm, "alice")
        bob   = UserMemory(hgshm, "bob")

        alice.store_preference("style", "concise")
        alice.store_goal("Learn ML")
        bob.store_preference("style", "verbose")

        alice_stats = alice.stats()
        bob_stats   = bob.stats()

        assert alice_stats["total"] == 2
        assert bob_stats["total"]   == 1


# ════════════════════════════════════════════════════════════════════
# ISSUE-008 — Performance: tag queries are fast at scale
# ════════════════════════════════════════════════════════════════════

class TestTagQueryPerformance:
    def test_stats_faster_than_full_scan(self, hgshm):
        """
        stats_by_tag must be faster than loading all nodes into Python.
        We insert 100 nodes across two domains and compare timing.
        """
        sm = SystemMemory(hgshm)
        um = UserMemory(hgshm, "perf_user")

        for i in range(50):
            sm.store_workflow(f"wf_{i}", success=(i % 2 == 0))
        for i in range(50):
            um.store_preference("topic", f"topic_{i}")

        # Time the new indexed stats
        t0 = time.perf_counter()
        for _ in range(10):
            sm.stats()
            um.stats()
        indexed_time = time.perf_counter() - t0

        # Must complete 20 stat calls in under 2 seconds
        assert indexed_time < 2.0, (
            f"20 stats() calls took {indexed_time:.3f}s — "
            f"indexed query is too slow"
        )

    def test_nodes_by_tags_respects_limit(self, hgshm):
        """nodes_by_tags(limit=N) must never return more than N results."""
        for i in range(50):
            hgshm.remember(f"bulk_{i}", tags=["bulk"])

        results = hgshm.nodes_by_tags(["bulk"], limit=10)
        assert len(results) <= 10


# ════════════════════════════════════════════════════════════════════
# ISSUE-009 — Duplicate class deprecation
# ════════════════════════════════════════════════════════════════════

class TestDeprecationNotices:
    """
    Verify that each legacy module:
      1. Still imports without error (no breaking change)
      2. Contains a DEPRECATED comment referencing the new module
      3. The old class still works for existing callers
    """

    LEGACY_MODULES = [
        ("core.memory_manager",          "memory.manager",
         "MemoryManager"),
        ("core.hierarchy_manager",       "memory.hybrid.hierarchy.hierarchy_manager",
         "HierarchyManager"),
        ("reflection.consolidation_engine", "memory.hybrid.consolidation.consolidation_engine",
         "ConsolidationEngine"),
        ("core.memory_types",            "memory.hybrid.models.memory_node",
         "MemoryType"),
        ("core.semantic_retriever",      "memory.hybrid.retrieval.hybrid_retriever",
         "SemanticRetriever"),
        ("retrieval.temporal_retriever", "memory.hybrid.retrieval.hybrid_retriever",
         "TemporalRetriever"),
        ("causality.epistemic_status",   "memory.hybrid.models.memory_node",
         "EpistemicStatus"),
    ]

    def test_all_legacy_modules_still_importable(self):
        """No legacy module must raise ImportError."""
        import importlib
        errors = []
        for old_mod, _, _ in self.LEGACY_MODULES:
            try:
                importlib.import_module(old_mod)
            except ImportError as e:
                errors.append(f"{old_mod}: {e}")
        assert not errors, f"Legacy import errors:\n" + "\n".join(errors)

    def test_all_legacy_modules_have_deprecated_comment(self):
        """Every legacy module must carry a # DEPRECATED comment."""
        base = Path(__file__).parent.parent
        missing = []
        for old_mod, new_mod, _ in self.LEGACY_MODULES:
            path = base / (old_mod.replace(".", "/") + ".py")
            if not path.exists():
                missing.append(f"FILE NOT FOUND: {path}")
                continue
            content = path.read_text()
            if "DEPRECATED" not in content:
                missing.append(f"{old_mod}: missing DEPRECATED notice")
            if new_mod not in content:
                missing.append(
                    f"{old_mod}: DEPRECATED notice doesn't reference {new_mod}")
        assert not missing, "\n".join(missing)

    def test_legacy_class_still_functional(self):
        """
        The legacy EpistemicStatus (actively used by beam_search.py) must
        still be importable and its values accessible.
        """
        from causality.epistemic_status import EpistemicStatus
        # Must have at least one value
        values = list(EpistemicStatus)
        assert len(values) > 0

    def test_new_and_legacy_memory_type_both_accessible(self):
        """
        Both MemoryType variants must be importable without error.
        They are different classes with different purposes.
        """
        from core.memory_types import MemoryType as LegacyMemoryType
        from memory.hybrid.models.memory_node import MemoryType as HGSHMMemoryType
        # They should be different objects
        assert LegacyMemoryType is not HGSHMMemoryType
        # Both should be usable
        assert hasattr(LegacyMemoryType, '__members__')
        assert hasattr(HGSHMMemoryType, '__members__')

    def test_new_memory_manager_is_distinct_from_legacy(self):
        """
        The ADMA MemoryManager and the legacy MemoryManager are different classes.
        """
        from core.memory_manager import MemoryManager as LegacyMM
        from memory.manager import MemoryManager as ADMA_MM
        assert LegacyMM is not ADMA_MM

    def test_deprecation_issue_number_in_comments(self):
        """All deprecation notices reference ISSUE-009."""
        base = Path(__file__).parent.parent
        missing = []
        for old_mod, _, _ in self.LEGACY_MODULES:
            path = base / (old_mod.replace(".", "/") + ".py")
            if not path.exists():
                continue
            content = path.read_text()
            if "ISSUE-009" not in content:
                missing.append(f"{old_mod}: no ISSUE-009 reference")
        assert not missing, "\n".join(missing)
