"""
Procedural Memory — Blix v0.3.8  (New module 6)

The beginnings of skill learning. ``agents.failure_memory.FailureMemory``
(v0.3.6) remembers what went WRONG; ``ProceduralMemory`` remembers what
went RIGHT, distilling successful task sequences into reusable
``Skill`` objects:

    Experience → Successful sequence → Reusable skill

    Skill(
        name="research_analysis",
        steps=["retrieve_documents", "summarize", "extract_insights", "update_knowledge"],
    )

This is deliberately lightweight: a ``Skill`` is a named, ordered list
of step descriptions (tool names or task-title fragments) plus
usage/success statistics — not a new planner and not an executable
program. The Planner can consult ``ProceduralMemory.find_matching_skill()``
to recognise "I've solved something like this before" and reuse the
known-good step sequence as a template, but actually decomposing /
executing the plan remains entirely the job of v0.3.5's
``planning.planner.Planner`` and ``agents.executor.AgentExecutor``.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

_STOP = {"a", "an", "the", "is", "are", "to", "for", "of", "in", "on", "and",
         "or", "with", "this", "that", "build", "create", "make"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """
    A reusable, named sequence of steps distilled from a successful run.

    Fields
    ------
    name:
        Short identifier, e.g. "research_analysis".
    steps:
        Ordered list of step descriptions (tool names or task-title fragments).
    goal_pattern:
        The original goal text this skill was learned from, used for
        similarity matching against future goals.
    use_count:
        How many times this skill has been reused.
    success_count:
        How many of those reuses succeeded.
    """

    name: str
    steps: list[str] = field(default_factory=list)
    goal_pattern: str = ""
    skill_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    use_count: int = 0
    success_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def success_rate(self) -> float:
        if self.use_count == 0:
            return 0.5
        return self.success_count / self.use_count

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "steps": self.steps,
            "goal_pattern": self.goal_pattern,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 3),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            name=d["name"], steps=d.get("steps", []), goal_pattern=d.get("goal_pattern", ""),
            skill_id=d.get("skill_id", uuid.uuid4().hex[:8]),
            use_count=d.get("use_count", 0), success_count=d.get("success_count", 0),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Procedural Memory
# ---------------------------------------------------------------------------


class ProceduralMemory:
    """
    Stores and matches learned ``Skill`` objects.

    Parameters
    ----------
    procedural_memory_file:
        Path to ``procedural_memory.json``.
    similarity_threshold:
        Minimum goal-text similarity to consider an existing skill a
        match for a new goal.
    min_steps_to_learn:
        Minimum sequence length before a successful run is considered
        "worth" distilling into a skill (very short sequences are too
        generic to be useful templates).
    """

    def __init__(
        self,
        procedural_memory_file: Path,
        similarity_threshold: float = 0.4,
        min_steps_to_learn: int = 2,
    ) -> None:
        self._file = procedural_memory_file
        self._threshold = similarity_threshold
        self._min_steps = min_steps_to_learn
        self._skills: dict[str, Skill] = {}
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
                s = Skill.from_dict(item)
                self._skills[s.skill_id] = s
            log.info("ProceduralMemory: loaded %d skill(s).", len(self._skills))
        except Exception as exc:
            log.warning("ProceduralMemory: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([s.to_dict() for s in self._skills.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_success(
        self, goal: str, steps: list[str], name: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        Distil a successful task sequence into a ``Skill``, or reinforce
        an existing matching skill if one already covers this kind of goal.

        Returns ``None`` if the sequence is too short to be worth learning.
        """
        if len(steps) < self._min_steps:
            return None

        existing = self.find_matching_skill(goal)
        if existing is not None:
            existing.use_count += 1
            existing.success_count += 1
            self._save()
            log.debug("ProceduralMemory: reinforced skill '%s' (id=%s)", existing.name, existing.skill_id)
            return existing

        skill_name = name or _derive_skill_name(goal)
        skill = Skill(name=skill_name, steps=steps, goal_pattern=goal, use_count=1, success_count=1)
        self._skills[skill.skill_id] = skill
        self._save()
        log.info("ProceduralMemory: learned new skill '%s' (%d step(s))", skill_name, len(steps))
        return skill

    def record_reuse_outcome(self, skill_id: str, success: bool) -> Optional[Skill]:
        """Record the outcome of reusing an existing skill (without re-learning steps)."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        skill.use_count += 1
        if success:
            skill.success_count += 1
        self._save()
        return skill

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_matching_skill(self, goal: str) -> Optional[Skill]:
        """Find the best-matching existing skill for a new goal, above the similarity threshold."""
        best: Optional[Skill] = None
        best_score = 0.0
        for skill in self._skills.values():
            score = _jaccard(goal, skill.goal_pattern)
            if score >= self._threshold and score > best_score:
                best, best_score = skill, score
        return best

    def suggest_steps(self, goal: str) -> Optional[list[str]]:
        """Convenience: return just the step template for the best-matching skill, if any."""
        skill = self.find_matching_skill(goal)
        return list(skill.steps) if skill else None

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def most_used(self, top_k: int = 5) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: -s.use_count)[:top_k]

    def most_reliable(self, top_k: int = 5, min_uses: int = 2) -> list[Skill]:
        confident = [s for s in self._skills.values() if s.use_count >= min_uses]
        return sorted(confident, key=lambda s: -s.success_rate)[:top_k]

    @property
    def count(self) -> int:
        return len(self._skills)


def _derive_skill_name(goal: str) -> str:
    """Best-effort short name derived from goal text, e.g. 'Research the history of X' -> 'research'."""
    tokens = list(_tokens(goal))
    if not tokens:
        return "general_skill"
    return "_".join(tokens[:2])
