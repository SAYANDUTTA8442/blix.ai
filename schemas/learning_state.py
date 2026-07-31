"""
Pydantic schema for tracking learning progress — Python 3.10 compatible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LearningState(BaseModel):
    """
    Tracks the user's evolving relationship with individual topics.

    Blix injects this into every prompt so it can calibrate explanation
    depth and avoid re-teaching material the user has already mastered.

    Attributes
    ----------
    topics_learned:
        Fully understood; Blix references without re-explaining.
    topics_in_progress:
        Currently being studied; Blix elaborates and scaffolds.
    weak_topics:
        Repeated confusion detected; Blix uses gentler, slower approach.
    strong_topics:
        High proficiency; Blix may use these as analogy anchors.
    """

    topics_learned: list[str] = Field(default_factory=list)
    topics_in_progress: list[str] = Field(default_factory=list)
    weak_topics: list[str] = Field(default_factory=list)
    strong_topics: list[str] = Field(default_factory=list)

    def all_topics(self) -> list[str]:
        """Return a deduplicated flat list of all tracked topics."""
        seen: set[str] = set()
        result: list[str] = []
        for topic in (
            self.topics_learned
            + self.topics_in_progress
            + self.weak_topics
            + self.strong_topics
        ):
            if topic not in seen:
                seen.add(topic)
                result.append(topic)
        return result

    def total_count(self) -> int:
        """Return total unique topics across all lists."""
        return len(self.all_topics())
