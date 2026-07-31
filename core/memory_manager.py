"""
MemoryManager — single source of truth for all persistent state.

Responsibilities
----------------
* Load / save conversations, profile, and learning state to JSON.
* Provide clean CRUD over ``MemoryEntry`` records.
* Keep the in-memory cache consistent with on-disk files.

Python 3.10 compatibility notes
--------------------------------
* ``from __future__ import annotations`` defers all annotation evaluation,
  allowing ``list[X]`` and ``Optional[X]`` syntax on Python 3.10.
* No ``match``/``case`` or 3.12+ syntax is used.

Design note
-----------
All I/O is synchronous and JSON-based.  The interface is intentionally
narrow so a future v2 can swap in SQLite or a vector DB behind the same
method signatures without touching callers.
"""
# DEPRECATED — core.memory_manager (ISSUE-009)
#
# This module is superseded by memory.manager.
# The class ``MemoryManager`` here is the v0.3.x implementation;
# ``memory.manager.MemoryManager`` is the v0.3.15+ HGSHM implementation.
#
# These are different classes with different APIs. Callers that need
# the v0.3.15+ version must update their imports:
#
#     # Old (this file — legacy):
#     from core.memory_manager import MemoryManager
#
#     # New (HGSHM-backed):
#     from memory.manager import MemoryManager
#
# This file will be removed in v0.4. Do not add new callers.
# Issue: https://github.com/blix/blix/issues/9
#


from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from schemas.memory_entry import MemoryEntry
from schemas.profile import Profile
from schemas.learning_state import LearningState
from utils.helpers import load_json, save_json
from utils.logger import get_logger

log = get_logger(__name__)


class MemoryManager:
    """
    Manages persistent storage for conversations, user profile, and
    learning state.

    Parameters
    ----------
    conversations_file:
        Path to ``conversations.json``.
    profile_file:
        Path to ``profile.json``.
    learning_state_file:
        Path to ``learning_state.json``.
    """

    def __init__(
        self,
        conversations_file: Path,
        profile_file: Path,
        learning_state_file: Path,
    ) -> None:
        self._conversations_file = conversations_file
        self._profile_file = profile_file
        self._learning_state_file = learning_state_file

        # In-memory caches populated during __init__
        self._memories: list[MemoryEntry] = []
        self._profile: Profile = Profile()
        self._learning_state: LearningState = LearningState()

        # Eagerly load all state on construction
        self.load_memories()
        self._load_profile()
        self._load_learning_state()

    # ------------------------------------------------------------------
    # Conversations — public CRUD
    # ------------------------------------------------------------------

    def load_memories(self) -> list[MemoryEntry]:
        """
        Load all conversation entries from disk into the in-memory cache.

        Malformed entries are skipped with a warning rather than raising,
        so a single corrupted record never prevents startup.

        Returns
        -------
        list[MemoryEntry]
            The loaded entries (also cached internally).
        """
        raw_list = load_json(self._conversations_file)
        if not isinstance(raw_list, list):
            log.warning("conversations.json had unexpected format; resetting to empty list.")
            raw_list = []

        entries: list[MemoryEntry] = []
        for raw in raw_list:
            try:
                entries.append(MemoryEntry.model_validate(raw))
            except ValidationError as exc:
                log.warning("Skipping malformed memory entry: %s", exc)

        self._memories = entries
        log.info("Loaded %d memory entries from disk.", len(entries))
        return entries

    def save_memories(self) -> None:
        """Persist the current in-memory cache to disk atomically."""
        data = [m.model_dump() for m in self._memories]
        save_json(self._conversations_file, data)
        log.debug("Saved %d memory entries.", len(self._memories))

    def add_memory(self, user_input: str, assistant_output: str) -> MemoryEntry:
        """
        Create a new ``MemoryEntry``, append to cache, and persist.

        The new entry's ``id`` is one greater than the current maximum,
        ensuring monotonically increasing IDs even after deletes.

        Parameters
        ----------
        user_input:
            Raw text from the user.
        assistant_output:
            Blix's response.

        Returns
        -------
        MemoryEntry
            The newly created entry.
        """
        next_id = (max(m.id for m in self._memories) + 1) if self._memories else 1
        entry = MemoryEntry(
            id=next_id,
            input=user_input,
            output=assistant_output,
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self._memories.append(entry)
        self.save_memories()
        log.info("Saved memory entry id=%d.", entry.id)
        return entry

    def update_memory(self, entry_id: int, **fields: object) -> Optional[MemoryEntry]:
        """
        Update fields on an existing entry identified by *entry_id*.

        Only the supplied keyword arguments are modified; other fields
        are preserved.

        Parameters
        ----------
        entry_id:
            The ``id`` of the entry to update.
        **fields:
            Keyword arguments matching ``MemoryEntry`` field names.

        Returns
        -------
        MemoryEntry or None
            The updated entry, or ``None`` if *entry_id* was not found.
        """
        for i, entry in enumerate(self._memories):
            if entry.id == entry_id:
                updated = entry.model_copy(update=dict(fields))
                self._memories[i] = updated
                self.save_memories()
                log.info("Updated memory entry id=%d.", entry_id)
                return updated
        log.warning("update_memory: id=%d not found.", entry_id)
        return None

    def delete_memory(self, entry_id: int) -> bool:
        """
        Remove the entry with *entry_id* from cache and disk.

        Parameters
        ----------
        entry_id:
            The ``id`` of the entry to delete.

        Returns
        -------
        bool
            ``True`` if an entry was removed, ``False`` if not found.
        """
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.id != entry_id]
        if len(self._memories) < before:
            self.save_memories()
            log.info("Deleted memory entry id=%d.", entry_id)
            return True
        log.warning("delete_memory: id=%d not found.", entry_id)
        return False

    def get_all_memories(self) -> list[MemoryEntry]:
        """Return the full in-memory cache in insertion (oldest-first) order."""
        return list(self._memories)

    def get_memory_by_id(self, entry_id: int) -> Optional[MemoryEntry]:
        """
        Return a single entry by *entry_id*, or ``None`` if not found.

        Parameters
        ----------
        entry_id:
            The ``id`` of the entry to fetch.
        """
        for entry in self._memories:
            if entry.id == entry_id:
                return entry
        return None

    def memory_count(self) -> int:
        """Return the total number of stored conversation entries."""
        return len(self._memories)

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def _load_profile(self) -> None:
        """Load profile from disk; fall back to empty defaults on error."""
        raw = load_json(self._profile_file)
        if isinstance(raw, dict) and raw:
            try:
                self._profile = Profile.model_validate(raw)
                log.info("Loaded user profile (name=%r).", self._profile.name)
            except ValidationError as exc:
                log.warning("Profile validation failed; using defaults. %s", exc)
        else:
            log.info("No profile found; using empty defaults.")

    def save_profile(self) -> None:
        """Persist the current profile to disk."""
        save_json(self._profile_file, self._profile.model_dump())
        log.debug("Saved profile.")

    @property
    def profile(self) -> Profile:
        """The current user profile (in-memory)."""
        return self._profile

    @profile.setter
    def profile(self, value: Profile) -> None:
        """Set the profile and immediately persist it."""
        self._profile = value
        self.save_profile()

    # ------------------------------------------------------------------
    # Learning State
    # ------------------------------------------------------------------

    def _load_learning_state(self) -> None:
        """Load learning state from disk; fall back to empty defaults."""
        raw = load_json(self._learning_state_file)
        if isinstance(raw, dict) and raw:
            try:
                self._learning_state = LearningState.model_validate(raw)
                log.info("Loaded learning state.")
            except ValidationError as exc:
                log.warning("LearningState validation failed; using defaults. %s", exc)
        else:
            log.info("No learning state found; using empty defaults.")

    def save_learning_state(self) -> None:
        """Persist the current learning state to disk."""
        save_json(self._learning_state_file, self._learning_state.model_dump())
        log.debug("Saved learning state.")

    @property
    def learning_state(self) -> LearningState:
        """The current learning state (in-memory)."""
        return self._learning_state

    @learning_state.setter
    def learning_state(self, value: LearningState) -> None:
        """Set the learning state and immediately persist it."""
        self._learning_state = value
        self.save_learning_state()
