"""
Multi-User Namespacing — Blix v0.3.1  (Issue 12)

Addresses: "Single-user assumption — one memory space, one profile, one graph."

This module does NOT rewrite every manager's constructor.  Instead it
provides ``UserNamespace``, a small path-resolution helper that every
v0.3 manager already accepts implicitly (they all take a ``Path`` to
their JSON file in ``__init__``).

Usage
-----
    ns = UserNamespace(base_dir=Path("memory"), user_id="sayan")
    mm = MemoryManager(
        conversations_file=ns.path("conversations.json"),
        profile_file=ns.path("profile.json"),
        learning_state_file=ns.path("learning_state.json"),
    )
    graph = MemoryGraph(graph_file=ns.path("graph.json"))
    pm = ProjectManager(projects_file=ns.path("projects.json"))

For the default single-user deployment, ``user_id=None`` (or "default")
reproduces the exact v0.3 layout — fully backwards compatible.

For multi-user deployments, each user gets an isolated subdirectory:
    memory/
        users/
            sayan/
                conversations.json
                graph.json
                ...
            alice/
                conversations.json
                graph.json
                ...

``UserRegistry`` tracks which users exist and provides lightweight
session-to-user resolution.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


_SLUG_RE = re.compile(r"[^a-z0-9_-]")


def _slugify(user_id: str) -> str:
    """Convert an arbitrary user_id into a filesystem-safe slug."""
    slug = user_id.strip().lower().replace(" ", "_")
    slug = _SLUG_RE.sub("", slug)
    return slug or "default"


# ---------------------------------------------------------------------------
# UserNamespace
# ---------------------------------------------------------------------------


class UserNamespace:
    """
    Resolves storage paths for a given user, isolating their memory space.

    Parameters
    ----------
    base_dir:
        Root memory directory (e.g. ``Path("memory")``).
    user_id:
        Unique user identifier.  If ``None`` or ``"default"``, paths
        resolve to ``base_dir`` directly — i.e. exactly the v0.3 single-user
        layout.  Otherwise paths resolve to ``base_dir/users/<slug>/``.
    """

    DEFAULT_USER = "default"

    def __init__(self, base_dir: Path, user_id: Optional[str] = None) -> None:
        self._base = base_dir
        self._user_id = user_id or self.DEFAULT_USER
        self._slug = _slugify(self._user_id)

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def slug(self) -> str:
        return self._slug

    @property
    def root(self) -> Path:
        """Root directory for this user's memory files."""
        if self._slug == self.DEFAULT_USER:
            return self._base
        return self._base / "users" / self._slug

    def path(self, relative: str) -> Path:
        """
        Resolve a relative filename (e.g. ``"conversations.json"`` or
        ``"hierarchy/sessions.json"``) to this user's namespaced path.
        """
        full = self.root / relative
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def __repr__(self) -> str:
        return f"UserNamespace(user_id={self._user_id!r}, root={self.root})"


# ---------------------------------------------------------------------------
# UserRegistry
# ---------------------------------------------------------------------------


@dataclass
class UserRecord:
    """Metadata about one registered user."""

    user_id: str
    slug: str
    display_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_active: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "slug": self.slug,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserRecord":
        return cls(**d)


class UserRegistry:
    """
    Tracks all known users and their namespaces.

    Persists to ``<base_dir>/users.json``.  This is the single piece of
    *shared* state across the otherwise-isolated per-user namespaces —
    it contains no personal data, only ids and timestamps.

    Parameters
    ----------
    base_dir:
        Root memory directory.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._file = base_dir / "users.json"
        self._users: dict[str, UserRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                rec = UserRecord.from_dict(item)
                self._users[rec.slug] = rec
            log.info("UserRegistry: loaded %d user(s).", len(self._users))
        except Exception as exc:
            log.warning("UserRegistry: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([u.to_dict() for u in self._users.values()], fh, indent=2)

    def register(self, user_id: str, display_name: str = "") -> UserNamespace:
        """
        Register a new user (idempotent) and return their namespace.
        """
        slug = _slugify(user_id)
        if slug not in self._users:
            self._users[slug] = UserRecord(
                user_id=user_id, slug=slug, display_name=display_name or user_id
            )
            self._save()
            log.info("UserRegistry: registered new user %r (slug=%s)", user_id, slug)
        return UserNamespace(self._base, user_id)

    def touch(self, user_id: str) -> None:
        """Update last_active timestamp for a user."""
        slug = _slugify(user_id)
        if slug in self._users:
            self._users[slug].last_active = datetime.now(timezone.utc).isoformat()
            self._save()

    def get(self, user_id: str) -> Optional[UserRecord]:
        return self._users.get(_slugify(user_id))

    def list_users(self) -> list[UserRecord]:
        return list(self._users.values())

    def namespace_for(self, user_id: Optional[str]) -> UserNamespace:
        """Get (and lazily register) a namespace for the given user."""
        if user_id is None:
            return UserNamespace(self._base, None)
        if self.get(user_id) is None:
            return self.register(user_id)
        return UserNamespace(self._base, user_id)

    @property
    def user_count(self) -> int:
        return len(self._users)
