"""
Project Memory System — Blix v0.3  (Feature 5)

Projects are first-class memory objects.  Each project tracks goals,
milestones, completed work, current status, and next actions.

The ``ProjectManager`` handles CRUD and persistence.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from schemas.memory_layers import ProjectSummary
from utils.logger import get_logger

log = get_logger(__name__)


class ProjectManager:
    """
    Manages the collection of ``ProjectSummary`` objects.

    Parameters
    ----------
    projects_file:
        Path to ``projects.json`` persistence file.
    """

    def __init__(self, projects_file: Path) -> None:
        self._file = projects_file
        self._projects: dict[str, ProjectSummary] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw_list = json.load(fh)
            for raw in raw_list:
                p = ProjectSummary.model_validate(raw)
                self._projects[p.project_name.lower()] = p
            log.info("ProjectManager loaded %d projects.", len(self._projects))
        except Exception as exc:
            log.warning("Could not load projects (%s); starting empty.", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        data = [p.model_dump() for p in self._projects.values()]
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_json_default, ensure_ascii=False)
        log.debug("ProjectManager saved %d projects.", len(self._projects))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_or_create(self, project_name: str) -> ProjectSummary:
        """Return the project with this name, creating it if necessary."""
        key = project_name.lower()
        if key not in self._projects:
            ps = ProjectSummary(
                id=f"project-{key.replace(' ', '_')}",
                project_name=project_name,
                summary=f"Project: {project_name}",
                source_ids=[],
            )
            self._projects[key] = ps
            self._save()
            log.info("Created project: %r", project_name)
        return self._projects[key]

    def get(self, project_name: str) -> Optional[ProjectSummary]:
        return self._projects.get(project_name.lower())

    def list_all(self, status: Optional[str] = None) -> list[ProjectSummary]:
        projects = list(self._projects.values())
        if status:
            projects = [p for p in projects if p.current_status == status]
        return projects

    def update(self, project_name: str, **fields: object) -> Optional[ProjectSummary]:
        """
        Update fields on an existing project.

        List fields (goals, milestones, completed_work, next_actions) are
        extended with unique new items rather than replaced, unless
        ``_replace=True`` is passed in fields.

        Returns the updated ``ProjectSummary`` or ``None`` if not found.
        """
        key = project_name.lower()
        ps = self._projects.get(key)
        if ps is None:
            return None

        replace = bool(fields.pop("_replace", False))
        updates: dict = {}

        for field, value in fields.items():
            existing = getattr(ps, field, None)
            if isinstance(existing, list) and isinstance(value, list) and not replace:
                merged = list(existing)
                for item in value:
                    if item not in merged:
                        merged.append(item)
                updates[field] = merged
            else:
                updates[field] = value

        updates["last_active"] = datetime.now(timezone.utc).replace(tzinfo=None)
        updated = ps.model_copy(update=updates)
        self._projects[key] = updated
        self._save()
        log.info("Updated project %r.", project_name)
        return updated

    def link_session(self, project_name: str, session_id: str) -> None:
        """Record that a session touched this project."""
        ps = self.get_or_create(project_name)
        key = project_name.lower()
        if session_id not in ps.related_session_ids:
            self._projects[key] = ps.model_copy(
                update={
                    "related_session_ids": ps.related_session_ids + [session_id],
                    "last_active": datetime.now(timezone.utc).replace(tzinfo=None),
                }
            )
            self._save()

    def record_progress(
        self,
        project_name: str,
        *,
        completed: Optional[list[str]] = None,
        next_actions: Optional[list[str]] = None,
        milestones: Optional[list[str]] = None,
        status: Optional[str] = None,
    ) -> ProjectSummary:
        """Convenience method to record progress on a project."""
        updates: dict = {}
        if completed:
            updates["completed_work"] = completed
        if next_actions:
            updates["next_actions"] = next_actions
        if milestones:
            updates["milestones"] = milestones
        if status:
            updates["current_status"] = status
            updates["_replace"] = False
        ps = self.update(project_name, **updates)
        return ps or self.get_or_create(project_name)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._projects)


def _json_default(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {type(obj)!r}")
