"""
Dynamic Profile Evolution — Blix v0.3  (Feature 4)

Extends the v0.2 Profile with:
* Versioned history (no silent overwrites)
* Per-field confidence scores
* Conflict resolution (keep-highest-confidence)
* Full audit trail of every update

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from schemas.profile import Profile
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Audit entry
# ---------------------------------------------------------------------------


class ProfileAuditEntry(BaseModel):
    """Records one change to the profile."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    field: str
    old_value: Optional[str] = None  # JSON-serialised
    new_value: str  # JSON-serialised
    source: str = Field(default="extraction", description="'extraction', 'manual', 'graph'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Versioned profile
# ---------------------------------------------------------------------------


class VersionedProfile(BaseModel):
    """
    Wraps ``Profile`` with version history and confidence scores.

    All profile mutations go through ``ProfileEvolver`` — never mutate
    the ``Profile`` directly when using v0.3 features.
    """

    version: int = Field(default=1, ge=1)
    profile: Profile = Field(default_factory=Profile)
    # field_name → confidence
    confidences: dict[str, float] = Field(default_factory=dict)
    audit: list[ProfileAuditEntry] = Field(default_factory=list)

    def get_confidence(self, field: str) -> float:
        return self.confidences.get(field, 0.0)


# ---------------------------------------------------------------------------
# Evolver
# ---------------------------------------------------------------------------


class ProfileEvolver:
    """
    Manages profile updates with conflict resolution and audit trail.

    Parameters
    ----------
    versioned_profile_file:
        Path to ``versioned_profile.json`` (separate from the legacy
        ``profile.json`` so v0.2 readers still work).
    """

    def __init__(self, versioned_profile_file: Path) -> None:
        self._file = versioned_profile_file
        self._vp: VersionedProfile = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> VersionedProfile:
        if not self._file.exists():
            return VersionedProfile()
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            vp = VersionedProfile.model_validate(raw)
            log.info("ProfileEvolver loaded version %d.", vp.version)
            return vp
        except Exception as exc:
            log.warning("Could not load versioned profile (%s); starting fresh.", exc)
            return VersionedProfile()

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(self._vp.model_dump(), fh, indent=2, default=_json_default, ensure_ascii=False)
        log.debug("VersionedProfile v%d saved.", self._vp.version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def profile(self) -> Profile:
        """The current base profile (for injection into prompts)."""
        return self._vp.profile

    @property
    def versioned(self) -> VersionedProfile:
        return self._vp

    def update(
        self,
        *,
        name: Optional[str] = None,
        education: Optional[str] = None,
        new_interests: Optional[list[str]] = None,
        new_projects: Optional[list[str]] = None,
        new_goals: Optional[list[str]] = None,
        confidence: float = 1.0,
        source: str = "extraction",
    ) -> bool:
        """
        Apply updates with conflict resolution.

        Scalar fields (name, education) are only overwritten if the incoming
        confidence is ≥ the stored confidence.  List fields are extended
        with unique items (no conflict possible — only additive).

        Returns
        -------
        bool
            True if any field was actually changed.
        """
        changed = False
        p = self._vp.profile
        confidences = dict(self._vp.confidences)
        audit: list[ProfileAuditEntry] = []

        def _scalar(field: str, current: str, proposed: Optional[str]) -> str:
            nonlocal changed
            if not proposed:
                return current
            stored_conf = confidences.get(field, 0.0)
            if current and confidence < stored_conf:
                log.debug("Profile conflict: %s kept (conf %.2f > %.2f)", field, stored_conf, confidence)
                return current
            if proposed != current:
                audit.append(ProfileAuditEntry(
                    field=field,
                    old_value=json.dumps(current),
                    new_value=json.dumps(proposed),
                    source=source,
                    confidence=confidence,
                ))
                confidences[field] = confidence
                changed = True
                return proposed
            return current

        def _list_extend(field: str, current: list[str], additions: Optional[list[str]]) -> list[str]:
            nonlocal changed
            if not additions:
                return current
            new_items = [x for x in additions if x and x not in current]
            if not new_items:
                return current
            audit.append(ProfileAuditEntry(
                field=field,
                old_value=json.dumps(current),
                new_value=json.dumps(current + new_items),
                source=source,
                confidence=confidence,
            ))
            changed = True
            return current + new_items

        new_name = _scalar("name", p.name, name)
        new_edu = _scalar("education", p.education, education)
        new_interests = _list_extend("interests", list(p.interests), new_interests)
        new_projects = _list_extend("projects", list(p.projects), new_projects)
        new_goals = _list_extend("goals", list(p.goals), new_goals)

        if changed:
            new_profile = Profile(
                name=new_name,
                education=new_edu,
                interests=new_interests,
                projects=new_projects,
                goals=new_goals,
                notes=list(p.notes),
            )
            self._vp = VersionedProfile(
                version=self._vp.version + 1,
                profile=new_profile,
                confidences=confidences,
                audit=self._vp.audit + audit,
            )
            self._save()
            log.info(
                "Profile updated → v%d (%d changes, source=%s)",
                self._vp.version,
                len(audit),
                source,
            )
        return changed

    def get_audit(self, field: Optional[str] = None) -> list[ProfileAuditEntry]:
        """Return audit trail, optionally filtered by field name."""
        if field:
            return [e for e in self._vp.audit if e.field == field]
        return list(self._vp.audit)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_default(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {type(obj)!r}")
