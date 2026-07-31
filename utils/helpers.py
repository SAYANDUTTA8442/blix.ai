"""
Miscellaneous helper utilities — Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file.

    Returns an empty list ``[]`` when *path* does not exist, so callers
    can treat a missing file as an empty collection without special-casing.

    Parameters
    ----------
    path:
        File to read.

    Returns
    -------
    Any
        Parsed JSON value, or ``[]`` if the file is absent.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """
    Serialise *data* to *path* as pretty-printed JSON.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    path:
        Destination file.
    data:
        JSON-serialisable object.
    indent:
        Indentation width (default 2).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default, ensure_ascii=False)


def _json_default(obj: Any) -> Any:
    """Custom JSON encoder for types not handled by stdlib ``json``."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON-serialisable")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def truncate(text: str, max_chars: int = 80) -> str:
    """
    Return *text* truncated to *max_chars* with a trailing ellipsis.

    Parameters
    ----------
    text:
        Source string.
    max_chars:
        Hard limit (default 80).
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def strip_whitespace(text: str) -> str:
    """Collapse runs of whitespace/newlines to a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def format_timestamp(dt: datetime) -> str:
    """Return a human-readable local timestamp string."""
    return dt.strftime("%Y-%m-%d %H:%M UTC")
