"""
api/deps.py — FastAPI dependency injection for BlixContext
"""

from __future__ import annotations

from typing import Optional

from api.context import BlixContext

# Module-level singleton — set once at startup by api/server.py
_context: Optional[BlixContext] = None


def set_context(ctx: BlixContext) -> None:
    global _context
    _context = ctx


def get_context() -> BlixContext:
    if _context is None:
        raise RuntimeError(
            "BlixContext has not been initialised. "
            "Call set_context() before handling requests."
        )
    return _context
