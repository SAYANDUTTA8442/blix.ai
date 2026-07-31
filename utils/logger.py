"""
Centralised logging setup for Blix — Python 3.10 compatible.

Every module should obtain its logger via ``get_logger(__name__)``.
The root logger is configured once on the first call; subsequent calls
just return the named child logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_root_configured = False


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return a configured named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Explicit log level for this logger.  Defaults to ``INFO``.

    Returns
    -------
    logging.Logger
    """
    global _root_configured
    if not _root_configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.WARNING)  # keep noisy third-party libs quiet
        _root_configured = True

    logger = logging.getLogger(name)
    effective_level = level if level is not None else logging.INFO
    if not logger.level:
        logger.setLevel(effective_level)
    return logger
