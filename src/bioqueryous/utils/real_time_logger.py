"""Lightweight logging shim for experiment-time observability."""

from __future__ import annotations

import logging
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Return a module-level logger configured for console output."""

    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("bioqueryous")
        if not _LOGGER.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            _LOGGER.addHandler(handler)
        _LOGGER.setLevel(logging.INFO)
    return _LOGGER
