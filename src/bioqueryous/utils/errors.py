"""Shared exception types used by the unified LLM client."""

from __future__ import annotations

DEFAULT_RETRY_DELAY: float = 1.0


class APIError(RuntimeError):
    """Raised when an upstream model provider returns a non-recoverable error."""


class CircuitBreakerOpenError(RuntimeError):
    """Raised when every model in a provider pool is temporarily unavailable."""
