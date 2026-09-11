"""Identity hypothesis layer failures — distinct from retrieval or fusion errors."""

from __future__ import annotations


class IdentityHypothesisError(ValueError):
    """Raised when identity hypothesis input violates domain invariants."""


class IdentityEvidenceUnavailableError(IdentityHypothesisError):
    """Raised when required source identity evidence cannot be loaded."""
