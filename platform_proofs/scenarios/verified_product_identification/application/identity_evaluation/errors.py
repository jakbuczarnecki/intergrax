"""Identity hypothesis evaluation failures — distinct from hypothesis formation."""

from __future__ import annotations


class IdentityHypothesisEvaluationError(ValueError):
    """Raised when identity hypothesis evaluation input violates invariants."""
