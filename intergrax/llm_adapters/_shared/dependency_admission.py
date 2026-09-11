# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""W2 dependency admission error classification for LLM resilience paths."""

from __future__ import annotations

from intergrax.contracts.dependency_concurrency_admission import (
    is_dependency_concurrency_admission_error,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundaryClosedError,
    DependencyAttemptReleaseInvariantError,
)


def is_non_retriable_dependency_admission_failure(exc: BaseException) -> bool:
    """Admission / boundary lifecycle failures must not retry, failover, or poison CB."""
    if is_dependency_concurrency_admission_error(exc):
        return True
    return isinstance(
        exc,
        (
            DependencyAttemptExecutionBoundaryClosedError,
            DependencyAttemptReleaseInvariantError,
        ),
    )
