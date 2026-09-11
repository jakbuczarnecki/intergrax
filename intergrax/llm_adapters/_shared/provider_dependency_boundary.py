# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Process-local shared provider ``DependencyAttemptExecutionBoundary`` (W2-B3)."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)

_process_provider_dependency_boundary: DependencyAttemptExecutionBoundary | None = None


def set_llm_provider_dependency_boundary(
    boundary: DependencyAttemptExecutionBoundary | None,
) -> None:
    """Tier-3 / runtime bootstrap: one shared boundary per process for all LLM adapters."""
    global _process_provider_dependency_boundary
    _process_provider_dependency_boundary = boundary


def get_llm_provider_dependency_boundary() -> DependencyAttemptExecutionBoundary | None:
    return _process_provider_dependency_boundary


def apply_llm_provider_dependency_boundary(adapter: LLMAdapter) -> None:
    boundary = _process_provider_dependency_boundary
    if boundary is not None:
        adapter.bind_provider_dependency_boundary(boundary)
