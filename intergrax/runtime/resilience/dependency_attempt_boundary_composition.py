# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned materialization: declarative dependency admission → execution boundary."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionConfiguration,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyPolicy,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)


class ToolDependencyAttemptBoundaryMaterializationError(RuntimeError):
    """Fail closed when strict production tool admission cannot be materialized."""


def materialize_tool_dependency_attempt_boundary(
    configuration: DependencyConcurrencyAdmissionConfiguration | None,
    *,
    production_mode: bool,
) -> DependencyAttemptExecutionBoundary | None:
    """Build ``DependencyAttemptExecutionBoundary`` from typed declarative configuration."""
    if configuration is None:
        if production_mode:
            raise ToolDependencyAttemptBoundaryMaterializationError(
                "strict production requires reliability_profile.dependency_concurrency_admission",
            )
        return None

    tool_bindings = [
        binding
        for binding in configuration.bindings
        if binding.identity.kind is DependencyConcurrencyKind.TOOL
    ]
    if production_mode and not tool_bindings:
        raise ToolDependencyAttemptBoundaryMaterializationError(
            "strict production requires at least one TOOL dependency concurrency binding",
        )

    if not configuration.bindings:
        if production_mode:
            raise ToolDependencyAttemptBoundaryMaterializationError(
                "strict production requires dependency concurrency admission bindings",
            )
        return None

    policies: dict[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy] = {}
    for binding in configuration.bindings:
        policies[binding.identity] = binding.policy

    admission = LocalDependencyConcurrencyAdmission(_policy_snapshot(policies))
    return DependencyAttemptExecutionBoundary(admission)


def _policy_snapshot(
    policies: Mapping[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy],
) -> Mapping[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy]:
    return dict(policies)


__all__ = [
    "ToolDependencyAttemptBoundaryMaterializationError",
    "materialize_tool_dependency_attempt_boundary",
]
