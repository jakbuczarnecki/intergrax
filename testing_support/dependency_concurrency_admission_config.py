# © Artur Czarnecki. All rights reserved.

"""Explicit typed dependency-concurrency configurations for tests (not production defaults)."""

from __future__ import annotations

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionConfiguration,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyBinding,
)


def tool_dependency_concurrency_admission_configuration(
    tool_id: str,
    *,
    max_concurrent_calls: int = 2,
    overload_mode: DependencyConcurrencyOverloadMode = DependencyConcurrencyOverloadMode.REJECT,
    wait_timeout_seconds: float | None = None,
) -> DependencyConcurrencyAdmissionConfiguration:
    """Build a single-TOOL admission configuration for deterministic test wiring."""
    identity = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value=tool_id,
    )
    policy = DependencyConcurrencyPolicy(
        max_concurrent_calls=max_concurrent_calls,
        overload_mode=overload_mode,
        wait_timeout_seconds=wait_timeout_seconds,
    )
    return DependencyConcurrencyAdmissionConfiguration(
        bindings=(
            DependencyConcurrencyPolicyBinding(identity=identity, policy=policy),
        ),
    )
