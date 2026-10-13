# © Artur Czarnecki. All rights reserved.

"""Strict host dependency-concurrency admission materialization (HARNESS-W4-R1 / PROD-Q-R1).

Declarative ``ReliabilityProfile.dependency_concurrency_admission`` remains the
configuration owner. When STRICT production hosts omit explicit bindings, this
module materializes one explicit TOOL policy per registered catalog tool id at
the sanctioned declarative-tool composition root (fail-closed when the registry
is empty).
"""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionConfiguration,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyBinding,
)

# Reference capacity aligned with product ``OrchestrationProfile.max_inflight_nodes``.
REFERENCE_STRICT_PRODUCT_TOOL_MAX_CONCURRENT_CALLS: int = 8


def build_dependency_concurrency_admission_for_tool_ids(
    tool_ids: tuple[str, ...],
    *,
    max_concurrent_calls: int = REFERENCE_STRICT_PRODUCT_TOOL_MAX_CONCURRENT_CALLS,
) -> DependencyConcurrencyAdmissionConfiguration:
    """Build explicit per-tool admission bindings for one sorted tool-id snapshot."""
    bindings: list[DependencyConcurrencyPolicyBinding] = []
    policy = DependencyConcurrencyPolicy(
        max_concurrent_calls=max_concurrent_calls,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
    )
    for tool_id in tool_ids:
        if not tool_id:
            continue
        bindings.append(
            DependencyConcurrencyPolicyBinding(
                identity=DependencyConcurrencyIdentity(
                    kind=DependencyConcurrencyKind.TOOL,
                    value=tool_id,
                ),
                policy=policy,
            ),
        )
    return DependencyConcurrencyAdmissionConfiguration(bindings=tuple(bindings))


def resolve_dependency_concurrency_admission_configuration(
    configured: DependencyConcurrencyAdmissionConfiguration | None,
    *,
    tool_wiring: ApplicationToolWiring,
    production_mode: bool,
) -> DependencyConcurrencyAdmissionConfiguration | None:
    """Resolve admission configuration for declarative strict production wiring."""
    if not production_mode:
        return configured
    if configured is not None:
        return configured
    tool_ids = tuple(sorted(tool_wiring.registry.tool_ids()))
    if not tool_ids:
        return None
    return build_dependency_concurrency_admission_for_tool_ids(tool_ids)


__all__ = [
    "REFERENCE_STRICT_PRODUCT_TOOL_MAX_CONCURRENT_CALLS",
    "build_dependency_concurrency_admission_for_tool_ids",
    "resolve_dependency_concurrency_admission_configuration",
]
