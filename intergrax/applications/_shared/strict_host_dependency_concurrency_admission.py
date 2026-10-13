# © Artur Czarnecki. All rights reserved.

"""Reference-host tool dependency-concurrency admission helpers (HARNESS-W4-R1 / PROD-Q-R1).

Declarative ``ReliabilityProfile.dependency_concurrency_admission`` is the sole
configuration owner. Shared declarative wiring must not synthesize policies from
the live tool registry.
"""

from __future__ import annotations

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionConfiguration,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyBinding,
)
from intergrax.skills.registry.tool_requirements import available_tool_ids_for_profile
from intergrax.tools.contracts.tool_profile import ToolProfile

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


def build_reference_host_tool_dependency_concurrency_admission(
    tool_profile: ToolProfile,
    *,
    max_concurrent_calls: int,
) -> DependencyConcurrencyAdmissionConfiguration | None:
    """Materialize explicit TOOL admission for reference host tool availability."""
    tool_ids = available_tool_ids_for_profile(tool_profile)
    if not tool_ids:
        return None
    return build_dependency_concurrency_admission_for_tool_ids(
        tool_ids,
        max_concurrent_calls=max_concurrent_calls,
    )


def resolve_dependency_concurrency_admission_configuration(
    configured: DependencyConcurrencyAdmissionConfiguration | None,
    *,
    production_mode: bool,
) -> DependencyConcurrencyAdmissionConfiguration | None:
    """Return profile-owned admission configuration without registry synthesis."""
    _ = production_mode
    return configured


__all__ = [
    "REFERENCE_STRICT_PRODUCT_TOOL_MAX_CONCURRENT_CALLS",
    "build_dependency_concurrency_admission_for_tool_ids",
    "build_reference_host_tool_dependency_concurrency_admission",
    "resolve_dependency_concurrency_admission_configuration",
]
