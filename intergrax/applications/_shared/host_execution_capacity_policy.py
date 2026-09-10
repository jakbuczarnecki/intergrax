# © Artur Czarnecki. All rights reserved.

"""Strict (production) host-local execution capacity guardrails (Enterprise Scale W0).

``max_parallel_nodes`` and ``max_inflight_nodes`` are **process-local** GraphExecutor
limits (asyncio semaphores per Nexus instance). They are not cluster-wide capacity.
"""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode


class HostExecutionCapacityPolicyError(ValueError):
    """Raised when strict execution mode lacks mandatory host-local capacity caps."""


def strict_host_execution_capacity_violations(
    env: ApplicationEnvironmentProfile,
) -> tuple[str, ...]:
    """Return conformance messages when strict mode omits orchestration capacity caps."""
    if env.execution_mode is not ExecutionMode.STRICT:
        return ()
    orch = env.orchestration_profile
    violations: list[str] = []
    if orch.max_parallel_nodes is None:
        violations.append(
            "execution_mode=strict requires orchestration_profile.max_parallel_nodes "
            "(process-local batch parallelism cap; not cluster-wide)",
        )
    if orch.max_inflight_nodes is None:
        violations.append(
            "execution_mode=strict requires orchestration_profile.max_inflight_nodes "
            "(process-local inflight cap; not cluster-wide)",
        )
    return tuple(violations)


def validate_strict_host_execution_capacity(env: ApplicationEnvironmentProfile) -> None:
    """Fail closed before Nexus/GraphExecutor composition in strict (production) mode."""
    violations = strict_host_execution_capacity_violations(env)
    if violations:
        raise HostExecutionCapacityPolicyError("; ".join(violations))
