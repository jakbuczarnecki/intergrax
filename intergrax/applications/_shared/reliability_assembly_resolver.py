# © Artur Czarnecki. All rights reserved.

"""Reliability assembly validation for Tier-3 hosts (Phase REL-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.reliability_wiring import ApplicationReliabilityWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.persistence_topology import (
    format_topology_mismatch_error,
    resolve_idempotency_store_topology,
    topology_satisfies,
)


@dataclass(frozen=True, slots=True)
class ReliabilityAssemblyValidationResult:
    """Outcome of reliability assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class ReliabilityAssemblyError(ValueError):
    """Raised when reliability assembly validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def validate_reliability_wiring(
    wiring: ApplicationReliabilityWiring,
    env: ApplicationEnvironmentProfile,
) -> ReliabilityAssemblyValidationResult:
    """Validate reliability artifacts match environment profile requirements."""
    errors: list[str] = []
    reliability = env.reliability_profile
    orchestration = env.orchestration_profile

    if reliability.long_running_scheduler_enabled and not orchestration.long_running_enabled:
        errors.append(
            "long_running_scheduler_enabled requires orchestration_profile.long_running_enabled",
        )

    if reliability.idempotency_enabled and wiring.idempotency_store is None:
        errors.append("idempotency_enabled requires idempotency_store")

    if not reliability.idempotency_enabled and wiring.idempotency_store is not None:
        errors.append("idempotency_enabled=False requires no idempotency_store")

    if reliability.idempotency_enabled:
        required_topology = env.meta.required_persistence_topology
        provided_topology = resolve_idempotency_store_topology(wiring.idempotency_store)
        if provided_topology is None:
            errors.append(
                format_topology_mismatch_error(
                    mechanism="idempotency",
                    required=required_topology,
                    provided=None,
                ),
            )
        elif not topology_satisfies(required_topology, provided_topology):
            errors.append(
                format_topology_mismatch_error(
                    mechanism="idempotency",
                    required=required_topology,
                    provided=provided_topology,
                ),
            )

    if (
        wiring.circuit_breaker_config.failure_threshold
        != reliability.circuit_breaker_failure_threshold
    ):
        errors.append("circuit_breaker_config must match reliability_profile.circuit_breaker_failure_threshold")

    return ReliabilityAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_reliability_assembly_valid(
    wiring: ApplicationReliabilityWiring,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Raise :class:`ReliabilityAssemblyError` when reliability validation fails."""
    result = validate_reliability_wiring(wiring, env)
    if not result.valid:
        raise ReliabilityAssemblyError(result.errors)
