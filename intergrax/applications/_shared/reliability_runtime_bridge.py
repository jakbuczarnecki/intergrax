# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile reliability fields to wiring options (Phase REL-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.nexus.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class ReliabilityWiringOptions:
    """Resolved reliability wiring flags for Tier-3 hosts."""

    idempotency_enabled: bool
    circuit_breaker_failure_threshold: int
    checkpoint_interval_steps: int
    long_running_scheduler_enabled: bool


def resolve_reliability_wiring_options(
    profile: ReliabilityProfile,
) -> ReliabilityWiringOptions:
    """Translate ``ReliabilityProfile`` into host wiring flags."""
    return ReliabilityWiringOptions(
        idempotency_enabled=profile.idempotency_enabled,
        circuit_breaker_failure_threshold=profile.circuit_breaker_failure_threshold,
        checkpoint_interval_steps=profile.checkpoint_interval_steps,
        long_running_scheduler_enabled=profile.long_running_scheduler_enabled,
    )


def apply_reliability_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: ReliabilityProfile,
    *,
    idempotency_store: IdempotencyStore | None = None,
) -> RuntimeConfig:
    """Record reliability posture on runtime config for downstream tool invocations."""
    options = resolve_reliability_wiring_options(profile)
    if options.idempotency_enabled:
        config.idempotency_store = idempotency_store
    else:
        config.idempotency_store = None
    return config


def apply_reliability_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
    *,
    idempotency_store: IdempotencyStore | None = None,
) -> RuntimeConfig:
    """Apply environment-declared reliability profile."""
    return apply_reliability_profile_to_runtime_config(
        config,
        env.reliability_profile,
        idempotency_store=idempotency_store,
    )
