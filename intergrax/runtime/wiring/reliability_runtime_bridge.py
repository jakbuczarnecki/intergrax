# © Artur Czarnecki. All rights reserved.

"""Map reliability profile fields to wiring options."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.host_profile_slices import ReliabilityProfile
from intergrax.contracts.resilience_policy import ResiliencePolicy, default_resilience_policy
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.nexus.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class ReliabilityWiringOptions:
    """Resolved reliability wiring flags."""

    idempotency_enabled: bool
    circuit_breaker_failure_threshold: int
    checkpoint_interval_steps: int
    long_running_scheduler_enabled: bool
    resilience_policy: ResiliencePolicy
    default_autonomy_level: str
    tenant_autonomy_ceiling: str | None


def resolve_reliability_wiring_options(
    profile: ReliabilityProfile,
) -> ReliabilityWiringOptions:
    """Translate ``ReliabilityProfile`` into host wiring flags."""
    policy = profile.resilience_policy or default_resilience_policy()
    ceiling = profile.tenant_autonomy_ceiling
    return ReliabilityWiringOptions(
        idempotency_enabled=profile.idempotency_enabled,
        circuit_breaker_failure_threshold=profile.circuit_breaker_failure_threshold,
        checkpoint_interval_steps=profile.checkpoint_interval_steps,
        long_running_scheduler_enabled=profile.long_running_scheduler_enabled,
        resilience_policy=policy,
        default_autonomy_level=profile.default_autonomy_level.value,
        tenant_autonomy_ceiling=ceiling.value if ceiling is not None else None,
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
