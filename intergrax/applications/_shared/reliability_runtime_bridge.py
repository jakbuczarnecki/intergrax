# © Artur Czarnecki. All rights reserved.

"""Compatibility wrapper — runtime bridges with application profile adapters."""

from __future__ import annotations

from intergrax.applications._shared.runtime_boundary_adapters import RELIABILITY_PROJECTION
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.execution.host_runtime_config import RuntimeConfig
from intergrax.runtime.wiring.reliability_runtime_bridge import (
    ReliabilityWiringOptions,
    apply_reliability_profile_to_runtime_config,
    resolve_reliability_wiring_options,
)

__all__ = [
    "ReliabilityWiringOptions",
    "apply_reliability_profile_to_runtime_config",
    "apply_reliability_profiles_from_environment",
    "resolve_reliability_wiring_options",
]


def apply_reliability_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
    *,
    idempotency_store: IdempotencyStore | None = None,
) -> RuntimeConfig:
    """Apply environment-declared reliability profile."""
    return apply_reliability_profile_to_runtime_config(
        config,
        RELIABILITY_PROJECTION.project(env.reliability_profile),
        idempotency_store=idempotency_store,
    )
