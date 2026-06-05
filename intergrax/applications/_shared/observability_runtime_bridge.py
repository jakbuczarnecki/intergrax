# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile observability fields to wiring options (Phase OBS-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ObservabilityProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class ObservabilityWiringOptions:
    """Resolved observability wiring flags for Tier-3 hosts."""

    use_in_memory_trace: bool
    enable_runtime_events: bool
    otel_enabled: bool
    metrics_plugins_enabled: bool
    debug_surface_override: bool | None


def resolve_observability_wiring_options(
    profile: ObservabilityProfile,
) -> ObservabilityWiringOptions:
    """Translate ``ObservabilityProfile`` into ``wire_nexus_observability`` flags."""
    return ObservabilityWiringOptions(
        use_in_memory_trace=not profile.trace_sqlite_enabled,
        enable_runtime_events=profile.trace_sqlite_enabled,
        otel_enabled=profile.otel_enabled,
        metrics_plugins_enabled=profile.metrics_plugins_enabled,
        debug_surface_override=profile.debug_surface_override,
    )


def apply_observability_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: ObservabilityProfile,
) -> RuntimeConfig:
    """Record observability posture on runtime config for downstream diagnostics."""
    options = resolve_observability_wiring_options(profile)
    if options.use_in_memory_trace:
        config.trace_db_path = None
    return config


def apply_observability_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared observability profile."""
    return apply_observability_profile_to_runtime_config(config, env.observability_profile)
