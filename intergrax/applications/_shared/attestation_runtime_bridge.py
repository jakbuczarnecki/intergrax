# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile EBE fields to RuntimeConfig (partner PoC)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.attestation.settings import resolve_execution_boundary_export_runtime
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.nexus.config import RuntimeConfig


def apply_attestation_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: ApplicationEnvironmentProfile,
    *,
    boundary_event_buffer: BoundaryEventBuffer | None = None,
) -> RuntimeConfig:
    """Attach execution boundary export settings and optional shared buffer."""
    export_settings = resolve_execution_boundary_export_runtime(
        profile.execution_boundary_export_profile,
    )
    if export_settings is not None:
        config.execution_boundary_export = export_settings
    if boundary_event_buffer is not None:
        config.boundary_event_buffer = boundary_event_buffer
    return config


def apply_attestation_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
    *,
    boundary_event_buffer: BoundaryEventBuffer | None = None,
) -> RuntimeConfig:
    return apply_attestation_profile_to_runtime_config(
        config,
        env,
        boundary_event_buffer=boundary_event_buffer,
    )
