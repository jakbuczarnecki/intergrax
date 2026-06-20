# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile EBE fields to RuntimeConfig (partner PoC)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.attestation.settings import (
    ExecutionBoundaryExportRuntimeSettings,
    resolve_execution_boundary_export_runtime,
)
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.host_attestation import resolve_host_attestation_sealer_from_env
from intergrax.runtime.nexus.config import RuntimeConfig


def build_boundary_event_buffer(
    profile: ApplicationEnvironmentProfile,
) -> BoundaryEventBuffer | None:
    export_profile = profile.execution_boundary_export_profile
    if export_profile is None or not export_profile.enabled:
        return None
    sealer = resolve_host_attestation_sealer_from_env(
        enabled=export_profile.host_signing_enabled,
        public_key_id=export_profile.host_signing_public_key_id,
    )
    return BoundaryEventBuffer(host_attestation_sealer=sealer)


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
