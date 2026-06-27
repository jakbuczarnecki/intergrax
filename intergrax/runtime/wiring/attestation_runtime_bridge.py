# © Artur Czarnecki. All rights reserved.

"""Map execution boundary export profile to RuntimeConfig."""

from __future__ import annotations

from intergrax.contracts.host_profile_slices import ExecutionBoundaryExportProfile
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.settings import resolve_execution_boundary_export_runtime
from intergrax.runtime.nexus.config import RuntimeConfig


def apply_attestation_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: RuntimeEnvironmentProfile | ExecutionBoundaryExportProfile | None,
    *,
    boundary_event_buffer: BoundaryEventBuffer | None = None,
) -> RuntimeConfig:
    """Attach execution boundary export settings and optional shared buffer."""
    export_profile: ExecutionBoundaryExportProfile | None
    if isinstance(profile, RuntimeEnvironmentProfile):
        export_profile = profile.execution_boundary_export_profile
    elif isinstance(profile, ExecutionBoundaryExportProfile):
        export_profile = profile
    else:
        export_profile = None
    export_settings = resolve_execution_boundary_export_runtime(export_profile)
    if export_settings is not None:
        config.execution_boundary_export = export_settings
    if boundary_event_buffer is not None:
        config.boundary_event_buffer = boundary_event_buffer
    return config
