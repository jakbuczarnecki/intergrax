# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile EBE fields to RuntimeConfig (partner PoC)."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.core.security.security_configuration import (
    DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
    SecurityConfigurationError,
)
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.host_attestation import resolve_host_attestation_sealer_from_env
from intergrax.runtime.attestation.settings import (
    resolve_execution_boundary_export_runtime,
)
from intergrax.runtime.execution.host_runtime_config import RuntimeConfig


def _is_strict_profile(profile: ApplicationEnvironmentProfile) -> bool:
    return profile.application_profile is ApplicationProfile.PRODUCT or profile.execution_mode is ExecutionMode.STRICT


def build_boundary_event_buffer(
    profile: ApplicationEnvironmentProfile,
    *,
    environ: Mapping[str, str] | None = None,
    lab_demo_seed_opt_in: bool = False,
) -> BoundaryEventBuffer | None:
    """Build the EBE buffer; host signing needs an explicit key in strict profiles (R027).

    The public demo seed is reachable only through the named lab and proof opt-in. Lab
    (non-strict) profiles are the explicit lab path and apply it; a strict profile never does,
    and an explicit ``lab_demo_seed_opt_in`` on a strict profile is a configuration error.
    """
    export_profile = profile.execution_boundary_export_profile
    if export_profile is None or not export_profile.enabled:
        return None
    strict = _is_strict_profile(profile)
    if strict and lab_demo_seed_opt_in:
        raise SecurityConfigurationError(
            item="host_attestation",
            reason_code=DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
            message="the lab demo-seed opt-in is not accepted on a strict profile",
        )
    sealer = resolve_host_attestation_sealer_from_env(
        enabled=export_profile.host_signing_enabled,
        public_key_id=export_profile.host_signing_public_key_id,
        environ=environ,
        lab_demo_seed_opt_in=not strict,
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
