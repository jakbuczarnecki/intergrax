# © Artur Czarnecki. All rights reserved.

"""Runtime settings resolved from Tier-3 ``ExecutionBoundaryExportProfile``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.runtime.attestation.attestation_policy import AttestationCaptureMode

if TYPE_CHECKING:
    from intergrax.contracts.host_profile_slices import ExecutionBoundaryExportProfile


@dataclass(frozen=True, slots=True)
class ExecutionBoundaryExportRuntimeSettings:
    enabled: bool
    capture_mode: AttestationCaptureMode
    allowlist: frozenset[str]
    include_canonical_io: bool = True
    step_level_enabled: bool = False
    host_signing_enabled: bool = False
    host_signing_public_key_id: str = "attestation-demo-host-1"


def resolve_execution_boundary_export_runtime(
    profile: ExecutionBoundaryExportProfile | None,
) -> ExecutionBoundaryExportRuntimeSettings | None:
    if profile is None or not profile.enabled:
        return None
    mode_raw = (profile.capture_mode or "side_effects_only").strip().lower()
    try:
        capture_mode = AttestationCaptureMode(mode_raw)
    except ValueError:
        capture_mode = AttestationCaptureMode.SIDE_EFFECTS_ONLY
    return ExecutionBoundaryExportRuntimeSettings(
        enabled=True,
        capture_mode=capture_mode,
        allowlist=frozenset(profile.allowlist),
        include_canonical_io=profile.include_canonical_io,
        step_level_enabled=profile.step_level_enabled,
        host_signing_enabled=profile.host_signing_enabled,
        host_signing_public_key_id=profile.host_signing_public_key_id,
    )
