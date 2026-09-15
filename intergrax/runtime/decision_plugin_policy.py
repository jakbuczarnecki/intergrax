# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision plugin load policy and production admission (metadata-only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import ConflictPolicy, EntryPointSpec, LoadIsolation
from intergrax.core.plugins.selection_ref import PlatformPluginSelectionRef
from intergrax.core.plugins.platform_qualification import (
    PluginQualificationResult,
    evaluate_external_package_entry_point_production_admission,
    resolve_host_platform_version,
)


@dataclass(frozen=True, slots=True)
class DecisionPluginLoadPolicy:
    """Shared Decision plugin load governance for all capability groups."""

    ep_name_conflict: ConflictPolicy = "error"
    on_load_failure: LoadIsolation = "isolate"
    require_production_admission: bool = False
    require_manifest_capability_binding: bool = False
    package_qualification_lookup: (
        Callable[[EntryPointSpec], PluginQualificationResult | None] | None
    ) = None
    platform_version: str | None = None
    requested_strategy_plugins: tuple[PlatformPluginSelectionRef, ...] | None = None
    requested_verification_stage_plugins: tuple[PlatformPluginSelectionRef, ...] | None = None
    requested_artifact_plugins: tuple[PlatformPluginSelectionRef, ...] | None = None
    requested_exposure_selection_strategy_plugins: (
        tuple[PlatformPluginSelectionRef, ...] | None
    ) = None


def production_admission_rejection_for_spec(
    spec: EntryPointSpec,
    policy: DecisionPluginLoadPolicy,
) -> PluginAdmissionRejection | None:
    if not policy.require_production_admission:
        return None

    platform_version = policy.platform_version or resolve_host_platform_version()
    lookup = policy.package_qualification_lookup
    qualification = lookup(spec) if lookup is not None else None
    admission = evaluate_external_package_entry_point_production_admission(
        spec,
        qualification,
        platform_version=platform_version,
    )
    if admission.admitted:
        return None
    return PluginAdmissionRejection(
        spec=spec,
        reason_code=PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED,
        reason=admission.reason,
        fail_closed=True,
    )
