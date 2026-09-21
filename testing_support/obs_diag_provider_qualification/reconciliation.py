# © Artur Czarnecki. All rights reserved.

"""Anti-drift reconciliation between manifest discovery and qualification matrix."""

from __future__ import annotations

from testing_support.obs_diag_provider_qualification.descriptor import (
    ObsDiagProviderClass,
    ObsDiagProviderQualificationDescriptor,
    ObsDiagProviderSupportStatus,
)
from testing_support.obs_diag_provider_qualification.discovery import (
    DiscoveredObsDiagProvider,
)


def obs_diag_external_provider_ids(
    discovered: tuple[DiscoveredObsDiagProvider, ...],
) -> frozenset[str]:
    return frozenset(row.provider_id for row in discovered)


def obs_diag_classified_external_provider_ids(
    classifications: tuple[ObsDiagProviderQualificationDescriptor, ...],
) -> frozenset[str]:
    return frozenset(
        row.provider_id
        for row in classifications
        if row.provider_class == ObsDiagProviderClass.EXTERNAL_VENDOR
    )


def obs_diag_anti_drift_delta(
    *,
    discovered: tuple[DiscoveredObsDiagProvider, ...],
    classifications: tuple[ObsDiagProviderQualificationDescriptor, ...],
) -> tuple[list[str], list[str]]:
    discovered_ids = obs_diag_external_provider_ids(discovered)
    classified_ids = obs_diag_classified_external_provider_ids(classifications)
    missing = sorted(discovered_ids - classified_ids)
    stale = sorted(classified_ids - discovered_ids)
    return missing, stale


def obs_diag_qualified_external_without_proof(
    classifications: tuple[ObsDiagProviderQualificationDescriptor, ...],
) -> list[str]:
    violations: list[str] = []
    for row in classifications:
        if row.provider_class != ObsDiagProviderClass.EXTERNAL_VENDOR:
            continue
        if row.declared_status != ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED:
            continue
        if row.live_proof_module is None or row.failure_recovery_proof_module is None:
            violations.append(row.provider_id)
    return violations


def obs_diag_telemetry_vendor_falsely_qualified(
    classifications: tuple[ObsDiagProviderQualificationDescriptor, ...],
) -> list[str]:
    violations: list[str] = []
    for row in classifications:
        if row.provider_class != ObsDiagProviderClass.EXTERNAL_VENDOR:
            continue
        if row.domain.value != "telemetry":
            continue
        if row.declared_status == ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED:
            violations.append(row.provider_id)
    return violations
