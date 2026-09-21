# © Artur Czarnecki. All rights reserved.

"""OBS/DIAG provider qualification matrix — manifest discovery + explicit classification."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationStatus

from testing_support.obs_diag_provider_qualification.descriptor import (
    ObsDiagProviderClass,
    ObsDiagProviderDomain,
    ObsDiagProviderQualificationDescriptor,
    ObsDiagProviderSupportStatus,
)
from testing_support.obs_diag_provider_qualification.discovery import (
    DiscoveredObsDiagProvider,
    discover_obs_diag_provider_surfaces,
)

_INTERNAL_REGISTRY: tuple[ObsDiagProviderQualificationDescriptor, ...] = (
    ObsDiagProviderQualificationDescriptor(
        provider_id="sqlite-file",
        domain=ObsDiagProviderDomain.PERSISTENCE,
        provider_class=ObsDiagProviderClass.PLATFORM_INTERNAL,
        contract="EvidencePersistencePort / DocumentStore (cross-process qualification)",
        integration_status="qualification-backend",
        adapter_exists=True,
        live_proof_module="tests/unit/runtime/architecture/test_obs_dg005_distributed_topology_qualification.py",
        failure_recovery_proof_module=(
            "tests/integration/providers/obs_diag/test_x5_sqlite_file_persistence_qualification.py"
        ),
        declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
        discovery_source="platform-internal-registry",
        delivery_or_durability_note="A writes → A.close() → B reads (file-backed)",
    ),
    ObsDiagProviderQualificationDescriptor(
        provider_id="in-process-async",
        domain=ObsDiagProviderDomain.TRANSPORT,
        provider_class=ObsDiagProviderClass.PLATFORM_INTERNAL,
        contract="colocated worker / async spine (non-broker)",
        integration_status="platform-internal",
        adapter_exists=True,
        live_proof_module="tests/integration/runtime/test_obs_universal_spine_async_e2e.py",
        failure_recovery_proof_module=None,
        declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
        discovery_source="platform-internal-registry",
        delivery_or_durability_note="in-process only; not external broker qualification",
    ),
)

_PLATFORM_SEMANTICS_REGISTRY: tuple[ObsDiagProviderQualificationDescriptor, ...] = (
    ObsDiagProviderQualificationDescriptor(
        provider_id="observability_export_semantics",
        domain=ObsDiagProviderDomain.TELEMETRY,
        provider_class=ObsDiagProviderClass.PLATFORM_EXPORT_SEMANTICS,
        contract="ObservabilityExporter derived export (platform behavior)",
        integration_status="platform-semantics",
        adapter_exists=True,
        live_proof_module="tests/unit/runtime/observability/test_harden_3c_export_failure_semantics.py",
        failure_recovery_proof_module="tests/unit/runtime/observability/test_export_policy.py",
        declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
        discovery_source="platform-export-semantics-registry",
        delivery_or_durability_note=(
            "fail-open export; canonical RuntimeEvent unchanged (HARDEN-3C); not live OTLP endpoint"
        ),
    ),
)

_EXTERNAL_CLASSIFICATION_OVERRIDES: dict[str, ObsDiagProviderQualificationDescriptor] = {
    "kafka": ObsDiagProviderQualificationDescriptor(
        provider_id="kafka",
        domain=ObsDiagProviderDomain.TRANSPORT,
        provider_class=ObsDiagProviderClass.EXTERNAL_VENDOR,
        contract="MessageProducer / MessageConsumer / TaskQueue (OBS spine)",
        integration_status=IntegrationStatus.STABLE.value,
        adapter_exists=True,
        live_proof_module="tests/integration/runtime/test_obs_universal_spine_cross_process_x4_e2e.py",
        failure_recovery_proof_module=(
            "tests/integration/providers/obs_diag/test_x5_kafka_transport_failure_recovery.py"
        ),
        declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
        discovery_source="manifest",
        delivery_or_durability_note=(
            "at-least-once; unreachable endpoint failure + fresh-provider recovery (X5)"
        ),
    ),
    "mongodb": ObsDiagProviderQualificationDescriptor(
        provider_id="mongodb",
        domain=ObsDiagProviderDomain.PERSISTENCE,
        provider_class=ObsDiagProviderClass.EXTERNAL_VENDOR,
        contract="DocumentStore → FunctionalEvidencePersistence / Problem persistence",
        integration_status=IntegrationStatus.STABLE.value,
        adapter_exists=True,
        live_proof_module="tests/system/functional_diagnostics_durability/",
        failure_recovery_proof_module=None,
        declared_status=ObsDiagProviderSupportStatus.SUPPORTED_NOT_QUALIFIED,
        discovery_source="manifest",
        delivery_or_durability_note="D1-R1 process-boundary qualified when INTERGRAX_MONGODB_URI set",
    ),
}


def _adapter_only_descriptor(
    discovered: DiscoveredObsDiagProvider,
) -> ObsDiagProviderQualificationDescriptor:
    return ObsDiagProviderQualificationDescriptor(
        provider_id=discovered.provider_id,
        domain=discovered.domain,
        provider_class=ObsDiagProviderClass.EXTERNAL_VENDOR,
        contract=f"Integration catalog adapter ({discovered.adapter_package})",
        integration_status=discovered.integration_status.value,
        adapter_exists=True,
        live_proof_module=None,
        failure_recovery_proof_module=None,
        declared_status=ObsDiagProviderSupportStatus.ADAPTER_ONLY,
        discovery_source=discovered.manifest_path,
        delivery_or_durability_note="catalog adapter; no OBS/DIAG live qualification executed",
    )


def _merge_override_discovery_source(
    override: ObsDiagProviderQualificationDescriptor,
    manifest_path: str,
) -> ObsDiagProviderQualificationDescriptor:
    return ObsDiagProviderQualificationDescriptor(
        provider_id=override.provider_id,
        domain=override.domain,
        provider_class=override.provider_class,
        contract=override.contract,
        integration_status=override.integration_status,
        adapter_exists=override.adapter_exists,
        live_proof_module=override.live_proof_module,
        failure_recovery_proof_module=override.failure_recovery_proof_module,
        declared_status=override.declared_status,
        discovery_source=manifest_path,
        delivery_or_durability_note=override.delivery_or_durability_note,
    )


def build_obs_diag_external_classifications(
    discovered: tuple[DiscoveredObsDiagProvider, ...],
) -> tuple[ObsDiagProviderQualificationDescriptor, ...]:
    rows: list[ObsDiagProviderQualificationDescriptor] = []
    for item in discovered:
        override = _EXTERNAL_CLASSIFICATION_OVERRIDES.get(item.provider_id)
        if override is not None:
            rows.append(_merge_override_discovery_source(override, item.manifest_path))
        else:
            rows.append(_adapter_only_descriptor(item))
    rows.sort(key=lambda row: row.provider_id)
    return tuple(rows)


_DISCOVERED_AT_HEAD = discover_obs_diag_provider_surfaces()
OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS = build_obs_diag_external_classifications(
    _DISCOVERED_AT_HEAD,
)

OBS_DIAG_X5_PROVIDER_INVENTORY: tuple[ObsDiagProviderQualificationDescriptor, ...] = (
    *_INTERNAL_REGISTRY,
    *_PLATFORM_SEMANTICS_REGISTRY,
    *OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS,
)
