# © Artur Czarnecki. All rights reserved.

"""Fresh OBS/DIAG-scoped provider inventory @ HEAD (manifest-driven, not memory)."""

from __future__ import annotations

from testing_support.obs_diag_provider_qualification.descriptor import (
    ObsDiagProviderQualificationDescriptor,
    ObsDiagProviderSupportStatus,
)

# Production OBS/DIAG spine paths (declared in architecture docs + X4 harness).
_SQLITE_FILE = ObsDiagProviderQualificationDescriptor(
    provider_id="sqlite-file",
    domain="persistence",
    contract="EvidencePersistencePort / DocumentStore (cross-process qualification)",
    integration_status="qualification-backend",
    adapter_exists=True,
    live_proof_module="tests/unit/runtime/architecture/test_obs_dg005_distributed_topology_qualification.py",
    failure_recovery_proof_module=(
        "tests/integration/providers/obs_diag/test_x5_sqlite_file_persistence_qualification.py"
    ),
    declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
    delivery_or_durability_note="A writes → A closes → B reads (file-backed)",
)

_KAFKA = ObsDiagProviderQualificationDescriptor(
    provider_id="kafka",
    domain="transport",
    contract="MessageProducer / MessageConsumer / TaskQueue (OBS spine)",
    integration_status="STABLE",
    adapter_exists=True,
    live_proof_module="tests/integration/runtime/test_obs_universal_spine_cross_process_x4_e2e.py",
    failure_recovery_proof_module=(
        "tests/integration/providers/obs_diag/test_x5_kafka_transport_failure_recovery.py"
    ),
    declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
    delivery_or_durability_note="at-least-once; platform idempotency on worker path (X4)",
)

_MONGODB_DOCSTORE = ObsDiagProviderQualificationDescriptor(
    provider_id="mongodb",
    domain="persistence",
    contract="DocumentStore → FunctionalEvidencePersistence / Problem persistence",
    integration_status="STABLE",
    adapter_exists=True,
    live_proof_module="tests/system/functional_diagnostics_durability/",
    failure_recovery_proof_module=None,
    declared_status=ObsDiagProviderSupportStatus.SUPPORTED_NOT_QUALIFIED,
    delivery_or_durability_note="D1-R1 process-boundary qualified when INTERGRAX_MONGODB_URI set",
)

_OTEL = ObsDiagProviderQualificationDescriptor(
    provider_id="otel",
    domain="telemetry",
    contract="ObservabilityExporter (derived export)",
    integration_status="STABLE",
    adapter_exists=True,
    live_proof_module="tests/unit/runtime/observability/test_harden_3c_export_failure_semantics.py",
    failure_recovery_proof_module="tests/unit/runtime/observability/test_export_policy.py",
    declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
    delivery_or_durability_note="fail-open export; canonical RuntimeEvent unchanged (HARDEN-3C)",
)

_OTEL_COLLECTOR = ObsDiagProviderQualificationDescriptor(
    provider_id="opentelemetry_collector",
    domain="telemetry",
    contract="ObservabilityExporter transport",
    integration_status="STABLE",
    adapter_exists=True,
    live_proof_module=None,
    failure_recovery_proof_module=None,
    declared_status=ObsDiagProviderSupportStatus.ADAPTER_ONLY,
)

_IN_MEMORY_TRANSPORT = ObsDiagProviderQualificationDescriptor(
    provider_id="in-process-async",
    domain="transport",
    contract="colocated worker / async spine (non-broker)",
    integration_status="platform-internal",
    adapter_exists=True,
    live_proof_module="tests/integration/runtime/test_obs_universal_spine_async_e2e.py",
    failure_recovery_proof_module=None,
    declared_status=ObsDiagProviderSupportStatus.SUPPORTED_QUALIFIED,
    delivery_or_durability_note="in-process only; not external broker qualification",
)

OBS_DIAG_X5_PROVIDER_INVENTORY: tuple[ObsDiagProviderQualificationDescriptor, ...] = (
    _SQLITE_FILE,
    _KAFKA,
    _MONGODB_DOCSTORE,
    _OTEL,
    _OTEL_COLLECTOR,
    _IN_MEMORY_TRANSPORT,
)
