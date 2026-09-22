# © Artur Czarnecki. All rights reserved.

"""Manifest-driven observability_backend vendor matrix with EC3 evidence categories."""

from __future__ import annotations

from testing_support.obs_diag_observability_vendor_qualification.descriptor import (
    ObsDiagProofKind,
    ObservabilityPlatformIsolationEvidence,
    ObservabilityQualifiedPathEvidence,
    ObservabilityQualifiedPathRow,
    ObservabilityVendorQualificationEvidence,
    ObservabilityVendorQualificationRow,
    ObservabilityVendorQualificationStatus,
    PlatformIsolationProofReference,
    QualifiedPathProofReference,
    VendorQualificationProofReference,
)
from testing_support.obs_diag_provider_qualification.discovery import (
    discover_obs_diag_provider_surfaces,
)

_CONTRACT_MIGRATION = VendorQualificationProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/integrations/providers/observability_backend/test_observability_provider_contract_migration.py",
)
_VENDOR_CONTRACT = VendorQualificationProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/runtime/integrations/test_observability_vendor_integration_contract.py",
)

_PLATFORM_EXPORT_NORMAL = QualifiedPathProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/runtime/observability/test_harden_3c_export_failure_semantics.py",
)
_PLATFORM_EXPORT_FAILURE = QualifiedPathProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/runtime/observability/test_harden_3d_exporter_health.py",
)
_PLATFORM_EXPORT_POLICY = QualifiedPathProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/runtime/observability/test_export_policy.py",
)
_PLATFORM_CANONICAL_ISOLATION = PlatformIsolationProofReference(
    kind=ObsDiagProofKind.INTEGRATION,
    module="tests/integration/runtime/test_diag_final_external_otel_e2e.py",
)

PLATFORM_CANONICAL_TRUTH_ISOLATION_EVIDENCE = ObservabilityPlatformIsolationEvidence(
    canonical_truth_isolation=_PLATFORM_CANONICAL_ISOLATION,
)

_EXTERNAL_OTLP_NORMAL = QualifiedPathProofReference(
    kind=ObsDiagProofKind.EXTERNAL_LIVE,
    module="tests/integration/runtime/test_diag_final_external_otel_e2e.py",
)
_EXTERNAL_OTLP_PRIVACY = QualifiedPathProofReference(
    kind=ObsDiagProofKind.EXTERNAL_LIVE,
    module="tests/integration/runtime/diag_final_otel_support.py",
)

_ELASTIC_RETRY = VendorQualificationProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/integrations/providers/observability_backend/test_elasticsearch_observability_retry.py",
)
_ELASTIC_FAILURE = VendorQualificationProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/integrations/providers/observability_backend/test_elasticsearch_observability_delivery_errors.py",
)
_ELASTIC_FAILED_SINK = VendorQualificationProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/integrations/providers/observability_backend/test_elasticsearch_observability_failed_delivery_sink.py",
)

_SENTRY_ISOLATION = VendorQualificationProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/integrations/providers/observability_backend/test_sentry.py",
)

_ELASTIC_PATH_RETRY = QualifiedPathProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/integrations/providers/observability_backend/test_elasticsearch_observability_retry.py",
)
_ELASTIC_PATH_FAILURE = QualifiedPathProofReference(
    kind=ObsDiagProofKind.UNIT_CONTRACT,
    module="tests/unit/integrations/providers/observability_backend/test_elasticsearch_observability_delivery_errors.py",
)

_ELASTICSEARCH_TRANSPORT_PATH_EVIDENCE = ObservabilityQualifiedPathEvidence(
    normal_delivery=_ELASTIC_PATH_RETRY,
    failure_isolation=_ELASTIC_PATH_FAILURE,
    recovery=_ELASTIC_PATH_RETRY,
    canonical_truth_isolation=None,
    privacy=None,
)

_EVIDENCE_OVERRIDES: dict[str, ObservabilityVendorQualificationEvidence] = {
    "elasticsearch": ObservabilityVendorQualificationEvidence(
        normal_delivery=_ELASTIC_RETRY,
        failure_isolation=_ELASTIC_FAILURE,
        recovery=_ELASTIC_RETRY,
        canonical_truth_isolation=None,
        privacy=None,
    ),
    "opensearch": ObservabilityVendorQualificationEvidence(
        normal_delivery=_CONTRACT_MIGRATION,
        failure_isolation=_CONTRACT_MIGRATION,
        recovery=None,
        canonical_truth_isolation=None,
        privacy=None,
    ),
    "sentry": ObservabilityVendorQualificationEvidence(
        normal_delivery=_SENTRY_ISOLATION,
        failure_isolation=_SENTRY_ISOLATION,
        recovery=None,
        canonical_truth_isolation=None,
        privacy=None,
    ),
}


def _default_adapter_evidence() -> ObservabilityVendorQualificationEvidence:
    return ObservabilityVendorQualificationEvidence(
        normal_delivery=None,
        failure_isolation=None,
        recovery=None,
        canonical_truth_isolation=None,
        privacy=None,
    )


def _contract_conformant_evidence() -> ObservabilityVendorQualificationEvidence:
    return ObservabilityVendorQualificationEvidence(
        normal_delivery=_CONTRACT_MIGRATION,
        failure_isolation=_VENDOR_CONTRACT,
        recovery=None,
        canonical_truth_isolation=None,
        privacy=None,
    )


def _status_for_evidence(
    evidence: ObservabilityVendorQualificationEvidence,
) -> ObservabilityVendorQualificationStatus:
    if evidence.missing_live_categories() == ():
        return ObservabilityVendorQualificationStatus.LIVE_QUALIFIED
    if evidence.normal_delivery is not None:
        return ObservabilityVendorQualificationStatus.CONTRACT_CONFORMANT
    return ObservabilityVendorQualificationStatus.ADAPTER_ONLY


def build_observability_vendor_inventory() -> tuple[ObservabilityVendorQualificationRow, ...]:
    discovered = discover_obs_diag_provider_surfaces()
    telemetry = [row for row in discovered if row.domain.value == "telemetry"]
    rows: list[ObservabilityVendorQualificationRow] = []
    for item in telemetry:
        evidence = _EVIDENCE_OVERRIDES.get(item.provider_id, _default_adapter_evidence())
        if item.provider_id not in _EVIDENCE_OVERRIDES:
            evidence = _contract_conformant_evidence()
        rows.append(
            ObservabilityVendorQualificationRow(
                provider_id=item.provider_id,
                manifest_path=item.manifest_path,
                integration_status=item.integration_status.value,
                contract_implemented=True,
                evidence=evidence,
                qualification=_status_for_evidence(evidence),
            ),
        )
    rows.sort(key=lambda row: row.provider_id)
    return tuple(rows)


OBSERVABILITY_VENDOR_INVENTORY: tuple[ObservabilityVendorQualificationRow, ...] = (
    build_observability_vendor_inventory()
)

OBSERVABILITY_QUALIFIED_PATHS: tuple[ObservabilityQualifiedPathRow, ...] = (
    ObservabilityQualifiedPathRow(
        path_id="platform_export_semantics",
        evidence=ObservabilityQualifiedPathEvidence(
            normal_delivery=_PLATFORM_EXPORT_NORMAL,
            failure_isolation=_PLATFORM_EXPORT_FAILURE,
            recovery=_PLATFORM_EXPORT_POLICY,
            canonical_truth_isolation=None,
            privacy=None,
        ),
        qualification=ObservabilityVendorQualificationStatus.LIVE_QUALIFIED,
        privacy_required=False,
        platform_isolation=PLATFORM_CANONICAL_TRUTH_ISOLATION_EVIDENCE,
    ),
    ObservabilityQualifiedPathRow(
        path_id="external_otlp_collector_slice",
        evidence=ObservabilityQualifiedPathEvidence(
            normal_delivery=_EXTERNAL_OTLP_NORMAL,
            failure_isolation=_EXTERNAL_OTLP_NORMAL,
            recovery=_EXTERNAL_OTLP_NORMAL,
            canonical_truth_isolation=_EXTERNAL_OTLP_NORMAL,
            privacy=_EXTERNAL_OTLP_PRIVACY,
        ),
        qualification=ObservabilityVendorQualificationStatus.LIVE_QUALIFIED,
    ),
    ObservabilityQualifiedPathRow(
        path_id="elasticsearch_transport_contracts",
        evidence=_ELASTICSEARCH_TRANSPORT_PATH_EVIDENCE,
        qualification=ObservabilityVendorQualificationStatus.CONTRACT_CONFORMANT,
    ),
)
