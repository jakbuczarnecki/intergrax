# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform causal evidence → observability export envelope mapping (DIAG-1)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.contracts.npsc5f_compatibility import (
    UnknownCausalEvidenceExportVersionError,
)
from intergrax.runtime.observability.causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    PlatformCausalEvidence,
)
from intergrax.runtime.observability.causal_evidence_legacy import (
    LegacyPlatformCausalEvidence,
)
from intergrax.runtime.observability.export_boundary import (
    CausalEvidenceExportSource,
    ExportRecordKind,
    ExportStatus,
    LegacyCausalEvidenceExportSource,
    ObservabilityExportEnvelope,
)


class CausalEvidenceExportVersion(StrEnum):
    V1_LEGACY = "causal_evidence_export_source.v1"
    V2_CANONICAL = "causal_evidence_export_source.v2"


def causal_evidence_export_source_from_evidence(
    evidence: PlatformCausalEvidence,
) -> CausalEvidenceExportSource:
    """Build canonical v2 export source (default write/export path)."""
    return CausalEvidenceExportSource(
        evidence_id=evidence.evidence_id,
        relation_kind=evidence.relation_kind.value,
        tenant_id=evidence.tenant_id,
        transport_provider=evidence.source.provider,
        transport_task_id=evidence.source.task_id,
        target_task_id=evidence.target.task_id,
        target_run_id=evidence.target.run_id,
        target_attempt_id=evidence.target.attempt_id,
        target_execution_id=evidence.target.execution_id,
        recorded_at=evidence.recorded_at,
    )


def legacy_causal_evidence_export_source_from_evidence(
    evidence: LegacyPlatformCausalEvidence,
) -> LegacyCausalEvidenceExportSource:
    """Explicit v1 export for legacy consumers only."""
    return LegacyCausalEvidenceExportSource(
        evidence_id=evidence.evidence_id,
        relation_kind=evidence.relation_kind.value,
        tenant_id=evidence.tenant_id,
        transport_provider=evidence.source.provider,
        transport_task_id=evidence.source.task_id,
        target_task_id=evidence.target.task_id,
        target_run_id=evidence.target.run_id,
        target_attempt_id=evidence.target.attempt_id,
        recorded_at=evidence.recorded_at,
    )


def export_source_for_version(
    evidence: PlatformCausalEvidence,
    *,
    requested_version: CausalEvidenceExportVersion,
) -> CausalEvidenceExportSource | LegacyCausalEvidenceExportSource:
    if requested_version == CausalEvidenceExportVersion.V2_CANONICAL:
        return causal_evidence_export_source_from_evidence(evidence)
    if requested_version == CausalEvidenceExportVersion.V1_LEGACY:
        raise UnknownCausalEvidenceExportVersionError(
            "v1 export requires LegacyPlatformCausalEvidence; "
            "use legacy_causal_evidence_export_source_from_evidence",
        )
    raise UnknownCausalEvidenceExportVersionError(
        f"unsupported causal evidence export version: {requested_version!r}",
    )


def envelope_from_causal_evidence_source(
    source: CausalEvidenceExportSource | LegacyCausalEvidenceExportSource,
) -> ObservabilityExportEnvelope:
    """Map a typed causal-evidence export source to the shared observability export path."""
    platform_schema = PLATFORM_CAUSAL_EVIDENCE_SCHEMA
    execution_id = ""
    if isinstance(source, CausalEvidenceExportSource):
        execution_id = source.target_execution_id
    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.DIAGNOSTIC,
        recorded_at=source.recorded_at,
        run_id=source.target_run_id,
        task_id=source.target_task_id,
        attempt_id=source.target_attempt_id,
        execution_id=execution_id,
        tenant_id=source.tenant_id,
        event_type=source.relation_kind,
        status=ExportStatus.SUCCEEDED,
        schema_id=platform_schema,
        source_schema_id=platform_schema,
        event_id=source.evidence_id,
        causal_evidence_source=source,
    )


def envelope_from_causal_evidence(
    evidence: PlatformCausalEvidence,
) -> ObservabilityExportEnvelope:
    """Map v2 causal evidence to canonical observability export projection."""
    return envelope_from_causal_evidence_source(
        causal_evidence_export_source_from_evidence(evidence),
    )
