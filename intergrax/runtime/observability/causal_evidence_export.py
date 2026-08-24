# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform causal evidence → observability export envelope mapping (DIAG-1)."""

from __future__ import annotations

from intergrax.runtime.observability.causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    PlatformCausalEvidence,
)
from intergrax.runtime.observability.export_boundary import (
    CausalEvidenceExportSource,
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
)


def causal_evidence_export_source_from_evidence(
    evidence: PlatformCausalEvidence,
) -> CausalEvidenceExportSource:
    """Build a typed export source that preserves full causal semantics."""
    return CausalEvidenceExportSource(
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


def envelope_from_causal_evidence_source(
    source: CausalEvidenceExportSource,
) -> ObservabilityExportEnvelope:
    """Map a typed causal-evidence export source to the shared observability export path."""
    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.DIAGNOSTIC,
        recorded_at=source.recorded_at,
        run_id=source.target_run_id,
        task_id=source.target_task_id,
        tenant_id=source.tenant_id,
        event_type=source.relation_kind,
        status=ExportStatus.SUCCEEDED,
        schema_id=PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
        source_schema_id=PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
        event_id=source.evidence_id,
        causal_evidence_source=source,
    )


def envelope_from_causal_evidence(
    evidence: PlatformCausalEvidence,
) -> ObservabilityExportEnvelope:
    """Map causal evidence to an optional observability export projection."""
    return envelope_from_causal_evidence_source(
        causal_evidence_export_source_from_evidence(evidence),
    )
