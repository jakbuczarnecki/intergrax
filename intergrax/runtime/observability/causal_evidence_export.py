# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform causal evidence → observability export envelope mapping (DIAG-1)."""

from __future__ import annotations

from intergrax.runtime.observability.causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    PlatformCausalEvidence,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
)


def envelope_from_causal_evidence(
    evidence: PlatformCausalEvidence,
) -> ObservabilityExportEnvelope:
    """Map causal evidence to the shared observability export path."""
    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.DIAGNOSTIC,
        recorded_at=evidence.recorded_at,
        run_id=evidence.target.run_id,
        task_id=evidence.target.task_id,
        tenant_id=evidence.tenant_id,
        event_type=evidence.relation_kind.value,
        status=ExportStatus.SUCCEEDED,
        schema_id=PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
        source_schema_id=PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
        event_id=evidence.evidence_id,
        counts={
            "transport_task_id_len": len(evidence.source.task_id),
        },
    )
