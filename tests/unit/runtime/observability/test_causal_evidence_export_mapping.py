# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.causal_evidence_export import (
    causal_evidence_export_source_from_evidence,
    envelope_from_causal_evidence,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    InMemoryObservabilityExporter,
)

pytestmark = pytest.mark.unit


def _causal_evidence() -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id="tenant-a",
        source=MessageBusTaskRef(
            provider="celery",
            task_id="celery-task-42",
            tenant_id="tenant-a",
        ),
        target=RuntimeExecutionRef(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            tenant_id="tenant-a",
        ),
    )


@pytest.mark.asyncio
async def test_causal_evidence_exports_through_observability_envelope() -> None:
    evidence = _causal_evidence()
    envelope = envelope_from_causal_evidence(evidence)
    exporter = InMemoryObservabilityExporter()

    await exporter.export(envelope)

    assert len(exporter.envelopes) == 1
    stored = exporter.envelopes[0]
    assert stored.record_kind == ExportRecordKind.DIAGNOSTIC
    assert stored.status == ExportStatus.SUCCEEDED
    assert stored.tenant_id == "tenant-a"
    assert stored.run_id == evidence.target.run_id
    assert stored.task_id == evidence.target.task_id
    assert stored.event_type == CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION.value
    assert stored.source_schema_id == "platform_causal_evidence.v1"
    assert stored.event_id == evidence.evidence_id


def test_causal_evidence_export_preserves_full_causal_semantics() -> None:
    evidence = _causal_evidence()
    source = causal_evidence_export_source_from_evidence(evidence)
    envelope = envelope_from_causal_evidence(evidence)

    assert envelope.causal_evidence_source is not None
    assert envelope.causal_evidence_source == source
    assert source.transport_provider == evidence.source.provider
    assert source.transport_task_id == evidence.source.task_id
    assert source.relation_kind == evidence.relation_kind.value
    assert source.target_task_id == evidence.target.task_id
    assert source.target_run_id == evidence.target.run_id
    assert source.target_attempt_id == evidence.target.attempt_id
    assert source.tenant_id == evidence.tenant_id
    assert source.evidence_id == evidence.evidence_id
