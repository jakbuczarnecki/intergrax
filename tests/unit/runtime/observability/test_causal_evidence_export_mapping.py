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
from intergrax.runtime.observability.causal_evidence_export import envelope_from_causal_evidence
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    InMemoryObservabilityExporter,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_causal_evidence_exports_through_observability_envelope() -> None:
    evidence = PlatformCausalEvidence(
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
