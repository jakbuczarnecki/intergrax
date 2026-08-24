# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_TRANSPORT_TASK_ID = "celery-task-9f3a2b1c"
_PROVIDER = "celery"


def _runtime_execution_ref(*, tenant_id: str = _TENANT_A) -> RuntimeExecutionRef:
    return RuntimeExecutionRef(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        tenant_id=tenant_id,
    )


def _message_bus_task_ref(*, tenant_id: str = _TENANT_A) -> MessageBusTaskRef:
    return MessageBusTaskRef(
        provider=_PROVIDER,
        task_id=_TRANSPORT_TASK_ID,
        tenant_id=tenant_id,
    )


def _causal_evidence(*, tenant_id: str = _TENANT_A) -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=tenant_id,
        source=_message_bus_task_ref(tenant_id=tenant_id),
        target=_runtime_execution_ref(tenant_id=tenant_id),
    )


def test_transport_and_runtime_identity_domains_remain_distinct() -> None:
    source = _message_bus_task_ref()
    target = _runtime_execution_ref()

    assert source.task_id == _TRANSPORT_TASK_ID
    assert source.task_id != str(target.task_id)


def test_transport_task_id_may_match_runtime_task_id_text_without_domain_collapse() -> None:
    runtime_task_id = mint_task_id()
    source = MessageBusTaskRef(
        provider=_PROVIDER,
        task_id=str(runtime_task_id),
        tenant_id=_TENANT_A,
    )
    target = RuntimeExecutionRef(
        task_id=runtime_task_id,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        tenant_id=_TENANT_A,
    )

    assert source.task_id == str(target.task_id)
    evidence = PlatformCausalEvidence(
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=_TENANT_A,
        source=source,
        target=target,
    )
    assert evidence.source.task_id == str(evidence.target.task_id)
    assert evidence.source is source
    assert evidence.target is target


def test_causal_evidence_points_to_canonical_execution_identity() -> None:
    target = _runtime_execution_ref()
    evidence = PlatformCausalEvidence(
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=_TENANT_A,
        source=_message_bus_task_ref(),
        target=target,
    )

    assert evidence.target.task_id == target.task_id
    assert evidence.target.run_id == target.run_id
    assert evidence.target.attempt_id == target.attempt_id
    assert evidence.source.provider == _PROVIDER
    assert evidence.source.task_id == _TRANSPORT_TASK_ID


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("task_id", "not-a-task-id"),
        ("run_id", "not-a-run-id"),
        ("attempt_id", "not-an-attempt-id"),
    ],
)
def test_invalid_canonical_execution_ids_fail_closed(
    field_name: str,
    invalid_value: str,
) -> None:
    payload = _runtime_execution_ref().model_dump()
    payload[field_name] = invalid_value
    with pytest.raises(ValidationError):
        RuntimeExecutionRef.model_validate(payload)


def test_missing_relation_side_fails_closed() -> None:
    with pytest.raises(ValidationError):
        PlatformCausalEvidence.model_validate(
            {
                "schema_version": PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
                "relation_kind": CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION.value,
                "tenant_id": _TENANT_A,
                "source": _message_bus_task_ref().model_dump(),
            }
        )
    with pytest.raises(ValidationError):
        PlatformCausalEvidence.model_validate(
            {
                "schema_version": PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
                "relation_kind": CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION.value,
                "tenant_id": _TENANT_A,
                "target": _runtime_execution_ref().model_dump(),
            }
        )


def test_tenant_mismatch_fails_closed() -> None:
    with pytest.raises(ValidationError):
        PlatformCausalEvidence(
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=_TENANT_A,
            source=_message_bus_task_ref(tenant_id=_TENANT_B),
            target=_runtime_execution_ref(tenant_id=_TENANT_A),
        )
    with pytest.raises(ValidationError):
        PlatformCausalEvidence(
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=_TENANT_A,
            source=_message_bus_task_ref(tenant_id=_TENANT_A),
            target=_runtime_execution_ref(tenant_id=_TENANT_B),
        )


def test_serialization_round_trip_preserves_semantic_fact() -> None:
    original = _causal_evidence()
    restored = PlatformCausalEvidence.model_validate_json(original.model_dump_json())

    assert restored.schema_version == PLATFORM_CAUSAL_EVIDENCE_SCHEMA
    assert restored.evidence_id == original.evidence_id
    assert restored.relation_kind == original.relation_kind
    assert restored.tenant_id == original.tenant_id
    assert restored.source == original.source
    assert restored.target == original.target
    assert restored.recorded_at == original.recorded_at


def test_runtime_event_still_requires_canonical_execution_identity() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvent(
            task_id="celery-task",
            run_id="run-1",
            attempt_id="attempt-1",
            event_type=RuntimeEventType.TASK_CREATED,
            phase=ExecutionPhase.INTAKE,
        )

    valid = RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
    )
    assert valid.task_id.startswith("task_")


def test_causal_evidence_forbids_extra_fields() -> None:
    payload = json.loads(_causal_evidence().model_dump_json())
    payload["correlation_id"] = "must-not-appear"
    with pytest.raises(ValidationError):
        PlatformCausalEvidence.model_validate(payload)
