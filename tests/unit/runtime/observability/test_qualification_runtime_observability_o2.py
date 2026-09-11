# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-O2 completion alignment observability tests."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id
from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionAlignmentDiagV1,
    CompletionMode,
    decode_completion_alignment_diag_v1,
)
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats
from intergrax.runtime.observability.qualification_runtime_trace import (
    O2_SUPPORTED_TRACE_SCHEMA_IDS,
    TaskTraceRuntimeDiagnosticPort,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentAssessment,
    CompletionAlignmentMismatchReason,
    CompletionAlignmentState,
    CompletionAlignmentStatus,
    assess_completion_alignment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    decide_completion_alignment_correction,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_telemetry import (
    alignment_direction_for_assessment,
    alignment_status_for_assessment,
    build_completion_alignment_diag_v1,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    RECONCILIATION_PHASE_TRACE_SCHEMA,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RUN_ID = mint_run_id()
_NODE = "node_incident_investigator"


def _finalize_trace_run(store: InMemoryRunTraceStore, *, run_id: str, tenant_id: str) -> None:
    store.finalize_run(
        run_id,
        RunMetadata(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id="user-1",
            session_id="session-1",
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=1, llm_usage={}),
        ),
    )


def _event_dicts(store: InMemoryRunTraceStore, *, run_id: str, tenant_id: str) -> tuple[dict[str, object], ...]:
    _finalize_trace_run(store, run_id=run_id, tenant_id=tenant_id)
    persisted = store.read_run(run_id, tenant_id)
    return tuple(asdict(item) for item in persisted.events)


def test_o2_schema_capability_marker() -> None:
    assert CompletionAlignmentDiagV1.schema_id() in O2_SUPPORTED_TRACE_SCHEMA_IDS


def test_match_emission() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
            has_supported_diagnosis=True,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    payload = build_completion_alignment_diag_v1(
        run_id=_RUN_ID,
        node_id=_NODE,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        supported_state_present=True,
        assessment=assessment,
        correctable=decision.alignment_correctable,
    )
    assert payload.alignment_status is AlignmentStatus.MATCH
    assert payload.alignment_direction is AlignmentDirection.NONE
    assert payload.correctable is False


def test_forward_detection() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode=COMPLETION_UNRESOLVED,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    payload = build_completion_alignment_diag_v1(
        run_id=_RUN_ID,
        node_id=_NODE,
        completion_mode=COMPLETION_UNRESOLVED,
        supported_state_present=True,
        assessment=assessment,
        correctable=decision.alignment_correctable,
    )
    assert payload.alignment_direction is AlignmentDirection.FORWARD
    assert payload.correctable is True
    assert payload.alignment_status is AlignmentStatus.MISMATCH


def test_reverse_detection() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
            has_supported_diagnosis=False,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    payload = build_completion_alignment_diag_v1(
        run_id=_RUN_ID,
        node_id=_NODE,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        supported_state_present=False,
        assessment=assessment,
        correctable=decision.alignment_correctable,
    )
    assert payload.alignment_direction is AlignmentDirection.REVERSE
    assert payload.correctable is True
    assert payload.mismatch_reason == (
        CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE.value
    )


def test_unknown_fail_closed() -> None:
    assessment = CompletionAlignmentAssessment(
        status=CompletionAlignmentStatus.MISALIGNED,
        mismatch_reason=CompletionAlignmentMismatchReason.UNKNOWN_COMPLETION_MODE,
    )
    payload = build_completion_alignment_diag_v1(
        run_id=_RUN_ID,
        node_id=_NODE,
        completion_mode="not_a_mode",
        supported_state_present=False,
        assessment=assessment,
        correctable=False,
    )
    assert alignment_direction_for_assessment(assessment) is AlignmentDirection.UNKNOWN
    assert alignment_status_for_assessment(assessment) is AlignmentStatus.MISMATCH
    assert payload.alignment_direction is AlignmentDirection.UNKNOWN
    assert payload.correctable is False


def test_readback_from_trace_event() -> None:
    store = InMemoryRunTraceStore()
    run_id = mint_run_id()
    trace_emitter = TaskTraceEmitter(run_id=run_id, attempt_id=mint_attempt_id())
    task = Task(tenant_id="tenant-1", user_id="u1", message="m")
    port = TaskTraceRuntimeDiagnosticPort(trace_emitter=trace_emitter, task=task)
    payload = CompletionAlignmentDiagV1(
        run_id=run_id,
        node_id=_NODE,
        completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS,
        alignment_status=AlignmentStatus.MISMATCH,
        alignment_direction=AlignmentDirection.REVERSE,
        mismatch_reason="supported_diagnosis_without_supported_state",
        correctable=True,
        supported_state_present=False,
        supported_hypothesis_id=None,
        supported_resolution=None,
    )
    port.emit_completion_alignment(payload=payload)
    for event in trace_emitter.events:
        store.append_event(event)
    events = _event_dicts(store, run_id=str(run_id), tenant_id="tenant-1")
    readback = read_typed_alignment_events(events)
    assert readback.status.value == "pass"
    assert len(readback.events) == 1
    assert readback.events[0].alignment_direction is AlignmentDirection.REVERSE
    assert events[0]["payload_schema_id"] == COMPLETION_ALIGNMENT_TRACE_SCHEMA
    roundtrip = decode_completion_alignment_diag_v1(payload.to_dict())
    assert roundtrip == payload


def test_readback_ignores_non_alignment_trace_events() -> None:
    events = (
        {
            "payload_schema_id": RECONCILIATION_PHASE_TRACE_SCHEMA,
            "payload": {
                "run_id": str(_RUN_ID),
                "validation_invalid": False,
                "entered_reconciliation": True,
                "phase": "entered",
            },
        },
        {
            "payload_schema_id": CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
            "payload": {
                "run_id": str(_RUN_ID),
                "node_id": _NODE,
                "attempt_index": 0,
                "max_iterations": 2,
            },
        },
        {
            "payload_schema_id": "incident.completion_alignment.v1",
            "payload": {
                "alignment_mismatch_detected": True,
                "alignment_direction": "model_overcommit",
                "alignment_correctable": True,
                "alignment_correction_attempted": False,
                "alignment_correction_succeeded": False,
                "alignment_correction_exhausted": False,
                "revision_authoritative_context_present": False,
            },
        },
    )
    readback = read_typed_alignment_events(events)
    assert readback.status.value == "pass"
    assert readback.events == ()
