# © Artur Czarnecki. All rights reserved.

"""R2 correction matrix — execution failure evidence contracts (DIAG R2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_failure_evidence import (
    ExecutionFailureEvidenceRecordResult,
    ExecutionFailureEvidenceRecordStatus,
    ExecutionFailureEvidenceRequest,
    ExecutionFailureKind,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFindingKind
from intergrax.runtime.diagnostics.execution_failure_analysis import (
    ExecutionFailureAnalyzer,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyScope
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    build_deterministic_problem_signature,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicExecutionFailureFindingSignature,
    DeterministicProblemSignature,
    ProblemGroupingSubject,
    ProblemGroupingSubjectFinding,
    ProblemGroupingSubjectFindingSource,
    normalize_assessment,
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_record_codec import (
    _decode_signature,
    _encode_signature,
    decode_problem_record,
    encode_problem_record,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import (
    MandatoryEvidencePersistenceError,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
    RuntimeEventExecutionFailureEvidenceRecorder,
)
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem


_EXECUTION_DIR = Path("intergrax/runtime/execution")


def test_a1_runtime_execution_ast_gate_no_private_nexus_publish() -> None:
    forbidden = (
        "_publish_terminal_runtime_event_with_active_identity",
        "NexusLoop._publish",
    )
    violations: list[str] = []
    for path in _EXECUTION_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                violations.append(f"{path.as_posix()}:{needle}")
    assert violations == []


def test_a1_orchestration_uses_public_terminal_publisher() -> None:
    source = Path("intergrax/runtime/execution/orchestration.py").read_text(
        encoding="utf-8"
    )
    assert "build_nexus_root_orchestration_terminal_publisher" in source
    assert "_publish_terminal_runtime_event_with_active_identity" not in source


def test_a2_recorder_no_persistence_returns_unavailable() -> None:
    bus = RuntimeEventBus(persistence=None)
    recorder = RuntimeEventExecutionFailureEvidenceRecorder(bus)
    result = recorder.record_failure(
        ExecutionFailureEvidenceRequest(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
            safe_summary="Execution delegate failed",
        ),
    )
    assert result.status is ExecutionFailureEvidenceRecordStatus.UNAVAILABLE
    assert result.event_id is None


def test_a3_a4_record_result_invariants() -> None:
    event_id = mint_execution_id()
    persisted = ExecutionFailureEvidenceRecordResult.persisted(event_id)
    assert persisted.status is ExecutionFailureEvidenceRecordStatus.PERSISTED
    assert persisted.event_id == event_id
    unavailable = ExecutionFailureEvidenceRecordResult.unavailable()
    assert unavailable.event_id is None
    with pytest.raises(ValueError):
        ExecutionFailureEvidenceRecordResult(
            status=ExecutionFailureEvidenceRecordStatus.PERSISTED,
            event_id=None,
        )
    with pytest.raises(ValueError):
        ExecutionFailureEvidenceRecordResult(
            status=ExecutionFailureEvidenceRecordStatus.UNAVAILABLE,
            event_id=event_id,
        )


def test_a5_recorder_enabled_missing_scope_fail_closed() -> None:
    import asyncio

    async def _run() -> None:
        runtime = ExecutionRuntime(
            delegate=_NoOpDelegate(),
            failure_evidence_recorder=RuntimeEventExecutionFailureEvidenceRecorder(
                RuntimeEventBus(persistence=InMemoryRuntimeEventStore()),
            ),
        )
        context = RootExecutionContext(
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            authority=None,
            tenant_id=None,
            task_id=None,
        )
        with pytest.raises(ValueError, match="failure evidence recorder requires"):
            await runtime.execute(object(), context)

    asyncio.run(_run())


class _NoOpDelegate:
    async def execute(self, request: object) -> object:
        return request


def test_a6_execution_failure_analyzer_return_type() -> None:
    hints = ExecutionFailureAnalyzer.analyze.__annotations__
    assert "return" in hints
    assert "DiagnosticFinding" in str(hints["return"])


def test_a8_execution_failure_distinct_grouping_signature() -> None:
    from intergrax.runtime.diagnostics.diagnostic_assessment import (
        DiagnosticAssessment,
        DiagnosticCertainty,
        DiagnosticFinding,
        DiagnosticPrecision,
        FailureBoundary,
    )

    execution_id = mint_execution_id()
    event_id = mint_execution_id()
    finding = DiagnosticFinding(
        kind=DiagnosticFindingKind.EXECUTION_FAILED,
        scope=LifecycleAnomalyScope.ATTEMPT,
        attempt_id=mint_attempt_id(),
        certainty=DiagnosticCertainty.PROVEN,
        claim="Canonical runtime evidence proves that this execution failed.",
        source_anomaly_kind=None,
        supporting_event_ids=(event_id,),
        supporting_evidence_ids=(),
        supporting_positions=(),
        execution_id=execution_id,
        precision=DiagnosticPrecision.EXECUTION_LEVEL,
        failure_boundary=FailureBoundary(
            execution_id=execution_id,
            supporting_event_id=event_id,
            certainty=DiagnosticCertainty.PROVEN,
            precision=DiagnosticPrecision.EXECUTION_LEVEL,
        ),
        execution_failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
    )
    assessment = DiagnosticAssessment(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=(finding,),
        limitations=(),
    )
    subject = normalize_assessment(assessment)
    assert (
        subject.findings[0].source
        is ProblemGroupingSubjectFindingSource.EXECUTION_FAILURE
    )
    signature = build_deterministic_problem_signature(subject)
    assert len(signature.findings) == 1
    assert type(signature.findings[0]) is DeterministicExecutionFailureFindingSignature


def test_a9_historical_v2_lifecycle_problem_record_decodes() -> None:
    problem = sample_problem(tenant_id="tenant-v2-hist")
    record = {
        "schema_version": "intergrax.diagnostic_problem.persistence.v2",
        "payload": encode_problem_record(problem)["payload"],
    }
    decoded = decode_problem_record(record)
    assert decoded.problem_id == problem.problem_id


def test_a10_v3_execution_failure_signature_round_trip() -> None:
    signature = DeterministicProblemSignature(
        findings=(
            DeterministicExecutionFailureFindingSignature(
                kind=DiagnosticFindingKind.EXECUTION_FAILED,
                execution_id=mint_execution_id(),
                execution_failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
            ),
        ),
        limitations=(),
    )
    round_trip = _decode_signature(_encode_signature(signature))
    assert round_trip == signature


def test_a11_malformed_lifecycle_signature_rejected() -> None:
    payload = {
        "findings": [
            {
                "source": "lifecycle",
                "kind": DiagnosticFindingKind.EVENT_AFTER_TERMINAL.value,
                "scope": LifecycleAnomalyScope.ATTEMPT.value,
            },
        ],
        "limitations": [],
    }
    with pytest.raises(ProblemPersistenceIntegrityError):
        _decode_signature(payload)


def test_a7_lifecycle_signature_still_requires_anomaly_kind_via_subject() -> None:
    subject = ProblemGroupingSubject(
        subject_ref=problem_grouping_subject_ref_for_execution(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        ),
        findings=(
            ProblemGroupingSubjectFinding(
                source=ProblemGroupingSubjectFindingSource.LIFECYCLE,
                kind=DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
                scope=LifecycleAnomalyScope.ATTEMPT,
                source_anomaly_kind=None,
            ),
        ),
        limitations=(),
    )
    with pytest.raises(
        ValueError, match="lifecycle grouping finding missing required fields"
    ):
        build_deterministic_problem_signature(subject)


def test_writer_emits_v3_problem_schema() -> None:
    problem = sample_problem(tenant_id="tenant-v3-write")
    encoded = encode_problem_record(problem)
    assert encoded["schema_version"] == "intergrax.diagnostic_problem.persistence.v3"


def test_persistence_outage_recorder_returns_unavailable() -> None:
    class _FailingPersistence(InMemoryRuntimeEventStore):
        def append(self, event: object, *, tenant_id: str | None = None) -> None:
            raise MandatoryEvidencePersistenceError("outage")

    bus = RuntimeEventBus(persistence=_FailingPersistence())
    recorder = RuntimeEventExecutionFailureEvidenceRecorder(bus)
    result = recorder.record_failure(
        ExecutionFailureEvidenceRequest(
            tenant_id="tenant-a",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
            safe_summary="Execution delegate failed",
        ),
    )
    assert result.status is ExecutionFailureEvidenceRecordStatus.UNAVAILABLE
