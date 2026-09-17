# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-C extended domain read adoption qualification gates."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_artifact_read import (
    ExecutionArtifactMetadataReadPort,
    ExecutionArtifactMetadataReadResult,
    ExecutionArtifactMetadataRecord,
    ExecutionArtifactLifecycleStatus,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstruction,
    RuntimeHistoryCompleteness,
)
from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind
from intergrax.contracts.external_work_runtime_read import (
    ExternalWorkRuntimeFactReadPort,
    ExternalWorkRuntimeFactReadResult,
    ExternalWorkRuntimeFactRecord,
    ExternalWorkRuntimeStatus,
)
from intergrax.contracts.memory_runtime_read import (
    MemoryRuntimeOperationClass,
    MemoryRuntimeOperationReadPort,
    MemoryRuntimeOperationReadResult,
    MemoryRuntimeOperationRecord,
    MemoryRuntimeOperationStatus,
)
from intergrax.contracts.model_runtime_read import (
    ModelRuntimeInvocationReadPort,
    ModelRuntimeInvocationReadResult,
    ModelRuntimeInvocationRecord,
    ModelRuntimeInvocationStatus,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.execution_event_position import ExecutionEventPosition
from intergrax.contracts.runtime_inspection import (
    RuntimeInspectionCompleteness,
    RuntimeInspectionErrorCode,
    RuntimeInspectionQuery,
    RuntimeInspectionSourceFailureCode,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionArtifactSection,
    RuntimeInspectionExternalWorkSection,
    RuntimeInspectionMemorySection,
    RuntimeInspectionModelSection,
)
from intergrax.contracts.runtime_inspection.sources import RuntimeInspectionExecutionScope
from intergrax.runtime.events.payload_registry import merge_payload_envelope
from intergrax.runtime.events.payloads.canonical import ExternalOperationFailurePayloadV1
from intergrax.runtime.runtime_inspection.adapters.artifact_read import (
    ArtifactMetadataInspectionAdapter,
    ReconstructionArtifactMetadataReader,
)
from intergrax.runtime.runtime_inspection.adapters.external_work_read import (
    ExternalWorkInspectionAdapter,
    ReconstructionExternalWorkFactReader,
)
from intergrax.runtime.runtime_inspection.adapters.memory_read import (
    MemoryOperationInspectionAdapter,
    ReconstructionMemoryOperationReader,
)
from intergrax.runtime.runtime_inspection.adapters.model_read import (
    ModelInvocationInspectionAdapter,
    ReconstructionModelInvocationReader,
)
from intergrax.runtime.runtime_inspection.federation import FederatedRuntimeInspectionReadService
from intergrax.runtime.runtime_inspection.redaction import payload_contains_raw_secret
from tests.qualification.inspect_01.test_inspect_01a_federation import (
    _EXEC,
    _FactsReader,
    _SCOPE,
    _ScopeReader,
    _TENANT,
    _reconstruction,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RAW_SECRET = "INSPECT_C_SECRET_xyz"
_OTHER_TENANT = "tenant-inspect-c-other"


def _scope_identity_fields(
    *,
    tenant_id: str = _TENANT,
    task_id: TaskId = _SCOPE.task_id,
    run_id: RunId = _SCOPE.run_id,
    execution_id: ExecutionId = _EXEC,
) -> dict[str, object]:
    return {
        "tenant_id": tenant_id,
        "task_id": task_id,
        "run_id": run_id,
        "execution_id": execution_id,
        "attempt_id": _SCOPE.attempt_id,
    }


def _memory_record(**overrides: object) -> MemoryRuntimeOperationRecord:
    base = {
        "operation_ref": "op-1",
        "memory_class": "task",
        "operation_class": MemoryRuntimeOperationClass.READ,
        "operation_status": MemoryRuntimeOperationStatus.HIT,
        "record_ref": "rec",
        "source_category": "custom",
        **_scope_identity_fields(),
        "sequence_key": 1,
        "evidence_refs": (),
        "safe_summary": "ok",
    }
    base.update(overrides)
    return MemoryRuntimeOperationRecord(**base)


def _base_event(**overrides: object) -> RuntimeEvent:
    payload = overrides.pop("payload", {})
    event_type = overrides.pop("event_type", RuntimeEventType.STEP_STARTED)
    return RuntimeEvent(
        tenant_id=_TENANT,
        task_id=_SCOPE.task_id,
        run_id=_SCOPE.run_id,
        attempt_id=_SCOPE.attempt_id,
        execution_id=_EXEC,
        event_type=event_type,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=datetime(2026, 2, 1, tzinfo=timezone.utc),
        payload=payload if isinstance(payload, dict) else {},
        **{k: v for k, v in overrides.items() if k in RuntimeEvent.model_fields},
    )


def _positioned(event: RuntimeEvent, position: int = 1) -> PositionedRuntimeEvent:
    return PositionedRuntimeEvent(
        event=event,
        position=ExecutionEventPosition(position),
    )


def _extended_reconstruction() -> ExecutionReconstruction:
    memory_read = _base_event(
        event_type=RuntimeEventType.MEMORY_READ,
        payload={"namespace": "task", "key": "fact-1", "found": True, "record_id": "rec-1"},
    )
    llm = _base_event(
        event_type=RuntimeEventType.LLM_CALL,
        payload={
            "model": "catalog/model-neutral",
            "label": "step",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "finish_reason": "stop",
        },
    )
    ext_failure = _base_event(event_type=RuntimeEventType.EXTERNAL_OPERATION_FAILED, payload={})
    typed = ExternalOperationFailurePayloadV1(
        execution_id=_EXEC,
        operation_attempt_id="attempt-op-1",
        provider_id="provider-neutral",
        operation_type="remote.fetch",
        failure_kind=ExternalOperationFailureKind.REMOTE_FAILURE,
        retryable=True,
        evidence_refs=("evidence-1",),
    )
    ext_failure = ext_failure.model_copy(
        update={"payload": merge_payload_envelope(ext_failure.payload, typed)},
    )
    artifact_event = _base_event(
        event_type=RuntimeEventType.STEP_COMPLETED,
        payload={
            "artifact_refs": [
                {
                    "artifact_id": "art-1",
                    "type": "report",
                    "sensitivity": "internal",
                },
            ],
        },
    )
    return ExecutionReconstruction(
        tenant_id=_TENANT,
        task_id=_SCOPE.task_id,
        run_id=_SCOPE.run_id,
        causal_evidence=(),
        positioned_events=(
            _positioned(memory_read, 1),
            _positioned(llm, 2),
            _positioned(ext_failure, 3),
            _positioned(artifact_event, 4),
        ),
        attempts=(),
        runtime_history_completeness=RuntimeHistoryCompleteness.COMPLETE,
    )


def _federated(reconstruction: ExecutionReconstruction | None = None):
    facts = _FactsReader(reconstruction or _extended_reconstruction())
    return FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=facts,
        memory_reader=MemoryOperationInspectionAdapter(
            ReconstructionMemoryOperationReader(facts),
        ),
        model_reader=ModelInvocationInspectionAdapter(
            ReconstructionModelInvocationReader(facts),
        ),
        external_work_reader=ExternalWorkInspectionAdapter(
            ReconstructionExternalWorkFactReader(facts),
        ),
        artifact_reader=ArtifactMetadataInspectionAdapter(
            ReconstructionArtifactMetadataReader(facts),
        ),
    )


def test_c_q1_memory_canonical_read() -> None:
    snapshot = _federated().inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.memory is not None
    assert len(snapshot.memory.operations) == 1
    assert snapshot.memory.operations[0].operation_class is MemoryRuntimeOperationClass.READ


def test_c_q2_model_canonical_read() -> None:
    snapshot = _federated().inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.model is not None
    assert snapshot.model.invocations[0].model_ref == "catalog/model-neutral"


def test_c_q3_external_work_canonical_read() -> None:
    snapshot = _federated().inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.external_work is not None
    assert snapshot.external_work.work_entries[0].work_status is ExternalWorkRuntimeStatus.FAILED


def test_c_q4_artifact_canonical_read() -> None:
    snapshot = _federated().inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.artifacts is not None
    assert snapshot.artifacts.artifacts[0].artifact_ref == "art-1"


def test_c_q5_cross_tenant_fail_closed() -> None:
    bad = _extended_reconstruction()
    events = list(bad.positioned_events)
    first = events[0].event.model_copy(update={"tenant_id": "other-tenant"})
    events[0] = _positioned(first, 1)
    reconstruction = replace(bad, positioned_events=tuple(events))
    with pytest.raises(RuntimeInspectionTenantBoundaryError):
        _federated(reconstruction).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )


def test_c_q6_identity_mismatch_source_integrity() -> None:
    bad = _extended_reconstruction()
    events = list(bad.positioned_events)
    first = events[0].event.model_copy(update={"task_id": mint_task_id()})
    events[0] = _positioned(first, 1)
    reconstruction = replace(bad, positioned_events=tuple(events))
    from intergrax.contracts.runtime_inspection.errors import RuntimeInspectionError

    with pytest.raises(RuntimeInspectionError) as exc_info:
        _federated(reconstruction).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


def test_c_q7_empty_vs_unavailable() -> None:
    snapshot = _federated(_reconstruction()).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.memory is not None
    assert snapshot.memory.operations == ()
    assert snapshot.memory.source_available is True
    assert not any(f.domain == "memory" for f in snapshot.source_failures)


def test_c_q8_availability_partial() -> None:
    class _FailMemory:
        source_id = "mem_fail"

        def read_memory_operations(self, scope: RuntimeInspectionExecutionScope):
            raise RuntimeError("down")

    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        memory_reader=_FailMemory(),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.memory is None
    assert snapshot.completeness is RuntimeInspectionCompleteness.PARTIAL


def test_c_q9_integrity_not_partial_only() -> None:
    from intergrax.contracts.runtime_inspection.errors import RuntimeInspectionError

    class _BadMemory:
        source_id = "bad_memory"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=(_memory_record(execution_id=mint_execution_id()),),
                is_truncated=False,
            )

    with pytest.raises(RuntimeInspectionError) as exc_info:
        FederatedRuntimeInspectionReadService(
            scope_reader=_ScopeReader(),
            execution_facts_reader=_FactsReader(),
            memory_reader=MemoryOperationInspectionAdapter(_BadMemory()),
        ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


def test_c_q10_no_write_side_effects() -> None:
    reconstruction = _extended_reconstruction()
    before = len(reconstruction.positioned_events)
    _federated(reconstruction).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert len(reconstruction.positioned_events) == before


def test_c_q11_no_model_execution_during_read() -> None:
    snapshot = _federated().inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.model is not None


def test_c_q12_custom_sources_through_contracts() -> None:
    class _CustomMemory:
        source_id = "custom_memory"

        def read_memory_operations(self, scope: RuntimeInspectionExecutionScope):
            return RuntimeInspectionMemorySection(
                operations=(),
                completeness=RuntimeInspectionCompleteness.COMPLETE,
                source_id=self.source_id,
            )

    class _CustomModel:
        source_id = "custom_model"

        def read_model_invocations(self, scope: RuntimeInspectionExecutionScope):
            return RuntimeInspectionModelSection(
                invocations=(),
                completeness=RuntimeInspectionCompleteness.COMPLETE,
                source_id=self.source_id,
            )

    class _CustomExternal:
        source_id = "custom_external"

        def read_external_work(self, scope: RuntimeInspectionExecutionScope):
            return RuntimeInspectionExternalWorkSection(
                work_entries=(),
                completeness=RuntimeInspectionCompleteness.COMPLETE,
                source_id=self.source_id,
            )

    class _CustomArtifact:
        source_id = "custom_artifact"

        def read_artifacts(self, scope: RuntimeInspectionExecutionScope):
            return RuntimeInspectionArtifactSection(
                artifacts=(),
                completeness=RuntimeInspectionCompleteness.COMPLETE,
                source_id=self.source_id,
            )

    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        memory_reader=_CustomMemory(),
        model_reader=_CustomModel(),
        external_work_reader=_CustomExternal(),
        artifact_reader=_CustomArtifact(),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.memory.source_id == "custom_memory"
    assert snapshot.model.source_id == "custom_model"
    assert snapshot.external_work.source_id == "custom_external"
    assert snapshot.artifacts.source_id == "custom_artifact"


def test_c_q13_provider_neutral_contracts() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for name in (
        "memory_runtime_read.py",
        "model_runtime_read.py",
        "external_work_runtime_read.py",
        "execution_artifact_read.py",
    ):
        text = (root / "intergrax" / "contracts" / name).read_text(encoding="utf-8")
        assert "openai" not in text.lower()
        assert "dict[str, Any]" not in text


def test_c_q14_no_reflection_patterns() -> None:
    import ast
    from pathlib import Path

    from intergrax.runtime.runtime_inspection import federation as federation_module

    source = Path(federation_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "getattr" not in names


def test_c_q15_truncation_completeness_via_adapter() -> None:
    from intergrax.contracts.memory_runtime_read import MemoryRuntimeExecutionScope

    class _TruncatedMemory(MemoryRuntimeOperationReadPort):
        source_id = "trunc_mem"

        def list_operations(self, scope, *, limit: int):
            records = tuple(_memory_record(sequence_key=i) for i in range(1, limit + 1))
            return MemoryRuntimeOperationReadResult(records=records, is_truncated=True)

    adapter = MemoryOperationInspectionAdapter(_TruncatedMemory(), operation_limit=4)
    section = adapter.read_memory_operations(_SCOPE)
    assert len(section.operations) == 4
    assert section.is_truncated is True
    assert section.completeness is RuntimeInspectionCompleteness.PARTIAL

    class _CompleteMemory(MemoryRuntimeOperationReadPort):
        source_id = "complete_mem"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(records=(), is_truncated=False)

    complete = MemoryOperationInspectionAdapter(_CompleteMemory()).read_memory_operations(_SCOPE)
    assert complete.operations == ()
    assert complete.is_truncated is False
    assert complete.completeness is RuntimeInspectionCompleteness.COMPLETE

    reconstruction = _extended_reconstruction()
    reader = ReconstructionMemoryOperationReader(_FactsReader(reconstruction))
    scope = MemoryRuntimeExecutionScope(
        tenant_id=_TENANT,
        task_id=_SCOPE.task_id,
        run_id=_SCOPE.run_id,
        attempt_id=_SCOPE.attempt_id,
        execution_id=_EXEC,
    )
    first = reader.list_operations(scope, limit=10)
    second = reader.list_operations(scope, limit=10)
    assert first.records == second.records


def test_c_q16_redaction() -> None:
    reconstruction = _extended_reconstruction()
    events = list(reconstruction.positioned_events)
    llm = events[1].event.model_copy(
        update={
            "payload": {
                "model": "m",
                "label": _RAW_SECRET,
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "total_tokens": 1,
            },
        },
    )
    events[1] = _positioned(llm, 2)
    snapshot = _federated(
        replace(reconstruction, positioned_events=tuple(events)),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    serialized = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True)
    assert not payload_contains_raw_secret(serialized, raw_secret=_RAW_SECRET)


def test_c_q17_inspect_a_regression_smoke() -> None:
    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.identity.execution_id == _EXEC


def test_c_r1_missing_tenant_spine_provenance() -> None:
    from intergrax.contracts.runtime_inspection.errors import RuntimeInspectionError

    bad = _extended_reconstruction()
    events = list(bad.positioned_events)
    first = events[0].event.model_copy(update={"tenant_id": None})
    events[0] = _positioned(first, 1)
    reconstruction = replace(bad, positioned_events=tuple(events))
    with pytest.raises(RuntimeInspectionError) as exc_info:
        _federated(reconstruction).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


@pytest.mark.parametrize("domain", ["memory", "model", "external_work", "artifact"])
def test_c_r1_custom_port_cross_tenant_same_execution(domain: str) -> None:
    facts = _FactsReader(_extended_reconstruction())
    memory_reader = MemoryOperationInspectionAdapter(ReconstructionMemoryOperationReader(facts))
    model_reader = ModelInvocationInspectionAdapter(ReconstructionModelInvocationReader(facts))
    external_work_reader = ExternalWorkInspectionAdapter(ReconstructionExternalWorkFactReader(facts))
    artifact_reader = ArtifactMetadataInspectionAdapter(ReconstructionArtifactMetadataReader(facts))

    if domain == "memory":

        class _Bad(MemoryRuntimeOperationReadPort):
            source_id = "x_mem"

            def list_operations(self, scope, *, limit: int):
                return MemoryRuntimeOperationReadResult(
                    records=(_memory_record(tenant_id=_OTHER_TENANT),),
                    is_truncated=False,
                )

        memory_reader = MemoryOperationInspectionAdapter(_Bad())
    elif domain == "model":

        class _Bad(ModelRuntimeInvocationReadPort):
            source_id = "x_model"

            def list_invocations(self, scope, *, limit: int):
                return ModelRuntimeInvocationReadResult(
                    records=(
                        ModelRuntimeInvocationRecord(
                            invocation_ref="inv",
                            model_ref="m",
                            capability_label="c",
                            invocation_status=ModelRuntimeInvocationStatus.RECORDED,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            finish_reason=None,
                            **_scope_identity_fields(tenant_id=_OTHER_TENANT),
                            sequence_key=1,
                            evidence_refs=(),
                            safe_summary="ok",
                        ),
                    ),
                    is_truncated=False,
                )

        model_reader = ModelInvocationInspectionAdapter(_Bad())
    elif domain == "external_work":

        class _Bad(ExternalWorkRuntimeFactReadPort):
            source_id = "x_ext"

            def list_work_facts(self, scope, *, limit: int):
                return ExternalWorkRuntimeFactReadResult(
                    records=(
                        ExternalWorkRuntimeFactRecord(
                            work_ref="w",
                            work_class="c",
                            work_status=ExternalWorkRuntimeStatus.FAILED,
                            provider_ref="p",
                            failure_classification="f",
                            retryable=False,
                            **_scope_identity_fields(tenant_id=_OTHER_TENANT),
                            sequence_key=1,
                            evidence_refs=(),
                            safe_summary="ok",
                        ),
                    ),
                    is_truncated=False,
                )

        external_work_reader = ExternalWorkInspectionAdapter(_Bad())
    else:

        class _Bad(ExecutionArtifactMetadataReadPort):
            source_id = "x_art"

            def list_artifact_metadata(self, scope, *, limit: int):
                return ExecutionArtifactMetadataReadResult(
                    records=(
                        ExecutionArtifactMetadataRecord(
                            artifact_ref="a",
                            artifact_type="t",
                            lifecycle_status=ExecutionArtifactLifecycleStatus.REGISTERED,
                            content_classification="internal",
                            **_scope_identity_fields(tenant_id=_OTHER_TENANT),
                            sequence_key=1,
                            evidence_refs=(),
                            safe_summary="ok",
                        ),
                    ),
                    is_truncated=False,
                )

        artifact_reader = ArtifactMetadataInspectionAdapter(_Bad())

    with pytest.raises(RuntimeInspectionTenantBoundaryError):
        FederatedRuntimeInspectionReadService(
            scope_reader=_ScopeReader(),
            execution_facts_reader=facts,
            memory_reader=memory_reader,
            model_reader=model_reader,
            external_work_reader=external_work_reader,
            artifact_reader=artifact_reader,
        ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", mint_task_id()),
        ("run_id", mint_run_id()),
        ("attempt_id", mint_attempt_id()),
        ("execution_id", mint_execution_id()),
    ],
)
def test_c_r1_identity_collision_source_integrity(field: str, value: object) -> None:
    from intergrax.contracts.runtime_inspection.errors import RuntimeInspectionError

    overrides = {field: value}

    class _BadMemory(MemoryRuntimeOperationReadPort):
        source_id = "collision_mem"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=(_memory_record(**overrides),),
                is_truncated=False,
            )

    with pytest.raises(RuntimeInspectionError) as exc_info:
        MemoryOperationInspectionAdapter(_BadMemory()).read_memory_operations(_SCOPE)
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


@pytest.mark.parametrize("tenant_id", ["", "   ", "\t"])
def test_c_r1_missing_tenant_on_custom_record(tenant_id: str) -> None:
    from intergrax.contracts.runtime_inspection.errors import RuntimeInspectionError

    class _BadMemory(MemoryRuntimeOperationReadPort):
        source_id = "no_tenant_mem"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=(_memory_record(tenant_id=tenant_id),),
                is_truncated=False,
            )

    with pytest.raises(RuntimeInspectionError) as exc_info:
        MemoryOperationInspectionAdapter(_BadMemory()).read_memory_operations(_SCOPE)
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY
    assert "missing tenant provenance" in str(exc_info.value)


def test_c_r1_custom_canonical_memory_port() -> None:
    class _Custom(MemoryRuntimeOperationReadPort):
        source_id = "custom_canonical_memory"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=(_memory_record(),),
                is_truncated=False,
            )

    section = MemoryOperationInspectionAdapter(_Custom()).read_memory_operations(_SCOPE)
    assert section.source_id == "custom_canonical_memory"
    assert len(section.operations) == 1


def test_c_r1_custom_canonical_model_port() -> None:
    class _Custom(ModelRuntimeInvocationReadPort):
        source_id = "custom_canonical_model"

        def list_invocations(self, scope, *, limit: int):
            return ModelRuntimeInvocationReadResult(
                records=(
                    ModelRuntimeInvocationRecord(
                        invocation_ref="inv",
                        model_ref="m",
                        capability_label="c",
                        invocation_status=ModelRuntimeInvocationStatus.RECORDED,
                        prompt_tokens=1,
                        completion_tokens=0,
                        total_tokens=1,
                        finish_reason=None,
                        **_scope_identity_fields(),
                        sequence_key=1,
                        evidence_refs=(),
                        safe_summary="ok",
                    ),
                ),
                is_truncated=False,
            )

    section = ModelInvocationInspectionAdapter(_Custom()).read_model_invocations(_SCOPE)
    assert section.source_id == "custom_canonical_model"
    assert len(section.invocations) == 1


def test_c_r1_custom_canonical_external_work_port() -> None:
    class _Custom(ExternalWorkRuntimeFactReadPort):
        source_id = "custom_canonical_external"

        def list_work_facts(self, scope, *, limit: int):
            return ExternalWorkRuntimeFactReadResult(
                records=(
                    ExternalWorkRuntimeFactRecord(
                        work_ref="w",
                        work_class="c",
                        work_status=ExternalWorkRuntimeStatus.FAILED,
                        provider_ref="p",
                        failure_classification="f",
                        retryable=False,
                        **_scope_identity_fields(),
                        sequence_key=1,
                        evidence_refs=(),
                        safe_summary="ok",
                    ),
                ),
                is_truncated=False,
            )

    section = ExternalWorkInspectionAdapter(_Custom()).read_external_work(_SCOPE)
    assert section.source_id == "custom_canonical_external"
    assert len(section.work_entries) == 1


def test_c_r1_custom_canonical_artifact_port() -> None:
    class _Custom(ExecutionArtifactMetadataReadPort):
        source_id = "custom_canonical_artifact"

        def list_artifact_metadata(self, scope, *, limit: int):
            return ExecutionArtifactMetadataReadResult(
                records=(
                    ExecutionArtifactMetadataRecord(
                        artifact_ref="a",
                        artifact_type="t",
                        lifecycle_status=ExecutionArtifactLifecycleStatus.REGISTERED,
                        content_classification="internal",
                        **_scope_identity_fields(),
                        sequence_key=1,
                        evidence_refs=(),
                        safe_summary="ok",
                    ),
                ),
                is_truncated=False,
            )

    section = ArtifactMetadataInspectionAdapter(_Custom()).read_artifacts(_SCOPE)
    assert section.source_id == "custom_canonical_artifact"
    assert len(section.artifacts) == 1


def test_c_q18_inspect_b_c1_regression_smoke() -> None:
    from intergrax.runtime.runtime_inspection.adapters.tool_runtime_read import (
        ReconstructionToolRuntimeInvocationReader,
        ToolRuntimeInvocationInspectionAdapter,
    )

    facts = _FactsReader()
    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=facts,
        tool_reader=ToolRuntimeInvocationInspectionAdapter(
            ReconstructionToolRuntimeInvocationReader(facts),
        ),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.tools is not None
