# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-C extended domain read adoption qualification gates."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_artifact_read import (
    ExecutionArtifactMetadataReadResult,
    ExecutionArtifactMetadataRecord,
    ExecutionArtifactLifecycleStatus,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
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
    ExternalWorkRuntimeFactReadResult,
    ExternalWorkRuntimeFactRecord,
    ExternalWorkRuntimeStatus,
)
from intergrax.contracts.memory_runtime_read import (
    MemoryRuntimeOperationClass,
    MemoryRuntimeOperationReadResult,
    MemoryRuntimeOperationRecord,
    MemoryRuntimeOperationStatus,
)
from intergrax.contracts.model_runtime_read import (
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
    wrong_exec = mint_execution_id()

    class _BadMemory:
        source_id = "bad_memory"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=(
                    MemoryRuntimeOperationRecord(
                        operation_ref="op-1",
                        memory_class="task",
                        operation_class=MemoryRuntimeOperationClass.READ,
                        operation_status=MemoryRuntimeOperationStatus.HIT,
                        record_ref="rec",
                        source_category="custom",
                        execution_id=wrong_exec,
                        attempt_id=_SCOPE.attempt_id,
                        sequence_key=1,
                        evidence_refs=(),
                        safe_summary="ok",
                    ),
                ),
                is_truncated=False,
            )

    from intergrax.contracts.runtime_inspection.errors import RuntimeInspectionError

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


def test_c_q15_deterministic_order() -> None:
    reconstruction = _extended_reconstruction()
    reader = ReconstructionMemoryOperationReader(_FactsReader(reconstruction))
    from intergrax.contracts.memory_runtime_read import MemoryRuntimeExecutionScope

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
