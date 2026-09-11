# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionPage,
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptDiscoveryPage,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageReader,
    ExecutionLineageRunScope,
    ExecutionLineageSegmentPage,
    ExecutionLineageSegmentRecord,
    ExecutionLineageSealRecord,
    ExecutionLineageSegmentLifecycle,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageCompleteness,
    ExecutionLineageReadStatus,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from tests.unit.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"


def _scope(*, task_id: str, run_id: str, attempt_id: str) -> ExecutionLineageAttemptScope:
    return build_execution_lineage_attempt_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )


def _build_reconstructor(
    *,
    lineage: ExecutionLineageReader | None,
    task_id: str,
    run_id: str,
    attempt_id: str,
    max_lineage_records: int = 10_000,
) -> ExecutionReconstructor:
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(
        PlatformCausalEvidence(
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=_TENANT,
            source=MessageBusTaskRef(provider="celery", task_id="t1", tenant_id=_TENANT),
            target=RuntimeExecutionRef(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                tenant_id=_TENANT,
            ),
            recorded_at=datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        ),
    )
    runtime_store.append(
        sample_runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        tenant_id=_TENANT,
    )
    return ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
        execution_lineage=lineage,
        max_lineage_records=max_lineage_records,
    )


def test_single_segment_parent_chain() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    e1, e2, e3 = mint_execution_id(), mint_execution_id(), mint_execution_id()
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, e1)
    lineage.admit_root(scope, e1, e1)
    lineage.admit_child(scope, e1, e2, e1)
    lineage.admit_child(scope, e1, e3, e2)

    reconstruction = _build_reconstructor(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).reconstruct_execution(_TENANT, task_id, run_id)

    attempt = reconstruction.attempts[0]
    assert attempt.lineage is not None
    assert attempt.lineage.read_status is ExecutionLineageReadStatus.AVAILABLE
    assert attempt.lineage.completeness is ExecutionLineageCompleteness.OPEN
    segment = attempt.lineage.segments[0]
    by_id = {row.execution_id: row.parent_execution_id for row in segment.admissions}
    assert by_id[e2] == e1
    assert by_id[e3] == e2


def test_multi_segment_resume_topology() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    e1, e2, e3, e4, e5 = [mint_execution_id() for _ in range(5)]
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, e1)
    lineage.admit_root(scope, e1, e1)
    lineage.admit_child(scope, e1, e2, e1)
    lineage.admit_child(scope, e1, e3, e2)
    lineage.close_segment_for_resume(scope, e1)
    lineage.open_segment(scope, e4, e1)
    lineage.admit_root(scope, e4, e4)
    lineage.admit_child(scope, e4, e5, e4)

    attempt = _build_reconstructor(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0]
    assert attempt.lineage is not None
    assert [segment.root_execution_id for segment in attempt.lineage.segments] == [e1, e4]
    s2 = attempt.lineage.segments[1]
    assert s2.predecessor_root_execution_id == e1
    root_admission = next(row for row in s2.admissions if row.execution_id == e4)
    assert root_admission.parent_execution_id is None


def test_nested_fan_out_parents() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, e1)
    lineage.admit_root(scope, e1, e1)
    lineage.admit_child(scope, e1, e2, e1)
    lineage.admit_child(scope, e1, e3, e1)
    lineage.admit_child(scope, e1, e4, e2)

    segment = _build_reconstructor(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0].lineage.segments[0]
    by_id = {row.execution_id: row.parent_execution_id for row in segment.admissions}
    assert by_id[e2] == e1
    assert by_id[e3] == e1
    assert by_id[e4] == e2


def test_degraded_attempt_is_partial() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    root = mint_execution_id()
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)
    lineage.mark_degraded(scope, "test")

    completeness = _build_reconstructor(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0].lineage.completeness
    assert completeness is ExecutionLineageCompleteness.PARTIAL


def test_failed_execution_can_be_complete_lineage() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    root = mint_execution_id()
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)
    lineage.seal_attempt(scope, ExecutionLineageAttemptClosureKind.FAILED)

    attempt_lineage = _build_reconstructor(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0].lineage
    assert attempt_lineage.completeness is ExecutionLineageCompleteness.COMPLETE
    assert attempt_lineage.closure_kind is ExecutionLineageAttemptClosureKind.FAILED


def test_truncated_when_max_records_exceeded() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    root = mint_execution_id()
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)
    for _ in range(4):
        child = mint_execution_id()
        lineage.admit_child(scope, root, child, root)

    completeness = _build_reconstructor(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        max_lineage_records=2,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0].lineage.completeness
    assert completeness is ExecutionLineageCompleteness.TRUNCATED


def test_lineage_backend_unavailable() -> None:
    class _UnavailableReader(InMemoryExecutionLineagePersistence):
        def read_attempt_lineage_state(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageAttemptState | None:
            raise ExecutionLineageUnavailableError("down")

    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = _UnavailableReader()
    read_status = _build_reconstructor(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0].lineage.read_status
    assert read_status is ExecutionLineageReadStatus.UNAVAILABLE


def test_duplicate_execution_admission_integrity_error() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    scope = _scope(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    e1, e2 = mint_execution_id(), mint_execution_id()
    duplicate = ExecutionLineageAdmissionRecord(
        scope=scope,
        segment_root_execution_id=e1,
        execution_id=e2,
        parent_execution_id=e1,
        admission_position=2,
    )
    root = ExecutionLineageAdmissionRecord(
        scope=scope,
        segment_root_execution_id=e1,
        execution_id=e1,
        parent_execution_id=None,
        admission_position=1,
    )

    class _DuplicateReader(ExecutionLineageReader):
        def list_admissions_for_attempt(
            self,
            scope: ExecutionLineageAttemptScope,
            limit: int,
            cursor: str | None = None,
        ) -> ExecutionLineageAdmissionPage:
            return ExecutionLineageAdmissionPage(
                admissions=(root, duplicate, duplicate),
            )

        def list_segments_for_attempt(
            self,
            scope: ExecutionLineageAttemptScope,
            limit: int,
            cursor: str | None = None,
        ) -> ExecutionLineageSegmentPage:
            return ExecutionLineageSegmentPage(
                segments=(
                    ExecutionLineageSegmentRecord(
                        scope=scope,
                        root_execution_id=e1,
                        predecessor_root_execution_id=None,
                        lifecycle=ExecutionLineageSegmentLifecycle.SEGMENT_OPEN,
                    ),
                ),
            )

        def read_attempt_lineage_state(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageAttemptState | None:
            return ExecutionLineageAttemptState(scope=scope, generation=1, next_admission_position=3)

        def read_seal(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageSealRecord | None:
            return None

        def read_discovery_run_state(
            self,
            run_scope: ExecutionLineageRunScope,
        ) -> None:
            return None

        def list_attempts_for_run(
            self,
            run_scope: ExecutionLineageRunScope,
            limit: int,
            cursor: str | None = None,
        ) -> ExecutionLineageAttemptDiscoveryPage:
            return ExecutionLineageAttemptDiscoveryPage(attempts=())

        def read_attempt_discovery_record(
            self,
            run_scope: ExecutionLineageRunScope,
            attempt_id: object,
        ) -> None:
            return None

    with pytest.raises(ExecutionReconstructionIntegrityError):
        _build_reconstructor(
            lineage=_DuplicateReader(),
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ).reconstruct_execution(_TENANT, task_id, run_id)


def test_backwards_compatible_without_lineage_reader() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    attempt = _build_reconstructor(
        lineage=None,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0]
    assert attempt.lineage is None
