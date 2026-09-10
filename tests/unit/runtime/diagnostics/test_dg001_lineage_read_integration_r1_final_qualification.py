# © Artur Czarnecki. All rights reserved.

"""DG-001 lineage read integration R1 final qualification matrix (D8–D13, C1–C15, §69–§73)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
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
    ExecutionLineageAttemptDiscoveryRecord,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageDiscoveryCoverageOrigin,
    ExecutionLineageDiscoveryRunState,
    ExecutionLineageIntegrityError,
    ExecutionLineagePersistence,
    ExecutionLineageReader,
    ExecutionLineageRunScope,
    ExecutionLineageSegmentLifecycle,
    ExecutionLineageSegmentPage,
    ExecutionLineageSegmentRecord,
    ExecutionLineageSealRecord,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
    build_execution_lineage_run_scope,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryPageV1,
    DocumentRecord,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageCompleteness,
    ExecutionLineageReadStatus,
    ExecutionLineageReconstructionIntegrityError,
    reconstruct_attempt_lineage,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionAttemptDiscoveryCompleteness,
    ExecutionAttemptDiscoveryReadStatus,
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.execution.lineage.codecs import (
    encode_execution_lineage_attempt_discovery_record,
    encode_execution_lineage_discovery_run_state,
)
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
    execution_lineage_discovery_partition_key,
)
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
from tests.unit.runtime.execution.lineage.lineage_test_helpers import (
    register_v1_attempt,
    seed_legacy_attempt_state,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TENANT_B = "tenant-b"


def _attempt_scope(
    *,
    tenant_id: str = _TENANT,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
) -> ExecutionLineageAttemptScope:
    return build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
    )


def _run_scope(scope: ExecutionLineageAttemptScope) -> ExecutionLineageRunScope:
    return build_execution_lineage_run_scope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )


def _document_persistence() -> DocumentStoreExecutionLineagePersistence:
    return DocumentStoreExecutionLineagePersistence(InMemoryDocumentStore())


@pytest.fixture(params=["memory", "document_store"])
def persistence(request: pytest.FixtureRequest) -> ExecutionLineagePersistence:
    if request.param == "memory":
        return InMemoryExecutionLineagePersistence()
    return _document_persistence()


def _reconstructor(
    *,
    lineage: ExecutionLineageReader | None,
    runtime_store: InMemoryRuntimeEventStore | None = None,
    causal_store: InMemoryCausalEvidencePersistence | None = None,
    max_lineage_records: int = 10_000,
    max_attempt_discovery_snapshot_retries: int = 8,
) -> ExecutionReconstructor:
    return ExecutionReconstructor(
        runtime_events=runtime_store or InMemoryRuntimeEventStore(),
        causal_evidence=causal_store or InMemoryCausalEvidencePersistence(),
        execution_lineage=lineage,
        max_lineage_records=max_lineage_records,
        max_attempt_discovery_snapshot_retries=max_attempt_discovery_snapshot_retries,
    )


def _append_runtime(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
) -> None:
    store.append(
        sample_runtime_event(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        tenant_id=tenant_id,
    )


def _append_causal(
    store: InMemoryCausalEvidencePersistence,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
) -> None:
    store.append(
        PlatformCausalEvidence(
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=tenant_id,
            source=MessageBusTaskRef(
                provider="celery", task_id="t1", tenant_id=tenant_id
            ),
            target=RuntimeExecutionRef(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                tenant_id=tenant_id,
            ),
            recorded_at=datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        ),
    )


def _execution_id_from_int(value: int) -> ExecutionId:
    return ExecutionId(f"exec_{value:032x}")


class _DelegateReader(ExecutionLineageReader):
    def __init__(self, inner: ExecutionLineageReader) -> None:
        self._inner = inner

    def list_admissions_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAdmissionPage:
        return self._inner.list_admissions_for_attempt(scope, limit, cursor=cursor)

    def list_segments_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageSegmentPage:
        return self._inner.list_segments_for_attempt(scope, limit, cursor=cursor)

    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        return self._inner.read_attempt_lineage_state(scope)

    def read_seal(
        self, scope: ExecutionLineageAttemptScope
    ) -> ExecutionLineageSealRecord | None:
        return self._inner.read_seal(scope)

    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        return self._inner.read_discovery_run_state(run_scope)

    def list_attempts_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAttemptDiscoveryPage:
        return self._inner.list_attempts_for_run(run_scope, limit, cursor=cursor)

    def read_attempt_discovery_record(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord | None:
        return self._inner.read_attempt_discovery_record(run_scope, attempt_id)


class _DiscoveryUnavailableReader(_DelegateReader):
    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        raise ExecutionLineageUnavailableError("discovery down")

    def list_attempts_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAttemptDiscoveryPage:
        raise ExecutionLineageUnavailableError("discovery down")

    def read_attempt_discovery_record(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord | None:
        raise ExecutionLineageUnavailableError("discovery down")


class _GenerationChurnDiscoveryReader(_DelegateReader):
    def __init__(
        self,
        inner: ExecutionLineageReader,
        *,
        churn_until_call: int = 0,
        always_churn: bool = False,
    ) -> None:
        super().__init__(inner)
        self._run_state_reads = 0
        self._churn_until_call = churn_until_call
        self._always_churn = always_churn

    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        state = self._inner.read_discovery_run_state(run_scope)
        self._run_state_reads += 1
        if state is None:
            return None
        if self._always_churn:
            return state.model_copy(
                update={"generation": state.generation + self._run_state_reads},
            )
        if self._run_state_reads <= self._churn_until_call:
            return state.model_copy(update={"generation": state.generation + 1})
        return state


class _OperationalOutageDocumentStore(InMemoryDocumentStore):
    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        raise RuntimeError("operational get failure")

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
    ) -> DocumentQueryPageV1:
        raise RuntimeError("operational query failure")


class _QueryOutageDocumentStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__()
        self._query_outage = False

    def enable_query_outage(self) -> None:
        self._query_outage = True

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
    ) -> DocumentQueryPageV1:
        if self._query_outage:
            raise RuntimeError("operational query failure")
        return super().query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
            row_key_upper_bound=row_key_upper_bound,
            data_equalities=data_equalities,
            sort=sort,
        )


class _AttemptStateUnavailableReader(_DelegateReader):
    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        raise ExecutionLineageUnavailableError("attempt state unavailable")


class _FromRunStartReader(_DelegateReader):
    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        return ExecutionLineageDiscoveryRunState(
            run_scope=run_scope,
            generation=1,
            next_discovery_position=1,
            coverage_contract_version=1,
            coverage_origin=ExecutionLineageDiscoveryCoverageOrigin.FROM_RUN_START,
        )

    def list_attempts_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAttemptDiscoveryPage:
        return ExecutionLineageAttemptDiscoveryPage(attempts=(), next_cursor=None)


def test_indexed_lineage_only_real_attempt(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    root = mint_execution_id()
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    persistence.open_attempt(scope, discovery_contract_version=1)
    persistence.open_segment(scope, root)

    reconstruction = _reconstructor(lineage=persistence).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert tuple(item.attempt_id for item in reconstruction.attempts) == (
        scope.attempt_id,
    )
    attempt = reconstruction.attempts[0]
    assert attempt.lineage is not None
    assert attempt.lineage.read_status is ExecutionLineageReadStatus.AVAILABLE
    assert attempt.lineage.discovery_contract_version == 1
    assert attempt.lineage.completeness is ExecutionLineageCompleteness.OPEN


def test_stale_discovery_only_not_surfaced(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    reconstruction = _reconstructor(lineage=persistence).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert reconstruction.attempt_count == 0


def test_discovery_unavailable_not_integrity_with_runtime_attempt(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    register_v1_attempt(persistence, scope)
    runtime_store = InMemoryRuntimeEventStore()
    _append_runtime(
        runtime_store,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    reconstruction = _reconstructor(
        lineage=_DiscoveryUnavailableReader(persistence),
        runtime_store=runtime_store,
    ).reconstruct_execution(scope.tenant_id, scope.task_id, scope.run_id)
    assert reconstruction.attempt_count == 1
    assert (
        reconstruction.attempt_discovery_read_status
        is ExecutionAttemptDiscoveryReadStatus.UNAVAILABLE
    )
    assert reconstruction.attempt_discovery_completeness is None


def test_discovery_only_unavailable_state_read_not_real_attempt(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    persistence.open_attempt(scope, discovery_contract_version=1)

    class _StateUnavailableReader(_DelegateReader):
        def read_attempt_lineage_state(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageAttemptState | None:
            raise ExecutionLineageUnavailableError("state down")

    reconstruction = _reconstructor(
        lineage=_StateUnavailableReader(persistence),
    ).reconstruct_execution(scope.tenant_id, scope.task_id, scope.run_id)
    assert reconstruction.attempt_count == 0


def test_d8_mixed_legacy_and_indexed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    a1, a2, a3 = mint_attempt_id(), mint_attempt_id(), mint_attempt_id()
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    scope_a1 = _attempt_scope(task_id=task_id, run_id=run_id, attempt_id=a1)
    scope_a2 = _attempt_scope(task_id=task_id, run_id=run_id, attempt_id=a2)
    scope_a3 = _attempt_scope(task_id=task_id, run_id=run_id, attempt_id=a3)
    seed_legacy_attempt_state(store, scope_a1)
    register_v1_attempt(lineage, scope_a2)
    register_v1_attempt(lineage, scope_a3)
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    _append_runtime(
        runtime_store, tenant_id=_TENANT, task_id=task_id, run_id=run_id, attempt_id=a1
    )
    _append_causal(
        causal_store, tenant_id=_TENANT, task_id=task_id, run_id=run_id, attempt_id=a1
    )

    reconstruction = _reconstructor(
        lineage=lineage,
        runtime_store=runtime_store,
        causal_store=causal_store,
    ).reconstruct_execution(_TENANT, task_id, run_id)
    assert {item.attempt_id for item in reconstruction.attempts} == {a1, a2, a3}
    assert (
        reconstruction.attempt_discovery_completeness
        is ExecutionAttemptDiscoveryCompleteness.LEGACY_UNKNOWN
    )
    ordered = [item.attempt_id for item in reconstruction.attempts]
    assert ordered.index(a2) < ordered.index(a1)
    assert ordered.index(a3) < ordered.index(a1)


def test_d9_legacy_same_attempt_resume() -> None:
    scope = _attempt_scope()
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    seed_legacy_attempt_state(store, scope)
    e1, e2 = mint_execution_id(), mint_execution_id()
    lineage.open_segment(scope, e1)
    lineage.admit_root(scope, e1, e1)
    lineage.close_segment_for_resume(scope, e1)
    lineage.open_segment(scope, e2, e1)
    state = lineage.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.discovery_contract_version is None


def test_d9_post_v1_same_attempt_resume(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    e1, e2 = mint_execution_id(), mint_execution_id()
    register_v1_attempt(persistence, scope)
    discovery_before = persistence.read_attempt_discovery_record(
        run_scope,
        scope.attempt_id,
    )
    assert discovery_before is not None
    persistence.open_segment(scope, e1)
    persistence.admit_root(scope, e1, e1)
    persistence.close_segment_for_resume(scope, e1)
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    persistence.open_segment(scope, e2, e1)
    after = persistence.read_attempt_discovery_record(run_scope, scope.attempt_id)
    assert after is not None
    assert after.discovery_position == discovery_before.discovery_position


def test_d10_paginated_run_discovery_full_proof(
    persistence: ExecutionLineagePersistence,
) -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    run_scope = build_execution_lineage_run_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
    )
    attempt_ids = [mint_attempt_id() for _ in range(5)]
    for attempt_id in attempt_ids:
        persistence.register_attempt_for_run(run_scope, attempt_id)
    collected: list[ExecutionLineageAttemptDiscoveryRecord] = []
    cursor: str | None = None
    while True:
        page = persistence.list_attempts_for_run(run_scope, limit=2, cursor=cursor)
        collected.extend(page.attempts)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert len(collected) == len(attempt_ids)
    assert len({item.attempt_id for item in collected}) == len(attempt_ids)
    assert [item.discovery_position for item in collected] == list(range(1, 6))


def test_d11_generation_changes_during_pagination(
    persistence: ExecutionLineagePersistence,
) -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    scopes = [
        _attempt_scope(task_id=task_id, run_id=run_id, attempt_id=mint_attempt_id())
        for _ in range(2)
    ]
    for scope in scopes:
        register_v1_attempt(persistence, scope)
    reader = _GenerationChurnDiscoveryReader(persistence, churn_until_call=1)
    reconstruction = _reconstructor(
        lineage=reader,
        max_attempt_discovery_snapshot_retries=4,
    ).reconstruct_execution(_TENANT, task_id, run_id)
    assert (
        reconstruction.attempt_discovery_read_status
        is ExecutionAttemptDiscoveryReadStatus.AVAILABLE
    )
    assert reconstruction.attempt_count == 2


def test_d12_persistent_discovery_churn(
    persistence: ExecutionLineagePersistence,
) -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    run_scope = build_execution_lineage_run_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
    )
    persistence.register_attempt_for_run(run_scope, mint_attempt_id())
    reader = _GenerationChurnDiscoveryReader(persistence, always_churn=True)
    reconstruction = _reconstructor(
        lineage=reader,
        max_attempt_discovery_snapshot_retries=3,
    ).reconstruct_execution(_TENANT, task_id, run_id)
    assert (
        reconstruction.attempt_discovery_read_status
        is ExecutionAttemptDiscoveryReadStatus.AVAILABLE
    )
    assert (
        reconstruction.attempt_discovery_completeness
        is ExecutionAttemptDiscoveryCompleteness.TRUNCATED
    )


def test_d13_stale_discovery_only(persistence: ExecutionLineagePersistence) -> None:
    test_stale_discovery_only_not_surfaced(persistence)


def test_c1_pure_legacy() -> None:
    scope = _attempt_scope()
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    seed_legacy_attempt_state(store, scope)
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    _append_runtime(
        runtime_store,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    _append_causal(
        causal_store,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    reconstruction = _reconstructor(
        lineage=lineage,
        runtime_store=runtime_store,
        causal_store=causal_store,
    ).reconstruct_execution(scope.tenant_id, scope.task_id, scope.run_id)
    assert reconstruction.attempt_count == 1
    assert (
        reconstruction.attempt_discovery_read_status
        is ExecutionAttemptDiscoveryReadStatus.AVAILABLE
    )
    assert (
        reconstruction.attempt_discovery_completeness
        is ExecutionAttemptDiscoveryCompleteness.LEGACY_UNKNOWN
    )


def test_c2_legacy_enrichment() -> None:
    test_c1_pure_legacy()


def test_c3_hidden_legacy_lineage_only_not_enumerable() -> None:
    scope = _attempt_scope()
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    seed_legacy_attempt_state(store, scope)
    reconstruction = _reconstructor(lineage=lineage).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert reconstruction.attempt_count == 0
    assert (
        reconstruction.attempt_discovery_completeness
        is ExecutionAttemptDiscoveryCompleteness.LEGACY_UNKNOWN
    )


def test_c4_mixed_run_legacy_unknown() -> None:
    test_d8_mixed_legacy_and_indexed()


def test_c5_post_v1_missing_index() -> None:
    scope = _attempt_scope()
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    register_v1_attempt(lineage, scope)
    partition = execution_lineage_discovery_partition_key(_run_scope(scope))
    store.delete(partition, f"attempt:{scope.attempt_id}")
    runtime_store = InMemoryRuntimeEventStore()
    _append_runtime(
        runtime_store,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        _reconstructor(
            lineage=lineage, runtime_store=runtime_store
        ).reconstruct_execution(
            scope.tenant_id,
            scope.task_id,
            scope.run_id,
        )


def test_c6_stale_discovery_row(persistence: ExecutionLineagePersistence) -> None:
    test_stale_discovery_only_not_surfaced(persistence)


def test_c7_stable_indexed_snapshot(persistence: ExecutionLineagePersistence) -> None:
    scope = _attempt_scope()
    register_v1_attempt(persistence, scope)
    reconstruction = _reconstructor(lineage=persistence).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert reconstruction.attempt_count == 1
    assert reconstruction.attempts[0].lineage is not None
    assert reconstruction.attempts[0].lineage.discovery_position == 1


def test_c8_churn_truncated(persistence: ExecutionLineagePersistence) -> None:
    test_d12_persistent_discovery_churn(persistence)


def test_c9_per_attempt_snapshot_retry(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)

    class _SealChurnReader(_DelegateReader):
        def __init__(self, inner: ExecutionLineageReader) -> None:
            super().__init__(inner)
            self._state_reads = 0

        def read_attempt_lineage_state(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageAttemptState | None:
            state = self._inner.read_attempt_lineage_state(scope)
            if state is None:
                return None
            self._state_reads += 1
            if self._state_reads == 1:
                return state.model_copy(update={"generation": state.generation})
            if self._state_reads == 2:
                return state.model_copy(update={"generation": state.generation + 1})
            return state

    lineage = reconstruct_attempt_lineage(
        _SealChurnReader(persistence),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        initial_lineage_page_limit=100,
        max_lineage_records=10_000,
        max_lineage_snapshot_retries=4,
    )
    assert lineage.read_status is ExecutionLineageReadStatus.AVAILABLE


def test_c10_segment_truncation(persistence: ExecutionLineagePersistence) -> None:
    scope = _attempt_scope()
    s1 = _execution_id_from_int(2)
    s2 = _execution_id_from_int(1)
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, s1)
    persistence.admit_root(scope, s1, s1)
    persistence.close_segment_for_resume(scope, s1)
    persistence.open_segment(scope, s2, s1)
    lineage = reconstruct_attempt_lineage(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        initial_lineage_page_limit=100,
        max_lineage_records=1,
    )
    assert lineage.completeness is ExecutionLineageCompleteness.TRUNCATED
    assert lineage.segments == ()


def test_c10_admission_truncation(persistence: ExecutionLineagePersistence) -> None:
    scope = _attempt_scope()
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    for _ in range(3):
        child = mint_execution_id()
        persistence.admit_child(scope, root, child, root)
    lineage = reconstruct_attempt_lineage(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        initial_lineage_page_limit=100,
        max_lineage_records=2,
    )
    assert lineage.completeness is ExecutionLineageCompleteness.TRUNCATED


def test_c11_state_seal_contradiction(persistence: ExecutionLineagePersistence) -> None:
    scope = _attempt_scope()
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    state = persistence.read_attempt_lineage_state(scope)
    assert state is not None
    tampered = state.model_copy(update={"sealed": False})

    class _TamperedReader(_DelegateReader):
        def read_attempt_lineage_state(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageAttemptState | None:
            return tampered

    with pytest.raises(ExecutionLineageReconstructionIntegrityError):
        reconstruct_attempt_lineage(
            _TamperedReader(persistence),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            initial_lineage_page_limit=100,
            max_lineage_records=10_000,
        )


@pytest.mark.parametrize(
    ("state_update", "seal"),
    [
        ({"sealed": False}, "present"),
        ({"sealed": True}, "absent"),
        (
            {"closure_kind": ExecutionLineageAttemptClosureKind.FAILED},
            "present",
        ),
        ({"degraded": True}, "present"),
    ],
    ids=[
        "state_unsealed_seal_exists",
        "state_sealed_seal_absent",
        "closure_kind_mismatch",
        "degraded_mismatch",
    ],
)
def test_c11_state_seal_contradiction_matrix(
    persistence: ExecutionLineagePersistence,
    state_update: dict[str, object],
    seal: str,
) -> None:
    scope = _attempt_scope()
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    state = persistence.read_attempt_lineage_state(scope)
    assert state is not None
    seal_record = persistence.read_seal(scope)
    tampered_state = state.model_copy(update=state_update)

    class _MatrixReader(_DelegateReader):
        def read_attempt_lineage_state(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageAttemptState | None:
            return tampered_state

        def read_seal(
            self, scope: ExecutionLineageAttemptScope
        ) -> ExecutionLineageSealRecord | None:
            if seal == "absent":
                return None
            return seal_record

    with pytest.raises(ExecutionLineageReconstructionIntegrityError):
        reconstruct_attempt_lineage(
            _MatrixReader(persistence),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            initial_lineage_page_limit=100,
            max_lineage_records=10_000,
        )


def test_c12_backend_unavailable() -> None:
    store = _OperationalOutageDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    scope = _attempt_scope()
    with pytest.raises(ExecutionLineageUnavailableError):
        lineage.read_attempt_lineage_state(scope)


def test_c13_legacy_same_attempt_resume() -> None:
    test_d9_legacy_same_attempt_resume()


def test_c14_new_post_v1_in_legacy_run() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    a1, a2 = mint_attempt_id(), mint_attempt_id()
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    scope_a1 = _attempt_scope(task_id=task_id, run_id=run_id, attempt_id=a1)
    scope_a2 = _attempt_scope(task_id=task_id, run_id=run_id, attempt_id=a2)
    seed_legacy_attempt_state(store, scope_a1)
    register_v1_attempt(lineage, scope_a2)
    runtime_store = InMemoryRuntimeEventStore()
    _append_runtime(
        runtime_store, tenant_id=_TENANT, task_id=task_id, run_id=run_id, attempt_id=a1
    )
    reconstruction = _reconstructor(
        lineage=lineage,
        runtime_store=runtime_store,
    ).reconstruct_execution(_TENANT, task_id, run_id)
    post_v1 = next(item for item in reconstruction.attempts if item.attempt_id == a2)
    assert post_v1.lineage is not None
    assert post_v1.lineage.discovery_contract_version == 1
    assert (
        reconstruction.attempt_discovery_completeness
        is ExecutionAttemptDiscoveryCompleteness.LEGACY_UNKNOWN
    )


def test_c15_from_run_start_reader_contract() -> None:
    scope = _attempt_scope()
    reconstruction = _reconstructor(
        lineage=_FromRunStartReader(_document_persistence())
    ).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert (
        reconstruction.attempt_discovery_completeness
        is ExecutionAttemptDiscoveryCompleteness.COMPLETE
    )


def test_section_69_multi_segment_truncation(
    persistence: ExecutionLineagePersistence,
) -> None:
    test_c10_segment_truncation(persistence)


def test_section_70_torn_seal_retry(persistence: ExecutionLineagePersistence) -> None:
    scope = _attempt_scope()
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    sealed_state = persistence.read_attempt_lineage_state(scope)
    assert sealed_state is not None

    class _TornSealReader(_DelegateReader):
        def __init__(self, inner: ExecutionLineageReader) -> None:
            super().__init__(inner)
            self._state_reads = 0

        def read_attempt_lineage_state(
            self,
            scope: ExecutionLineageAttemptScope,
        ) -> ExecutionLineageAttemptState | None:
            state = self._inner.read_attempt_lineage_state(scope)
            if state is None:
                return None
            self._state_reads += 1
            if self._state_reads == 1:
                return state.model_copy(
                    update={
                        "generation": state.generation,
                        "sealed": False,
                        "closure_kind": None,
                    },
                )
            if self._state_reads == 2:
                return state.model_copy(
                    update={"generation": state.generation + 1},
                )
            return state

    lineage = reconstruct_attempt_lineage(
        _TornSealReader(persistence),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        initial_lineage_page_limit=100,
        max_lineage_records=10_000,
        max_lineage_snapshot_retries=4,
    )
    assert lineage.read_status is ExecutionLineageReadStatus.AVAILABLE
    assert lineage.completeness is ExecutionLineageCompleteness.COMPLETE
    assert lineage.closure_kind is ExecutionLineageAttemptClosureKind.COMPLETED


def test_section_71_stable_contradiction(
    persistence: ExecutionLineagePersistence,
) -> None:
    test_c11_state_seal_contradiction(persistence)


def test_section_72_root_admission_crash_open_then_partial(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    open_lineage = reconstruct_attempt_lineage(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        initial_lineage_page_limit=100,
        max_lineage_records=10_000,
    )
    assert open_lineage.completeness is ExecutionLineageCompleteness.OPEN
    persistence.mark_degraded(scope, "crash")
    new_root = mint_execution_id()
    persistence.open_segment(scope, new_root, root)
    partial = reconstruct_attempt_lineage(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        initial_lineage_page_limit=100,
        max_lineage_records=10_000,
    )
    assert partial.completeness is ExecutionLineageCompleteness.PARTIAL


def test_section_73_documentstore_get_outage() -> None:
    store = _OperationalOutageDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    scope = _attempt_scope()
    with pytest.raises(ExecutionLineageUnavailableError):
        lineage.read_attempt_lineage_state(scope)


def test_section_73_documentstore_query_outage() -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    store = _QueryOutageDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    lineage.register_attempt_for_run(run_scope, scope.attempt_id)
    store.enable_query_outage()
    with pytest.raises(ExecutionLineageUnavailableError):
        lineage.list_attempts_for_run(run_scope, 100)


def test_section_73_documentstore_outage() -> None:
    test_section_73_documentstore_get_outage()


def test_parentless_non_root_admission_integrity(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    e1, e2 = mint_execution_id(), mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, e1)
    persistence.admit_root(scope, e1, e1)

    class _CorruptReader(_DelegateReader):
        def list_admissions_for_attempt(
            self,
            scope: ExecutionLineageAttemptScope,
            limit: int,
            cursor: str | None = None,
        ) -> ExecutionLineageAdmissionPage:
            root = ExecutionLineageAdmissionRecord(
                scope=scope,
                segment_root_execution_id=e1,
                execution_id=e1,
                parent_execution_id=None,
                admission_position=1,
            )
            corrupt = ExecutionLineageAdmissionRecord.model_construct(
                scope=scope,
                segment_root_execution_id=e1,
                execution_id=e2,
                parent_execution_id=None,
                admission_position=2,
            )
            return ExecutionLineageAdmissionPage(admissions=(root, corrupt))

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

    with pytest.raises(ExecutionLineageReconstructionIntegrityError):
        reconstruct_attempt_lineage(
            _CorruptReader(persistence),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            initial_lineage_page_limit=100,
            max_lineage_records=10_000,
        )


def test_discovery_run_scope_validation() -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    lineage.register_attempt_for_run(run_scope, scope.attempt_id)
    partition = execution_lineage_discovery_partition_key(run_scope)
    state = lineage.read_discovery_run_state(run_scope)
    assert state is not None
    wrong_scope = build_execution_lineage_run_scope(
        tenant_id=scope.tenant_id,
        task_id=mint_task_id(),
        run_id=scope.run_id,
    )
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key="meta:discovery_run",
            data=encode_execution_lineage_discovery_run_state(
                state.model_copy(update={"run_scope": wrong_scope}),
            ),
        ),
    )
    with pytest.raises(ExecutionLineageIntegrityError):
        lineage.read_discovery_run_state(run_scope)


def test_discovery_point_identity_validation() -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    record = lineage.register_attempt_for_run(run_scope, scope.attempt_id)
    partition = execution_lineage_discovery_partition_key(run_scope)
    wrong_attempt = mint_attempt_id()
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=f"attempt:{scope.attempt_id}",
            data=encode_execution_lineage_attempt_discovery_record(
                record.model_copy(update={"attempt_id": wrong_attempt}),
            ),
        ),
    )
    with pytest.raises(ExecutionLineageIntegrityError):
        lineage.read_attempt_discovery_record(run_scope, scope.attempt_id)


def test_discovery_row_meta_atomic_invariant() -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    lineage.register_attempt_for_run(run_scope, scope.attempt_id)
    partition = execution_lineage_discovery_partition_key(run_scope)
    store.delete(partition, "meta:discovery_run")
    with pytest.raises(ExecutionLineageIntegrityError):
        lineage.register_attempt_for_run(run_scope, scope.attempt_id)
    with pytest.raises(ExecutionReconstructionIntegrityError):
        _reconstructor(lineage=lineage).reconstruct_execution(
            scope.tenant_id,
            scope.task_id,
            scope.run_id,
        )


def test_tenant_isolation(persistence: ExecutionLineagePersistence) -> None:
    scope_a = _attempt_scope(tenant_id=_TENANT)
    scope_b = _attempt_scope(
        tenant_id=_TENANT_B,
        task_id=scope_a.task_id,
        run_id=scope_a.run_id,
        attempt_id=scope_a.attempt_id,
    )
    register_v1_attempt(persistence, scope_a)
    reconstruction = _reconstructor(lineage=persistence).reconstruct_execution(
        _TENANT_B,
        scope_b.task_id,
        scope_b.run_id,
    )
    assert reconstruction.attempt_count == 0


def test_provider_conformance_indexed_real_attempt(
    persistence: ExecutionLineagePersistence,
) -> None:
    test_indexed_lineage_only_real_attempt(persistence)


def test_from_run_start_candidate_state_unavailable_never_reports_complete(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)

    class _FromRunStartIndexedReader(_DelegateReader):
        def read_discovery_run_state(
            self,
            run_scope: ExecutionLineageRunScope,
        ) -> ExecutionLineageDiscoveryRunState | None:
            state = self._inner.read_discovery_run_state(run_scope)
            if state is None:
                return None
            return state.model_copy(
                update={
                    "coverage_origin": (
                        ExecutionLineageDiscoveryCoverageOrigin.FROM_RUN_START
                    ),
                    "coverage_contract_version": 1,
                },
            )

    reader = _AttemptStateUnavailableReader(
        _FromRunStartIndexedReader(persistence),
    )
    reconstruction = _reconstructor(lineage=reader).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert reconstruction.attempt_count == 0
    assert (
        reconstruction.attempt_discovery_read_status
        is ExecutionAttemptDiscoveryReadStatus.UNAVAILABLE
    )
    assert reconstruction.attempt_discovery_completeness is None
    assert (
        reconstruction.attempt_discovery_completeness
        is not ExecutionAttemptDiscoveryCompleteness.COMPLETE
    )


def test_legacy_discovery_only_candidate_state_unavailable_not_legacy_unknown(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    reader = _AttemptStateUnavailableReader(persistence)
    reconstruction = _reconstructor(lineage=reader).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert reconstruction.attempt_count == 0
    assert (
        reconstruction.attempt_discovery_read_status
        is ExecutionAttemptDiscoveryReadStatus.UNAVAILABLE
    )
    assert reconstruction.attempt_discovery_completeness is None


def test_truncated_segments_broken_admission_parent_integrity(
    persistence: ExecutionLineagePersistence,
) -> None:
    scope = _attempt_scope()
    e1, e2, e3 = mint_execution_id(), mint_execution_id(), mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, e1)
    persistence.admit_root(scope, e1, e1)

    class _BrokenParentReader(_DelegateReader):
        def list_admissions_for_attempt(
            self,
            scope: ExecutionLineageAttemptScope,
            limit: int,
            cursor: str | None = None,
        ) -> ExecutionLineageAdmissionPage:
            root = ExecutionLineageAdmissionRecord(
                scope=scope,
                segment_root_execution_id=e1,
                execution_id=e1,
                parent_execution_id=None,
                admission_position=1,
            )
            orphan = ExecutionLineageAdmissionRecord(
                scope=scope,
                segment_root_execution_id=e1,
                execution_id=e3,
                parent_execution_id=e2,
                admission_position=3,
            )
            return ExecutionLineageAdmissionPage(admissions=(root, orphan))

        def list_segments_for_attempt(
            self,
            scope: ExecutionLineageAttemptScope,
            limit: int,
            cursor: str | None = None,
        ) -> ExecutionLineageSegmentPage:
            segment = ExecutionLineageSegmentRecord(
                scope=scope,
                root_execution_id=e1,
                predecessor_root_execution_id=None,
                lifecycle=ExecutionLineageSegmentLifecycle.SEGMENT_OPEN,
            )
            extra = ExecutionLineageSegmentRecord(
                scope=scope,
                root_execution_id=e2,
                predecessor_root_execution_id=e1,
                lifecycle=ExecutionLineageSegmentLifecycle.SEGMENT_OPEN,
            )
            tail = ExecutionLineageSegmentRecord(
                scope=scope,
                root_execution_id=e3,
                predecessor_root_execution_id=e2,
                lifecycle=ExecutionLineageSegmentLifecycle.SEGMENT_OPEN,
            )
            return ExecutionLineageSegmentPage(segments=(segment, extra, tail))

    with pytest.raises(ExecutionLineageReconstructionIntegrityError):
        reconstruct_attempt_lineage(
            _BrokenParentReader(persistence),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            initial_lineage_page_limit=100,
            max_lineage_records=2,
        )


def _put_discovery_run_meta(
    store: InMemoryDocumentStore,
    run_scope: ExecutionLineageRunScope,
    *,
    coverage_origin: ExecutionLineageDiscoveryCoverageOrigin | None,
    coverage_contract_version: int | None,
    next_discovery_position: int,
) -> None:
    partition = execution_lineage_discovery_partition_key(run_scope)
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key="meta:discovery_run",
            data=encode_execution_lineage_discovery_run_state(
                ExecutionLineageDiscoveryRunState(
                    run_scope=run_scope,
                    generation=1,
                    next_discovery_position=next_discovery_position,
                    coverage_contract_version=coverage_contract_version,
                    coverage_origin=coverage_origin,
                ),
            ),
        ),
    )


def test_empty_discovery_run_meta_coverage_none_next_one_integrity() -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    store = InMemoryDocumentStore()
    _put_discovery_run_meta(
        store,
        run_scope,
        coverage_origin=None,
        coverage_contract_version=None,
        next_discovery_position=1,
    )
    lineage = DocumentStoreExecutionLineagePersistence(store)
    with pytest.raises(ExecutionReconstructionIntegrityError):
        _reconstructor(lineage=lineage).reconstruct_execution(
            scope.tenant_id,
            scope.task_id,
            scope.run_id,
        )


def test_empty_discovery_run_meta_coverage_none_next_gt_one_integrity() -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    store = InMemoryDocumentStore()
    _put_discovery_run_meta(
        store,
        run_scope,
        coverage_origin=None,
        coverage_contract_version=None,
        next_discovery_position=2,
    )
    lineage = DocumentStoreExecutionLineagePersistence(store)
    with pytest.raises(ExecutionReconstructionIntegrityError):
        _reconstructor(lineage=lineage).reconstruct_execution(
            scope.tenant_id,
            scope.task_id,
            scope.run_id,
        )


def test_empty_from_run_start_discovery_run_meta_complete() -> None:
    scope = _attempt_scope()
    reconstruction = _reconstructor(
        lineage=_FromRunStartReader(_document_persistence()),
    ).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert (
        reconstruction.attempt_discovery_completeness
        is ExecutionAttemptDiscoveryCompleteness.COMPLETE
    )


def test_idempotent_register_corrupt_run_meta_scope() -> None:
    scope = _attempt_scope()
    run_scope = _run_scope(scope)
    store = InMemoryDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    lineage.register_attempt_for_run(run_scope, scope.attempt_id)
    state = lineage.read_discovery_run_state(run_scope)
    assert state is not None
    wrong_scope = build_execution_lineage_run_scope(
        tenant_id=scope.tenant_id,
        task_id=mint_task_id(),
        run_id=scope.run_id,
    )
    partition = execution_lineage_discovery_partition_key(run_scope)
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key="meta:discovery_run",
            data=encode_execution_lineage_discovery_run_state(
                state.model_copy(update={"run_scope": wrong_scope}),
            ),
        ),
    )
    with pytest.raises(ExecutionLineageIntegrityError):
        lineage.register_attempt_for_run(run_scope, scope.attempt_id)


def test_diagnostics_reconstruction_query_outage_metadata() -> None:
    scope = _attempt_scope()
    store = _QueryOutageDocumentStore()
    lineage = DocumentStoreExecutionLineagePersistence(store)
    register_v1_attempt(lineage, scope)
    store.enable_query_outage()
    reconstruction = _reconstructor(lineage=lineage).reconstruct_execution(
        scope.tenant_id,
        scope.task_id,
        scope.run_id,
    )
    assert (
        reconstruction.attempt_discovery_read_status
        is ExecutionAttemptDiscoveryReadStatus.UNAVAILABLE
    )
    assert reconstruction.attempt_discovery_completeness is None


def test_documentstore_operational_oserror_translation() -> None:
    class _OSErrorDocumentStore(InMemoryDocumentStore):
        def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
            raise OSError("operational os failure")

    lineage = DocumentStoreExecutionLineagePersistence(_OSErrorDocumentStore())
    with pytest.raises(ExecutionLineageUnavailableError):
        lineage.read_attempt_lineage_state(_attempt_scope())
