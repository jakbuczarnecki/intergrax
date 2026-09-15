# © Artur Czarnecki. All rights reserved.

"""OBS-ASOF-REBASE — historical execution query qualification (E-axis)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    BitemporalKnowledgeBasis,
    KnowledgeOrderingScope,
    KnowledgeRevisionWatermark,
    SystemTimeBasis,
    ValidTimeBasis,
)
from intergrax.contracts.execution_identity import (
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.historical_reconstruction import ExecutionHistoricalReconstructionRequest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.execution_position import AsOfBoundary, ExecutionEventPosition
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.historical_reconstruction import HistoricalReconstructionService
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.contracts.test_bitemporal_revision_ordering import (
    _InMemoryRevisionOrderingAuthority,
)
from tests.unit.runtime.events.test_asof_projection import _append_sequence, _event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_T0 = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)


def _watermark() -> KnowledgeRevisionWatermark:
    scope = KnowledgeOrderingScope(tenant_id=_TENANT)
    authority = _InMemoryRevisionOrderingAuthority()
    return authority.watermark(scope)


def _bitemporal_query() -> BitemporalKnowledgeBasis:
    return BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.instant(_T0),
        system_time=SystemTimeBasis.instant(_T0),
    )


def _historical_request(*, run_id: RunId, position: int) -> ExecutionHistoricalReconstructionRequest:
    return ExecutionHistoricalReconstructionRequest(
        tenant_id=_TENANT,
        run_id=run_id,
        execution_as_of=AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(position)),
        knowledge_watermark=_watermark(),
        bitemporal_query=_bitemporal_query(),
    )


def _service(store: InMemoryRuntimeEventStore | DocumentBackedRuntimeEventStore) -> HistoricalReconstructionService:
    return HistoricalReconstructionService(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
        revision_ordering=_InMemoryRevisionOrderingAuthority(),
    )


def _reconstruct_at_position_2(
    service: HistoricalReconstructionService,
    *,
    run_id: RunId,
) -> object:
    return service.reconstruct(
        _historical_request(run_id=run_id, position=2),
        revision_reader=_EmptyRevisionReader(),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=lambda state, _revision: state,
        initial_state=(),
    )


class _EmptyRevisionReader:
    def load_revision(self, revision_id: object) -> object:
        raise AssertionError("no revisions configured for E-axis tests")


@pytest.mark.unit
def test_obs_asof_rebase_post_append_immunity_at_historical_service() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[RuntimeEventType.TASK_CREATED, RuntimeEventType.PLAN_CREATED],
    )
    service = _service(store)
    first = _reconstruct_at_position_2(service, run_id=run_id)
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.STEP_STARTED,
            RuntimeEventType.STEP_COMPLETED,
            RuntimeEventType.TASK_COMPLETED,
        ],
    )
    second = _reconstruct_at_position_2(service, run_id=run_id)
    assert first == second
    assert first.execution_projection.last_included_position == ExecutionEventPosition(2)


@pytest.mark.unit
def test_obs_asof_rebase_retry_attempt_visibility_by_boundary() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=[RuntimeEventType.TASK_CREATED, RuntimeEventType.TASK_FAILED],
    )
    boundary_before_retry = AsOfBoundary(
        run_id=run_id,
        position=positioned[-1].position,
    )
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        event_types=[RuntimeEventType.RETRY_STARTED, RuntimeEventType.PLAN_CREATED],
    )
    service = _service(store)

    before = service.reconstruct(
        ExecutionHistoricalReconstructionRequest(
            tenant_id=_TENANT,
            run_id=run_id,
            execution_as_of=boundary_before_retry,
            knowledge_watermark=_watermark(),
            bitemporal_query=_bitemporal_query(),
        ),
        revision_reader=_EmptyRevisionReader(),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=lambda state, _revision: state,
        initial_state=(),
    )
    positioned_after = store.list_positioned_for_run(run_id, tenant_id=_TENANT)
    after_boundary = AsOfBoundary(run_id=run_id, position=positioned_after[-1].position)
    after = service.reconstruct(
        ExecutionHistoricalReconstructionRequest(
            tenant_id=_TENANT,
            run_id=run_id,
            execution_as_of=after_boundary,
            knowledge_watermark=_watermark(),
            bitemporal_query=_bitemporal_query(),
        ),
        revision_reader=_EmptyRevisionReader(),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=lambda state, _revision: state,
        initial_state=(),
    )

    assert len(before.execution_projection.attempts) == 1
    assert before.execution_projection.attempts[0].attempt_id == attempt_a1
    assert len(after.execution_projection.attempts) == 2
    assert {row.attempt_id for row in after.execution_projection.attempts} == {
        attempt_a1,
        attempt_a2,
    }


@pytest.mark.unit
def test_obs_asof_rebase_later_execution_identity_absent_before_acceptance() -> None:
    """Child execution facts appear only after their accepted position (E-axis via journal prefix)."""
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_execution = mint_execution_id()
    child_execution = mint_execution_id()
    positioned = [
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=root_execution,
                event_type=RuntimeEventType.TASK_CREATED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=root_execution,
                event_type=RuntimeEventType.PLAN_CREATED,
            ),
            tenant_id=_TENANT,
        ),
    ]
    boundary = AsOfBoundary(run_id=run_id, position=positioned[1].position)
    store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=child_execution,
            event_type=RuntimeEventType.STEP_STARTED,
        ),
        tenant_id=_TENANT,
    )
    service = _service(store)
    before = service.reconstruct(
        ExecutionHistoricalReconstructionRequest(
            tenant_id=_TENANT,
            run_id=run_id,
            execution_as_of=boundary,
            knowledge_watermark=_watermark(),
            bitemporal_query=_bitemporal_query(),
        ),
        revision_reader=_EmptyRevisionReader(),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=lambda state, _revision: state,
        initial_state=(),
    )
    execution_ids_before = {
        row.event.execution_id for row in before.execution_reconstruction.positioned_events
    }
    assert execution_ids_before == {root_execution}
    assert child_execution not in execution_ids_before


@pytest.mark.unit
def test_obs_asof_rebase_custom_neutral_runtime_store() -> None:
    doc_store = InMemoryDocumentStore()
    store = DocumentBackedRuntimeEventStore(doc_store)
    run_id = mint_run_id()
    event = sample_runtime_event(tenant_id=_TENANT, run_id=run_id)
    store.append(event, tenant_id=_TENANT)
    service = _service(store)
    result = service.reconstruct(
        _historical_request(run_id=run_id, position=1),
        revision_reader=_EmptyRevisionReader(),
        revision_bitemporal_basis=lambda revision: revision.basis,
        reducer=lambda state, _revision: state,
        initial_state=(),
    )
    assert result.execution_projection.last_included_position == ExecutionEventPosition(1)

