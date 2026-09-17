# © Artur Czarnecki. All rights reserved.

"""OBS-ASOF-REBASE-R1 — E-scoped lineage reconstruction integrity."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_event_position import AsOfBoundary, ExecutionEventPosition
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAsOfReader,
    ExecutionLineageAttemptScope,
    ExecutionLineageReader,
    build_execution_lineage_run_scope,
)
from intergrax.runtime.events.execution_position import as_of_boundary_for_positioned
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import (
    ExecutionAttemptDiscoveryCompleteness,
    ExecutionAttemptDiscoveryReadStatus,
    ExecutionLineageReadStatus,
    ExecutionReconstructor,
)
from testing_support.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt
from tests.unit.runtime.events.test_asof_projection import _append_sequence

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_REPO_ROOT = Path(__file__).resolve().parents[5]
_RECONSTRUCTION_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "reconstruction"
    / "execution_reconstruction.py"
)


def _reconstructor(
    *,
    store: InMemoryRuntimeEventStore,
    lineage: InMemoryExecutionLineagePersistence | None = None,
    lineage_as_of: ExecutionLineageAsOfReader | None = None,
) -> ExecutionReconstructor:
    return ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
        execution_lineage=lineage,
        execution_lineage_as_of=lineage_as_of,
    )


def _boundary_at(positioned, index: int) -> AsOfBoundary:
    return as_of_boundary_for_positioned(positioned[index])


def test_future_attempt_discovery_does_not_leak_at_execution_boundary() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.PLAN_CREATED,
        ],
    )
    boundary = _boundary_at(positioned, 1)
    lineage = InMemoryExecutionLineagePersistence()
    scope_a1 = ExecutionLineageAttemptScope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    register_v1_attempt(lineage, scope_a1)
    reconstructor = _reconstructor(store=store, lineage=lineage)

    before = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    run_scope = build_execution_lineage_run_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
    )
    lineage.register_attempt_for_run(run_scope, attempt_a2)
    lineage.open_attempt(
        ExecutionLineageAttemptScope(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a2,
        ),
        discovery_contract_version=1,
    )

    after = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    assert before.attempts == after.attempts
    assert {a.attempt_id for a in after.attempts} == {attempt_a1}
    assert after.attempt_discovery_read_status is (
        ExecutionAttemptDiscoveryReadStatus.NOT_APPLICABLE
    )
    assert after.attempt_discovery_completeness is (
        ExecutionAttemptDiscoveryCompleteness.NOT_APPLICABLE
    )


def test_future_child_lineage_does_not_alter_historical_attempt_at_boundary() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.PLAN_CREATED,
        ],
    )
    boundary = _boundary_at(positioned, 1)
    lineage = InMemoryExecutionLineagePersistence()
    scope = ExecutionLineageAttemptScope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    register_v1_attempt(lineage, scope)
    root, child = mint_execution_id(), mint_execution_id()
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)
    reconstructor = _reconstructor(store=store, lineage=lineage)

    before = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    lineage.admit_child(scope, root, child, root)
    after = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    assert before == after
    assert before.attempts[0].lineage is None


def test_append_immunity_full_reconstruction_with_lineage_configured() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.PLAN_CREATED,
        ],
    )
    boundary = _boundary_at(positioned, 1)
    lineage = InMemoryExecutionLineagePersistence()
    register_v1_attempt(
        lineage,
        ExecutionLineageAttemptScope(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a1,
        ),
    )
    reconstructor = _reconstructor(store=store, lineage=lineage)
    first = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )

    attempt_a2 = mint_attempt_id()
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        event_types=[RuntimeEventType.PLAN_CREATED],
    )
    run_scope = build_execution_lineage_run_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
    )
    lineage.register_attempt_for_run(run_scope, attempt_a2)
    lineage.open_attempt(
        ExecutionLineageAttemptScope(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a2,
        ),
        discovery_contract_version=1,
    )

    second = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    assert first == second


def test_retry_attempt_lineage_later_does_not_leak_before_boundary() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.PLAN_CREATED,
            RuntimeEventType.TASK_FAILED,
        ],
    )
    boundary = _boundary_at(positioned, 2)
    lineage = InMemoryExecutionLineagePersistence()
    register_v1_attempt(
        lineage,
        ExecutionLineageAttemptScope(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a1,
        ),
    )
    reconstructor = _reconstructor(store=store, lineage=lineage)
    at_boundary = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        event_types=[RuntimeEventType.PLAN_CREATED],
    )
    run_scope = build_execution_lineage_run_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
    )
    lineage.register_attempt_for_run(run_scope, attempt_a2)
    lineage.open_attempt(
        ExecutionLineageAttemptScope(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a2,
        ),
        discovery_contract_version=1,
    )
    after_retry = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    assert at_boundary == after_retry
    assert {a.attempt_id for a in after_retry.attempts} == {attempt_a1}


def test_current_reconstruction_still_uses_lineage() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=[RuntimeEventType.TASK_CREATED, RuntimeEventType.PLAN_CREATED],
    )
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        event_types=[RuntimeEventType.PLAN_CREATED],
    )
    lineage = InMemoryExecutionLineagePersistence()
    for attempt in (attempt_a1, attempt_a2):
        register_v1_attempt(
            lineage,
            ExecutionLineageAttemptScope(
                tenant_id=_TENANT,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt,
            ),
        )
    reconstructor = _reconstructor(store=store, lineage=lineage)
    current = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert {a.attempt_id for a in current.attempts} == {attempt_a1, attempt_a2}
    assert current.attempt_discovery_read_status is (
        ExecutionAttemptDiscoveryReadStatus.AVAILABLE
    )
    assert any(
        a.lineage is not None
        and a.lineage.read_status is ExecutionLineageReadStatus.AVAILABLE
        for a in current.attempts
    )


class _PassthroughLineageAsOfReader(ExecutionLineageAsOfReader):
    def __init__(self, persistence: InMemoryExecutionLineagePersistence) -> None:
        self._persistence = persistence

    def reader_at_execution_boundary(
        self,
        boundary: AsOfBoundary,
    ) -> ExecutionLineageReader:
        return self._persistence


def test_execution_lineage_as_of_reader_enriches_historical_when_injected() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.PLAN_CREATED,
        ],
    )
    boundary = _boundary_at(positioned, 1)
    lineage = InMemoryExecutionLineagePersistence()
    scope = ExecutionLineageAttemptScope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    register_v1_attempt(lineage, scope)
    root = mint_execution_id()
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)
    as_of_reader = _PassthroughLineageAsOfReader(lineage)
    reconstructor = _reconstructor(
        store=store,
        lineage=lineage,
        lineage_as_of=as_of_reader,
    )
    historical = reconstructor.reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    assert {a.attempt_id for a in historical.attempts} == {attempt_a1}
    assert historical.attempt_discovery_read_status is (
        ExecutionAttemptDiscoveryReadStatus.AVAILABLE
    )
    assert historical.attempts[0].lineage is not None
    assert historical.attempts[0].lineage.read_status is ExecutionLineageReadStatus.AVAILABLE


def test_architecture_gate_no_current_lineage_load_when_as_of_without_capability() -> None:
    text = _RECONSTRUCTION_MODULE.read_text(encoding="utf-8")
    assert "_resolve_lineage_enrichment_for_reconstruction" in text
    assert "_run_discovery_snapshot_not_applicable_at_execution_boundary" in text
    tree = ast.parse(text)
    forbidden_compare = "discovery_position" in text and "<= boundary.position" in text
    assert not forbidden_compare


def test_architecture_gate_no_timestamp_lineage_filter_in_lineage_resolution() -> None:
    text = _RECONSTRUCTION_MODULE.read_text(encoding="utf-8")
    start = text.index("def _resolve_lineage_enrichment_for_reconstruction")
    end = text.index("def _load_run_discovery_snapshot", start)
    block = text[start:end]
    assert "datetime" not in block
    assert "created_at" not in block
    assert "recorded_at" not in block
