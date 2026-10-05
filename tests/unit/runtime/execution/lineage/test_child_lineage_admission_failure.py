# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.delegation_authority import (
    resolve_root_parent_execution_authority,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_execution_id,
    mint_task_id,
    peek_active_parent_execution_id,
    require_active_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageIntegrityError,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.identity_authority import mint_root_execution_identity
from intergrax.runtime.execution.lineage.active_lineage import (
    peek_attempt_lineage_degradation,
)
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    deactivate_root_execution_lineage,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    peek_active_execution_authority,
    require_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.observability.reconstruction.execution_lineage_reconstruction import (
    ExecutionLineageCompleteness,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)

_TENANT = "tenant-a"


class _ChildAdmissionOutage(InMemoryExecutionLineagePersistence):
    def __init__(self) -> None:
        super().__init__()
        self.admit_child_call_count = 0
        self.mark_degraded_call_count = 0

    def admit_child(self, *args: object, **kwargs: object) -> object:
        self.admit_child_call_count += 1
        raise ExecutionLineageUnavailableError("child admission unavailable")

    def mark_degraded(self, *args: object, **kwargs: object) -> object:
        self.mark_degraded_call_count += 1
        return super().mark_degraded(*args, **kwargs)


class _FlakyChildAdmission(InMemoryExecutionLineagePersistence):
    def __init__(self) -> None:
        super().__init__()
        self.admit_child_call_count = 0

    def admit_child(self, *args: object, **kwargs: object) -> object:
        self.admit_child_call_count += 1
        if self.admit_child_call_count == 1:
            raise ExecutionLineageUnavailableError("first child admission unavailable")
        return super().admit_child(*args, **kwargs)


class _MarkDegradedOutage(_ChildAdmissionOutage):
    def mark_degraded(self, *args: object, **kwargs: object) -> object:
        self.mark_degraded_call_count += 1
        raise ExecutionLineageUnavailableError("mark_degraded unavailable")


def _activate_root_runner(
    persistence: InMemoryExecutionLineagePersistence,
) -> tuple[
    object,
    object,
    object,
    object,
    object,
    ChildExecutionRunner,
    object,
    object,
]:
    identity = mint_root_execution_identity()
    root = identity.execution_id
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=root,
    )
    persistence.admit_root(scope, root, root)
    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=root,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    budget_token = bind_root_execution_budget(execution_id=root, ledger=ledger)
    runner = ChildExecutionRunner(ledger=ledger)
    return (
        scope,
        root,
        lineage_token,
        degradation_token,
        authority_token,
        runner,
        identity_token,
        budget_token,
    )


@pytest.mark.asyncio
async def test_child_admission_unavailable_blocks_delegate_and_marks_degraded() -> None:
    persistence = _ChildAdmissionOutage()
    (
        scope,
        root,
        lineage_token,
        degradation_token,
        authority_token,
        runner,
        identity_token,
        budget_token,
    ) = _activate_root_runner(persistence)

    delegate_called = False

    class _Delegate:
        async def execute(self, _request: object) -> str:
            nonlocal delegate_called
            delegate_called = True
            return "child"

    try:
        with pytest.raises(ExecutionLineageUnavailableError):
            await runner.execute(request=object(), delegate=_Delegate())
        degradation = peek_attempt_lineage_degradation()
        assert degradation is not None
        assert degradation.degraded is True
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    assert delegate_called is False
    assert persistence.admit_child_call_count == 1
    assert persistence.mark_degraded_call_count == 1
    state = persistence.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.degraded is True
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    assert len(page.admissions) == 1
    assert page.admissions[0].execution_id == root


@pytest.mark.asyncio
async def test_mark_degraded_unavailable_still_blocks_delegate() -> None:
    persistence = _MarkDegradedOutage()
    (
        _scope,
        _root,
        lineage_token,
        degradation_token,
        authority_token,
        runner,
        identity_token,
        budget_token,
    ) = _activate_root_runner(persistence)

    delegate_called = False

    class _Delegate:
        async def execute(self, _request: object) -> str:
            nonlocal delegate_called
            delegate_called = True
            return "child"

    try:
        with pytest.raises(ExecutionLineageUnavailableError):
            await runner.execute(request=object(), delegate=_Delegate())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    assert delegate_called is False
    assert persistence.admit_child_call_count == 1
    assert persistence.mark_degraded_call_count == 1


@pytest.mark.asyncio
async def test_sibling_after_failed_child_admission_may_execute_when_durable() -> None:
    persistence = _FlakyChildAdmission()
    (
        scope,
        root,
        lineage_token,
        degradation_token,
        authority_token,
        runner,
        identity_token,
        budget_token,
    ) = _activate_root_runner(persistence)

    first_delegate_called = False
    second_delegate_called = False

    class _FirstDelegate:
        async def execute(self, _request: object) -> str:
            nonlocal first_delegate_called
            first_delegate_called = True
            return "first"

    class _SecondDelegate:
        async def execute(self, _request: object) -> str:
            nonlocal second_delegate_called
            second_delegate_called = True
            return "second"

    try:
        with pytest.raises(ExecutionLineageUnavailableError):
            await runner.execute(request=object(), delegate=_FirstDelegate())
        degradation = peek_attempt_lineage_degradation()
        assert degradation is not None
        assert degradation.degraded is True
        result = await runner.execute(request=object(), delegate=_SecondDelegate())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    assert first_delegate_called is False
    assert second_delegate_called is True
    assert result == "second"
    assert persistence.admit_child_call_count == 2
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    child_records = [item for item in page.admissions if item.parent_execution_id == root]
    assert len(child_records) == 1


@pytest.mark.asyncio
async def test_failed_child_admission_reconstruction_honest() -> None:
    persistence = _ChildAdmissionOutage()
    (
        scope,
        root,
        lineage_token,
        degradation_token,
        authority_token,
        runner,
        identity_token,
        budget_token,
    ) = _activate_root_runner(persistence)

    class _Delegate:
        async def execute(self, _request: object) -> str:
            return "child"

    try:
        with pytest.raises(ExecutionLineageUnavailableError):
            await runner.execute(request=object(), delegate=_Delegate())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(
        PlatformCausalEvidence(
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=_TENANT,
            source=MessageBusTaskRef(provider="celery", task_id="t1", tenant_id=_TENANT),
            target=RuntimeExecutionRef(
                task_id=scope.task_id,
                run_id=scope.run_id,
                attempt_id=scope.attempt_id,
                execution_id=root,
                tenant_id=_TENANT,
            ),
            recorded_at=datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        )
    )
    runtime_store.append(
        sample_runtime_event(
            tenant_id=_TENANT,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
        ),
        tenant_id=_TENANT,
    )
    attempt_lineage = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
        execution_lineage=persistence,
    ).reconstruct_execution(
        _TENANT,
        scope.task_id,
        scope.run_id,
    ).attempts[0].lineage
    assert attempt_lineage.completeness is ExecutionLineageCompleteness.PARTIAL
    child_edges = [
        row
        for row in attempt_lineage.segments[0].admissions
        if row.parent_execution_id == root
    ]
    assert child_edges == []


@pytest.mark.asyncio
async def test_child_budget_released_after_failed_lineage_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = _ChildAdmissionOutage()
    (
        scope,
        _root,
        lineage_token,
        degradation_token,
        authority_token,
        runner,
        identity_token,
        budget_token,
    ) = _activate_root_runner(persistence)
    budget_state = peek_active_execution_budget()
    assert budget_state is not None
    ledger = budget_state.ledger
    child_id = mint_execution_id()
    monkeypatch.setattr(
        "intergrax.runtime.execution.child.mint_child_execution_id",
        lambda: child_id,
    )

    class _Delegate:
        async def execute(self, _request: object) -> str:
            return "child"

    try:
        with pytest.raises(ExecutionLineageUnavailableError):
            await runner.execute(request=object(), delegate=_Delegate())
        snapshot = ledger.export_snapshot(scope.attempt_id)
        child_records = [
            record for record in snapshot.records if record.execution_id == child_id
        ]
        assert len(child_records) == 1
        assert child_records[0].released is True
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)


@pytest.mark.asyncio
async def test_parent_identity_and_authority_restored_after_failed_admission() -> None:
    persistence = _ChildAdmissionOutage()
    (
        _scope,
        root,
        lineage_token,
        degradation_token,
        authority_token,
        runner,
        identity_token,
        budget_token,
    ) = _activate_root_runner(persistence)
    parent_authority = require_active_execution_authority()

    class _Delegate:
        async def execute(self, _request: object) -> str:
            return "child"

    try:
        with pytest.raises(ExecutionLineageUnavailableError):
            await runner.execute(request=object(), delegate=_Delegate())
        assert require_active_execution_id() == root
        assert peek_active_parent_execution_id() is None
        assert peek_active_execution_authority() == parent_authority
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)


@pytest.mark.asyncio
async def test_conflicting_parent_still_blocks_child_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = InMemoryExecutionLineagePersistence()
    identity = mint_root_execution_identity()
    root = identity.execution_id
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    child = mint_execution_id()
    other_parent = mint_execution_id()
    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=root,
    )
    persistence.admit_root(scope, root, root)
    persistence.admit_child(scope, root, other_parent, root)
    persistence.admit_child(scope, root, child, other_parent)
    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=root,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    budget_token = bind_root_execution_budget(execution_id=root, ledger=ledger)
    runner = ChildExecutionRunner(ledger=ledger)
    monkeypatch.setattr(
        "intergrax.runtime.execution.child.mint_child_execution_id",
        lambda: child,
    )

    delegate_called = False

    class _Delegate:
        async def execute(self, _request: object) -> str:
            nonlocal delegate_called
            delegate_called = True
            return "child"

    try:
        with pytest.raises(ExecutionLineageIntegrityError):
            await runner.execute(request=object(), delegate=_Delegate())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    assert delegate_called is False
