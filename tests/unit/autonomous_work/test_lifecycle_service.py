# © Artur Czarnecki. All rights reserved.

"""AW-2B — Authoritative worker lifecycle transition service tests."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.autonomous_work.lifecycle import (
    AutonomousWorkInvalidLifecycleTransition,
    AutonomousWorkLifecycleClockError,
    AutonomousWorkLifecycleStateConflict,
    WorkerLifecycleService,
    WorkerLifecycleTransitionPolicy,
    WorkerLifecycleTransitionRequest,
)
from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityNotFound,
    AutonomousWorkRepositoryCapabilities,
    AutonomousWorkRevisionConflict,
    WorkerInstanceRepository,
)
from intergrax.contracts.autonomous_work import (
    DefinitionRevision,
    Revision,
    WorkerInstance,
    WorkerLifecycleState,
    initial_revision,
    mint_worker_definition_id,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.references import (
    PrincipalBindingRef,
    WorkspaceContextRef,
)

pytestmark = pytest.mark.unit

_UTC = timezone.utc
_LIFECYCLE_MODULE = Path("intergrax/autonomous_work/lifecycle.py")

_CANONICAL_ALLOWED_TRANSITIONS: dict[
    WorkerLifecycleState, frozenset[WorkerLifecycleState]
] = {
    WorkerLifecycleState.PROVISIONING: frozenset(
        {
            WorkerLifecycleState.ACTIVE,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.ACTIVE: frozenset(
        {
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.IDLE: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.WORKING: frozenset(
        {
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.WAITING_EXTERNAL: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.WAITING_FOR_HUMAN: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.RECOVERING: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.DEGRADED: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.PAUSED: frozenset(
        {
            WorkerLifecycleState.ACTIVE,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.QUARANTINED: frozenset(
        {
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.STOPPED: frozenset(),
}


@dataclass
class _MutableClock:
    current: datetime

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


def _worker_instance(**overrides: object) -> WorkerInstance:
    now = datetime(2026, 9, 2, 12, 0, tzinfo=_UTC)
    payload = {
        "worker_instance_id": mint_worker_instance_id(),
        "worker_definition_id": mint_worker_definition_id(),
        "definition_revision": DefinitionRevision(1),
        "lifecycle_state": WorkerLifecycleState.PROVISIONING,
        "principal_binding_ref": PrincipalBindingRef("binding/order-ops-1"),
        "workspace_context_ref": WorkspaceContextRef("workspace/order-ops"),
        "active_responsibility_refs": (),
        "active_goal_refs": (),
        "created_at": now,
        "updated_at": now,
        "revision": initial_revision(),
    }
    payload.update(overrides)
    return WorkerInstance(**payload)


def _request(
    worker: WorkerInstance,
    *,
    target_state: WorkerLifecycleState,
    reason: str = "test-transition",
    requested_at: datetime | None = None,
) -> WorkerLifecycleTransitionRequest:
    return WorkerLifecycleTransitionRequest(
        worker_instance_id=worker.worker_instance_id,
        expected_revision=worker.revision,
        expected_state=worker.lifecycle_state,
        target_state=target_state,
        transition_reason=reason,
        requested_at=requested_at or worker.updated_at,
    )


def _service(
    repository: InMemoryWorkerInstanceRepository | None = None,
    *,
    clock: _MutableClock | Callable[[], datetime] | None = None,
) -> tuple[WorkerLifecycleService, InMemoryWorkerInstanceRepository, _MutableClock]:
    repo = repository or InMemoryWorkerInstanceRepository()
    mutable_clock = (
        clock if isinstance(clock, _MutableClock) else _MutableClock(datetime(2026, 9, 2, 12, 0, tzinfo=_UTC))
    )
    service = WorkerLifecycleService(repository=repo, clock=mutable_clock)
    return service, repo, mutable_clock


class _CustomWorkerInstanceRepository:
    """Structural typing adapter for pluginability gate."""

    def __init__(self, delegate: InMemoryWorkerInstanceRepository) -> None:
        self._delegate = delegate
        self.replace_calls = 0

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return self._delegate.capabilities

    def create(self, instance: WorkerInstance) -> WorkerInstance:
        return self._delegate.create(instance)

    def get(self, *, worker_instance_id: WorkerInstanceId) -> WorkerInstance | None:
        return self._delegate.get(worker_instance_id=worker_instance_id)

    def replace(
        self,
        instance: WorkerInstance,
        *,
        expected_revision: Revision,
    ) -> WorkerInstance:
        self.replace_calls += 1
        return self._delegate.replace(instance, expected_revision=expected_revision)


def test_transition_policy_covers_all_states() -> None:
    policy = WorkerLifecycleTransitionPolicy()
    for state in WorkerLifecycleState:
        assert isinstance(policy.allowed_targets(state), frozenset)


def test_transition_matrix_matches_canonical_allow_list() -> None:
    policy = WorkerLifecycleTransitionPolicy()
    for from_state in WorkerLifecycleState:
        expected_targets = _CANONICAL_ALLOWED_TRANSITIONS[from_state]
        assert policy.allowed_targets(from_state) == expected_targets
        for to_state in WorkerLifecycleState:
            allowed = policy.can_transition(from_state, to_state)
            if from_state == to_state:
                assert allowed is False
            else:
                assert allowed is (to_state in expected_targets)


def test_stopped_is_terminal() -> None:
    policy = WorkerLifecycleTransitionPolicy()
    for to_state in WorkerLifecycleState:
        assert not policy.can_transition(WorkerLifecycleState.STOPPED, to_state)


def test_quarantined_fail_closed_without_direct_working_or_active_release() -> None:
    policy = WorkerLifecycleTransitionPolicy()
    assert not policy.can_transition(
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.WORKING,
    )
    assert not policy.can_transition(
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.ACTIVE,
    )
    assert policy.can_transition(
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.PAUSED,
    )


def test_valid_transition_preserves_identity_and_advances_revision() -> None:
    service, repo, clock = _service()
    created = repo.create(_worker_instance())
    clock.advance(timedelta(minutes=5))

    result = service.transition(
        _request(created, target_state=WorkerLifecycleState.ACTIVE),
    )

    assert result.changed is True
    assert result.previous_state is WorkerLifecycleState.PROVISIONING
    assert result.current_state is WorkerLifecycleState.ACTIVE
    persisted = result.worker_instance
    assert persisted.worker_instance_id == created.worker_instance_id
    assert persisted.worker_definition_id == created.worker_definition_id
    assert persisted.definition_revision == created.definition_revision
    assert persisted.principal_binding_ref == created.principal_binding_ref
    assert persisted.workspace_context_ref == created.workspace_context_ref
    assert persisted.active_responsibility_refs == created.active_responsibility_refs
    assert persisted.active_goal_refs == created.active_goal_refs
    assert persisted.created_at == created.created_at
    assert persisted.updated_at == clock.current
    assert persisted.revision == Revision(created.revision.value + 1)


def test_invalid_transition_is_deterministic_and_leaves_repository_unchanged() -> None:
    service, repo, _clock = _service()
    created = repo.create(
        _worker_instance(lifecycle_state=WorkerLifecycleState.ACTIVE),
    )

    with pytest.raises(AutonomousWorkInvalidLifecycleTransition) as exc_info:
        service.transition(
            _request(created, target_state=WorkerLifecycleState.PROVISIONING),
        )

    error = exc_info.value
    assert error.worker_instance_id == created.worker_instance_id
    assert error.from_state is WorkerLifecycleState.ACTIVE
    assert error.to_state is WorkerLifecycleState.PROVISIONING
    unchanged = repo.get(worker_instance_id=created.worker_instance_id)
    assert unchanged is not None
    assert unchanged == created


def test_same_state_request_is_no_op_without_revision_churn() -> None:
    service, repo, clock = _service()
    created = repo.create(
        _worker_instance(lifecycle_state=WorkerLifecycleState.IDLE),
    )
    custom_repo = _CustomWorkerInstanceRepository(repo)
    service = WorkerLifecycleService(repository=custom_repo, clock=clock)

    result = service.transition(
        _request(created, target_state=WorkerLifecycleState.IDLE),
    )

    assert result.changed is False
    assert result.worker_instance == created
    assert custom_repo.replace_calls == 0


def test_stale_revision_conflict_is_propagated() -> None:
    service, repo, clock = _service()
    created = repo.create(_worker_instance())
    clock.advance(timedelta(minutes=1))
    first = service.transition(
        _request(created, target_state=WorkerLifecycleState.ACTIVE),
    )
    clock.advance(timedelta(minutes=1))

    with pytest.raises(AutonomousWorkRevisionConflict):
        service.transition(
            WorkerLifecycleTransitionRequest(
                worker_instance_id=created.worker_instance_id,
                expected_revision=created.revision,
                expected_state=WorkerLifecycleState.ACTIVE,
                target_state=WorkerLifecycleState.IDLE,
                transition_reason="stale-write",
                requested_at=clock.current,
            ),
        )

    persisted = repo.get(worker_instance_id=created.worker_instance_id)
    assert persisted == first.worker_instance


def test_expected_state_mismatch_is_rejected_without_mutation() -> None:
    service, repo, _clock = _service()
    created = repo.create(
        _worker_instance(lifecycle_state=WorkerLifecycleState.IDLE),
    )

    with pytest.raises(AutonomousWorkLifecycleStateConflict) as exc_info:
        service.transition(
            WorkerLifecycleTransitionRequest(
                worker_instance_id=created.worker_instance_id,
                expected_revision=created.revision,
                expected_state=WorkerLifecycleState.WORKING,
                target_state=WorkerLifecycleState.PAUSED,
                transition_reason="wrong-assumption",
                requested_at=created.updated_at,
            ),
        )

    error = exc_info.value
    assert error.expected_state is WorkerLifecycleState.WORKING
    assert error.actual_state is WorkerLifecycleState.IDLE
    assert repo.get(worker_instance_id=created.worker_instance_id) == created


def test_rehydration_preserves_persisted_lifecycle_after_service_reconstruction() -> None:
    repo = InMemoryWorkerInstanceRepository()
    clock = _MutableClock(datetime(2026, 9, 2, 12, 0, tzinfo=_UTC))
    first_service = WorkerLifecycleService(repository=repo, clock=clock)
    created = repo.create(_worker_instance())

    clock.advance(timedelta(minutes=1))
    active = first_service.transition(
        _request(created, target_state=WorkerLifecycleState.ACTIVE),
    ).worker_instance
    clock.advance(timedelta(minutes=1))
    waiting = first_service.transition(
        _request(active, target_state=WorkerLifecycleState.WAITING_EXTERNAL),
    ).worker_instance

    second_service = WorkerLifecycleService(repository=repo, clock=clock)
    rehydrated = second_service.get_current(
        worker_instance_id=waiting.worker_instance_id,
    )

    assert rehydrated.lifecycle_state is WorkerLifecycleState.WAITING_EXTERNAL
    assert rehydrated.revision == waiting.revision
    assert rehydrated.worker_instance_id == waiting.worker_instance_id


def test_backward_clock_is_rejected() -> None:
    service, repo, clock = _service()
    created = repo.create(_worker_instance())
    clock.advance(timedelta(minutes=5))

    clock.current = created.updated_at - timedelta(seconds=1)
    with pytest.raises(AutonomousWorkLifecycleClockError):
        service.transition(
            _request(created, target_state=WorkerLifecycleState.ACTIVE),
        )


def test_get_current_raises_when_worker_missing() -> None:
    service, _repo, _clock = _service()
    worker_id = mint_worker_instance_id()
    with pytest.raises(AutonomousWorkEntityNotFound):
        service.get_current(worker_instance_id=worker_id)


def test_custom_repository_is_injectable_without_concrete_adapter_dependency() -> None:
    repo = InMemoryWorkerInstanceRepository()
    custom_repo = _CustomWorkerInstanceRepository(repo)
    clock = _MutableClock(datetime(2026, 9, 2, 12, 0, tzinfo=_UTC))
    service = WorkerLifecycleService(repository=custom_repo, clock=clock)
    created = repo.create(_worker_instance())
    clock.advance(timedelta(minutes=1))

    result = service.transition(
        _request(created, target_state=WorkerLifecycleState.ACTIVE),
    )

    assert result.current_state is WorkerLifecycleState.ACTIVE
    assert custom_repo.replace_calls == 1
    assert isinstance(custom_repo, WorkerInstanceRepository)


def test_lifecycle_service_architecture_gates() -> None:
    source = _LIFECYCLE_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "intergrax.autonomous_work.in_memory_repository" not in imported_modules
    assert not any(module.startswith("intergrax.runtime") for module in imported_modules)
    assert not any(module.startswith("agents") for module in imported_modules)

    lifecycle_params = inspect.signature(WorkerLifecycleService.__init__).parameters
    repository_annotation = lifecycle_params["repository"].annotation
    assert "WorkerInstanceRepository" in str(repository_annotation)

    public_names = (
        "WorkerLifecycleService",
        "WorkerLifecycleTransitionPolicy",
        "AutonomousWorkInvalidLifecycleTransition",
        "AutonomousWorkLifecycleStateConflict",
        "AutonomousWorkLifecycleClockError",
        "WorkerLifecycleTransitionRequest",
        "WorkerLifecycleTransitionResult",
    )
    import intergrax.autonomous_work.lifecycle as lifecycle_module

    for name in public_names:
        assert hasattr(lifecycle_module, name)

    lifecycle_state_source = Path(
        "intergrax/contracts/autonomous_work/lifecycle.py"
    ).read_text(encoding="utf-8")
    assert "class WorkerLifecycleState" in lifecycle_state_source
    assert "WorkerLifecycleState" in source
    assert source.count("class WorkerLifecycleState") == 0
