# © Artur Czarnecki. All rights reserved.

"""AW-4A — Worker wake-up admission service tests."""

from __future__ import annotations

import ast
import importlib
import threading
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerInstanceRepository,
    InMemoryWorkerWakeUpReceiptRepository,
)
from intergrax.autonomous_work.repository import AutonomousWorkEntityNotFound
from intergrax.autonomous_work.wake_up_service import (
    WorkerWakeUpEligibilityPolicy,
    WorkerWakeUpService,
)
from intergrax.contracts.autonomous_work import (
    WakeUpId,
    WorkerLifecycleState,
    WorkerWakeUpDisposition,
    WorkerWakeUpSignal,
    WorkerWakeUpSourceKind,
    mint_wake_up_id,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import WakeUpSourceRef
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = timezone.utc
_NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _FixedClock:
    now_value: datetime

    def now(self) -> datetime:
        return self.now_value


def _service(
    *,
    worker_repo: InMemoryWorkerInstanceRepository | None = None,
    continuity_repo: InMemoryWorkContinuityStateRepository | None = None,
    receipt_repo: InMemoryWorkerWakeUpReceiptRepository | None = None,
    now: datetime = _NOW,
) -> tuple[
    WorkerWakeUpService,
    InMemoryWorkerInstanceRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerWakeUpReceiptRepository,
]:
    worker_repo = worker_repo or InMemoryWorkerInstanceRepository()
    continuity_repo = continuity_repo or InMemoryWorkContinuityStateRepository()
    receipt_repo = receipt_repo or InMemoryWorkerWakeUpReceiptRepository()
    service = WorkerWakeUpService(
        worker_instance_repository=worker_repo,
        continuity_state_repository=continuity_repo,
        wake_up_receipt_repository=receipt_repo,
        clock=_FixedClock(now),
    )
    return service, worker_repo, continuity_repo, receipt_repo


def _seed_worker(
    worker_repo: InMemoryWorkerInstanceRepository,
    *,
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.IDLE,
) -> str:
    instance = contract_suite.worker_instance(lifecycle_state=lifecycle_state)
    worker_repo.create(instance)
    return instance.worker_instance_id


def _signal(
    *,
    worker_instance_id: str,
    wake_up_id: WakeUpId | None = None,
    source_kind: WorkerWakeUpSourceKind = WorkerWakeUpSourceKind.EXTERNAL_EVENT,
) -> WorkerWakeUpSignal:
    wake_id = wake_up_id or mint_wake_up_id()
    return WorkerWakeUpSignal(
        wake_up_id=wake_id,
        worker_instance_id=worker_instance_id,
        source_kind=source_kind,
        source_ref=WakeUpSourceRef("source/external/event-1"),
        occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
        delivery_identity=wake_id,
        correlation_ref=None,
    )


def test_idle_worker_accepts_wake_up_and_restores_continuity() -> None:
    service, worker_repo, continuity_repo, _ = _service()
    worker_id = _seed_worker(worker_repo, lifecycle_state=WorkerLifecycleState.IDLE)
    continuity = contract_suite.continuity_state(worker_instance_ref=worker_id)
    continuity_repo.create(continuity)
    result = service.accept(_signal(worker_instance_id=worker_id))
    assert result.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert result.context is not None
    assert result.context.continuity_state == continuity
    assert result.context.worker_instance.lifecycle_state == WorkerLifecycleState.IDLE


def test_duplicate_delivery_returns_duplicate_with_same_receipt() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(worker_repo)
    wake_id = mint_wake_up_id()
    signal = _signal(worker_instance_id=worker_id, wake_up_id=wake_id)
    first = service.accept(signal)
    second = service.accept(signal)
    assert first.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert second.disposition == WorkerWakeUpDisposition.DUPLICATE
    assert second.context is not None
    assert second.context.receipt == first.context.receipt
    assert second.context.wake_up_signal == first.context.wake_up_signal


def test_conflicting_replay_returns_conflict_with_canonical_context() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(worker_repo)
    wake_id = mint_wake_up_id()
    first = service.accept(
        WorkerWakeUpSignal(
            wake_up_id=wake_id,
            worker_instance_id=worker_id,
            source_kind=WorkerWakeUpSourceKind.QUEUE_DELIVERY,
            source_ref=WakeUpSourceRef("queue/order-123"),
            occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
            delivery_identity=mint_wake_up_id(),
            correlation_ref=None,
        )
    )
    second = service.accept(
        WorkerWakeUpSignal(
            wake_up_id=wake_id,
            worker_instance_id=worker_id,
            source_kind=WorkerWakeUpSourceKind.OPERATOR,
            source_ref=WakeUpSourceRef("operator/manual"),
            occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
            delivery_identity=mint_wake_up_id(),
            correlation_ref=None,
        )
    )
    assert first.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert second.disposition == WorkerWakeUpDisposition.CONFLICT
    assert second.context is not None
    assert second.context.receipt == first.context.receipt
    assert second.context.wake_up_signal == first.context.wake_up_signal


def test_transport_redelivery_with_different_delivery_identity_is_duplicate() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(worker_repo)
    wake_id = mint_wake_up_id()
    first = service.accept(
        WorkerWakeUpSignal(
            wake_up_id=wake_id,
            worker_instance_id=worker_id,
            source_kind=WorkerWakeUpSourceKind.QUEUE_DELIVERY,
            source_ref=WakeUpSourceRef("queue/order-123"),
            occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
            delivery_identity=mint_wake_up_id(),
            correlation_ref=None,
        )
    )
    second = service.accept(
        WorkerWakeUpSignal(
            wake_up_id=wake_id,
            worker_instance_id=worker_id,
            source_kind=WorkerWakeUpSourceKind.QUEUE_DELIVERY,
            source_ref=WakeUpSourceRef("queue/order-123"),
            occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
            delivery_identity=mint_wake_up_id(),
            correlation_ref=None,
        )
    )
    assert first.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert second.disposition == WorkerWakeUpDisposition.DUPLICATE


def test_concurrent_duplicate_exactly_one_accepted() -> None:
    _, worker_repo, _, receipt_repo = _service()
    worker_id = _seed_worker(worker_repo)
    wake_id = mint_wake_up_id()
    signal = _signal(worker_instance_id=worker_id, wake_up_id=wake_id)
    dispositions: list[WorkerWakeUpDisposition] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        local_service, _, _, _ = _service(
            worker_repo=worker_repo,
            receipt_repo=receipt_repo,
        )
        barrier.wait(timeout=5)
        dispositions.append(local_service.accept(signal).disposition)

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert dispositions.count(WorkerWakeUpDisposition.ACCEPTED) == 1
    assert dispositions.count(WorkerWakeUpDisposition.DUPLICATE) == 1


def test_distinct_wake_up_ids_both_accepted() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(worker_repo)
    first = service.accept(_signal(worker_instance_id=worker_id, wake_up_id=mint_wake_up_id()))
    second = service.accept(_signal(worker_instance_id=worker_id, wake_up_id=mint_wake_up_id()))
    assert first.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert second.disposition == WorkerWakeUpDisposition.ACCEPTED


def test_cross_worker_isolation_same_wake_up_id() -> None:
    service, worker_repo, _, _ = _service()
    worker_a = _seed_worker(worker_repo)
    worker_b = _seed_worker(worker_repo)
    wake_id = mint_wake_up_id()
    assert service.accept(_signal(worker_instance_id=worker_a, wake_up_id=wake_id)).disposition == (
        WorkerWakeUpDisposition.ACCEPTED
    )
    assert service.accept(_signal(worker_instance_id=worker_b, wake_up_id=wake_id)).disposition == (
        WorkerWakeUpDisposition.ACCEPTED
    )


def test_paused_worker_rejects_ordinary_wake_up() -> None:
    service, worker_repo, _, receipt_repo = _service()
    worker_id = _seed_worker(worker_repo, lifecycle_state=WorkerLifecycleState.PAUSED)
    signal = _signal(worker_instance_id=worker_id)
    result = service.accept(signal)
    assert result.disposition == WorkerWakeUpDisposition.NOT_ELIGIBLE
    assert receipt_repo.get(
        worker_instance_id=worker_id,
        wake_up_id=signal.wake_up_id,
    ) is None


def test_quarantined_worker_rejects_wake_up() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(worker_repo, lifecycle_state=WorkerLifecycleState.QUARANTINED)
    result = service.accept(_signal(worker_instance_id=worker_id))
    assert result.disposition == WorkerWakeUpDisposition.REJECTED


def test_stopped_worker_rejects_wake_up() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(worker_repo, lifecycle_state=WorkerLifecycleState.STOPPED)
    result = service.accept(_signal(worker_instance_id=worker_id))
    assert result.disposition == WorkerWakeUpDisposition.REJECTED


def test_waiting_external_accepts_dependency_recovery_only() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(
        worker_repo,
        lifecycle_state=WorkerLifecycleState.WAITING_EXTERNAL,
    )
    accepted = service.accept(
        _signal(
            worker_instance_id=worker_id,
            source_kind=WorkerWakeUpSourceKind.DEPENDENCY_RECOVERY,
        )
    )
    rejected = service.accept(_signal(worker_instance_id=worker_id))
    assert accepted.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert rejected.disposition == WorkerWakeUpDisposition.NOT_ELIGIBLE


def test_waiting_for_human_accepts_human_continuation_only() -> None:
    service, worker_repo, _, _ = _service()
    worker_id = _seed_worker(
        worker_repo,
        lifecycle_state=WorkerLifecycleState.WAITING_FOR_HUMAN,
    )
    accepted = service.accept(
        _signal(
            worker_instance_id=worker_id,
            source_kind=WorkerWakeUpSourceKind.HUMAN_CONTINUATION,
        )
    )
    rejected = service.accept(_signal(worker_instance_id=worker_id))
    assert accepted.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert rejected.disposition == WorkerWakeUpDisposition.NOT_ELIGIBLE


def test_missing_worker_raises_not_found() -> None:
    service, _, _, _ = _service()
    with pytest.raises(AutonomousWorkEntityNotFound):
        service.accept(_signal(worker_instance_id=mint_worker_instance_id()))


@pytest.mark.parametrize(
    ("invalid_wake_up_id", "message"),
    [
        ("bad-id", "WakeUpId must start with 'wkup_'"),
        ("wkup_not32hexsuffix", "WakeUpId suffix must match"),
    ],
)
def test_malformed_wake_up_id_rejected(invalid_wake_up_id: str, message: str) -> None:
    worker_id = mint_worker_instance_id()
    with pytest.raises(ValueError, match=message):
        _signal(worker_instance_id=worker_id, wake_up_id=WakeUpId(invalid_wake_up_id))


def test_wake_up_signal_has_no_authority_fields() -> None:
    field_names = {field.name for field in fields(WorkerWakeUpSignal)}
    forbidden = {"permissions", "authority", "principal", "role", "credentials", "metadata"}
    assert forbidden.isdisjoint(field_names)


def test_wake_up_service_has_no_llm_or_execution_imports() -> None:
    module = importlib.import_module("intergrax.autonomous_work.wake_up_service")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported).lower()
    forbidden = (
        "openai",
        "anthropic",
        "langchain",
        "runtime.events",
        "runtime.task",
        "agents.",
        "execution_authority_admission",
        "principal_binding_resolver",
    )
    for token in forbidden:
        assert token not in joined


def test_wake_up_service_has_no_provider_imports() -> None:
    module = importlib.import_module("intergrax.autonomous_work.wake_up_service")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "postgresql" not in alias.name
                assert "in_memory" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert "postgresql" not in node.module
            assert "in_memory_repository" not in node.module


def test_wake_up_service_does_not_mutate_lifecycle() -> None:
    module = importlib.import_module("intergrax.autonomous_work.wake_up_service")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "WorkerLifecycleService" not in source
    assert ".transition(" not in source


def test_wake_up_contracts_do_not_import_runtime_services() -> None:
    module = importlib.import_module("intergrax.contracts.autonomous_work.wake_up")
    assert module.__file__ is not None
    with open(module.__file__, encoding="utf-8") as handle:
        source = handle.read()
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("from ") and not stripped.startswith("import "):
            continue
        assert "intergrax.runtime" not in stripped
        assert "agents." not in stripped


def test_eligibility_policy_matrix() -> None:
    policy = WorkerWakeUpEligibilityPolicy()
    assert policy.evaluate(
        lifecycle_state=WorkerLifecycleState.IDLE,
        source_kind=WorkerWakeUpSourceKind.SCHEDULE,
    ).eligible
    assert not policy.evaluate(
        lifecycle_state=WorkerLifecycleState.PAUSED,
        source_kind=WorkerWakeUpSourceKind.OPERATOR,
    ).eligible
    assert policy.evaluate(
        lifecycle_state=WorkerLifecycleState.WAITING_EXTERNAL,
        source_kind=WorkerWakeUpSourceKind.RECOVERY_TIMER,
    ).eligible
