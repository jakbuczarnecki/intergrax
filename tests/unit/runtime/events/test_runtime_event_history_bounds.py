# © Artur Czarnecki. All rights reserved.

"""OBS-RUNTIME-HISTORY-BOUNDS: process-local RuntimeEventBus history tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_history import (
    DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY,
    RuntimeEventHistoryPolicy,
    RuntimeEventHistoryRetention,
)
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event_history import (
    BoundedRuntimeEventHistory,
    PlatformOwnedRuntimeEventHistoryBuffer,
)
from intergrax.runtime.nexus.orchestration.task_finisher import build_nexus_task_result
from intergrax.runtime.nexus.response.final_response_composer import (
    FinalResponseComposer,
)
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.contracts.execution_identity import mint_attempt_id
from testing_support.builder import build_task_for_tests, canonical_run_id_for_tests
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.observability.event_delivery.in_memory_sink import (
    InMemoryEventSink,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_EVENT_BUS_PATH = _REPO / "intergrax" / "runtime" / "events" / "event_bus.py"
_TASK_FINISHER_PATH = (
    _REPO / "intergrax" / "runtime" / "nexus" / "orchestration" / "task_finisher.py"
)
_NEXUS_LOOP_PATH = _REPO / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_HISTORY_IMPL_PATH = (
    _REPO / "intergrax" / "runtime" / "events" / "runtime_event_history.py"
)


def _event(
    *,
    label: str,
    run_id: str | None = None,
    task_id: str | None = None,
) -> RuntimeEvent:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
    )

    rid = run_id or mint_run_id()
    tid = task_id or mint_task_id()
    return RuntimeEvent.model_validate(
        {
            "tenant_id": "tenant-a",
            "task_id": tid,
            "run_id": rid,
            "attempt_id": mint_attempt_id(),
            "execution_id": mint_execution_id(),
            "event_id": mint_event_id(),
            "event_type": RuntimeEventType.STEP_STARTED,
            "phase": ExecutionPhase.STEP_EXECUTION,
            "payload": {"label": label},
        },
    )


class RecordingHistoryStrategy:
    __slots__ = ("windows",)

    def __init__(self) -> None:
        self.windows: list[tuple[RuntimeEvent, ...]] = []

    def on_history_window(self, window: tuple[RuntimeEvent, ...]) -> None:
        self.windows.append(window)


class HoardingHistoryStrategy:
    __slots__ = ("_vault",)

    def __init__(self) -> None:
        self._vault: list[RuntimeEvent] = []

    def on_history_window(self, window: tuple[RuntimeEvent, ...]) -> None:
        self._vault.extend(window)

    @property
    def vault_size(self) -> int:
        return len(self._vault)


class CustomBoundedHistory:
    """Plugin buffer without subclassing default implementations."""

    __slots__ = ("_capacity", "_items")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: list[RuntimeEvent] = []

    def retention(self) -> RuntimeEventHistoryRetention:
        return RuntimeEventHistoryRetention(mode="bounded", capacity=self._capacity)

    def append(self, event: RuntimeEvent) -> None:
        if len(self._items) >= self._capacity:
            self._items.pop(0)
        self._items.append(event)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()


class LyingBoundedHistory:
    """Declares bounded retention but keeps a private vault (composition probe passes)."""

    __slots__ = ("_capacity", "_vault")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._vault: list[RuntimeEvent] = []

    def retention(self) -> RuntimeEventHistoryRetention:
        return RuntimeEventHistoryRetention(mode="bounded", capacity=self._capacity)

    def append(self, event: RuntimeEvent) -> None:
        self._vault.append(event)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._vault[-self._capacity :])

    def clear(self) -> None:
        self._vault.clear()


class UnsafeUnboundedHistory:
    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[RuntimeEvent] = []

    def retention(self) -> RuntimeEventHistoryRetention:
        return RuntimeEventHistoryRetention(
            mode="bounded",
            capacity=DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY,
        )

    def append(self, event: RuntimeEvent) -> None:
        self._items.append(event)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()


def test_default_bus_is_bounded() -> None:
    bus = RuntimeEventBus()
    retention = bus._history_buffer.retention()
    assert retention.mode == "bounded"
    assert retention.capacity == DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY
    for index in range(DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY + 50):
        bus.record(_event(label=str(index)))
    assert len(bus.history) == DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY


def test_disabled_history() -> None:
    store = InMemoryRuntimeEventStore()
    sink = InMemoryEventSink()
    seen: list[RuntimeEvent] = []
    bus = RuntimeEventBus(
        persistence=store,
        record_history=False,
        event_sink=sink,
    )
    bus.subscribe(lambda event: seen.append(event))

    event = _event(label="only")
    bus.record(event, tenant_id="tenant-a")
    assert bus._history_buffer.retention().mode == "disabled"
    assert bus.history == []
    assert bus.event_count == 1
    assert len(store.list_for_run(event.run_id, tenant_id="tenant-a")) == 1
    assert len(sink.records) == 1
    assert len(seen) == 1


def test_bounded_overflow_keeps_latest() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(3))
    events = [_event(label=str(i)) for i in range(5)]
    for item in events:
        bus.record(item)
    snapshot = bus.history
    assert [e.payload["label"] for e in snapshot] == ["2", "3", "4"]


def test_large_volume_stays_bounded() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(16))
    for index in range(10_000):
        bus.record(_event(label=str(index)))
    assert len(bus.history) == 16


@pytest.mark.asyncio
async def test_publish_respects_bound() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    await bus.publish(_event(label="a"))
    await bus.publish(_event(label="b"))
    await bus.publish(_event(label="c"))
    assert [e.payload["label"] for e in bus.history] == ["b", "c"]


def test_mixed_record_and_publish_respects_bound() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    bus.record(_event(label="r1"))
    asyncio.run(bus.publish(_event(label="p1")))
    bus.record(_event(label="r2"))
    assert [e.payload["label"] for e in bus.history] == ["p1", "r2"]


def test_clear_history() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(4))
    bus.record(_event(label="x"))
    bus.clear_history()
    assert bus.history == []
    assert bus.event_count == 1
    bus.record(_event(label="y"))
    assert len(bus.history) == 1
    assert bus.event_count == 2


def test_history_snapshot_is_copy() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(4))
    bus.record(_event(label="one"))
    first = bus.history
    second = bus.history
    assert first is not second
    assert first[0].event_id == second[0].event_id


def test_persistence_independent_of_local_eviction() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    bus = RuntimeEventBus(
        persistence=store,
        history_policy=RuntimeEventHistoryPolicy.bounded(1),
    )
    for label in ("E1", "E2", "E3"):
        event = _event(label=label, run_id=run_id).model_copy(
            update={"task_id": task_id}
        )
        bus.record(event, tenant_id="tenant-a")
    assert [e.payload["label"] for e in bus.history] == ["E3"]
    persisted = store.list_for_run(run_id, tenant_id="tenant-a")
    assert [e.payload["label"] for e in persisted] == ["E1", "E2", "E3"]


def test_reconstruction_unaffected_by_local_eviction() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    bus = RuntimeEventBus(
        persistence=store,
        history_policy=RuntimeEventHistoryPolicy.bounded(1),
    )
    for label in ("E1", "E2", "E3"):
        event = _event(label=label, run_id=run_id)
        event = event.model_copy(update={"task_id": task_id})
        bus.record(event, tenant_id="tenant-a")
    recon = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    view = recon.reconstruct_execution("tenant-a", task_id, run_id)
    assert len(view.positioned_events) == 3


def test_sink_and_handlers_unaffected_by_capacity() -> None:
    store = InMemoryRuntimeEventStore()
    sink = InMemoryEventSink()
    handled: list[str] = []
    bus = RuntimeEventBus(
        persistence=store,
        history_policy=RuntimeEventHistoryPolicy.bounded(1),
        event_sink=sink,
    )
    bus.subscribe(lambda event: handled.append(str(event.payload["label"])))
    for label in ("a", "b", "c"):
        bus.record(_event(label=label), tenant_id="tenant-a")
    assert len(sink.records) == 3
    assert handled == ["a", "b", "c"]
    assert len(bus.history) == 1


def test_conflicting_record_history_and_policy() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RuntimeEventBus(
            record_history=True, history_policy=RuntimeEventHistoryPolicy.disabled()
        )


def test_conflicting_buffer_and_policy() -> None:
    with pytest.raises(ValueError, match="history_buffer custom storage injection"):
        RuntimeEventBus(
            history_buffer=BoundedRuntimeEventHistory(4),
            history_policy=RuntimeEventHistoryPolicy.bounded(4),
        )


def test_invalid_bounded_policy_capacity() -> None:
    with pytest.raises(ValueError, match="max_events must be > 0"):
        RuntimeEventHistoryPolicy.bounded(0)


@pytest.mark.parametrize(
    ("capacity", "match"),
    [
        (True, "positive int"),
        (-1, "> 0"),
        (1.5, "positive int"),
        ("512", "positive int"),
    ],
)
def test_invalid_bounded_policy_capacity_types(capacity: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RuntimeEventHistoryPolicy.bounded(capacity)  # type: ignore[arg-type]


def test_invalid_bounded_runtime_event_history_capacity() -> None:
    with pytest.raises(ValueError, match="positive int"):
        BoundedRuntimeEventHistory(True)  # type: ignore[arg-type]


def test_custom_history_strategy_receives_bounded_windows() -> None:
    strategy = RecordingHistoryStrategy()
    bus = RuntimeEventBus(
        history_policy=RuntimeEventHistoryPolicy.bounded(2),
        history_strategy=strategy,
    )
    for label in ("1", "2", "3"):
        bus.record(_event(label=label))
    assert [e.payload["label"] for e in bus.history] == ["2", "3"]
    assert isinstance(bus._history_buffer, PlatformOwnedRuntimeEventHistoryBuffer)
    assert len(strategy.windows[-1]) == 2


def test_platform_retention_ignores_strategy_side_storage() -> None:
    strategy = HoardingHistoryStrategy()
    bus = RuntimeEventBus(
        history_policy=RuntimeEventHistoryPolicy.bounded(2),
        history_strategy=strategy,
    )
    for index in range(100):
        bus.record(_event(label=str(index)))
    assert len(bus.history) == 2
    assert strategy.vault_size > 2


def test_history_buffer_injection_removed() -> None:
    with pytest.raises(ValueError, match="history_buffer custom storage injection"):
        RuntimeEventBus(history_buffer=CustomBoundedHistory(2))


def test_platform_retention_bound_two_after_overflow() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    for index in range(100):
        bus.record(_event(label=str(index)))
    assert len(bus.history) == 2
    buffer = bus._history_buffer
    assert isinstance(buffer, PlatformOwnedRuntimeEventHistoryBuffer)
    assert len(buffer.snapshot()) == 2


def test_unsafe_unbounded_history_rejected_at_composition() -> None:
    with pytest.raises(ValueError, match="history_buffer custom storage injection"):
        RuntimeEventBus(history_buffer=UnsafeUnboundedHistory())


def test_runtime_events_exceeds_history_capacity_in_task_summary(tmp_path) -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    seed = "runtime-events-overflow"
    task = build_task_for_tests(seed=seed, tenant_id="t1", user_id="u1", message="m")
    run_id = canonical_run_id_for_tests(seed)
    scope = bus.open_runtime_event_metric_scope(task.task_id, run_id)
    try:
        for label in ("a", "b", "c", "d", "e"):
            bus.record(_event(label=label, task_id=task.task_id, run_id=run_id))
        assert len(bus.history) == 2
        assert bus.event_count == 5
        result = build_nexus_task_result(
            task,
            TaskTraceEmitter(run_id=run_id, attempt_id=mint_attempt_id()),
            answer="ok",
            executions=[],
            validation=ValidationResult(valid=True),
            plan=None,
            retry_records=[],
            graph_id="g1",
            composer=FinalResponseComposer(),
            event_bus=bus,
            shadow_manager=ShadowWorkspaceManager(root=tmp_path / "shadow"),
            sandbox_manager=SandboxSessionManager(root=tmp_path / "sandbox"),
            runtime_events_count=scope.count(),
        )
    finally:
        scope.close()
    assert result.summary.metrics.runtime_events == 5


def test_history_disabled_scoped_metric_still_counts() -> None:
    bus = RuntimeEventBus(record_history=False)
    task_id = mint_task_id()
    run_id = mint_run_id()
    scope = bus.open_runtime_event_metric_scope(task_id, run_id)
    for label in ("a", "b", "c"):
        bus.record(
            _event(label=label, task_id=task_id, run_id=run_id),
            tenant_id="tenant-a",
        )
    assert bus.history == []
    assert scope.count() == 3
    scope.close()


def test_two_tasks_sequential_scoped_metrics() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    task_a = mint_task_id()
    run_a = mint_run_id()
    task_b = mint_task_id()
    run_b = mint_run_id()
    scope_a = bus.open_runtime_event_metric_scope(task_a, run_a)
    for label in ("a1", "a2", "a3"):
        bus.record(_event(label=label, task_id=task_a, run_id=run_a))
    count_a = scope_a.count()
    scope_a.close()
    scope_b = bus.open_runtime_event_metric_scope(task_b, run_b)
    for label in ("b1", "b2", "b3", "b4", "b5"):
        bus.record(_event(label=label, task_id=task_b, run_id=run_b))
    count_b = scope_b.count()
    scope_b.close()
    assert count_a == 3
    assert count_b == 5


def test_two_tasks_concurrent_scoped_metrics() -> None:
    import threading

    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    task_a = mint_task_id()
    run_a = mint_run_id()
    task_b = mint_task_id()
    run_b = mint_run_id()
    scope_a = bus.open_runtime_event_metric_scope(task_a, run_a)
    scope_b = bus.open_runtime_event_metric_scope(task_b, run_b)

    def emit(task_id: str, run_id: str, prefix: str, total: int) -> None:
        for index in range(total):
            bus.record(_event(label=f"{prefix}{index}", task_id=task_id, run_id=run_id))

    thread_a = threading.Thread(target=emit, args=(task_a, run_a, "A", 4))
    thread_b = threading.Thread(target=emit, args=(task_b, run_b, "B", 7))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()
    assert scope_a.count() == 4
    assert scope_b.count() == 7
    scope_a.close()
    scope_b.close()


def test_scoped_metric_ignores_other_task_run_events() -> None:
    bus = RuntimeEventBus()
    task_id = mint_task_id()
    run_id = mint_run_id()
    scope = bus.open_runtime_event_metric_scope(task_id, run_id)
    bus.record(_event(label="in", task_id=task_id, run_id=run_id))
    bus.record(_event(label="out"))
    assert scope.count() == 1
    scope.close()


def test_task_finisher_does_not_use_history_length_for_runtime_events() -> None:
    source = _TASK_FINISHER_PATH.read_text(encoding="utf-8")
    assert "len(event_bus.history)" not in source
    assert "event_bus.event_count" not in source


def test_nexus_loop_has_no_shared_runtime_event_baseline() -> None:
    source = _NEXUS_LOOP_PATH.read_text(encoding="utf-8")
    assert "_runtime_event_count_baseline" not in source
    assert "self._runtime_event_metric_scope" not in source


def test_platform_owns_retention_envelope_in_history_resolver() -> None:
    source = _HISTORY_IMPL_PATH.read_text(encoding="utf-8")
    assert "class PlatformOwnedRuntimeEventHistoryBuffer" in source
    assert "on_history_window" in source
    assert "_sync_strategy" not in source


def test_event_bus_history_uses_platform_snapshot_only() -> None:
    source = _EVENT_BUS_PATH.read_text(encoding="utf-8")
    assert "return list(self._history_buffer.snapshot())" in source
    assert "strategy.snapshot" not in source


def test_event_bus_has_no_unbounded_list_storage() -> None:
    source = _EVENT_BUS_PATH.read_text(encoding="utf-8")
    assert "_history: List" not in source
    assert "_history.append" not in source
    assert "_record_history" not in source


def test_history_implementations_are_not_persistence_ports() -> None:
    from intergrax.contracts.execution_evidence.persistence_port import (
        EvidencePersistencePort,
    )

    assert not isinstance(BoundedRuntimeEventHistory(4), EvidencePersistencePort)
    from intergrax.runtime.events.runtime_event_history import (
        DisabledRuntimeEventHistory,
    )

    assert not isinstance(DisabledRuntimeEventHistory(), EvidencePersistencePort)
