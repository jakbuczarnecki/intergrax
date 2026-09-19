# © Artur Czarnecki. All rights reserved.

"""HARNESS-02-R1 qualification proofs Q01–Q17."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    reset_active_execution_deadline_scope,
)
from intergrax.contracts.execution_deadline.admission import (
    ExecutionProtectedWorkAdmissionPort,
    ExecutionProtectedWorkAdmissionResult,
)
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.runtime.execution.deadline_provider_guard import (
    ExecutionProtectedWorkDeniedError,
    resolve_active_provider_timeout_seconds,
)
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.contracts.execution_retry import (
    ExecutionFailureClassification,
    ExecutionFailureKind,
    ExecutionRetryEligibilityRequest,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.execution.deadline_authority import (
    ExecutionDeadlineAuthorityResolver,
    InMemoryExecutionDeadlinePersistence,
    decode_execution_deadline_authority_snapshot,
    encode_execution_deadline_authority_snapshot,
)
from intergrax.runtime.execution.deadline_authority.persistence import (
    ExecutionDeadlineCodecError,
    KvExecutionDeadlinePersistence,
)
from intergrax.runtime.execution.deadline_authority.projection import (
    project_deadline_at_utc,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    ComposedProtectedWorkAdmission,
    StaticCancellationView,
)
from intergrax.runtime.execution.retry.policy import evaluate_execution_retry_eligibility
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = pytest.mark.unit


class _FakeUtcClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now_utc(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class _FakeMonotonicClock:
    def __init__(self, value: float = 100.0) -> None:
        self._value = value

    def monotonic(self) -> float:
        return self._value

    def advance(self, seconds: float) -> None:
        self._value += seconds


class _KV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str, key: str) -> bytes | None:
        with self._lock:
            return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        with self._lock:
            self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        with self._lock:
            self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        del ttl_seconds
        with self._lock:
            current = self._data.get((tenant_id, key))
            if expected is None and current is not None:
                return False
            if expected is not None and current != expected:
                return False
            self._data[(tenant_id, key)] = new_value
            return True


def _resolver(
    persistence: InMemoryExecutionDeadlinePersistence,
    *,
    utc: _FakeUtcClock,
    monotonic: _FakeMonotonicClock,
) -> ExecutionDeadlineAuthorityResolver:
    return ExecutionDeadlineAuthorityResolver(
        persistence,
        utc_clock=utc,
        monotonic_clock=monotonic,
    )


def test_q01_root_creates_authority_once_cas() -> None:
    persistence = InMemoryExecutionDeadlinePersistence()
    run_id = mint_run_id()
    utc = _FakeUtcClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    resolver = _resolver(persistence, utc=utc, monotonic=_FakeMonotonicClock())
    first = resolver.resolve_for_root(
        tenant_id="t1",
        run_id=run_id,
        run_budget=RunBudget(max_wall_time_seconds=30.0),
        existing_run_materialized=False,
    )
    second = resolver.resolve_for_root(
        tenant_id="t1",
        run_id=run_id,
        run_budget=RunBudget(max_wall_time_seconds=999.0),
        existing_run_materialized=True,
    )
    assert first.snapshot.deadline_at_utc == second.snapshot.deadline_at_utc
    assert persistence.load(tenant_id="t1", run_id=run_id) is not None


def test_q02_resume_preserves_deadline() -> None:
    persistence = InMemoryExecutionDeadlinePersistence()
    run_id = mint_run_id()
    utc = _FakeUtcClock(datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc))
    resolver = _resolver(persistence, utc=utc, monotonic=_FakeMonotonicClock())
    created = resolver.resolve_for_root(
        tenant_id="t1",
        run_id=run_id,
        run_budget=RunBudget(max_wall_time_seconds=60.0),
        existing_run_materialized=False,
    )
    resumed = resolver.resolve_for_root(
        tenant_id="t1",
        run_id=run_id,
        run_budget=RunBudget(max_wall_time_seconds=60.0),
        existing_run_materialized=True,
    )
    assert resumed.snapshot.deadline_at_utc == created.snapshot.deadline_at_utc


def test_q04_new_run_new_authority() -> None:
    persistence = InMemoryExecutionDeadlinePersistence()
    utc = _FakeUtcClock(datetime(2026, 3, 1, tzinfo=timezone.utc))
    resolver = _resolver(persistence, utc=utc, monotonic=_FakeMonotonicClock())
    run_a = mint_run_id()
    run_b = mint_run_id()
    a = resolver.resolve_for_root(
        tenant_id="t1",
        run_id=run_a,
        run_budget=RunBudget(max_wall_time_seconds=10.0),
        existing_run_materialized=False,
    )
    utc.advance(5.0)
    b = resolver.resolve_for_root(
        tenant_id="t1",
        run_id=run_b,
        run_budget=RunBudget(max_wall_time_seconds=10.0),
        existing_run_materialized=False,
    )
    assert a.snapshot.deadline_at_utc != b.snapshot.deadline_at_utc


def test_q13_missing_authority_on_existing_run_fails_closed() -> None:
    persistence = InMemoryExecutionDeadlinePersistence()
    run_id = mint_run_id()
    utc = _FakeUtcClock(datetime(2026, 4, 1, tzinfo=timezone.utc))
    resolver = _resolver(persistence, utc=utc, monotonic=_FakeMonotonicClock())
    with pytest.raises(Exception, match="missing durable execution deadline"):
        resolver.resolve_for_root(
            tenant_id="t1",
            run_id=run_id,
            run_budget=RunBudget(max_wall_time_seconds=5.0),
            existing_run_materialized=True,
        )


def test_q14_parallel_workers_same_deadline() -> None:
    kv = _KV()
    persistence = KvExecutionDeadlinePersistence(kv)
    run_id = mint_run_id()
    utc = _FakeUtcClock(datetime(2026, 5, 1, tzinfo=timezone.utc))
    barrier = threading.Barrier(2)
    results: list[datetime | None] = []

    def worker() -> None:
        resolver = _resolver(
            persistence,  # type: ignore[arg-type]
            utc=utc,
            monotonic=_FakeMonotonicClock(),
        )
        barrier.wait()
        resolution = resolver.resolve_for_root(
            tenant_id="t1",
            run_id=run_id,
            run_budget=RunBudget(max_wall_time_seconds=12.0),
            existing_run_materialized=False,
        )
        results.append(resolution.snapshot.deadline_at_utc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(results) == 2
    assert results[0] == results[1]


def test_q12_retry_backoff_respects_deadline_utc() -> None:
    deadline = datetime(2026, 6, 1, 12, 0, 30, tzinfo=timezone.utc)
    now = datetime(2026, 6, 1, 12, 0, 28, tzinfo=timezone.utc)
    result = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=ExecutionFailureClassification(
                kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
            ),
            attempt_number=1,
            max_attempts=3,
            deadline_at_utc=deadline,
            now_utc=now,
            proposed_backoff_seconds=5.0,
        ),
    )
    assert result.action.value == "fail"


@dataclass(frozen=True)
class _AlwaysAvailableContributor(ExecutionProtectedWorkAdmissionPort):
    def assert_can_start_protected_work(self) -> ExecutionProtectedWorkAdmissionResult:
        return ExecutionProtectedWorkAdmissionResult.AVAILABLE


def test_q17_custom_available_cannot_override_expired() -> None:
    monotonic = _FakeMonotonicClock(1.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=1.0,
    )
    composed = ComposedProtectedWorkAdmission(
        canonical=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        contributors=(_AlwaysAvailableContributor(),),
    )
    assert (
        composed.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.EXPIRED
    )


def test_q17_custom_available_cannot_override_cancelled() -> None:
    monotonic = _FakeMonotonicClock()
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=None,
        remaining_seconds=float("inf"),
        is_expired=False,
        global_deadline_monotonic=None,
    )
    composed = ComposedProtectedWorkAdmission(
        canonical=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=True, reason="operator"),
            monotonic_clock=monotonic,
        ),
        contributors=(_AlwaysAvailableContributor(),),
    )
    assert (
        composed.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.CANCELLED
    )


def test_q11_provider_timeout_bounded_by_remaining() -> None:
    monotonic = _FakeMonotonicClock(100.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc),
        remaining_seconds=4.0,
        is_expired=False,
        global_deadline_monotonic=104.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    try:
        assert resolve_active_provider_timeout_seconds(30.0) == 4.0
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_q10_expired_blocks_llm_execute() -> None:
    monotonic = _FakeMonotonicClock(1.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=1.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    physical_calls = 0

    class _ProbeAdapter(LLMAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.provider = "openai"

        @property
        def context_window_tokens(self) -> int:
            return 8192

        def generate_messages(self, messages):  # type: ignore[no-untyped-def]
            del messages
            return self._execute(lambda: "ok")

    adapter = _ProbeAdapter()

    def _physical() -> str:
        nonlocal physical_calls
        physical_calls += 1
        return "ok"

    try:
        with pytest.raises(ExecutionProtectedWorkDeniedError):
            adapter._execute(_physical)
        assert physical_calls == 0
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_corrupt_snapshot_fail_closed() -> None:
    with pytest.raises(ExecutionDeadlineCodecError):
        decode_execution_deadline_authority_snapshot(b"{not-json")


def test_naive_deadline_rejected_on_decode() -> None:
    from intergrax.contracts.execution_deadline.snapshot import (
        EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION,
        ExecutionDeadlineAuthoritySnapshot,
    )

    snapshot = ExecutionDeadlineAuthoritySnapshot(
        schema_version=EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION,
        run_id=mint_run_id(),
        deadline_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        authority_created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        policy_max_wall_time_seconds=1.0,
    )
    raw = encode_execution_deadline_authority_snapshot(snapshot)
    tampered = raw.decode("utf-8").replace("+00:00", "")
    with pytest.raises(ExecutionDeadlineCodecError):
        decode_execution_deadline_authority_snapshot(tampered.encode("utf-8"))


def test_expired_projection_non_negative_remaining() -> None:
    projection = project_deadline_at_utc(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        utc_clock=_FakeUtcClock(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        monotonic_clock=_FakeMonotonicClock(),
    )
    assert projection.is_expired
    assert projection.remaining_seconds == 0.0
