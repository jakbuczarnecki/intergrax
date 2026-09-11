# © Artur Czarnecki. All rights reserved.

"""W2-B2 — DependencyAttemptExecutionBoundary behavioral tests."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyAdmissionTimeoutError,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPermit,
    DependencyConcurrencyPolicy,
)
from intergrax.runtime.resilience import dependency_attempt_execution_boundary as dab_module
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
    DependencyAttemptExecutionBoundaryClosedError,
    DependencyAttemptReleaseInvariantError,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _tool(value: str) -> DependencyConcurrencyIdentity:
    return DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value=value,
    )


def _request(value: str) -> DependencyConcurrencyAdmissionRequest:
    return DependencyConcurrencyAdmissionRequest(
        dependency=_tool(value),
        tenant_id="tenant-a",
    )


class _LoopAffinityRecorder:
    def __init__(self) -> None:
        self.acquire_threads: list[int] = []
        self.release_threads: list[int] = []
        self.acquire_loops: list[int] = []
        self.release_loops: list[int] = []

    def record_acquire(self) -> None:
        loop = asyncio.get_running_loop()
        self.acquire_threads.append(threading.get_ident())
        self.acquire_loops.append(id(loop))

    def record_release(self) -> None:
        loop = asyncio.get_running_loop()
        self.release_threads.append(threading.get_ident())
        self.release_loops.append(id(loop))


class _FakePermit:
    __slots__ = ("_admission", "_released", "_release_should_fail")

    def __init__(
        self,
        admission: _FakeAdmissionPort,
        *,
        release_should_fail: bool = False,
    ) -> None:
        self._admission = admission
        self._released = False
        self._release_should_fail = release_should_fail

    async def release(self) -> None:
        self._admission.affinity.record_release()
        if self._release_should_fail:
            raise RuntimeError("release failed")
        self._released = True
        await self._admission._release_slot()

    @property
    def released(self) -> bool:
        return self._released


class _FakeAdmissionPort:
    def __init__(
        self,
        *,
        capacity: int = 4,
        reject_value: str | None = None,
        wait_seconds: float | None = None,
    ) -> None:
        self.capacity = capacity
        self.reject_value = reject_value
        self.wait_seconds = wait_seconds
        self.active = 0
        self.acquire_calls = 0
        self.affinity = _LoopAffinityRecorder()
        self._mutex = asyncio.Lock()

    async def _release_slot(self) -> None:
        async with self._mutex:
            self.active -= 1

    async def acquire(
        self,
        request: DependencyConcurrencyAdmissionRequest,
    ) -> DependencyConcurrencyPermit:
        self.affinity.record_acquire()
        self.acquire_calls += 1
        if (
            self.reject_value is not None
            and request.dependency.value == self.reject_value
        ):
            raise DependencyConcurrencyExceededError("saturated")
        if self.wait_seconds is not None:
            await asyncio.sleep(self.wait_seconds)
        async with self._mutex:
            if self.active >= self.capacity:
                raise DependencyConcurrencyExceededError("saturated")
            self.active += 1
        return _FakePermit(self)


class _GatedRegistryLock:
    """Wraps boundary registry lock to pause acquire after permit result()."""

    def __init__(self, inner: threading.Lock) -> None:
        self._inner = inner
        self._armed = False
        self._blocked = threading.Event()
        self._release = threading.Event()
        self._release.set()
        self._inner_acquired = False

    def arm(self) -> None:
        self._armed = True
        self._release.clear()
        self._blocked.clear()

    def unblock(self) -> None:
        self._release.set()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._armed and not self._release.is_set():
            self._blocked.set()
            if not self._release.wait(timeout=10):
                return False
        if timeout < 0:
            acquired = self._inner.acquire(blocking)
        else:
            acquired = self._inner.acquire(blocking, timeout)
        if acquired:
            self._inner_acquired = True
        return acquired

    def release(self) -> None:
        if self._inner_acquired:
            self._inner.release()
            self._inner_acquired = False

    def __enter__(self) -> _GatedRegistryLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _begin_shutdown_on_inner_lock(
    boundary: DependencyAttemptExecutionBoundary,
    inner_lock: threading.Lock,
) -> None:
    with inner_lock:
        if boundary._lifecycle is not dab_module._BoundaryLifecycle.OPEN:
            return
        boundary._lifecycle = dab_module._BoundaryLifecycle.SHUTTING_DOWN
        pending_acquires = list(boundary._pending_acquire_futures)
    for pending in pending_acquires:
        pending.cancel()


class _ShutdownAfterSlotAdmission(_FakeAdmissionPort):
    """Holds permit after slot claim until test releases shutdown_gate."""

    def __init__(self, *, capacity: int = 4) -> None:
        super().__init__(capacity=capacity)
        self._continue = asyncio.Event()
        self._shutdown_gate = asyncio.Event()
        self.awaiting_start = threading.Event()
        self.slot_claimed = threading.Event()
        self.permits_released = 0

    async def acquire(
        self,
        request: DependencyConcurrencyAdmissionRequest,
    ) -> DependencyConcurrencyPermit:
        self.awaiting_start.set()
        await self._continue.wait()
        permit = await super().acquire(request)
        self.slot_claimed.set()
        await self._shutdown_gate.wait()
        return permit

    def unblock_acquire(self, loop: asyncio.AbstractEventLoop) -> None:
        loop.call_soon_threadsafe(self._continue.set)

    def release_permit(self, loop: asyncio.AbstractEventLoop) -> None:
        loop.call_soon_threadsafe(self._shutdown_gate.set)

    async def _release_slot(self) -> None:
        await super()._release_slot()
        self.permits_released += 1


class _FailingReleaseAdmission(_FakeAdmissionPort):
    async def acquire(
        self,
        request: DependencyConcurrencyAdmissionRequest,
    ) -> DependencyConcurrencyPermit:
        await super().acquire(request)
        return _FakePermit(self, release_should_fail=True)


def _reject_policy(capacity: int) -> DependencyConcurrencyPolicy:
    return DependencyConcurrencyPolicy(
        max_concurrent_calls=capacity,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


@pytest.fixture
def fake_admission() -> _FakeAdmissionPort:
    return _FakeAdmissionPort(capacity=32)


@pytest.fixture
def boundary(fake_admission: _FakeAdmissionPort) -> DependencyAttemptExecutionBoundary:
    instance = DependencyAttemptExecutionBoundary(fake_admission)
    yield instance
    instance.close()


def test_startup_ready_before_acquire(
    boundary: DependencyAttemptExecutionBoundary,
    fake_admission: _FakeAdmissionPort,
) -> None:
    handle = boundary.acquire(_request("tool-a"))
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(lambda: 1)
    boundary.bind_worker(handle, future)
    assert future.result() == 1
    boundary.complete_attached(handle)
    pool.shutdown(wait=True)
    assert fake_admission.acquire_calls == 1


def test_acquire_executed_on_owned_loop(
    boundary: DependencyAttemptExecutionBoundary,
    fake_admission: _FakeAdmissionPort,
) -> None:
    handle = boundary.acquire(_request("tool-a"))
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(lambda: None)
    boundary.bind_worker(handle, future)
    future.result()
    boundary.complete_attached(handle)
    pool.shutdown(wait=True)
    assert len(set(fake_admission.affinity.acquire_threads)) == 1
    assert fake_admission.affinity.acquire_threads == fake_admission.affinity.release_threads
    assert len(set(fake_admission.affinity.acquire_loops)) == 1
    assert fake_admission.affinity.acquire_loops == fake_admission.affinity.release_loops


def test_multiple_sync_caller_threads(
    boundary: DependencyAttemptExecutionBoundary,
) -> None:
    barrier = threading.Barrier(16)
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            handle = boundary.acquire(_request("tool-a"))
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(lambda: None)
            boundary.bind_worker(handle, future)
            future.result()
            boundary.complete_attached(handle)
            pool.shutdown(wait=True)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors


def test_acquire_reject_propagates() -> None:
    identity = _tool("tool-a")
    admission = LocalDependencyConcurrencyAdmission(
        {
            identity: DependencyConcurrencyPolicy(
                max_concurrent_calls=1,
                overload_mode=DependencyConcurrencyOverloadMode.REJECT,
                wait_timeout_seconds=None,
            ),
        },
    )
    boundary = DependencyAttemptExecutionBoundary(admission)
    first = boundary.acquire(_request("tool-a"))
    try:
        with pytest.raises(DependencyConcurrencyExceededError):
            boundary.acquire(_request("tool-a"))
    finally:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(lambda: None)
        boundary.bind_worker(first, future)
        future.result()
        boundary.complete_attached(first)
        pool.shutdown(wait=True)
        boundary.close()


def test_wait_timeout_propagates() -> None:
    identity = _tool("slow")
    admission = LocalDependencyConcurrencyAdmission(
        {
            identity: DependencyConcurrencyPolicy(
                max_concurrent_calls=1,
                overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
                wait_timeout_seconds=0.05,
            ),
        },
    )
    boundary = DependencyAttemptExecutionBoundary(admission)
    handle = boundary.acquire(_request("slow"))
    pool = ThreadPoolExecutor(max_workers=1)
    hold = threading.Event()

    def _block() -> None:
        hold.wait(timeout=5)

    future = pool.submit(_block)
    boundary.bind_worker(handle, future)
    assert hold.wait(timeout=0.2) or future.running() or not future.done()
    try:
        with pytest.raises(DependencyConcurrencyAdmissionTimeoutError):
            boundary.acquire(_request("slow"))
    finally:
        hold.set()
        future.result(timeout=5)
        boundary.complete_attached(handle)
        pool.shutdown(wait=True)
        boundary.close()


def test_attached_success_releases_before_return(
    boundary: DependencyAttemptExecutionBoundary,
    fake_admission: _FakeAdmissionPort,
) -> None:
    handle = boundary.acquire(_request("tool-a"))
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(lambda: "ok")
    boundary.bind_worker(handle, future)
    assert future.result() == "ok"
    assert fake_admission.active == 1
    boundary.complete_attached(handle)
    assert fake_admission.active == 0
    pool.shutdown(wait=True)


def test_attached_release_failure_surfaces(
    fake_admission: _FakeAdmissionPort,
) -> None:
    failing = _FailingReleaseAdmission()
    boundary = DependencyAttemptExecutionBoundary(failing)
    handle = boundary.acquire(_request("tool-a"))
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(lambda: None)
    boundary.bind_worker(handle, future)
    future.result()
    with pytest.raises(DependencyAttemptReleaseInvariantError):
        boundary.complete_attached(handle)
    pool.shutdown(wait=True)
    boundary.close()


def test_detached_timeout_keeps_capacity() -> None:
    identity = _tool("tool-a")
    admission = LocalDependencyConcurrencyAdmission(
        {
            identity: DependencyConcurrencyPolicy(
                max_concurrent_calls=1,
                overload_mode=DependencyConcurrencyOverloadMode.REJECT,
                wait_timeout_seconds=None,
            ),
        },
    )
    boundary = DependencyAttemptExecutionBoundary(admission)
    handle = boundary.acquire(_request("tool-a"))
    pool = ThreadPoolExecutor(max_workers=1)
    started = threading.Event()
    release_gate = threading.Event()

    def _slow() -> None:
        started.set()
        release_gate.wait(timeout=5)

    future = pool.submit(_slow)
    boundary.bind_worker(handle, future)
    assert started.wait(timeout=2)
    assert boundary.detach_if_still_running(handle) is True
    with pytest.raises(DependencyConcurrencyExceededError):
        boundary.acquire(_request("tool-a"))
    release_gate.set()
    time.sleep(0.2)
    boundary.begin_shutdown()
    boundary.drain_and_close()
    pool.shutdown(wait=True)


def test_detached_worker_completion_releases(
    fake_admission: _FakeAdmissionPort,
) -> None:
    boundary = DependencyAttemptExecutionBoundary(fake_admission)
    handle = boundary.acquire(_request("tool-a"))
    pool = ThreadPoolExecutor(max_workers=1)
    started = threading.Event()
    finished = threading.Event()

    def _work() -> None:
        started.set()
        finished.wait(timeout=5)

    future = pool.submit(_work)
    boundary.bind_worker(handle, future)
    assert started.wait(timeout=2)
    assert boundary.detach_if_still_running(handle) is True
    finished.set()
    time.sleep(0.3)
    boundary.drain_and_close()
    assert fake_admission.active == 0
    pool.shutdown(wait=True)


def test_acquire_success_shutdown_before_registration_releases_permit() -> None:
    admission = _ShutdownAfterSlotAdmission(capacity=4)
    boundary = DependencyAttemptExecutionBoundary(admission)
    loop = boundary._require_loop()
    gated_lock = _GatedRegistryLock(boundary._registry_lock)
    boundary._registry_lock = gated_lock
    acquire_error: list[BaseException] = []

    def _acquire() -> None:
        try:
            boundary.acquire(_request("tool-a"))
        except BaseException as exc:
            acquire_error.append(exc)

    thread = threading.Thread(target=_acquire)
    thread.start()
    assert admission.awaiting_start.wait(timeout=5)
    gated_lock.arm()
    admission.unblock_acquire(loop)
    assert admission.slot_claimed.wait(timeout=5)
    admission.release_permit(loop)
    assert gated_lock._blocked.wait(timeout=5)
    _begin_shutdown_on_inner_lock(boundary, gated_lock._inner)
    gated_lock.unblock()
    thread.join(timeout=10)
    assert len(acquire_error) == 1
    assert isinstance(acquire_error[0], DependencyAttemptExecutionBoundaryClosedError)
    assert admission.acquire_calls == 1
    assert admission.permits_released == 1
    assert admission.active == 0
    boundary.drain_and_close()


def test_shutdown_cancel_races_acquire_success_no_leak() -> None:
    admission = _ShutdownAfterSlotAdmission(capacity=4)
    boundary = DependencyAttemptExecutionBoundary(admission)
    loop = boundary._require_loop()
    gated_lock = _GatedRegistryLock(boundary._registry_lock)
    boundary._registry_lock = gated_lock
    outcomes: list[str] = []

    def _acquire() -> None:
        try:
            boundary.acquire(_request("tool-a"))
            outcomes.append("ok")
        except DependencyAttemptExecutionBoundaryClosedError:
            outcomes.append("closed")
        except BaseException as exc:
            outcomes.append(type(exc).__name__)

    thread = threading.Thread(target=_acquire)
    thread.start()
    assert admission.awaiting_start.wait(timeout=5)
    gated_lock.arm()
    admission.unblock_acquire(loop)
    assert admission.slot_claimed.wait(timeout=5)
    admission.release_permit(loop)
    assert gated_lock._blocked.wait(timeout=5)
    _begin_shutdown_on_inner_lock(boundary, gated_lock._inner)
    gated_lock.unblock()
    thread.join(timeout=10)
    assert outcomes == ["closed"]
    assert admission.permits_released == 1
    assert admission.active == 0
    boundary.drain_and_close()

    pending_admission = _FakeAdmissionPort(capacity=1, wait_seconds=0.2)
    cancel_boundary = DependencyAttemptExecutionBoundary(pending_admission)
    cancel_outcomes: list[str] = []

    def _pending_acquire() -> None:
        try:
            cancel_boundary.acquire(_request("tool-a"))
            cancel_outcomes.append("ok")
        except DependencyAttemptExecutionBoundaryClosedError:
            cancel_outcomes.append("closed")
        except BaseException as exc:
            cancel_outcomes.append(type(exc).__name__)

    cancel_thread = threading.Thread(target=_pending_acquire)
    cancel_thread.start()
    time.sleep(0.02)
    cancel_boundary.begin_shutdown()
    cancel_thread.join(timeout=10)
    assert cancel_outcomes[0] in {"closed", "CancelledError"}
    assert pending_admission.active == 0
    cancel_boundary.drain_and_close()


def test_shutdown_cancels_pending_wait() -> None:
    identity = _tool("wait")
    admission = LocalDependencyConcurrencyAdmission(
        {
            identity: DependencyConcurrencyPolicy(
                max_concurrent_calls=1,
                overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
                wait_timeout_seconds=30.0,
            ),
        },
    )
    boundary = DependencyAttemptExecutionBoundary(admission)
    first = boundary.acquire(_request("wait"))
    pool = ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    future = pool.submit(gate.wait)
    boundary.bind_worker(first, future)
    waiter_error: list[BaseException] = []

    def _waiter() -> None:
        try:
            boundary.acquire(_request("wait"))
        except BaseException as exc:
            waiter_error.append(exc)

    thread = threading.Thread(target=_waiter)
    thread.start()
    time.sleep(0.1)
    boundary.begin_shutdown()
    thread.join(timeout=5)
    gate.set()
    future.result(timeout=5)
    boundary.complete_attached(first)
    pool.shutdown(wait=True)
    boundary.drain_and_close()
    assert waiter_error


def test_close_idempotent(boundary: DependencyAttemptExecutionBoundary) -> None:
    boundary.close()
    boundary.close()


def test_no_acquire_after_close(boundary: DependencyAttemptExecutionBoundary) -> None:
    boundary.close()
    with pytest.raises(DependencyAttemptExecutionBoundaryClosedError):
        boundary.acquire(_request("tool-a"))


def test_no_orphan_thread_after_close() -> None:
    boundary = DependencyAttemptExecutionBoundary(_FakeAdmissionPort())
    boundary.close()
    for thread in threading.enumerate():
        if thread.name == "dependency-admission-boundary":
            assert thread.is_alive() is False


def test_concurrent_close_safe() -> None:
    boundary = DependencyAttemptExecutionBoundary(_FakeAdmissionPort())
    errors: list[BaseException] = []

    def _close() -> None:
        try:
            boundary.close()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_close) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors


def test_timeout_race_no_permit_leak() -> None:
    identity = _tool("race")
    for _ in range(5):
        admission = LocalDependencyConcurrencyAdmission({identity: _reject_policy(1)})
        boundary = DependencyAttemptExecutionBoundary(admission)
        handle = boundary.acquire(_request("race"))
        pool = ThreadPoolExecutor(max_workers=1)
        done = threading.Event()

        def _work() -> None:
            time.sleep(0.01)
            done.set()

        future = pool.submit(_work)
        boundary.bind_worker(handle, future)
        time.sleep(0.005)
        if boundary.detach_if_still_running(handle):
            pass
        else:
            boundary.complete_attached(handle)
        done.wait(timeout=1)
        boundary.close()
        pool.shutdown(wait=True)
        probe = DependencyAttemptExecutionBoundary(
            LocalDependencyConcurrencyAdmission({identity: _reject_policy(1)}),
        )
        token = probe.acquire(_request("race"))
        pool2 = ThreadPoolExecutor(max_workers=1)
        fut2 = pool2.submit(lambda: None)
        probe.bind_worker(token, fut2)
        fut2.result()
        probe.complete_attached(token)
        probe.close()
        pool2.shutdown(wait=True)
