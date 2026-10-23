# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sync physical dependency attempt boundary over async admission (W2-B2)."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as ConcurrentFuturesTimeoutError
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionPort,
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyAdmissionTimeoutError,
    DependencyConcurrencyPermit,
)

if TYPE_CHECKING:
    from concurrent.futures import Future as ThreadSafeFuture


class DependencyAttemptExecutionBoundaryClosedError(RuntimeError):
    """Boundary no longer accepts new physical dependency attempts."""


class DependencyAttemptReleaseInvariantError(RuntimeError):
    """Permit release failed for a physical dependency attempt (fail-closed)."""


class _AcquireAbandonedBeforeDelivery(Exception):
    """Async acquire granted after sync waiter abandoned (permit released on loop)."""


class _AcquireHandoffState(StrEnum):
    """Permit ownership between async grant completion and sync caller acceptance."""

    WAITING = "WAITING"
    GRANT_AVAILABLE = "GRANT_AVAILABLE"
    CALLER_OWNS = "CALLER_OWNS"
    ABANDONED = "ABANDONED"
    RELEASE_OBLIGATION_CLAIMED = "RELEASE_OBLIGATION_CLAIMED"
    RELEASE_PENDING = "RELEASE_PENDING"
    RELEASED = "RELEASED"


@dataclass(slots=True)
class _AcquirePermitHandoff:
    """Single linearization point for grant vs abandon cleanup (BA-F02)."""

    _state: _AcquireHandoffState = _AcquireHandoffState.WAITING
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim_successful_grant(self) -> bool:
        """Caller linearization immediately after a successful ``result()``."""
        with self._lock:
            if self._state not in (
                _AcquireHandoffState.WAITING,
                _AcquireHandoffState.GRANT_AVAILABLE,
            ):
                return False
            self._state = _AcquireHandoffState.CALLER_OWNS
            return True

    def mark_abandoned(self) -> None:
        with self._lock:
            if self._state is _AcquireHandoffState.CALLER_OWNS:
                return
            if self._state in (
                _AcquireHandoffState.RELEASED,
                _AcquireHandoffState.RELEASE_OBLIGATION_CLAIMED,
                _AcquireHandoffState.RELEASE_PENDING,
            ):
                return
            self._state = _AcquireHandoffState.ABANDONED

    def is_abandoned(self) -> bool:
        with self._lock:
            return self._state is _AcquireHandoffState.ABANDONED

    def is_release_obligation_claimed(self) -> bool:
        with self._lock:
            return self._state is _AcquireHandoffState.RELEASE_OBLIGATION_CLAIMED

    def is_release_complete(self) -> bool:
        with self._lock:
            return self._state is _AcquireHandoffState.RELEASED

    def is_release_pending(self) -> bool:
        with self._lock:
            return self._state is _AcquireHandoffState.RELEASE_PENDING

    def try_claim_release_obligation(self) -> bool:
        """Claim (or re-enter) abandoned cleanup before permit observation."""
        with self._lock:
            if self._state in (
                _AcquireHandoffState.CALLER_OWNS,
                _AcquireHandoffState.RELEASED,
                _AcquireHandoffState.RELEASE_PENDING,
            ):
                return False
            if self._state is _AcquireHandoffState.RELEASE_OBLIGATION_CLAIMED:
                return True
            if self._state in (
                _AcquireHandoffState.ABANDONED,
                _AcquireHandoffState.GRANT_AVAILABLE,
            ):
                self._state = _AcquireHandoffState.RELEASE_OBLIGATION_CLAIMED
                return True
            return False

    def complete_release(self) -> None:
        with self._lock:
            self._state = _AcquireHandoffState.RELEASED

    def mark_release_pending(self) -> None:
        with self._lock:
            if self._state is _AcquireHandoffState.RELEASED:
                return
            self._state = _AcquireHandoffState.RELEASE_PENDING

    def on_physical_grant(
        self,
        permit: DependencyConcurrencyPermit,
    ) -> Literal["deliver", "release_in_coro", "noop"]:
        """Linearize physical grant vs sync abandon on the admission event loop."""
        del permit
        with self._lock:
            if self._state is _AcquireHandoffState.WAITING:
                self._state = _AcquireHandoffState.GRANT_AVAILABLE
                return "deliver"
            if self._state in (
                _AcquireHandoffState.ABANDONED,
                _AcquireHandoffState.RELEASE_OBLIGATION_CLAIMED,
            ):
                self._state = _AcquireHandoffState.RELEASE_PENDING
                return "release_in_coro"
            if self._state is _AcquireHandoffState.CALLER_OWNS:
                raise RuntimeError("invariant: physical grant after caller ownership")
            if self._state in (
                _AcquireHandoffState.RELEASED,
                _AcquireHandoffState.RELEASE_PENDING,
            ):
                return "noop"
            raise RuntimeError(f"unexpected acquire handoff state: {self._state!r}")


@dataclass(frozen=True, slots=True)
class DependencyAttemptHandle:
    """Opaque sync handle tying one permit to one worker attempt."""

    _token: int


class _BoundaryLifecycle(StrEnum):
    OPEN = "OPEN"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    CLOSED = "CLOSED"


class _AttemptPhase(StrEnum):
    ACQUIRED = "ACQUIRED"
    ATTACHED = "ATTACHED"
    DETACHED = "DETACHED"


class _AttemptRecord:
    __slots__ = (
        "permit",
        "phase",
        "worker_future",
        "release_claimed",
        "release_future",
    )

    def __init__(self, permit: DependencyConcurrencyPermit) -> None:
        self.permit = permit
        self.phase = _AttemptPhase.ACQUIRED
        self.worker_future: ConcurrentFuture[object] | None = None
        self.release_claimed = False
        self.release_future: ThreadSafeFuture[object] | None = None


class DependencyAttemptExecutionBoundary:
    """Bridges sync tool physical attempts to async dependency admission."""

    _STARTUP_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        admission: DependencyConcurrencyAdmissionPort,
    ) -> None:
        if not isinstance(admission, DependencyConcurrencyAdmissionPort):
            raise TypeError("admission must implement DependencyConcurrencyAdmissionPort")
        self._admission = admission
        self._registry_lock = threading.Lock()
        self._lifecycle = _BoundaryLifecycle.OPEN
        self._next_token = 0
        self._attempts: dict[int, _AttemptRecord] = {}
        self._pending_acquire_futures: set[ConcurrentFuture[DependencyConcurrencyPermit]] = (
            set()
        )
        self._pending_async_release_futures: set[ConcurrentFuture[object]] = set()
        self._detached_pending_release: set[int] = set()
        self._release_failures: list[BaseException] = []
        self._drain_condition = threading.Condition(self._registry_lock)
        self._loop_ready = threading.Event()
        self._loop_exited = threading.Event()
        self._close_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._admission_thread: threading.Thread | None = None
        self._thread = threading.Thread(
            target=self._run_admission_loop,
            name="dependency-admission-boundary",
            daemon=False,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=self._STARTUP_TIMEOUT_SECONDS):
            raise RuntimeError(
                "DependencyAttemptExecutionBoundary failed to start admission loop"
            )

    def acquire(
        self,
        request: DependencyConcurrencyAdmissionRequest,
        *,
        acquire_timeout_seconds: float | None = None,
    ) -> DependencyAttemptHandle:
        """Block until admission acquire completes or raises a typed admission error."""
        self._require_accepts_attempts()
        loop = self._require_loop()
        handoff = _AcquirePermitHandoff()

        async def _acquire_coro() -> DependencyConcurrencyPermit:
            permit = await self._admission.acquire(request)
            grant_action = handoff.on_physical_grant(permit)
            if grant_action == "release_in_coro":
                await self._release_permit_for_abandoned_handoff(permit, handoff)
                raise _AcquireAbandonedBeforeDelivery()
            if grant_action == "noop":
                raise _AcquireAbandonedBeforeDelivery()
            return permit

        acquire_future = asyncio.run_coroutine_threadsafe(_acquire_coro(), loop)

        def _on_acquire_terminal(done_future: ConcurrentFuture[object]) -> None:
            del done_future
            if not acquire_future.done():
                return
            if acquire_future.cancelled():
                if (
                    handoff.is_abandoned()
                    or handoff.is_release_obligation_claimed()
                    or handoff.is_release_pending()
                ):
                    self._maybe_release_abandoned_acquire_permit(
                        acquire_future,
                        handoff,
                    )
                return
            exc = acquire_future.exception()
            if exc is not None:
                if isinstance(exc, _AcquireAbandonedBeforeDelivery):
                    return
                if (
                    handoff.is_release_obligation_claimed()
                    and not handoff.is_release_complete()
                ):
                    self._maybe_release_abandoned_acquire_permit(
                        acquire_future,
                        handoff,
                    )
                return
            if not (
                handoff.is_abandoned()
                or handoff.is_release_obligation_claimed()
            ):
                return
            self._maybe_release_abandoned_acquire_permit(acquire_future, handoff)

        acquire_future.add_done_callback(_on_acquire_terminal)
        with self._registry_lock:
            self._pending_acquire_futures.add(acquire_future)
        try:
            if acquire_timeout_seconds is None:
                permit = acquire_future.result()
            elif acquire_timeout_seconds <= 0:
                acquire_future.cancel()
                handoff.mark_abandoned()
                self._maybe_release_abandoned_acquire_permit(acquire_future, handoff)
                raise DependencyConcurrencyAdmissionTimeoutError(
                    "dependency admission exceeded remaining execution budget",
                )
            else:
                try:
                    permit = acquire_future.result(timeout=acquire_timeout_seconds)
                except ConcurrentFuturesTimeoutError:
                    acquire_future.cancel()
                    handoff.mark_abandoned()
                    self._maybe_release_abandoned_acquire_permit(
                        acquire_future,
                        handoff,
                    )
                    raise DependencyConcurrencyAdmissionTimeoutError(
                        "dependency admission exceeded remaining execution budget",
                    ) from None
        finally:
            with self._registry_lock:
                self._pending_acquire_futures.discard(acquire_future)

        if not handoff.claim_successful_grant():
            self._execute_permit_release(permit, wait=False)
            raise RuntimeError(
                "invariant: acquire permit granted after abandon handoff"
            )

        with self._registry_lock:
            if self._lifecycle is _BoundaryLifecycle.OPEN:
                token = self._next_token
                self._next_token += 1
                self._attempts[token] = _AttemptRecord(permit)
                return DependencyAttemptHandle(_token=token)

        release_future = self._schedule_permit_release(permit)
        try:
            release_future.result()
        except BaseException as exc:
            raise DependencyAttemptReleaseInvariantError(
                "dependency permit release failed after shutdown race"
            ) from exc
        raise DependencyAttemptExecutionBoundaryClosedError(
            "dependency attempt boundary is not accepting new attempts"
        )

    def release_after_submit_failure(self, handle: DependencyAttemptHandle) -> None:
        """Release permit when pool submit failed after successful acquire."""
        record = self._require_record(handle)
        if record.worker_future is not None:
            raise RuntimeError("invariant: submit failure with bound worker future")
        self._release_attached(handle, record)

    def bind_worker(
        self,
        handle: DependencyAttemptHandle,
        worker_future: ConcurrentFuture[object],
    ) -> None:
        """Associate the pool worker future; enables detached completion handling."""
        with self._registry_lock:
            record = self._require_record_locked(handle)
            if record.worker_future is not None:
                raise RuntimeError("invariant: worker future already bound")
            record.worker_future = worker_future
            record.phase = _AttemptPhase.ATTACHED

        def _worker_done_callback(done_future: ConcurrentFuture[object]) -> None:
            del done_future
            try:
                self._on_worker_terminal(handle)
            except BaseException as exc:
                self._record_lifecycle_failure(exc)

        worker_future.add_done_callback(_worker_done_callback)

    def detach_if_still_running(self, handle: DependencyAttemptHandle) -> bool:
        """Mark attempt detached when caller timed out while worker still runs."""
        with self._registry_lock:
            record = self._require_record_locked(handle)
            worker = record.worker_future
            if worker is None:
                raise RuntimeError("invariant: detach without worker future")
            if worker.done():
                return False
            record.phase = _AttemptPhase.DETACHED
            self._detached_pending_release.add(handle._token)
            return True

    def complete_direct(self, handle: DependencyAttemptHandle) -> None:
        """Release permit after a direct synchronous physical attempt (no worker future)."""
        with self._registry_lock:
            record = self._require_record_locked(handle)
            if record.worker_future is not None:
                raise RuntimeError("invariant: complete_direct with bound worker future")
            if record.release_claimed:
                return
            record.release_claimed = True
        self._release_attached(handle, record)

    def complete_attached(self, handle: DependencyAttemptHandle) -> None:
        """Release permit after worker terminal while caller remains attached."""
        with self._registry_lock:
            record = self._require_record_locked(handle)
            worker = record.worker_future
            if worker is None:
                raise RuntimeError("invariant: complete attached without worker")
            if not worker.done():
                raise RuntimeError("invariant: complete attached before worker terminal")
            if record.release_claimed:
                return
            record.release_claimed = True
            if record.phase is _AttemptPhase.DETACHED:
                raise RuntimeError(
                    "invariant: complete attached called for detached attempt"
                )
            self._detached_pending_release.discard(handle._token)

        self._release_attached(handle, record)

    def begin_shutdown(self) -> None:
        """Stop accepting attempts; cancel pending admission waits."""
        with self._registry_lock:
            if self._lifecycle is not _BoundaryLifecycle.OPEN:
                return
            self._lifecycle = _BoundaryLifecycle.SHUTTING_DOWN
            pending_acquires = list(self._pending_acquire_futures)
        for pending in pending_acquires:
            pending.cancel()

    def drain_and_close(self) -> None:
        """After physical workers drained, finish releases and stop admission loop."""
        with self._close_lock:
            with self._registry_lock:
                if self._lifecycle is _BoundaryLifecycle.CLOSED:
                    return
                self._lifecycle = _BoundaryLifecycle.SHUTTING_DOWN

            self._wait_for_detached_releases()
            self._wait_for_pending_async_releases()
            failures = self._snapshot_release_failures()
            if failures:
                raise DependencyAttemptReleaseInvariantError(
                    f"dependency attempt boundary drain failed: {failures[0]!r}"
                ) from failures[0]

            loop = self._require_loop()
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=self._STARTUP_TIMEOUT_SECONDS)
            if not self._loop_exited.wait(timeout=self._STARTUP_TIMEOUT_SECONDS):
                raise RuntimeError(
                    "DependencyAttemptExecutionBoundary admission loop did not exit"
                )
            if self._thread.is_alive():
                raise RuntimeError(
                    "DependencyAttemptExecutionBoundary admission thread did not stop"
                )

            with self._registry_lock:
                self._lifecycle = _BoundaryLifecycle.CLOSED

    def close(self) -> None:
        """Idempotent close when no separate pool drain is required (tests only)."""
        self.begin_shutdown()
        self.drain_and_close()

    def _run_admission_loop(self) -> None:
        self._admission_thread = threading.current_thread()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        loop.run_forever()
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
        self._loop_exited.set()

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None:
            raise RuntimeError("invariant: admission loop not initialized")
        return loop

    def _require_accepts_attempts(self) -> None:
        with self._registry_lock:
            if self._lifecycle is not _BoundaryLifecycle.OPEN:
                raise DependencyAttemptExecutionBoundaryClosedError(
                    "dependency attempt boundary is not accepting new attempts"
                )

    def _register_attempt(self, permit: DependencyConcurrencyPermit) -> int:
        with self._registry_lock:
            token = self._next_token
            self._next_token += 1
            self._attempts[token] = _AttemptRecord(permit)
            return token

    def _require_record(self, handle: DependencyAttemptHandle) -> _AttemptRecord:
        with self._registry_lock:
            return self._require_record_locked(handle)

    def _require_record_locked(
        self,
        handle: DependencyAttemptHandle,
    ) -> _AttemptRecord:
        record = self._attempts.get(handle._token)
        if record is None:
            raise RuntimeError("unknown dependency attempt handle")
        return record

    def _on_worker_terminal(self, handle: DependencyAttemptHandle) -> None:
        with self._registry_lock:
            record = self._attempts.get(handle._token)
            if record is None:
                return
            if record.release_claimed:
                return
            if record.phase is _AttemptPhase.ATTACHED:
                return
            if record.phase is not _AttemptPhase.DETACHED:
                return
            record.release_claimed = True
            self._detached_pending_release.discard(handle._token)

        release_future = self._schedule_permit_release(record.permit)
        with self._registry_lock:
            record.release_future = release_future

        def _release_done(release_cf: ConcurrentFuture[object]) -> None:
            try:
                release_cf.result()
            except BaseException as exc:
                self._record_lifecycle_failure(exc)
            finally:
                with self._registry_lock:
                    self._attempts.pop(handle._token, None)
                    self._drain_condition.notify_all()

        release_future.add_done_callback(_release_done)

    def _release_attached(
        self,
        handle: DependencyAttemptHandle,
        record: _AttemptRecord,
    ) -> None:
        release_future = self._schedule_permit_release(record.permit)
        with self._registry_lock:
            record.release_future = release_future
        try:
            release_future.result()
        except BaseException as exc:
            raise DependencyAttemptReleaseInvariantError(
                "dependency permit release failed on attached path"
            ) from exc
        finally:
            with self._registry_lock:
                self._attempts.pop(handle._token, None)
                self._drain_condition.notify_all()

    def _maybe_release_abandoned_acquire_permit(
        self,
        acquire_future: ConcurrentFuture[DependencyConcurrencyPermit],
        handoff: _AcquirePermitHandoff,
    ) -> None:
        """Release a permit when the sync waiter abandoned an async acquire."""
        if handoff.is_release_pending() or handoff.is_release_complete():
            return
        if not handoff.try_claim_release_obligation():
            return
        if not acquire_future.done():
            return
        if acquire_future.cancelled():
            return
        try:
            permit = acquire_future.result(timeout=0)
        except _AcquireAbandonedBeforeDelivery:
            return
        except BaseException:
            return
        self._schedule_handoff_permit_release(permit, handoff)

    async def _release_permit_for_abandoned_handoff(
        self,
        permit: DependencyConcurrencyPermit,
        handoff: _AcquirePermitHandoff,
    ) -> None:
        release_task = asyncio.create_task(permit.release())
        try:
            await asyncio.shield(release_task)
            handoff.complete_release()
        except asyncio.CancelledError:
            release_task.add_done_callback(
                lambda done_task: self._handoff_async_release_terminal(
                    done_task,
                    handoff,
                ),
            )
            raise

    def _handoff_async_release_terminal(
        self,
        release_task: asyncio.Task[object],
        handoff: _AcquirePermitHandoff,
    ) -> None:
        try:
            release_task.result()
            handoff.complete_release()
        except BaseException as exc:
            self._record_lifecycle_failure(exc)

    def _schedule_handoff_permit_release(
        self,
        permit: DependencyConcurrencyPermit,
        handoff: _AcquirePermitHandoff,
    ) -> None:
        handoff.mark_release_pending()
        release_future = self._schedule_permit_release(permit)

        def _release_done(release_cf: ConcurrentFuture[object]) -> None:
            try:
                release_cf.result()
                handoff.complete_release()
            except BaseException as exc:
                self._record_lifecycle_failure(exc)
            finally:
                with self._registry_lock:
                    self._pending_async_release_futures.discard(release_cf)

        with self._registry_lock:
            self._pending_async_release_futures.add(release_future)
        release_future.add_done_callback(_release_done)

    def _execute_permit_release(
        self,
        permit: DependencyConcurrencyPermit,
        *,
        wait: bool,
    ) -> None:
        release_future = self._schedule_permit_release(permit)
        on_admission_loop = (
            self._admission_thread is not None
            and threading.current_thread() is self._admission_thread
        )
        if on_admission_loop or not wait:
            with self._registry_lock:
                self._pending_async_release_futures.add(release_future)

            def _release_done(release_cf: ConcurrentFuture[object]) -> None:
                try:
                    release_cf.result()
                except BaseException as exc:
                    self._record_lifecycle_failure(exc)
                finally:
                    with self._registry_lock:
                        self._pending_async_release_futures.discard(release_cf)

            release_future.add_done_callback(_release_done)
            return
        try:
            release_future.result()
        except BaseException as exc:
            self._record_lifecycle_failure(exc)

    def _schedule_permit_release(
        self,
        permit: DependencyConcurrencyPermit,
    ) -> ConcurrentFuture[object]:
        loop = self._require_loop()

        async def _release_coro() -> None:
            await permit.release()

        return asyncio.run_coroutine_threadsafe(_release_coro(), loop)

    def _wait_for_pending_async_releases(self) -> None:
        deadline = time.monotonic() + self._STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            with self._registry_lock:
                pending = list(self._pending_async_release_futures)
            if not pending:
                return
            for release_future in pending:
                try:
                    release_future.result(timeout=0.05)
                except BaseException as exc:
                    self._record_lifecycle_failure(exc)
            time.sleep(0.01)
        raise RuntimeError(
            "DependencyAttemptExecutionBoundary pending async releases did not drain"
        )

    def _wait_for_detached_releases(self) -> None:
        while True:
            with self._registry_lock:
                if not self._attempts:
                    break
                release_futures = [
                    record.release_future
                    for record in self._attempts.values()
                    if record.release_future is not None
                ]
            for release_future in release_futures:
                release_future.result()
            with self._registry_lock:
                if not self._attempts:
                    break
                if all(
                    record.release_future is not None
                    for record in self._attempts.values()
                ):
                    continue
                self._drain_condition.wait(timeout=0.05)

    def _record_lifecycle_failure(self, exc: BaseException) -> None:
        with self._registry_lock:
            self._release_failures.append(exc)

    def _snapshot_release_failures(self) -> list[BaseException]:
        with self._registry_lock:
            return list(self._release_failures)
