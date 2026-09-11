# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sync physical dependency attempt boundary over async admission (W2-B2)."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionPort,
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyPermit,
)

if TYPE_CHECKING:
    from concurrent.futures import Future as ThreadSafeFuture


class DependencyAttemptExecutionBoundaryClosedError(RuntimeError):
    """Boundary no longer accepts new physical dependency attempts."""


class DependencyAttemptReleaseInvariantError(RuntimeError):
    """Permit release failed for a physical dependency attempt (fail-closed)."""


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
        self._detached_pending_release: set[int] = set()
        self._release_failures: list[BaseException] = []
        self._drain_condition = threading.Condition(self._registry_lock)
        self._loop_ready = threading.Event()
        self._loop_exited = threading.Event()
        self._close_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
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
    ) -> DependencyAttemptHandle:
        """Block until admission acquire completes or raises a typed admission error."""
        self._require_accepts_attempts()
        loop = self._require_loop()

        async def _acquire_coro() -> DependencyConcurrencyPermit:
            return await self._admission.acquire(request)

        acquire_future = asyncio.run_coroutine_threadsafe(_acquire_coro(), loop)
        with self._registry_lock:
            self._pending_acquire_futures.add(acquire_future)
        try:
            permit = acquire_future.result()
        finally:
            with self._registry_lock:
                self._pending_acquire_futures.discard(acquire_future)

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

    def _schedule_permit_release(
        self,
        permit: DependencyConcurrencyPermit,
    ) -> ConcurrentFuture[object]:
        loop = self._require_loop()

        async def _release_coro() -> None:
            await permit.release()

        return asyncio.run_coroutine_threadsafe(_release_coro(), loop)

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
