# © Artur Czarnecki. All rights reserved.

"""PDC-D4-R007 (APP-HOST-5E): OS-subprocess kill and supervised restart.

A hosted application owns a real OS worker process. The test hard-kills the worker by its
recorded PID; the application observes the death, asks for a restart through the control
coordinator, and the existing ``HostedApplicationSupervisor`` launches a replacement engine
and worker. No second supervisor is involved.

Process safety: every worker is self-bounded (see ``_worker_process.py``), started with a
recorded ``Popen`` handle, and cleaned up by exact handle in ``finally``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationContext,
    HostedApplicationEventType,
    HostedApplicationExitKind,
    HostedApplicationProfile,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
    RestartPolicy,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting.control import HostedApplicationControlCoordinator
from tests.qualification.pdc_x.d4._support import (
    APP_ID,
    RecordingPublisher,
    SystemClock,
    build_engine,
)

pytestmark = [pytest.mark.integration]

_WORKER_SCRIPT = Path(__file__).with_name("_worker_process.py")
_WORKER_MAX_SECONDS = 60.0
_WAIT_SECONDS = 20.0
_FACTORY_ID = "tests.qualification.pdc_x.d4.crash_restart.worker_runtime_factory"


class _SupervisedWorkerRuntime:
    """Hosted runtime that owns one OS worker and requests a restart if it dies."""

    def __init__(self, control: HostedApplicationControlCoordinator, workers: list[subprocess.Popen[str]]):
        self._control = control
        self._workers = workers
        self._process: subprocess.Popen[str] | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self, context: HostedApplicationContext) -> None:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, test-owned script
            [sys.executable, "-u", str(_WORKER_SCRIPT), str(_WORKER_MAX_SECONDS)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._workers.append(process)
        self._process = process
        assert process.stdout is not None
        banner = await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), timeout=_WAIT_SECONDS)
        assert banner.strip() == "ready"
        self._monitor = asyncio.ensure_future(self._watch(process))

    async def _watch(self, process: subprocess.Popen[str]) -> None:
        while process.poll() is None:
            await asyncio.sleep(0.05)
        if not self._stopping:
            self._control.request_restart("worker_process_exited", source_id="worker_monitor")

    async def stop(self, context: HostedApplicationContext) -> None:
        self._stopping = True
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, 5.0)
        if process is not None and process.stdout is not None:
            process.stdout.close()

    async def ready(self, context: HostedApplicationContext) -> bool:
        return self._process is not None and self._process.poll() is None


async def _wait_until(predicate, description: str) -> None:
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {description}")
        await asyncio.sleep(0.05)


def _ready_count(publisher: RecordingPublisher) -> int:
    return len(publisher.of_type(HostedApplicationEventType.APPLICATION_READY))


def _cleanup(workers: list[subprocess.Popen[str]]) -> None:
    """Kill only the exact worker handles this test started."""
    for process in workers:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - last-resort diagnostics
            pass
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()


def _build_supervisor(
    tmp_path: Path,
    workers: list[subprocess.Popen[str]],
    *,
    max_attempts: int,
) -> tuple[HostedApplicationSupervisor, RecordingPublisher, HostedApplicationControlCoordinator]:
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()

    def factory_for_attempt() -> object:
        return _SupervisedWorkerRuntime(control, workers)

    profile = HostedApplicationProfile(
        application_id=APP_ID,
        application_factory=factory_for_attempt,
        application_factory_id=_FACTORY_ID,
        restart=RestartPolicy(
            mode=RestartMode.ON_FAILURE,
            max_attempts=max_attempts,
            initial_backoff_seconds=0.05,
            max_backoff_seconds=0.2,
            jitter_ratio=0.0,
        ),
    )
    definition = resolve_hosted_application_definition(profile)
    counter = iter(range(1000))

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        engine, _, _ = build_engine(
            launch.definition,
            tmp_path,
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
        )
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: f"worker-attempt-{next(counter)}",
    )
    return supervisor, publisher, control


async def test_r007_killed_worker_process_is_restarted_by_the_supervisor(tmp_path: Path) -> None:
    workers: list[subprocess.Popen[str]] = []
    supervisor, publisher, control = _build_supervisor(tmp_path, workers, max_attempts=3)
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(lambda: _ready_count(publisher) == 1, "first instance ready")
        assert len(workers) == 1 and workers[0].poll() is None
        victim = workers[0]
        victim_pid = victim.pid
        assert victim_pid != os.getpid()

        victim.kill()  # hard kill of exactly the recorded worker PID

        await _wait_until(
            lambda: _ready_count(publisher) == 2,
            "replacement instance ready after supervised restart",
        )
        assert len(workers) == 2 and workers[1].poll() is None
        replacement = workers[1]
        assert replacement.pid not in {victim_pid, os.getpid()}
        assert victim.poll() is not None and victim.returncode != 0, "victim must be dead (killed)"

        control.request_shutdown("test.done")
        result = await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("test.abort")
            await asyncio.gather(run, return_exceptions=True)
        _cleanup(workers)

    kinds = [attempt.exit_record.exit_kind for attempt in result.attempts if attempt.exit_record]
    assert kinds == [HostedApplicationExitKind.RESTART_REQUESTED, HostedApplicationExitKind.CLEAN_STOP]
    assert result.restart_exhausted is False
    assert result.final_exit.exit_kind is HostedApplicationExitKind.CLEAN_STOP
    assert all(attempt.cleanup_verified for attempt in result.attempts)
    # Each attempt ran as its own engine instance with the same profile digest.
    assert [attempt.instance_id for attempt in result.attempts] == ["worker-attempt-0", "worker-attempt-1"]
    assert {attempt.exit_record.profile_digest for attempt in result.attempts if attempt.exit_record} == {
        result.profile_digest
    }
    restart_events = [
        event.event_type
        for event in publisher.events
        if event.instance_id == "supervisor" and event.event_type is not HostedApplicationEventType.LIFECYCLE_METRIC
    ]
    assert restart_events == [
        HostedApplicationEventType.RESTART_REQUESTED,
        HostedApplicationEventType.RESTART_SCHEDULED,
        HostedApplicationEventType.RESTART_STARTED,
    ]
    assert all(process.poll() is not None for process in workers), "no worker may outlive the test"
    assert APP_ID == result.application_id


async def test_r007_repeated_worker_kills_exhaust_the_restart_budget(tmp_path: Path) -> None:
    workers: list[subprocess.Popen[str]] = []
    supervisor, publisher, control = _build_supervisor(tmp_path, workers, max_attempts=1)
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(lambda: _ready_count(publisher) == 1, "first instance ready")
        workers[0].kill()
        await _wait_until(lambda: _ready_count(publisher) == 2, "replacement instance ready")
        workers[1].kill()
        result = await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("test.abort")
            await asyncio.gather(run, return_exceptions=True)
        _cleanup(workers)

    assert len(workers) == 2, "the budget allows exactly one replacement"
    assert result.restart_exhausted is True
    assert result.final_exit.exit_kind is HostedApplicationExitKind.RESTART_REQUESTED
    assert publisher.of_type(HostedApplicationEventType.RESTART_EXHAUSTED)
    assert all(process.poll() is not None for process in workers)
