# © Artur Czarnecki. All rights reserved.

"""Shared support for the PDC-X D4 host-runtime qualification suite.

The suite drives the real ``HostedApplicationEngine``, ``HostedApplicationSupervisor``
and ``FileHostedApplicationInstanceGuard``. Only the wall clock and the event sink
are replaced by small in-process adapters.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from intergrax.hosting import (
    HostedApplicationComponentHealth,
    HostedApplicationComponentRegistration,
    HostedApplicationComponentState,
    HostedApplicationContext,
    HostedApplicationDefinition,
    HostedApplicationEngine,
    HostedApplicationEvent,
    HostedApplicationEventType,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
    HostedApplicationProfile,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.context import HostedApplicationLogger
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.instance.file_guard import FileHostedApplicationInstanceGuard

APP_ID = "pdc_d4_app"


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class NoopLogger(HostedApplicationLogger):
    def debug(self, message: str, **fields: object) -> None:
        return None

    def info(self, message: str, **fields: object) -> None:
        return None

    def warning(self, message: str, **fields: object) -> None:
        return None

    def error(self, message: str, **fields: object) -> None:
        return None


@dataclass
class RecordingPublisher:
    """Event sink standing in for the canonical observability port."""

    events: list[HostedApplicationEvent] = field(default_factory=list)

    async def publish(self, event: HostedApplicationEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: HostedApplicationEventType) -> list[HostedApplicationEvent]:
        return [event for event in self.events if event.event_type is event_type]


class OrderedComponent:
    """Component that logs start/stop into a shared journal and can fail or degrade."""

    component_id = "ordered"

    def __init__(
        self,
        component_id: str,
        journal: list[str],
        *,
        fail_start: bool = False,
        healthy: bool = True,
    ) -> None:
        self.component_id = component_id
        self._journal = journal
        self._fail_start = fail_start
        self.healthy = healthy
        self.started = False

    async def start(self, context: HostedApplicationContext) -> None:
        self._journal.append(f"start:{self.component_id}")
        if self._fail_start:
            raise RuntimeError(f"{self.component_id} start failed")
        self.started = True

    async def stop(self, context: HostedApplicationContext) -> None:
        self._journal.append(f"stop:{self.component_id}")
        self.started = False

    async def health(self, context: HostedApplicationContext) -> HostedApplicationComponentHealth:
        return HostedApplicationComponentHealth(
            component_id=self.component_id,
            enabled=True,
            required=False,
            state=(
                HostedApplicationComponentState.READY
                if self.started and self.healthy
                else HostedApplicationComponentState.DEGRADED
            ),
            healthy=self.healthy,
            ready=self.started and self.healthy,
        )


class JournalRuntime:
    """Minimal application runtime recording its own start/stop."""

    def __init__(self, journal: list[str] | None = None) -> None:
        self.journal = journal if journal is not None else []
        self.started = False

    async def start(self, context: HostedApplicationContext) -> None:
        self.journal.append("start:runtime")
        self.started = True

    async def stop(self, context: HostedApplicationContext) -> None:
        self.journal.append("stop:runtime")
        self.started = False

    async def ready(self, context: HostedApplicationContext) -> bool:
        return self.started


_RUNTIME_HOLDER: dict[str, JournalRuntime] = {}


def journal_runtime_factory() -> JournalRuntime:
    runtime = _RUNTIME_HOLDER.get("runtime")
    if runtime is None:
        runtime = JournalRuntime()
        _RUNTIME_HOLDER["runtime"] = runtime
    return runtime


def make_profile(
    *,
    application_id: str = APP_ID,
    components: tuple[HostedApplicationComponentRegistration, ...] = (),
    runtime: JournalRuntime | None = None,
    **policies: object,
) -> HostedApplicationProfile:
    if runtime is None:
        _RUNTIME_HOLDER.pop("runtime", None)
    else:
        _RUNTIME_HOLDER["runtime"] = runtime
    return HostedApplicationProfile(
        application_id=application_id,
        application_factory=journal_runtime_factory,
        application_factory_id="tests.qualification.pdc_x.d4._support.journal_runtime_factory",
        components=components,
        **policies,  # type: ignore[arg-type]
    )


def build_paths(root: Path) -> HostedApplicationPaths:
    data_home = root / "data"
    run_directory = root / "run"
    data_home.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    return HostedApplicationPaths(data_home=data_home, run_directory=run_directory)


def build_engine(
    definition: HostedApplicationDefinition,
    root: Path,
    *,
    publisher: RecordingPublisher | None = None,
    control: HostedApplicationControlCoordinator | None = None,
    instance_id: str = "instance-001",
    active_work_controller: object | None = None,
) -> tuple[HostedApplicationEngine, RecordingPublisher, HostedApplicationControlCoordinator]:
    """Build a real engine over a real file instance guard rooted at ``root``."""
    clock = SystemClock()
    publisher = publisher or RecordingPublisher()
    control = control or HostedApplicationControlCoordinator(clock=clock)
    process_identity = HostedApplicationProcessIdentity(
        process_id=os.getpid(),
        started_at=clock.now(),
        host_id="pdc-d4-host",
    )
    paths = build_paths(root)
    guard = FileHostedApplicationInstanceGuard(
        run_directory=paths.run_directory,
        instance_policy=definition.instance_policy,
        process_identity=process_identity,
        clock=clock,
    )
    engine = HostedApplicationEngine(
        definition=definition,
        instance_id=instance_id,
        paths=paths,
        process_identity=process_identity,
        clock=clock,
        logger=NoopLogger(),
        shutdown=control,
        event_publisher=publisher,
        instance_guard=guard,
        active_work_controller=active_work_controller,  # type: ignore[arg-type]
        health_poll_interval_seconds=0.05,
    )
    return engine, publisher, control


def definition_of(profile: HostedApplicationProfile) -> HostedApplicationDefinition:
    return resolve_hosted_application_definition(profile)


class AsyncioActiveWork:
    """Active-work controller over asyncio tasks (the hosting drain port)."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._intake_open = True
        self.completed: list[str] = []
        self.cancelled: list[str] = []

    def submit(self, name: str, work: Callable[[], Awaitable[None]]) -> asyncio.Task[None]:
        if not self._intake_open:
            raise RuntimeError("intake closed")

        async def runner() -> None:
            try:
                await work()
                self.completed.append(name)
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise

        task = asyncio.ensure_future(runner())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def stop_intake(self) -> None:
        self._intake_open = False

    def active_work_count(self) -> int:
        return sum(1 for task in self._tasks if not task.done())

    async def wait_for_idle(self) -> None:
        pending = [task for task in self._tasks if not task.done()]
        if pending:
            # asyncio.wait never cancels the awaited tasks when the drain budget expires.
            await asyncio.wait(pending)

    async def cancel_active_work(self) -> None:
        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
