# © Artur Czarnecki. All rights reserved.

"""Shared fakes for hosting engine unit tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from intergrax.hosting import (
    HostedApplicationComponentHealth,
    HostedApplicationComponentRegistration,
    HostedApplicationComponentState,
    HostedApplicationContext,
    HostedApplicationEvent,
    HostedApplicationEventPublisher,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
    HostedApplicationProfile,
    HostedApplicationShutdownCoordinator,
    HostedApplicationShutdownRequestSnapshot,
)
from intergrax.hosting.contracts.context import HostedApplicationClock, HostedApplicationLogger
from intergrax.hosting.engine.ports import (
    HostedApplicationInstanceGuardPort,
    HostedApplicationInstanceIdentity,
    HostedApplicationInstanceLeasePort,
    HostedApplicationRuntime,
)


class FixedClock(HostedApplicationClock):
    def __init__(self, moment: datetime | None = None) -> None:
        self._moment = moment or datetime(2026, 7, 14, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._moment

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._moment = self._moment + timedelta(seconds=seconds)


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
class RecordingPublisher(HostedApplicationEventPublisher):
    events: list[HostedApplicationEvent] = field(default_factory=list)

    async def publish(self, event: HostedApplicationEvent) -> None:
        self.events.append(event)


class FakeLease(HostedApplicationInstanceLeasePort):
    def __init__(self, *, valid: bool = True) -> None:
        self._valid = valid
        self.released = False

    def is_valid(self) -> bool:
        return self._valid and not self.released

    async def release(self) -> None:
        self.released = True


class FakeInstanceGuard(HostedApplicationInstanceGuardPort):
    def __init__(self, lease: FakeLease | None = None, *, fail_acquire: bool = False) -> None:
        self.lease = lease or FakeLease()
        self.acquire_count = 0
        self.fail_acquire = fail_acquire

    async def acquire(
        self,
        identity: HostedApplicationInstanceIdentity,
    ) -> HostedApplicationInstanceLeasePort:
        self.acquire_count += 1
        self.last_identity = identity
        if self.fail_acquire:
            raise RuntimeError("instance acquire rejected")
        return self.lease


class FakeShutdownCoordinator(HostedApplicationShutdownCoordinator):
    def __init__(self) -> None:
        self._requested = False
        self._event = asyncio.Event()
        self._snapshot: HostedApplicationShutdownRequestSnapshot | None = None

    def is_shutdown_requested(self) -> bool:
        return self._requested

    def current_request(self) -> HostedApplicationShutdownRequestSnapshot | None:
        return self._snapshot

    def request_shutdown(
        self,
        reason_code: str,
        *,
        deadline_at: datetime | None = None,
    ) -> HostedApplicationShutdownRequestSnapshot:
        self._requested = True
        self._snapshot = HostedApplicationShutdownRequestSnapshot(
            reason_code=reason_code,
            requested_at=datetime.now(UTC),
            deadline_at=deadline_at,
        )
        self._event.set()
        return self._snapshot

    async def wait_until_requested(self) -> HostedApplicationShutdownRequestSnapshot:
        await self._event.wait()
        assert self._snapshot is not None
        return self._snapshot


class FakeRuntime:
    def __init__(self, *, ready: bool = True, fail_start: bool = False) -> None:
        self.ready_value = ready
        self.fail_start = fail_start
        self.started = False
        self.stopped = False
        self.start_count = 0
        self.stop_count = 0

    async def start(self, context: HostedApplicationContext) -> None:
        self.start_count += 1
        if self.fail_start:
            raise RuntimeError("runtime start failed")
        self.started = True

    async def stop(self, context: HostedApplicationContext) -> None:
        self.stop_count += 1
        self.stopped = True

    async def ready(self, context: HostedApplicationContext) -> bool:
        return self.ready_value and self.started


def runtime_factory(runtime: FakeRuntime | None = None) -> HostedApplicationRuntime:
    holder = runtime or FakeRuntime()
    return holder  # type: ignore[return-value]


def runtime_factory_with_context(
    context: HostedApplicationContext,
    runtime: FakeRuntime | None = None,
) -> HostedApplicationRuntime:
    return runtime_factory(runtime)


async def async_runtime_factory() -> HostedApplicationRuntime:
    return FakeRuntime()  # type: ignore[return-value]


async def async_runtime_factory_with_context(
    context: HostedApplicationContext,
) -> HostedApplicationRuntime:
    return FakeRuntime()  # type: ignore[return-value]


class FakeComponent:
    def __init__(
        self,
        component_id: str,
        *,
        fail_start: bool = False,
        required: bool = False,
        healthy: bool = True,
        ready: bool = True,
        start_delay: float = 0.0,
    ) -> None:
        self.component_id = component_id
        self.fail_start = fail_start
        self.required = required
        self.healthy = healthy
        self.ready = ready
        self.start_delay = start_delay
        self.started = False
        self.stopped = False
        self.start_order: int | None = None

    async def start(self, context: HostedApplicationContext) -> None:
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if self.fail_start:
            raise RuntimeError(f"{self.component_id} start failed")
        self.started = True

    async def stop(self, context: HostedApplicationContext) -> None:
        self.stopped = True

    async def health(self, context: HostedApplicationContext) -> HostedApplicationComponentHealth:
        return HostedApplicationComponentHealth(
            component_id=self.component_id,
            enabled=True,
            required=self.required,
            state=HostedApplicationComponentState.READY if self.started else HostedApplicationComponentState.CREATED,
            healthy=self.healthy,
            ready=self.ready and self.started,
        )


def build_engine_paths() -> HostedApplicationPaths:
    return HostedApplicationPaths(
        data_home=__import__("pathlib").Path("data/test_app"),
        run_directory=__import__("pathlib").Path("data/test_app/run"),
    )


def build_process_identity(clock: FixedClock) -> HostedApplicationProcessIdentity:
    return HostedApplicationProcessIdentity(
        process_id=1000,
        host_id="host-test",
        started_at=clock.now(),
    )


_RUNTIME_HOLDER: dict[str, FakeRuntime] = {}


def test_app_runtime_factory() -> HostedApplicationRuntime:
    runtime = _RUNTIME_HOLDER.get("runtime") or FakeRuntime()
    _RUNTIME_HOLDER["runtime"] = runtime
    return runtime  # type: ignore[return-value]


def minimal_profile_with_runtime(runtime: FakeRuntime | None = None) -> HostedApplicationProfile:
    if runtime is not None:
        _RUNTIME_HOLDER["runtime"] = runtime
    else:
        _RUNTIME_HOLDER.pop("runtime", None)
    return HostedApplicationProfile(
        application_id="test_app",
        application_factory=test_app_runtime_factory,
        application_factory_id="tests.unit.hosting.engine._fakes.test_app_runtime_factory",
    )


def component_registration(
    component: FakeComponent,
    *,
    required: bool = False,
    dependencies: tuple[str, ...] = (),
    failure_action=None,
) -> HostedApplicationComponentRegistration:
    kwargs: dict = {
        "component": component,
        "required": required,
        "dependencies": dependencies,
    }
    if failure_action is not None:
        kwargs["failure_action"] = failure_action
    return HostedApplicationComponentRegistration(**kwargs)
