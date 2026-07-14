# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

import pytest

from intergrax.hosting import (
    HostedApplicationContext,
    HostedApplicationHook,
    HostedApplicationHookPoint,
    HostedApplicationProfile,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder
from intergrax.hosting.engine.hooks import HookCoordinator
from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry
from tests.unit.hosting.engine._fakes import FixedClock, NoopLogger, build_engine_paths, build_process_identity
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit


def _build_hook_coordinator(
    hooks: tuple[HostedApplicationHook, ...],
    *,
    events: list[HostedApplicationEvent] | None = None,
) -> tuple[HookCoordinator, HostedApplicationContext]:
    captured = events if events is not None else []

    async def publish(event: HostedApplicationEvent) -> None:
        captured.append(event)

    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        hooks=__import__("intergrax.hosting", fromlist=["HostedApplicationHooks"]).HostedApplicationHooks(
            before_start=hooks,
        ),
    )
    definition = resolve_hosted_application_definition(profile)
    clock = FixedClock()
    diagnostics = DiagnosticsRecorder(
        clock=clock,
        application_id="test_app",
        instance_id="instance-001",
        profile_digest=definition.profile_digest,
        definition_digest=definition.definition_digest,
    )
    observer_tasks = ObserverTaskRegistry(diagnostics)
    coordinator = HookCoordinator(
        definition,
        definition.lifecycle_policy,
        diagnostics,
        observer_tasks,
        publish,
    )
    from intergrax.hosting.engine.lifecycle import HostedApplicationLifecycleController
    from intergrax.hosting.services import HostedApplicationServiceRegistry

    lifecycle = HostedApplicationLifecycleController(clock)
    context = HostedApplicationContext(
        application_id="test_app",
        instance_id="instance-001",
        profile=profile.public_view(),
        profile_digest=definition.profile_digest,
        paths=build_engine_paths(),
        process_identity=build_process_identity(clock),
        services=HostedApplicationServiceRegistry(),
        clock=clock,
        logger=NoopLogger(),
        event_publisher=__import__("tests.unit.hosting.engine._fakes", fromlist=["RecordingPublisher"]).RecordingPublisher(),
        shutdown=__import__("tests.unit.hosting.engine._fakes", fromlist=["FakeShutdownCoordinator"]).FakeShutdownCoordinator(),
        lifecycle=lifecycle,
    )
    return coordinator, context


@pytest.mark.asyncio
async def test_sync_hook_invoked_exactly_once_off_event_loop() -> None:
    calls = {"count": 0, "thread": None}

    def handler(_context: HostedApplicationContext) -> None:
        calls["count"] += 1
        calls["thread"] = threading.current_thread().ident

    hook = HostedApplicationHook(
        hook_id="sync_hook",
        handler=handler,
        handler_id="tests.sync_hook",
    )
    coordinator, context = _build_hook_coordinator((hook,))
    main_thread = threading.current_thread().ident
    await coordinator.execute_blocking(HostedApplicationHookPoint.BEFORE_START, context)
    assert calls["count"] == 1
    assert calls["thread"] != main_thread


@pytest.mark.asyncio
async def test_async_hook_invoked_exactly_once() -> None:
    calls = {"count": 0}

    async def handler(_context: HostedApplicationContext) -> None:
        calls["count"] += 1

    hook = HostedApplicationHook(
        hook_id="async_hook",
        handler=handler,
        handler_id="tests.async_hook",
    )
    coordinator, context = _build_hook_coordinator((hook,))
    await coordinator.execute_blocking(HostedApplicationHookPoint.BEFORE_START, context)
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_sync_hook_returning_awaitable_executes_once() -> None:
    calls = {"sync": 0, "inner": 0}

    def handler(_context: HostedApplicationContext):
        calls["sync"] += 1

        async def inner() -> None:
            calls["inner"] += 1

        return inner()

    hook = HostedApplicationHook(
        hook_id="sync_awaitable_hook",
        handler=handler,
        handler_id="tests.sync_awaitable_hook",
    )
    coordinator, context = _build_hook_coordinator((hook,))
    await coordinator.execute_blocking(HostedApplicationHookPoint.BEFORE_START, context)
    assert calls["sync"] == 1
    assert calls["inner"] == 1


@pytest.mark.asyncio
async def test_hook_lifecycle_events_emitted_in_order() -> None:
    events: list[HostedApplicationEvent] = []

    async def handler(_context: HostedApplicationContext) -> None:
        return None

    hook = HostedApplicationHook(
        hook_id="tracked_hook",
        handler=handler,
        handler_id="tests.tracked_hook",
        source_id="profile",
    )
    coordinator, context = _build_hook_coordinator((hook,), events=events)
    await coordinator.execute_blocking(HostedApplicationHookPoint.BEFORE_START, context)
    hook_events = [event for event in events if event.event_type.value.startswith("hosting.hook.")]
    assert [event.event_type for event in hook_events] == [
        HostedApplicationEventType.HOOK_STARTED,
        HostedApplicationEventType.HOOK_COMPLETED,
    ]
    assert hook_events[0].payload["hook_id"] == "tracked_hook"
