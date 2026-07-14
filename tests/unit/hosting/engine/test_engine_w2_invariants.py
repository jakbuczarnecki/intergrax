# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
import warnings

import pytest

from intergrax.hosting import (
    HostedApplicationEngine,
    HostedApplicationEvent,
    HostedApplicationEventSubscription,
    HostedApplicationEventType,
    HostedApplicationHook,
    HostedApplicationHooks,
    HostedApplicationLifecycleState,
    HostedApplicationProfile,
    resolve_hosted_application_definition,
)
from intergrax.hosting.engine.diagnostics import HostedApplicationFailurePhase
from intergrax.hosting.errors import HostedApplicationStartupError
from tests.unit.hosting.engine._fakes import (
    FakeComponent,
    FakeInstanceGuard,
    FakeRuntime,
    FixedClock,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
    build_process_identity,
    component_registration,
    minimal_profile_with_runtime,
)

pytestmark = pytest.mark.unit


def _build_engine(
    profile: HostedApplicationProfile | None = None,
    *,
    runtime: FakeRuntime | None = None,
    publisher: RecordingPublisher | None = None,
    guard: FakeInstanceGuard | None = None,
) -> HostedApplicationEngine:
    clock = FixedClock()
    resolved_profile = profile or minimal_profile_with_runtime(runtime)
    definition = resolve_hosted_application_definition(resolved_profile)
    return HostedApplicationEngine(
        definition=definition,
        instance_id="instance-001",
        paths=build_engine_paths(),
        process_identity=build_process_identity(clock),
        clock=clock,
        logger=NoopLogger(),
        shutdown=__import__(
            "tests.unit.hosting.engine._fakes",
            fromlist=["FakeShutdownCoordinator"],
        ).FakeShutdownCoordinator(),
        event_publisher=publisher or RecordingPublisher(),
        instance_guard=guard or FakeInstanceGuard(),
        health_poll_interval_seconds=0.01,
    )


def test_source_profile_metadata_mutation_isolated() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
        metadata={"tier": "local"},
    )
    definition = resolve_hosted_application_definition(profile)
    original_digest = definition.definition_digest
    original_view = definition.public_view()
    object.__setattr__(profile, "metadata", {"tier": "mutated"})
    assert definition.profile_public_snapshot.metadata == {"tier": "local"}
    assert definition.public_view().profile.metadata == {"tier": "local"}
    assert definition.definition_digest == original_digest
    assert original_view.profile.metadata == {"tier": "local"}


def test_nested_metadata_mutation_after_resolution_is_isolated() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
        metadata={"nested": {"key": "value"}},
    )
    definition = resolve_hosted_application_definition(profile)
    view = definition.public_view()
    payload = view.model_dump(mode="python")
    nested = payload["profile"]["metadata"]["nested"]
    if isinstance(nested, dict):
        nested["key"] = "mutated"
    assert definition.public_view().profile.metadata == {"nested": {"key": "value"}}


def test_public_view_metadata_mutation_does_not_affect_next_view() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
        metadata={"flag": True},
    )
    definition = resolve_hosted_application_definition(profile)
    first = definition.public_view()
    dumped = first.model_dump(mode="python")
    metadata = dumped["profile"]["metadata"]
    if isinstance(metadata, dict):
        metadata["flag"] = False
    second = definition.public_view()
    assert second.profile.metadata == {"flag": True}


def test_definition_mappings_reject_mutation() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
        components=(component_registration(FakeComponent("worker")),),
    )
    definition = resolve_hosted_application_definition(profile)
    with pytest.raises(TypeError):
        definition.enabled_components["worker"] = next(iter(definition.enabled_components.values()))  # type: ignore[index]


def test_definition_digest_aligned_with_snapshot() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
    )
    first = resolve_hosted_application_definition(profile)
    second = resolve_hosted_application_definition(profile)
    assert first.definition_digest == second.definition_digest
    assert first.profile_digest == second.profile_digest


@pytest.mark.asyncio
async def test_multiple_blocking_components_classified_health_evaluation() -> None:
    worker = FakeComponent("worker", required=True, healthy=False, ready=False)
    cache = FakeComponent("cache", required=True, healthy=False, ready=False)
    base = minimal_profile_with_runtime(FakeRuntime())
    profile = HostedApplicationProfile(
        application_id=base.application_id,
        application_factory=base.application_factory,
        application_factory_id=base.application_factory_id,
        components=(
            component_registration(worker, required=True),
            component_registration(cache, required=True),
        ),
    )
    engine = _build_engine(profile)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.current_failure is not None
    assert diagnostics.current_failure.phase is HostedApplicationFailurePhase.HEALTH_EVALUATION
    assert diagnostics.current_failure.reason_code == "blocking_components"
    assert set(diagnostics.health.blocking_component_ids) == {"worker", "cache"}


@pytest.mark.asyncio
async def test_health_poll_failure_clears_readiness_and_recovers() -> None:
    runtime = FakeRuntime()
    engine = _build_engine(runtime=runtime)
    await engine.start()
    assert engine.health_snapshot().ready
    assert engine.health_snapshot().accepting_new_work

    async def failing_ready(_context) -> bool:
        raise RuntimeError("ready probe failed")

    runtime.ready = failing_ready  # type: ignore[method-assign]
    await engine._refresh_health()
    snapshot = engine.health_snapshot()
    assert not snapshot.ready
    assert not snapshot.accepting_new_work
    assert snapshot.health_evaluation_failed

    async def healthy_ready(_context) -> bool:
        return True

    runtime.ready = healthy_ready  # type: ignore[method-assign]
    await engine._refresh_health()
    recovered = engine.health_snapshot()
    assert recovered.ready
    assert recovered.accepting_new_work
    assert not recovered.health_evaluation_failed

    result = await engine.stop(reason_code="test.complete")
    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED


@pytest.mark.asyncio
async def test_accepting_new_work_never_true_when_ready_false() -> None:
    coordinator_engine = _build_engine(runtime=FakeRuntime(ready=False))
    with pytest.raises(HostedApplicationStartupError):
        await coordinator_engine.start()
    health = coordinator_engine.health_snapshot()
    assert not health.ready
    assert not health.accepting_new_work


@pytest.mark.asyncio
async def test_stopped_event_carries_stopped_lifecycle_state() -> None:
    publisher = RecordingPublisher()
    engine = _build_engine(publisher=publisher)
    await engine.start()
    await engine.stop(reason_code="test.complete")
    stopped_events = [
        event
        for event in publisher.events
        if event.event_type is HostedApplicationEventType.APPLICATION_STOPPED
    ]
    assert len(stopped_events) == 1
    assert stopped_events[0].lifecycle_state is HostedApplicationLifecycleState.STOPPED


@pytest.mark.asyncio
async def test_lifecycle_events_carry_matching_states() -> None:
    publisher = RecordingPublisher()
    engine = _build_engine(publisher=publisher)
    await engine.start()
    await engine.stop(reason_code="test.complete")
    expected = {
        HostedApplicationEventType.APPLICATION_STARTING: HostedApplicationLifecycleState.STARTING,
        HostedApplicationEventType.APPLICATION_READY: HostedApplicationLifecycleState.READY,
        HostedApplicationEventType.APPLICATION_STOPPING: HostedApplicationLifecycleState.STOPPING,
        HostedApplicationEventType.APPLICATION_STOPPED: HostedApplicationLifecycleState.STOPPED,
    }
    for event in publisher.events:
        if event.event_type in expected:
            assert event.lifecycle_state is expected[event.event_type]


@pytest.mark.asyncio
async def test_startup_abort_executes_before_stop() -> None:
    calls: list[str] = []

    async def before_stop(_context) -> None:
        calls.append("before_stop")

    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
        hooks=HostedApplicationHooks(
            before_stop=(
                HostedApplicationHook(
                    hook_id="before_stop",
                    handler=before_stop,
                    handler_id="tests.before_stop",
                ),
            ),
        ),
    )
    runtime = FakeRuntime(start_delay=0.1)
    engine = _build_engine(profile, runtime=runtime)
    start_task = asyncio.create_task(engine.start())
    stop_task = asyncio.create_task(engine.stop(reason_code="startup.abort"))
    await asyncio.gather(start_task, stop_task, return_exceptions=True)
    assert "before_stop" in calls


@pytest.mark.asyncio
async def test_after_stop_hook_event_subscriptions_drained() -> None:
    hook_events: list[HostedApplicationEventType] = []
    terminal_events: list[HostedApplicationEventType] = []

    async def after_stop_hook(context) -> None:
        await context.event_publisher.publish(
            HostedApplicationEvent(
                event_type=HostedApplicationEventType.HOOK_STARTED,
                application_id=context.application_id,
                instance_id=context.instance_id,
                lifecycle_state=context.lifecycle.snapshot().state,
                payload={"hook_id": "after_stop", "hook_point": "after_stop"},
            )
        )

    def terminal_handler(event: HostedApplicationEvent) -> None:
        terminal_events.append(event.event_type)

    def hook_handler(event: HostedApplicationEvent) -> None:
        hook_events.append(event.event_type)

    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=minimal_profile_with_runtime().application_factory,
        hooks=HostedApplicationHooks(
            after_stop=(
                HostedApplicationHook(
                    hook_id="after_stop",
                    handler=after_stop_hook,
                    handler_id="tests.after_stop",
                ),
            ),
        ),
        event_subscriptions=(
            HostedApplicationEventSubscription(
                subscription_id="hook_sub",
                event_types=(
                    HostedApplicationEventType.HOOK_STARTED,
                    HostedApplicationEventType.HOOK_COMPLETED,
                    HostedApplicationEventType.HOOK_FAILED,
                ),
                handler=hook_handler,
                handler_id="tests.hook_sub",
                source_id="profile",
            ),
            HostedApplicationEventSubscription(
                subscription_id="terminal_sub",
                event_types=(HostedApplicationEventType.APPLICATION_STOPPED,),
                handler=terminal_handler,
                handler_id="tests.terminal_sub",
                source_id="profile",
            ),
        ),
    )
    engine = _build_engine(profile)
    await engine.start()
    result = await engine.stop(reason_code="test.complete")
    assert HostedApplicationEventType.HOOK_STARTED in hook_events
    assert HostedApplicationEventType.APPLICATION_STOPPED in terminal_events
    assert result.diagnostics.observer_task_count == 0


@pytest.mark.asyncio
async def test_registry_rejection_emits_no_coroutine_warning() -> None:
    from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder
    from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry

    clock = FixedClock()
    diagnostics = DiagnosticsRecorder(
        clock=clock,
        application_id="test_app",
        instance_id="instance-001",
        profile_digest="sha256:" + "a" * 64,
        definition_digest="sha256:" + "b" * 64,
    )
    registry = ObserverTaskRegistry(diagnostics)
    registry.close_to_new_tasks()

    async def noop() -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry.schedule(lambda: noop(), phase=HostedApplicationFailurePhase.AFTER_STOP_OBSERVER, source_id="noop")
        await asyncio.sleep(0)
    assert not any("coroutine" in str(item.message).lower() for item in caught)
