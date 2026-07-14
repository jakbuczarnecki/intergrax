# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

import pytest

from intergrax.hosting import HostedApplicationEvent, HostedApplicationEventSubscription, HostedApplicationEventType, HostedApplicationProfile
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder
from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry
from intergrax.hosting.eventing import HostingEventDispatcher
from tests.unit.hosting.engine._fakes import FixedClock, RecordingPublisher
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_sync_subscriber_invoked_exactly_once_off_event_loop() -> None:
    calls = {"count": 0, "thread": None}

    def handler(_event: HostedApplicationEvent) -> None:
        calls["count"] += 1
        calls["thread"] = threading.current_thread().ident

    subscription = HostedApplicationEventSubscription(
        subscription_id="sub_sync",
        event_types=(HostedApplicationEventType.APPLICATION_READY,),
        handler=handler,
        handler_id="tests.sub_sync",
        source_id="profile",
    )
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        event_subscriptions=(subscription,),
    )
    from intergrax.hosting import resolve_hosted_application_definition

    definition = resolve_hosted_application_definition(profile)
    clock = FixedClock()
    diagnostics = DiagnosticsRecorder(
        clock=clock,
        application_id="test_app",
        instance_id="instance-001",
        profile_digest=definition.profile_digest,
        definition_digest=definition.definition_digest,
    )
    dispatcher = HostingEventDispatcher(
        RecordingPublisher(),
        definition.event_subscriptions,
        diagnostics,
        ObserverTaskRegistry(diagnostics),
    )
    main_thread = threading.current_thread().ident
    await dispatcher.publish(
        HostedApplicationEvent(
            event_type=HostedApplicationEventType.APPLICATION_READY,
            application_id="test_app",
            instance_id="instance-001",
            lifecycle_state=__import__("intergrax.hosting", fromlist=["HostedApplicationLifecycleState"]).HostedApplicationLifecycleState.READY,
        )
    )
    await __import__("asyncio").sleep(0.05)
    assert calls["count"] == 1
    assert calls["thread"] != main_thread


@pytest.mark.asyncio
async def test_subscriber_order_uses_declaration_index() -> None:
    order: list[str] = []

    def make_handler(name: str):
        def handler(_event: HostedApplicationEvent) -> None:
            order.append(name)

        return handler

    subscriptions = (
        HostedApplicationEventSubscription(
            subscription_id="sub_z",
            event_types=(HostedApplicationEventType.APPLICATION_READY,),
            handler=make_handler("z"),
            handler_id="tests.sub_z",
            source_id="profile",
            priority=0,
        ),
        HostedApplicationEventSubscription(
            subscription_id="sub_a",
            event_types=(HostedApplicationEventType.APPLICATION_READY,),
            handler=make_handler("a"),
            handler_id="tests.sub_a",
            source_id="profile",
            priority=0,
        ),
    )
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        event_subscriptions=subscriptions,
    )
    from intergrax.hosting import resolve_hosted_application_definition

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
    dispatcher = HostingEventDispatcher(
        RecordingPublisher(),
        definition.event_subscriptions,
        diagnostics,
        observer_tasks,
    )
    await dispatcher.publish(
        HostedApplicationEvent(
            event_type=HostedApplicationEventType.APPLICATION_READY,
            application_id="test_app",
            instance_id="instance-001",
            lifecycle_state=__import__("intergrax.hosting", fromlist=["HostedApplicationLifecycleState"]).HostedApplicationLifecycleState.READY,
        )
    )
    await observer_tasks.drain(1.0)
    assert order == ["z", "a"]
