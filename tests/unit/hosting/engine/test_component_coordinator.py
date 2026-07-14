# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.hosting import (
    HostedApplicationComponentRegistration,
    HostedApplicationContext,
    HostedApplicationProfile,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.policies import ComponentFailureAction
from intergrax.hosting.engine.components import ComponentCoordinator
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder
from intergrax.hosting.errors import HostedApplicationComponentError
from intergrax.hosting.eventing import HostingEventDispatcher
from tests.unit.hosting.engine._fakes import (
    FakeComponent,
    FixedClock,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
    build_process_identity,
    component_registration,
)
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit


def _build_component_coordinator(
    components: tuple[HostedApplicationComponentRegistration, ...],
) -> tuple[ComponentCoordinator, HostedApplicationContext]:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        components=components,
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
    publisher = RecordingPublisher()
    from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry

    dispatcher = HostingEventDispatcher(
        publisher,
        definition.event_subscriptions,
        diagnostics,
        ObserverTaskRegistry(diagnostics),
    )
    coordinator = ComponentCoordinator(
        definition=definition,
        lifecycle_policy=definition.lifecycle_policy,
        diagnostics=diagnostics,
        publish_event=dispatcher,
    )
    from intergrax.hosting.engine.lifecycle import HostedApplicationLifecycleController
    from intergrax.hosting.services import HostedApplicationServiceRegistry
    from tests.unit.hosting.engine._fakes import FakeShutdownCoordinator

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
        event_publisher=dispatcher,
        shutdown=FakeShutdownCoordinator(),
        lifecycle=lifecycle,
    )
    return coordinator, context


def test_declaration_order_preserved_for_independent_components() -> None:
    zebra = FakeComponent("zebra")
    alpha = FakeComponent("alpha")
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        components=(
            component_registration(zebra),
            component_registration(alpha),
        ),
    )
    definition = resolve_hosted_application_definition(profile)
    assert definition.component_start_order == ("zebra", "alpha")


@pytest.mark.asyncio
async def test_required_dependency_skip_is_fatal() -> None:
    parent = FakeComponent("parent", fail_start=True)
    child = FakeComponent("child", required=True)
    coordinator, context = _build_component_coordinator(
        (
            component_registration(
                parent,
                required=False,
                failure_action=ComponentFailureAction.IGNORE_WITH_DIAGNOSTIC,
            ),
            component_registration(child, required=True, dependencies=("parent",)),
        )
    )
    with pytest.raises(HostedApplicationComponentError):
        await coordinator.start_phase(context, ("parent", "child"))


@pytest.mark.asyncio
async def test_mark_degraded_dependency_skip_continues() -> None:
    parent = FakeComponent("parent", fail_start=True)
    child = FakeComponent("child")
    coordinator, context = _build_component_coordinator(
        (
            component_registration(parent, required=False),
            component_registration(
                child,
                required=False,
                dependencies=("parent",),
                failure_action=ComponentFailureAction.MARK_DEGRADED,
            ),
        )
    )
    await coordinator.start_phase(context, ("parent", "child"))
    assert "child" in coordinator.degraded_component_ids
