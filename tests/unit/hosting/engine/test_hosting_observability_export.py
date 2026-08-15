# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from intergrax.contracts.event_severity import EventSeverity
from intergrax.hosting import HostedApplicationLifecycleState
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder
from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry
from intergrax.hosting.eventing import (
    HostingEventDispatcher,
    ObservabilityHostedApplicationEventPublisher,
    hosted_event_to_platform_export_source,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    InMemoryObservabilityExporter,
    ObservabilityExportEnvelope,
    envelope_from_platform_observability_source,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from tests.unit.hosting.engine._fakes import FixedClock, RecordingPublisher

pytestmark = pytest.mark.unit


def _sample_event(
    *,
    event_id: str = "E1",
    correlation_id: str = "C1",
    causation_id: str = "cause-1",
    event_type: HostedApplicationEventType = HostedApplicationEventType.APPLICATION_READY,
    payload: dict[str, object] | None = None,
) -> HostedApplicationEvent:
    return HostedApplicationEvent(
        event_id=event_id,
        event_type=event_type,
        occurred_at=datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC),
        application_id="test_app",
        instance_id="instance-001",
        lifecycle_state=HostedApplicationLifecycleState.READY,
        correlation_id=correlation_id,
        causation_id=causation_id,
        payload=payload or {},
    )


def test_hosting_projection_preserves_identity_without_execution_fields() -> None:
    event = _sample_event()
    source = hosted_event_to_platform_export_source(event)
    envelope = envelope_from_platform_observability_source(source)

    assert envelope.record_kind == ExportRecordKind.PLATFORM_SIGNAL
    assert envelope.event_id == "E1"
    assert envelope.correlation_id == "C1"
    assert envelope.task_id == ""
    assert envelope.run_id == ""
    assert envelope.source_schema_id == event.schema_id
    assert envelope.application_attributes is not None
    attrs = envelope.application_attributes.to_safe_attributes()
    assert attrs["hosting.application_id"] == "test_app"
    assert attrs["hosting.instance_id"] == "instance-001"
    assert attrs["hosting.causation_id"] == "cause-1"
    assert attrs["hosting.occurred_at"] == event.occurred_at.isoformat()


@pytest.mark.asyncio
async def test_two_hosting_events_do_not_mint_execution_identity() -> None:
    exporter = InMemoryObservabilityExporter()
    policy = ObservabilityExportPolicy(enabled=True)
    publisher = ObservabilityHostedApplicationEventPublisher(exporter, policy=policy)

    first = _sample_event(event_id="E1", correlation_id="C1", causation_id="")
    second = _sample_event(
        event_id="E2",
        correlation_id="C1",
        causation_id="E1",
        event_type=HostedApplicationEventType.COMPONENT_STARTED,
    )

    await publisher.publish(first)
    await publisher.publish(second)

    assert len(exporter.envelopes) == 2
    assert exporter.envelopes[0].event_id == "E1"
    assert exporter.envelopes[1].event_id == "E2"
    assert exporter.envelopes[1].sanitized_application_attributes is not None
    assert (
        exporter.envelopes[1].sanitized_application_attributes.attributes["hosting.causation_id"]
        == "E1"
    )
    for envelope in exporter.envelopes:
        assert envelope.task_id == ""
        assert envelope.run_id == ""


@pytest.mark.asyncio
async def test_export_disabled_skips_exporter_call() -> None:
    exporter = InMemoryObservabilityExporter()
    publisher = ObservabilityHostedApplicationEventPublisher(
        exporter,
        policy=ObservabilityExportPolicy(enabled=False),
    )

    await publisher.publish(_sample_event())

    assert exporter.envelopes == []


@pytest.mark.asyncio
async def test_metadata_only_policy_exports_safe_hosting_envelope() -> None:
    exporter = InMemoryObservabilityExporter()
    publisher = ObservabilityHostedApplicationEventPublisher(
        exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    await publisher.publish(_sample_event())

    assert len(exporter.envelopes) == 1
    exported = exporter.envelopes[0]
    assert exported.record_kind == ExportRecordKind.PLATFORM_SIGNAL
    assert exported.application_attributes is None
    assert exported.sanitized_application_attributes is not None
    assert exported.sanitized_application_attributes.attributes["hosting.application_id"] == "test_app"


def test_raw_hosting_payload_is_not_exported() -> None:
    event = _sample_event(
        payload={
            "prompt": "secret prompt",
            "content": "secret content",
            "component_id": "worker-1",
        }
    )
    source = hosted_event_to_platform_export_source(event)
    envelope = envelope_from_platform_observability_source(source)

    serialized = envelope.model_dump_json()
    assert "secret prompt" not in serialized
    assert "secret content" not in serialized
    assert "component_id" not in serialized


@pytest.mark.asyncio
async def test_dispatcher_subscriptions_receive_unmodified_authoring_envelope() -> None:
    received: list[HostedApplicationEvent] = []

    def handler(event: HostedApplicationEvent) -> None:
        received.append(event)

    from intergrax.hosting import HostedApplicationEventSubscription, resolve_hosted_application_definition
    from intergrax.hosting import HostedApplicationProfile
    from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

    subscription = HostedApplicationEventSubscription(
        subscription_id="sub_ready",
        event_types=(HostedApplicationEventType.APPLICATION_READY,),
        handler=handler,
        handler_id="tests.sub_ready",
    )
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        event_subscriptions=(subscription,),
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
    dispatcher = HostingEventDispatcher(
        RecordingPublisher(),
        definition.event_subscriptions,
        diagnostics,
        observer_tasks,
    )
    event = _sample_event()
    await dispatcher.publish(event)
    await observer_tasks.drain(1.0)

    assert len(received) == 1
    assert received[0] is event


@pytest.mark.asyncio
async def test_export_failure_does_not_break_hosting_publication() -> None:
    class _FailingExporter:
        async def export(self, envelope: ObservabilityExportEnvelope) -> None:
            raise RuntimeError("export failed")

    publisher = ObservabilityHostedApplicationEventPublisher(
        _FailingExporter(),
        policy=ObservabilityExportPolicy(enabled=True),
    )

    await publisher.publish(_sample_event())


@pytest.mark.asyncio
async def test_hosting_publisher_does_not_emit_runtime_event_or_domain_signal() -> None:
    exporter = InMemoryObservabilityExporter()
    publisher = ObservabilityHostedApplicationEventPublisher(
        exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    with patch("intergrax.runtime.events.signals.emit_domain_signal") as emit_domain_signal:
        await publisher.publish(_sample_event())

    emit_domain_signal.assert_not_called()
    assert len(exporter.envelopes) == 1
    assert exporter.envelopes[0].record_kind == ExportRecordKind.PLATFORM_SIGNAL


@pytest.mark.asyncio
async def test_try_export_applies_policy_before_exporter() -> None:
    exporter = InMemoryObservabilityExporter()
    event = _sample_event()
    envelope = envelope_from_platform_observability_source(
        hosted_event_to_platform_export_source(event)
    )

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is True
    assert len(exporter.envelopes) == 1
