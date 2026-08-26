# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosting event dispatcher and platform observability export bridge (TRACE-1B-HOS-FIX)."""

from __future__ import annotations

from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.hosting.engine.callbacks import invoke_callback
from intergrax.hosting.engine.definition import ResolvedEventSubscription
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder, HostedApplicationFailurePhase
from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry
from intergrax.runtime.observability.export_attributes import ApplicationObservabilityAttributes
from intergrax.runtime.observability.export_boundary import (
    NoOpObservabilityExporter,
    ObservabilityExporter,
    PlatformObservabilityExportSource,
    envelope_from_platform_observability_source,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)


class HostingObservabilityAttributes(ApplicationObservabilityAttributes):
    """Typed safe hosting metadata for platform observability export."""

    namespace: str = "hosting"
    application_id: str
    instance_id: str
    lifecycle_state: str
    severity: str
    occurred_at: str
    causation_id: str = ""


def hosted_event_to_platform_export_source(
    event: HostedApplicationEvent,
) -> PlatformObservabilityExportSource:
    """Project a hosting authoring envelope to a typed non-execution export source."""
    attributes = HostingObservabilityAttributes(
        application_id=event.application_id,
        instance_id=event.instance_id,
        lifecycle_state=event.lifecycle_state.value,
        severity=event.severity.value,
        occurred_at=event.occurred_at.isoformat(),
        causation_id=event.causation_id,
    )
    return PlatformObservabilityExportSource(
        event_id=event.event_id,
        source_schema_id=event.schema_id,
        event_type=event.event_type.value,
        occurred_at=event.occurred_at,
        correlation_id=event.correlation_id or event.event_id,
        application_attributes=attributes,
    )


class ObservabilityHostedApplicationEventPublisher:
    """Publish hosting events through the existing observability export path."""

    def __init__(
        self,
        exporter: ObservabilityExporter | None = None,
        *,
        policy: ObservabilityExportPolicy | None = None,
    ) -> None:
        self._exporter = exporter or NoOpObservabilityExporter()
        self._policy = policy or ObservabilityExportPolicy()

    async def publish(self, event: HostedApplicationEvent) -> None:
        source = hosted_event_to_platform_export_source(event)
        envelope = envelope_from_platform_observability_source(source)
        await try_export_observability_envelope(
            envelope,
            exporter=self._exporter,
            policy=self._policy,
        )


class CompositeHostedApplicationEventPublisher:
    """Fan-out hosting events to multiple downstream publishers in order."""

    def __init__(self, publishers: tuple[HostedApplicationEventPublisher, ...]) -> None:
        if not publishers:
            raise ValueError("publishers must not be empty")
        self._publishers = publishers

    async def publish(self, event: HostedApplicationEvent) -> None:
        for publisher in self._publishers:
            await publisher.publish(event)


class HostingEventDispatcher:
    """Combine downstream publisher, profile subscriptions and observer tracking."""

    def __init__(
        self,
        downstream: HostedApplicationEventPublisher,
        subscriptions: tuple[ResolvedEventSubscription, ...],
        diagnostics: DiagnosticsRecorder,
        observer_tasks: ObserverTaskRegistry,
    ) -> None:
        self._downstream = downstream
        self._subscriptions = tuple(
            sorted(
                subscriptions,
                key=lambda item: (
                    item.subscription.priority,
                    item.subscription.source_id,
                    item.declaration_index,
                ),
            )
        )
        self._diagnostics = diagnostics
        self._observer_tasks = observer_tasks

    async def publish(self, event: HostedApplicationEvent) -> None:
        try:
            await self._downstream.publish(event)
        except Exception as exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.EVENT_PUBLISH,
                source_kind="event_publisher",
                source_id="downstream",
                exc=exc,
                reason_code="event_publish_failed",
            )
        for resolved in self._matching_subscriptions(event.event_type):
            self._observer_tasks.schedule(
                lambda resolved=resolved, event=event: self._invoke_subscription(resolved, event),
                phase=HostedApplicationFailurePhase.EVENT_SUBSCRIBER,
                source_id=resolved.subscription.subscription_id,
            )

    def _matching_subscriptions(
        self,
        event_type: HostedApplicationEventType,
    ) -> tuple[ResolvedEventSubscription, ...]:
        return tuple(
            resolved
            for resolved in self._subscriptions
            if event_type in resolved.subscription.event_types
        )

    async def _invoke_subscription(
        self,
        resolved: ResolvedEventSubscription,
        event: HostedApplicationEvent,
    ) -> None:
        handler = resolved.subscription.handler
        try:
            await invoke_callback(handler, event)
        except Exception as exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.EVENT_SUBSCRIBER,
                source_kind="event_subscription",
                source_id=resolved.subscription.subscription_id,
                exc=exc,
                reason_code="event_subscriber_failed",
            )
