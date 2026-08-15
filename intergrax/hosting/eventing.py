# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosting event dispatcher and runtime event-spine bridge (APP-HOST-3B)."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, JsonValue

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId, mint_attempt_id
from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.hosting.engine.callbacks import invoke_callback
from intergrax.hosting.engine.definition import ResolvedEventSubscription
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder, HostedApplicationFailurePhase
from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_kind_registry import register_event_kind
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.signals import emit_domain_signal

HOSTING_DOMAIN_EVENT_KIND = "applications.hosting.event"
HOSTING_EVENT_PAYLOAD_SCHEMA_ID = "intergrax.hosting.event.v1"


def _hosting_canonical_id(prefix: str, key: str) -> str:
    return f"{prefix}_{uuid5(NAMESPACE_URL, key).hex}"


class HostedApplicationEventPayloadV1(RuntimeEventPayload):
    """Typed runtime payload preserving the safe hosting event envelope."""

    schema_id = HOSTING_EVENT_PAYLOAD_SCHEMA_ID

    hosted_event_id: str
    hosted_event_schema_id: str
    hosted_event_schema_version: str
    hosted_event_type: str
    occurred_at: str
    application_id: str
    instance_id: str
    lifecycle_state: str
    severity: str
    correlation_id: str = ""
    causation_id: str = ""
    safe_payload: dict[str, JsonValue] = Field(default_factory=dict)

    def redact(self) -> HostedApplicationEventPayloadV1:
        return self


def register_hosting_domain_signal() -> None:
    """Register hosting payload schema and domain event kind (idempotent)."""
    register_payload_schema(HostedApplicationEventPayloadV1, extension=True)
    register_event_kind(HOSTING_DOMAIN_EVENT_KIND, HOSTING_EVENT_PAYLOAD_SCHEMA_ID)


def hosted_event_to_payload(event: HostedApplicationEvent) -> HostedApplicationEventPayloadV1:
    return HostedApplicationEventPayloadV1(
        hosted_event_id=event.event_id,
        hosted_event_schema_id=event.schema_id,
        hosted_event_schema_version=event.schema_version,
        hosted_event_type=event.event_type.value,
        occurred_at=event.occurred_at.isoformat(),
        application_id=event.application_id,
        instance_id=event.instance_id,
        lifecycle_state=event.lifecycle_state.value,
        severity=event.severity.value,
        correlation_id=event.correlation_id,
        causation_id=event.causation_id,
        safe_payload=dict(event.payload),
    )


def build_hosting_emit_context(
    event: HostedApplicationEvent,
    bus: object | None,
    *,
    production_mode: bool = False,
) -> EmitContext:
    """Build synthetic spine correlation fields for hosting domain signals.

    The synthetic ``task_id`` is a legacy spine correlation field and is not an
    Intergrax application ``Task``.
    """
    return EmitContext(
        task_id=TaskId(_hosting_canonical_id("task", event.application_id)),
        run_id=RunId(_hosting_canonical_id("run", event.instance_id)),
        attempt_id=mint_attempt_id(),
        correlation_id=event.correlation_id or event.event_id,
        parent_event_id=event.causation_id or None,
        bus=bus,  # type: ignore[arg-type]
        production_mode=production_mode,
    )


class RuntimeSpineHostedApplicationEventPublisher:
    """Publish hosting events through the existing runtime event spine."""

    def __init__(self, bus: object | None = None, *, production_mode: bool = False) -> None:
        register_hosting_domain_signal()
        self._bus = bus
        self._production_mode = production_mode

    async def publish(self, event: HostedApplicationEvent) -> None:
        payload = hosted_event_to_payload(event)
        ctx = build_hosting_emit_context(event, self._bus, production_mode=self._production_mode)
        try:
            emit_domain_signal(
                ctx,
                kind=HOSTING_DOMAIN_EVENT_KIND,
                payload=payload,
                severity=event.severity,
                phase=ExecutionPhase.APPLICATION_HOSTING,
            )
        except Exception as exc:
            raise RuntimeError("hosting spine publication failed") from exc


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
