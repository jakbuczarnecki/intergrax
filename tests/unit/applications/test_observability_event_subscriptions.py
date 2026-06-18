# © Artur Czarnecki. All rights reserved.

"""OBS-EVOL-9.10: declarative ObservabilityProfile event bus subscriptions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.applications._shared.event_subscription_registry import (
    clear_event_subscription_handlers,
    register_event_subscription_handler,
)
from intergrax.applications._shared.observability_wiring import (
    EventSubscriptionWiringError,
    wire_observability_event_subscriptions,
)
from intergrax.applications.contracts.environment_profile import (
    EventSubscriptionSpec,
    ObservabilityProfile,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import EventCategory
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _LegalFlagV1(RuntimeEventPayload):
    schema_id = "agents.legal.flag.v1"
    ok: bool = True


@pytest.fixture(autouse=True)
def _clear_handlers() -> None:
    clear_event_subscription_handlers()
    yield
    clear_event_subscription_handlers()


def test_event_subscription_spec_requires_filter_when_enabled() -> None:
    with pytest.raises(ValidationError, match="requires at least one filter"):
        ObservabilityProfile(
            event_subscriptions=[
                EventSubscriptionSpec(
                    subscription_id="sub.bad",
                    handler_id="noop",
                    enabled=True,
                )
            ]
        )


def test_wire_observability_event_subscriptions_by_kind_prefix() -> None:
    seen: list[str] = []

    def _capture(event) -> None:
        seen.append(event.event_kind or "")

    register_event_subscription_handler("legal.capture", _capture)
    bus = RuntimeEventBus(record_history=True)
    profile = ObservabilityProfile(
        event_subscriptions=[
            EventSubscriptionSpec(
                subscription_id="sub.legal",
                handler_id="legal.capture",
                kind_prefix="agents.legal.",
            )
        ]
    )
    wiring = wire_observability_event_subscriptions(bus, profile)
    assert wiring.subscription_ids == ("sub.legal",)

    from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry

    clear_event_kind_registry()
    register_extension_runtime_payload(
        _LegalFlagV1,
        event_kind="agents.legal.clause_flagged",
    )
    ctx = EmitContext(task_id="t1", run_id="r1", bus=bus)
    emit_domain_signal(ctx, kind="agents.legal.clause_flagged", payload=_LegalFlagV1())
    assert seen == ["agents.legal.clause_flagged"]
    clear_event_kind_registry()


def test_wire_observability_event_subscriptions_rejects_unknown_handler() -> None:
    bus = RuntimeEventBus()
    profile = ObservabilityProfile(
        event_subscriptions=[
            EventSubscriptionSpec(
                subscription_id="sub.missing",
                handler_id="missing.handler",
                kind_prefix="agents.",
            )
        ]
    )
    with pytest.raises(EventSubscriptionWiringError, match="missing.handler"):
        wire_observability_event_subscriptions(bus, profile)


def test_wire_observability_event_subscriptions_uses_extra_handlers() -> None:
    seen: list[RuntimeEventType] = []
    bus = RuntimeEventBus(record_history=True)
    profile = ObservabilityProfile(
        event_subscriptions=[
            EventSubscriptionSpec(
                subscription_id="sub.task",
                handler_id="inline.task_created",
                event_types=[RuntimeEventType.TASK_CREATED],
            )
        ]
    )
    wire_observability_event_subscriptions(
        bus,
        profile,
        extra_handlers={
            "inline.task_created": lambda event: seen.append(event.event_type),
        },
    )
    from intergrax.runtime.events.runtime_event import RuntimeEvent

    bus.record(
        RuntimeEvent(
            task_id="t1",
            run_id="r1",
            event_type=RuntimeEventType.TASK_CREATED,
            phase=ExecutionPhase.INTAKE,
        )
    )
    assert seen == [RuntimeEventType.TASK_CREATED]


def test_observability_profile_rejects_duplicate_subscription_ids() -> None:
    spec = EventSubscriptionSpec(
        subscription_id="sub.dup",
        handler_id="h1",
        categories=[EventCategory.TOOL],
    )
    with pytest.raises(ValidationError, match="duplicate event subscription_id"):
        ObservabilityProfile(event_subscriptions=[spec, spec])
