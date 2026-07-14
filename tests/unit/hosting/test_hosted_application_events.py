# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import SecretStr, ValidationError

from intergrax.hosting import (
    HostedApplicationEvent,
    HostedApplicationEventSubscription,
    HostedApplicationEventType,
    HostedApplicationLifecycleState,
    HostedApplicationProfile,
)
from intergrax.hosting.contracts.events import (
    HOSTED_APPLICATION_EVENT_SCHEMA_ID,
    HOSTED_APPLICATION_EVENT_SCHEMA_VERSION,
)
from intergrax.contracts.event_severity import EventSeverity
from tests.unit.hosting._helpers import record_hosting_diagnostic_handler
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("event_type", list(HostedApplicationEventType))
def test_all_required_event_types_exist(event_type: HostedApplicationEventType) -> None:
    assert event_type.value.startswith("hosting.")


def test_injectable_event_id_and_timestamp() -> None:
    moment = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    event = HostedApplicationEvent(
        event_id="evt-001",
        event_type=HostedApplicationEventType.APPLICATION_READY,
        occurred_at=moment,
        application_id="my_application",
        instance_id="instance-001",
        lifecycle_state=HostedApplicationLifecycleState.READY,
        correlation_id="corr-1",
        causation_id="cause-1",
    )
    assert event.event_id == "evt-001"
    assert event.occurred_at == moment


def test_timezone_aware_default_timestamp() -> None:
    event = HostedApplicationEvent(
        event_type=HostedApplicationEventType.APPLICATION_READY,
        application_id="my_application",
        instance_id="instance-001",
        lifecycle_state=HostedApplicationLifecycleState.READY,
    )
    assert event.occurred_at.tzinfo is not None


def test_safe_payload_validation() -> None:
    with pytest.raises(ValidationError):
        HostedApplicationEvent(
            event_type=HostedApplicationEventType.APPLICATION_FAILED,
            application_id="my_application",
            instance_id="instance-001",
            lifecycle_state=HostedApplicationLifecycleState.FAILED,
            payload={"secret": SecretStr("x")},  # type: ignore[arg-type]
        )


def test_runtime_subscriber_absent_from_dump_schema_repr() -> None:
    subscription = HostedApplicationEventSubscription(
        subscription_id="diag",
        event_types=(HostedApplicationEventType.APPLICATION_FAILED,),
        handler=record_hosting_diagnostic_handler,
    )
    assert "handler" not in subscription.model_dump()
    assert "handler" not in subscription.model_json_schema().get("properties", {})
    assert "handler=" not in repr(subscription)


def test_duplicate_subscription_ids_rejected() -> None:
    subscription = HostedApplicationEventSubscription(
        subscription_id="diag",
        event_types=(HostedApplicationEventType.APPLICATION_FAILED,),
        handler=record_hosting_diagnostic_handler,
    )
    with pytest.raises(ValidationError, match="duplicate subscription_id"):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=sample_application_factory,
            event_subscriptions=(subscription, subscription),
        )


def test_duplicate_event_types_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate event_type"):
        HostedApplicationEventSubscription(
            subscription_id="diag",
            event_types=(
                HostedApplicationEventType.APPLICATION_FAILED,
                HostedApplicationEventType.APPLICATION_FAILED,
            ),
            handler=record_hosting_diagnostic_handler,
        )


def test_event_rejects_invalid_application_id() -> None:
    with pytest.raises(ValidationError, match="application_id"):
        HostedApplicationEvent(
            event_type=HostedApplicationEventType.APPLICATION_READY,
            application_id="1invalid",
            instance_id="instance-001",
            lifecycle_state=HostedApplicationLifecycleState.READY,
        )


def test_event_rejects_empty_event_id() -> None:
    with pytest.raises(ValidationError, match="event_id"):
        HostedApplicationEvent(
            event_id="",
            event_type=HostedApplicationEventType.APPLICATION_READY,
            application_id="my_application",
            instance_id="instance-001",
            lifecycle_state=HostedApplicationLifecycleState.READY,
        )


def test_event_rejects_naive_occurred_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HostedApplicationEvent(
            event_type=HostedApplicationEventType.APPLICATION_READY,
            occurred_at=datetime(2026, 7, 14, 12, 0),
            application_id="my_application",
            instance_id="instance-001",
            lifecycle_state=HostedApplicationLifecycleState.READY,
        )


def test_stable_schema_id_and_version() -> None:
    event = HostedApplicationEvent(
        event_type=HostedApplicationEventType.APPLICATION_READY,
        application_id="my_application",
        instance_id="instance-001",
        lifecycle_state=HostedApplicationLifecycleState.READY,
        severity=EventSeverity.INFO,
    )
    assert event.schema_id == HOSTED_APPLICATION_EVENT_SCHEMA_ID
    assert event.schema_version == HOSTED_APPLICATION_EVENT_SCHEMA_VERSION


def test_sync_event_handler_accepted() -> None:
    subscription = HostedApplicationEventSubscription(
        subscription_id="sync",
        event_types=(HostedApplicationEventType.APPLICATION_READY,),
        handler=record_hosting_diagnostic_handler,
        handler_id="tests.unit.hosting._helpers.record_hosting_diagnostic_handler",
    )
    assert subscription.subscription_id == "sync"


async def test_async_event_handler_accepted() -> None:
    subscription = HostedApplicationEventSubscription(
        subscription_id="async",
        event_types=(HostedApplicationEventType.APPLICATION_READY,),
        handler=record_hosting_diagnostic_handler,
        handler_id="tests.unit.hosting._helpers.record_hosting_diagnostic_handler",
    )
    assert subscription.subscription_id == "async"
