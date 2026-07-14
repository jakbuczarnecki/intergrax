# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from intergrax.hosting import (
    HostedApplicationComponent,
    HostedApplicationComponentHealth,
    HostedApplicationComponentRegistration,
    HostedApplicationComponentState,
    HostedApplicationProfile,
)
from tests.unit.hosting._helpers import SampleComponent
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit


def test_runtime_checkable_component_protocol() -> None:
    component = SampleComponent()
    assert isinstance(component, HostedApplicationComponent)


def test_safe_health_model() -> None:
    health = HostedApplicationComponentHealth(
        component_id="background_worker",
        enabled=True,
        required=True,
        state=HostedApplicationComponentState.READY,
        healthy=True,
        ready=True,
        detail_code="ok",
        safe_message="ready",
    )
    assert health.model_dump()["safe_message"] == "ready"


def test_component_runtime_object_absent_from_dump_schema_repr() -> None:
    registration = HostedApplicationComponentRegistration(component=SampleComponent())
    assert "component" not in registration.model_dump()
    assert "component" not in registration.model_json_schema().get("properties", {})
    assert "component=" not in repr(registration)


def test_required_disabled_registration_rejected() -> None:
    with pytest.raises(ValidationError, match="required component cannot be disabled"):
        HostedApplicationComponentRegistration(
            component=SampleComponent(),
            enabled=False,
            required=True,
        )


def test_self_dependency_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        HostedApplicationComponentRegistration(
            component=SampleComponent(),
            dependencies=("background_worker",),
        )


def test_duplicate_dependencies_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate dependency"):
        HostedApplicationComponentRegistration(
            component=SampleComponent(),
            dependencies=("dep_a", "dep_a"),
        )


def test_component_health_rejects_invalid_component_id() -> None:
    with pytest.raises(ValidationError, match="component_id"):
        HostedApplicationComponentHealth(
            component_id="",
            enabled=True,
            required=True,
            state=HostedApplicationComponentState.READY,
            healthy=True,
            ready=True,
        )


def test_component_health_rejects_invalid_detail_code() -> None:
    with pytest.raises(ValidationError, match="detail_code"):
        HostedApplicationComponentHealth(
            component_id="background_worker",
            enabled=True,
            required=True,
            state=HostedApplicationComponentState.READY,
            healthy=True,
            ready=True,
            detail_code="bad detail",
        )


def test_component_health_rejects_control_characters_in_safe_message() -> None:
    with pytest.raises(ValidationError, match="safe_message"):
        HostedApplicationComponentHealth(
            component_id="background_worker",
            enabled=True,
            required=True,
            state=HostedApplicationComponentState.READY,
            healthy=True,
            ready=True,
            safe_message="bad\x00message",
        )


def test_component_health_rejects_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HostedApplicationComponentHealth(
            component_id="background_worker",
            enabled=True,
            required=True,
            state=HostedApplicationComponentState.READY,
            healthy=True,
            ready=True,
            last_check_at=datetime(2026, 7, 14, 12, 0),
        )


def test_duplicate_component_ids_rejected_by_profile() -> None:
    with pytest.raises(ValidationError, match="duplicate component_id"):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=sample_application_factory,
            components=(
                HostedApplicationComponentRegistration(component=SampleComponent()),
                HostedApplicationComponentRegistration(
                    component=SampleComponent(),
                    component_id="background_worker",
                ),
            ),
        )
