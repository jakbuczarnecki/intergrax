# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.hosting import HostedApplicationProfile
from tests.unit.hosting._helpers import build_complete_profile, build_minimal_profile
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit


def test_minimal_profile_still_constructs() -> None:
    profile = build_minimal_profile()
    assert profile.hooks.before_start == ()
    assert profile.components == ()
    assert profile.event_subscriptions == ()


def test_complete_profile_constructs() -> None:
    profile = build_complete_profile()
    assert profile.hooks.before_ready[0].hook_id == "warm_cache"
    assert profile.components[0].component_id == "background_worker"


def test_public_view_contains_every_w1_domain() -> None:
    public_view = build_complete_profile().public_view()
    dumped = public_view.model_dump()
    for field in (
        "spec_version",
        "identity",
        "metadata",
        "hooks",
        "components",
        "lifecycle",
        "shutdown",
        "restart",
        "component_failure",
        "hook_failure",
        "instance",
        "event_subscriptions",
    ):
        assert field in dumped


def test_no_runtime_object_in_public_view() -> None:
    public_view = build_complete_profile().public_view()
    serialized = public_view.model_dump(mode="json")
    assert "application_factory" not in serialized
    assert "handler" not in serialized
    assert "component" not in serialized


def test_equivalent_complete_profile_produces_same_digest() -> None:
    first = build_complete_profile()
    second = build_complete_profile()
    assert first.profile_digest() == second.profile_digest()


def test_each_public_configuration_category_affects_digest() -> None:
    base = build_minimal_profile()
    changed_hooks = base.model_copy(
        update={"hooks": build_complete_profile().hooks},
    )
    assert base.profile_digest() != changed_hooks.profile_digest()


def test_interaction_profile_plugins_context_fields_absent() -> None:
    assert "interaction" not in HostedApplicationProfile.model_fields
    assert "plugins" not in HostedApplicationProfile.model_fields
    assert "context" not in HostedApplicationProfile.model_fields
    assert "environment" not in HostedApplicationProfile.model_fields


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=sample_application_factory,
            plugins=(),  # type: ignore[call-arg]
        )
