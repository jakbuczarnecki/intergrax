# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from intergrax.hosting import (
    HostedApplicationHook,
    HostedApplicationProfile,
)
from tests.unit.hosting._helpers import build_complete_profile, warm_cache_handler
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit


def test_hook_handler_absent_from_profile_json_schema() -> None:
    schema = HostedApplicationProfile.model_json_schema()
    assert "application_factory" not in schema.get("properties", {})
    hook_def = schema["$defs"]["HostedApplicationHook"]
    assert "handler" not in hook_def.get("properties", {})
    subscription_def = schema["$defs"]["HostedApplicationEventSubscription"]
    assert "handler" not in subscription_def.get("properties", {})


def test_component_reference_absent_from_profile_json_schema() -> None:
    schema = HostedApplicationProfile.model_json_schema()
    serialized = json.dumps(schema)
    assert '"component"' not in serialized or "component_id" in serialized


def test_complete_profile_public_view_serializes() -> None:
    payload = build_complete_profile().public_view().model_dump(mode="json")
    json.dumps(payload, sort_keys=True)


def test_hook_order_affects_digest() -> None:
    profile_a = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
        hooks=__import__("intergrax.hosting", fromlist=["HostedApplicationHooks"]).HostedApplicationHooks(
            before_ready=(
                HostedApplicationHook(hook_id="first", handler=warm_cache_handler, priority=0),
                HostedApplicationHook(
                    hook_id="second",
                    handler=warm_cache_handler,
                    handler_id="tests.second",
                    priority=0,
                ),
            )
        ),
    )
    profile_b = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
        hooks=__import__("intergrax.hosting", fromlist=["HostedApplicationHooks"]).HostedApplicationHooks(
            before_ready=(
                HostedApplicationHook(
                    hook_id="second",
                    handler=warm_cache_handler,
                    handler_id="tests.second",
                    priority=0,
                ),
                HostedApplicationHook(hook_id="first", handler=warm_cache_handler, priority=0),
            )
        ),
    )
    assert profile_a.profile_digest() != profile_b.profile_digest()
