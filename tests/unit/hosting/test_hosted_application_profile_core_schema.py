# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Schema and deterministic projection tests for hosted application profile (APP-HOST-1A.1)."""

from __future__ import annotations

import json
import re

import pytest

from intergrax.hosting import (
    HOSTED_APPLICATION_PROFILE_SPEC_VERSION,
    HostedApplicationIdentity,
    HostedApplicationProfile,
    HostedApplicationProfilePublicView,
)
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _build_profile(
    *,
    application_id: str = "my_application",
    application_factory=sample_application_factory,
    application_factory_id: str | None = None,
    metadata: dict | None = None,
) -> HostedApplicationProfile:
    kwargs: dict = {
        "application_id": application_id,
        "application_factory": application_factory,
    }
    if application_factory_id is not None:
        kwargs["application_factory_id"] = application_factory_id
    if metadata is not None:
        kwargs["metadata"] = metadata
    return HostedApplicationProfile(**kwargs)


def test_application_factory_absent_from_model_dump() -> None:
    profile = _build_profile()
    dumped = profile.model_dump()
    assert "application_factory" not in dumped


def test_application_factory_absent_from_model_json_schema() -> None:
    schema = HostedApplicationProfile.model_json_schema()
    assert "application_factory" not in schema.get("properties", {})


def test_application_factory_absent_from_public_view() -> None:
    profile = _build_profile()
    public_view = profile.public_view()
    dumped = public_view.model_dump()
    assert "application_factory" not in dumped
    assert set(dumped) == {
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
    }


def test_callable_repr_and_address_absent_from_repr_and_serialized_output() -> None:
    profile = _build_profile()
    profile_repr = repr(profile)
    assert "application_factory=" not in profile_repr
    assert "0x" not in profile_repr

    payload = profile.public_view().model_dump(mode="json")
    serialized = json.dumps(payload)
    assert "application_factory" not in payload
    assert "application_factory" not in payload["identity"]
    assert "0x" not in serialized


def test_public_view_contains_exact_fields() -> None:
    profile = _build_profile(metadata={"tier": "local"})
    public_view = profile.public_view()
    assert isinstance(public_view, HostedApplicationProfilePublicView)
    assert public_view.spec_version == HOSTED_APPLICATION_PROFILE_SPEC_VERSION
    assert public_view.identity == HostedApplicationIdentity(
        application_id="my_application",
        application_factory_id=profile.application_factory_id or "",
    )
    assert public_view.metadata == {"tier": "local"}
    assert public_view.hooks == ()
    assert public_view.components == ()
    assert public_view.event_subscriptions == ()


def test_public_view_json_serialization_is_deterministic() -> None:
    profile = _build_profile(metadata={"b": 2, "a": 1, "nested": {"z": 3, "y": 4}})
    first = json.dumps(profile.public_view().model_dump(mode="json"), sort_keys=True)
    second = json.dumps(profile.public_view().model_dump(mode="json"), sort_keys=True)
    assert first == second


def test_equivalent_metadata_insertion_order_produces_same_digest() -> None:
    profile_a = _build_profile(
        metadata={"alpha": 1, "beta": {"gamma": 2, "delta": 3}},
    )
    profile_b = _build_profile(
        metadata={"beta": {"delta": 3, "gamma": 2}, "alpha": 1},
    )
    assert profile_a.profile_digest() == profile_b.profile_digest()


def test_different_metadata_produces_different_digest() -> None:
    profile_a = _build_profile(metadata={"mode": "foreground"})
    profile_b = _build_profile(metadata={"mode": "background"})
    assert profile_a.profile_digest() != profile_b.profile_digest()


def test_different_application_factory_id_produces_different_digest() -> None:
    profile_a = _build_profile(application_factory_id="package.module.factory_a")
    profile_b = _build_profile(application_factory_id="package.module.factory_b")
    assert profile_a.profile_digest() != profile_b.profile_digest()


def test_different_callables_same_explicit_factory_id_produce_same_digest() -> None:
    profile_a = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
        application_factory_id="stable.factory.id",
    )
    profile_b = HostedApplicationProfile(
        application_id="my_application",
        application_factory=lambda: None,
        application_factory_id="stable.factory.id",
    )
    assert profile_a.profile_digest() == profile_b.profile_digest()


def test_digest_matches_sha256_format() -> None:
    profile = _build_profile(metadata={"proof": True})
    assert _DIGEST_PATTERN.match(profile.profile_digest())


def test_public_package_exports_work() -> None:
    from intergrax import hosting

    assert hosting.HostedApplicationProfile is HostedApplicationProfile
    assert hosting.HostedApplicationIdentity is HostedApplicationIdentity
    assert hosting.HostedApplicationProfilePublicView is HostedApplicationProfilePublicView
    assert hosting.HOSTED_APPLICATION_PROFILE_SPEC_VERSION == "1.0"


def test_no_application_environment_profile_field_introduced() -> None:
    assert "environment" not in HostedApplicationProfile.model_fields
    assert "ApplicationEnvironmentProfile" not in HostedApplicationProfile.model_json_schema()["title"]
