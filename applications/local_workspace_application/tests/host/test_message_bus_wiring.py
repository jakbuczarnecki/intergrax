# © Artur Czarnecki. All rights reserved.

"""Tests for LKW Kafka message bus wiring (LKW.4E)."""

from __future__ import annotations

import pytest

from intergrax.integrations.registry.catalog_manifests import REDIS
from local_workspace_application.host.message_bus_wiring import (
    local_workspace_message_bus_enabled,
    materialize_local_workspace_message_bus_profile,
)
from local_workspace_application.host.environment_profile import build_local_workspace_integration_profile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_message_bus_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", raising=False)
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", raising=False)
    assert local_workspace_message_bus_enabled() is False
    profile = build_local_workspace_integration_profile()
    assert profile.message_bus is None


def test_message_bus_profile_materializes_kafka_and_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("INTERGRAX_REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("INTERGRAX_KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")

    profile = build_local_workspace_integration_profile()

    assert profile.key_value_cache is not None
    assert profile.key_value_cache.manifest.slug == REDIS.slug
    binding = profile.message_bus
    assert binding is not None
    assert binding.manifest is not None
    assert binding.manifest.slug == "kafka"
    assert binding.instance is None


def test_kafka_message_bus_enabled_via_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", raising=False)
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "true")
    assert local_workspace_message_bus_enabled() is True


def test_materialize_is_idempotent_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", raising=False)
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", raising=False)
    profile = build_local_workspace_integration_profile()
    assert materialize_local_workspace_message_bus_profile(profile) == profile


def test_kafka_enabled_integration_profile_is_package_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("INTERGRAX_REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("INTERGRAX_KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")

    from intergrax.applications._shared.package_wiring import build_application_package
    from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

    profile = build_local_workspace_integration_profile()
    env = LOCAL_WORKSPACE_APPLICATION_MANIFEST.resolved_environment().model_copy(
        update={"integration_profile": profile},
    )

    package = build_application_package(LOCAL_WORKSPACE_APPLICATION_MANIFEST, env)

    assert package.distribution.checksum
