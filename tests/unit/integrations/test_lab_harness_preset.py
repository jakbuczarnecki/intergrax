# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.registry.catalog_manifests import LAB_JSON, LOG, OTEL, QDRANT, REDIS, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile
from lab_application.host.integration_wiring import build_lab_integration_profile


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_harness_preset_defaults_include_otel() -> None:
    profile = IntegrationProfile.lab_harness_preset()
    assert profile.relational_store is not None
    assert profile.relational_store.resolved_slug() == SQLITE.slug
    assert profile.notification_channel is not None
    assert profile.notification_channel.resolved_slug() == LOG.slug
    assert profile.interaction_surface is not None
    assert profile.interaction_surface.resolved_slug() == LAB_JSON.slug
    assert profile.observability_backend is not None
    assert profile.observability_backend.resolved_slug() == OTEL.slug
    assert OTEL.slug in profile.options


def test_lab_harness_preset_optional_redis_and_qdrant() -> None:
    profile = IntegrationProfile.lab_harness_preset(enable_redis=True, enable_qdrant=True)
    assert profile.key_value_cache is not None
    assert profile.key_value_cache.resolved_slug() == REDIS.slug
    assert profile.vector_store is not None
    assert profile.vector_store.resolved_slug() == QDRANT.slug


def test_build_lab_integration_profile_uses_preset() -> None:
    profile = build_lab_integration_profile()
    assert profile.observability_backend is not None
    assert profile.observability_backend.resolved_slug() == OTEL.slug


def test_build_lab_integration_profile_can_disable_otel() -> None:
    profile = build_lab_integration_profile(otel_enabled=False)
    assert profile.observability_backend is None
    assert OTEL.slug not in profile.options
