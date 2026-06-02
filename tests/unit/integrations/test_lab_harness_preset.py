# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug
from lab_application.host.integration_wiring import build_lab_integration_profile


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_harness_preset_defaults_include_otel() -> None:
    profile = IntegrationProfile.lab_harness_preset()
    assert profile.relational_store == IntegrationSlug.SQLITE
    assert profile.notification_channel == IntegrationSlug.LOG
    assert profile.interaction_surface == IntegrationSlug.LAB_JSON
    assert profile.observability_backend == IntegrationSlug.OTEL
    assert IntegrationSlug.OTEL in profile.options


def test_lab_harness_preset_optional_redis_and_qdrant() -> None:
    profile = IntegrationProfile.lab_harness_preset(enable_redis=True, enable_qdrant=True)
    assert profile.key_value_cache == IntegrationSlug.REDIS
    assert profile.vector_store == IntegrationSlug.QDRANT


def test_build_lab_integration_profile_uses_preset() -> None:
    profile = build_lab_integration_profile()
    assert profile.observability_backend == IntegrationSlug.OTEL


def test_build_lab_integration_profile_can_disable_otel() -> None:
    profile = build_lab_integration_profile(otel_enabled=False)
    assert profile.observability_backend is None
    assert IntegrationSlug.OTEL not in profile.options
