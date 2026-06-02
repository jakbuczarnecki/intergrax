# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.skill_wiring import build_application_skill_wiring, lab_skill_profile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug
from lab_application.host.integration_wiring import build_lab_integration_profile


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_integration_profile_otel_by_default() -> None:
    profile = build_lab_integration_profile()
    assert profile.relational_store == IntegrationSlug.SQLITE
    assert profile.observability_backend == IntegrationSlug.OTEL
    assert IntegrationSlug.OTEL in profile.options


def test_lab_integration_profile_otel_when_explicitly_enabled() -> None:
    profile = build_lab_integration_profile(otel_enabled=True)
    assert profile.observability_backend == IntegrationSlug.OTEL


def test_lab_skill_profile_includes_harness_bundle() -> None:
    profile = lab_skill_profile()
    assert "harness" in profile.enabled_bundles
    wiring = build_application_skill_wiring(profile)
    assert wiring.registry.has("harness.tool_smoke")


def test_harness_environment_profile_is_distinct_from_vendor_harness() -> None:
    env = IntegrationProfile.harness_environment()
    vendor = IntegrationProfile.harness_lab()
    assert env.observability_backend == IntegrationSlug.OTEL
    assert vendor.observability_backend == IntegrationSlug.SENTRY
