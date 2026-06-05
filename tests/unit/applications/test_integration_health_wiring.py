# © Artur Czarnecki. All rights reserved.

"""INT-2: Integration health probes at Tier-3 bootstrap."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.integration_health_wiring import (
    integration_health_summary,
    probe_integration_profile_health,
)
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_probe_integration_profile_health_lab_profile() -> None:
    register_default_integrations()
    profile = IntegrationProfile.lab()
    health = probe_integration_profile_health(profile)

    assert len(health) >= 1
    slugs = {item.slug for item in health}
    assert "sqlite" in slugs or "log" in slugs


def test_wire_application_environment_includes_integration_health() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    wiring = wire_application_environment(build_lab_manifest(settings), env)

    assert len(wiring.integration_health) >= 1
    summary = integration_health_summary(wiring.integration_health)
    assert "integrations healthy" in summary
