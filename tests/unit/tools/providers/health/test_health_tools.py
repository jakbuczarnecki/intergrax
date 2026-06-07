# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.health.contracts import HealthCheckIntegrationInput, HealthCheckProfileInput
from intergrax.tools.providers.health.service import health_check_integration, health_check_profile
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _integrations_catalog() -> None:
    register_default_integrations()

def test_health_check_integration_unknown_slug() -> None:
    out = health_check_integration(
        ToolWiringContext(),
        HealthCheckIntegrationInput(slug="definitely-not-a-real-slug-xyz"),
    )
    assert out.status.healthy is False
    assert out.status.detail == "slug_not_found"


def test_health_check_profile_requires_integration_profile() -> None:
    with pytest.raises(RuntimeError, match="integration_profile_not_configured"):
        health_check_profile(ToolWiringContext(), HealthCheckProfileInput())


def test_health_check_profile_reports_configured_slots() -> None:
    profile = IntegrationProfile(issue_tracker="jira")
    ctx = ToolWiringContext(integration_profile=profile)
    out = health_check_profile(ctx, HealthCheckProfileInput())
    assert out.healthy_count + out.unhealthy_count == len(out.statuses)
