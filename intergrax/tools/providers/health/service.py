# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.health import health_check_all, health_check_catalog_slugs
from intergrax.integrations.contracts.base import UnknownIntegrationError
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.health.contracts import (
    HealthCheckIntegrationInput,
    HealthCheckIntegrationOutput,
    HealthCheckProfileInput,
    HealthCheckProfileOutput,
    HealthStatusOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

HEALTH_CHECK_INTEGRATION_TOOL_ID = "health.check_integration"
HEALTH_CHECK_PROFILE_TOOL_ID = "health.check_profile"


def _status_output(slug: str, healthy: bool, detail: str = "") -> HealthStatusOutput:
    return HealthStatusOutput(slug=slug, healthy=healthy, detail=detail)


def _require_integration_profile(ctx: ToolWiringContext) -> IntegrationProfile:
    profile = ctx.integration_profile
    if profile is None:
        raise RuntimeError("integration_profile_not_configured")
    if not isinstance(profile, IntegrationProfile):
        raise RuntimeError("integration_profile_invalid_type")
    return profile


def health_check_integration(
    ctx: ToolWiringContext,
    params: HealthCheckIntegrationInput,
) -> HealthCheckIntegrationOutput:
    slug = params.slug.strip().lower()
    try:
        results = health_check_catalog_slugs([slug])
    except UnknownIntegrationError:
        return HealthCheckIntegrationOutput(
            status=_status_output(slug, healthy=False, detail="slug_not_found"),
        )
    if not results:
        return HealthCheckIntegrationOutput(
            status=_status_output(slug, healthy=False, detail="slug_not_found"),
        )
    item = results[0]
    return HealthCheckIntegrationOutput(
        status=_status_output(item.slug, item.healthy, item.detail),
    )


def health_check_profile(ctx: ToolWiringContext, _params: HealthCheckProfileInput) -> HealthCheckProfileOutput:
    profile = _require_integration_profile(ctx)
    results = health_check_all(profile)
    statuses = [_status_output(item.slug, item.healthy, item.detail) for item in results]
    healthy_count = sum(1 for item in statuses if item.healthy)
    return HealthCheckProfileOutput(
        statuses=statuses,
        healthy_count=healthy_count,
        unhealthy_count=len(statuses) - healthy_count,
    )
