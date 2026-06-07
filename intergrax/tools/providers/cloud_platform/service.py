# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.tools.providers.cloud_platform.contracts import (
    CloudPlatformHealthInput,
    CloudPlatformHealthOutput,
    CloudPlatformResolveInput,
    CloudPlatformResolveOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

CLOUD_PLATFORM_HEALTH_TOOL_ID = "cloud_platform.health"
CLOUD_PLATFORM_RESOLVE_TOOL_ID = "cloud_platform.resolve"


def _require_platform(ctx: ToolWiringContext) -> CloudPlatform:
    platform = ctx.cloud_platform
    if platform is None:
        raise RuntimeError("cloud_platform_not_configured")
    if not isinstance(platform, CloudPlatform):
        raise RuntimeError("cloud_platform_invalid_type")
    return platform


def cloud_platform_health(ctx: ToolWiringContext, _params: CloudPlatformHealthInput) -> CloudPlatformHealthOutput:
    platform = _require_platform(ctx)
    status = platform.health()
    region = platform.default_region
    return CloudPlatformHealthOutput(
        used=True,
        slug=platform.slug,
        healthy=status.healthy,
        default_region=str(region) if region else "",
        detail=status.detail,
    )


def cloud_platform_resolve(
    ctx: ToolWiringContext,
    params: CloudPlatformResolveInput,
) -> CloudPlatformResolveOutput:
    platform = _require_platform(ctx)
    category = params.category.strip()
    slug = platform.resolve(category)
    if slug is None:
        return CloudPlatformResolveOutput(
            used=True,
            category=category,
            resolved_slug="",
            reason="category_not_resolved",
        )
    return CloudPlatformResolveOutput(
        used=True,
        category=category,
        resolved_slug=str(slug),
        reason="ok",
    )
