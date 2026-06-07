# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.cloud_platform.contracts import (
    CloudPlatformHealthInput,
    CloudPlatformHealthOutput,
    CloudPlatformResolveInput,
    CloudPlatformResolveOutput,
)
from intergrax.tools.providers.cloud_platform.handlers import CloudPlatformHealthHandler, CloudPlatformResolveHandler
from intergrax.tools.providers.cloud_platform.service import (
    CLOUD_PLATFORM_HEALTH_TOOL_ID,
    CLOUD_PLATFORM_RESOLVE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CLOUD_PLATFORM_BUNDLE_ID = "cloud_platform"
CLOUD_PLATFORM_TOOL_IDS: tuple[str, ...] = (
    CLOUD_PLATFORM_HEALTH_TOOL_ID,
    CLOUD_PLATFORM_RESOLVE_TOOL_ID,
)


def register_cloud_platform_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CLOUD_PLATFORM_HEALTH_TOOL_ID,
            name=CLOUD_PLATFORM_HEALTH_TOOL_ID,
            description="Run a startup health probe on the configured cloud platform facade.",
            description_short="Probe cloud platform.",
            input_schema=CloudPlatformHealthInput,
            output_schema=CloudPlatformHealthOutput,
            error_mapping={},
            side_effects=False,
            category="cloud_platform",
            risk_level=ToolRiskLevel.LOW,
            tags=("cloud_platform", "integration", "probe"),
        ),
        CloudPlatformHealthHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CLOUD_PLATFORM_RESOLVE_TOOL_ID,
            name=CLOUD_PLATFORM_RESOLVE_TOOL_ID,
            description="Resolve default integration slug for a category on the configured cloud platform.",
            description_short="Resolve cloud default slug.",
            input_schema=CloudPlatformResolveInput,
            output_schema=CloudPlatformResolveOutput,
            error_mapping={},
            side_effects=False,
            category="cloud_platform",
            risk_level=ToolRiskLevel.LOW,
            tags=("cloud_platform", "integration"),
        ),
        CloudPlatformResolveHandler(ctx),
    )
