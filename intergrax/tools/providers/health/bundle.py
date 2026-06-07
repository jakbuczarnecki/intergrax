# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.health.contracts import (
    HealthCheckIntegrationInput,
    HealthCheckIntegrationOutput,
    HealthCheckProfileInput,
    HealthCheckProfileOutput,
)
from intergrax.tools.providers.health.handlers import HealthCheckIntegrationHandler, HealthCheckProfileHandler
from intergrax.tools.providers.health.service import (
    HEALTH_CHECK_INTEGRATION_TOOL_ID,
    HEALTH_CHECK_PROFILE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

HEALTH_BUNDLE_ID = "health"
HEALTH_TOOL_IDS: tuple[str, ...] = (HEALTH_CHECK_INTEGRATION_TOOL_ID, HEALTH_CHECK_PROFILE_TOOL_ID)


def register_health_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=HEALTH_CHECK_INTEGRATION_TOOL_ID,
            name=HEALTH_CHECK_INTEGRATION_TOOL_ID,
            description="Run a health probe for a single integration catalog slug.",
            description_short="Probe integration slug.",
            input_schema=HealthCheckIntegrationInput,
            output_schema=HealthCheckIntegrationOutput,
            error_mapping={},
            side_effects=False,
            category="health",
            risk_level=ToolRiskLevel.LOW,
            tags=("health", "integration", "probe"),
        ),
        HealthCheckIntegrationHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HEALTH_CHECK_PROFILE_TOOL_ID,
            name=HEALTH_CHECK_PROFILE_TOOL_ID,
            description="Run health probes for all integrations configured in the host IntegrationProfile.",
            description_short="Probe integration profile.",
            input_schema=HealthCheckProfileInput,
            output_schema=HealthCheckProfileOutput,
            error_mapping={},
            side_effects=False,
            category="health",
            risk_level=ToolRiskLevel.LOW,
            tags=("health", "integration", "probe"),
        ),
        HealthCheckProfileHandler(ctx),
    )
