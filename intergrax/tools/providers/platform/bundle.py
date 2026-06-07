# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.platform.contracts import (
    PlatformEvaluateFeatureFlagInput,
    PlatformFeatureFlagOutput,
    PlatformGetSecretInput,
    PlatformGetSecretOutput,
    PlatformGetWorkflowRunInput,
    PlatformListCheckSuitesInput,
    PlatformListCheckSuitesOutput,
    PlatformDeleteSecretInput,
    PlatformDeleteSecretOutput,
    PlatformPutSecretInput,
    PlatformPutSecretOutput,
    PlatformWorkflowRunOutput,
)
from intergrax.tools.providers.platform.handlers import (
    PlatformDeleteSecretHandler,
    PlatformEvaluateFeatureFlagHandler,
    PlatformGetSecretHandler,
    PlatformGetWorkflowRunHandler,
    PlatformListCheckSuitesHandler,
    PlatformPutSecretHandler,
)
from intergrax.tools.providers.platform.service import (
    PLATFORM_DELETE_SECRET_TOOL_ID,
    PLATFORM_EVALUATE_FEATURE_FLAG_TOOL_ID,
    PLATFORM_GET_SECRET_TOOL_ID,
    PLATFORM_GET_WORKFLOW_RUN_TOOL_ID,
    PLATFORM_LIST_CHECK_SUITES_TOOL_ID,
    PLATFORM_PUT_SECRET_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

PLATFORM_BUNDLE_ID = "platform"
PLATFORM_TOOL_IDS: tuple[str, ...] = (
    PLATFORM_GET_SECRET_TOOL_ID,
    PLATFORM_PUT_SECRET_TOOL_ID,
    PLATFORM_DELETE_SECRET_TOOL_ID,
    PLATFORM_EVALUATE_FEATURE_FLAG_TOOL_ID,
    PLATFORM_GET_WORKFLOW_RUN_TOOL_ID,
    PLATFORM_LIST_CHECK_SUITES_TOOL_ID,
)


def register_platform_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=PLATFORM_GET_SECRET_TOOL_ID,
            name=PLATFORM_GET_SECRET_TOOL_ID,
            description="Read a tenant-scoped secret from the configured secrets store.",
            description_short="Get secret.",
            input_schema=PlatformGetSecretInput,
            output_schema=PlatformGetSecretOutput,
            error_mapping={},
            side_effects=False,
            category="platform",
            risk_level=ToolRiskLevel.HIGH,
            tags=("platform", "secrets"),
        ),
        PlatformGetSecretHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=PLATFORM_PUT_SECRET_TOOL_ID,
            name=PLATFORM_PUT_SECRET_TOOL_ID,
            description="Create or update a tenant-scoped secret in the configured secrets store.",
            description_short="Put secret.",
            input_schema=PlatformPutSecretInput,
            output_schema=PlatformPutSecretOutput,
            error_mapping={},
            side_effects=True,
            category="platform",
            risk_level=ToolRiskLevel.CRITICAL,
            tags=("platform", "secrets"),
        ),
        PlatformPutSecretHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=PLATFORM_DELETE_SECRET_TOOL_ID,
            name=PLATFORM_DELETE_SECRET_TOOL_ID,
            description="Delete a tenant-scoped secret from the configured secrets store.",
            description_short="Delete secret.",
            input_schema=PlatformDeleteSecretInput,
            output_schema=PlatformDeleteSecretOutput,
            error_mapping={},
            side_effects=True,
            category="platform",
            risk_level=ToolRiskLevel.CRITICAL,
            tags=("platform", "secrets"),
        ),
        PlatformDeleteSecretHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=PLATFORM_EVALUATE_FEATURE_FLAG_TOOL_ID,
            name=PLATFORM_EVALUATE_FEATURE_FLAG_TOOL_ID,
            description="Evaluate a feature flag for a tenant/user context.",
            description_short="Evaluate feature flag.",
            input_schema=PlatformEvaluateFeatureFlagInput,
            output_schema=PlatformFeatureFlagOutput,
            error_mapping={},
            side_effects=False,
            category="platform",
            risk_level=ToolRiskLevel.LOW,
            tags=("platform", "feature_flag"),
        ),
        PlatformEvaluateFeatureFlagHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=PLATFORM_GET_WORKFLOW_RUN_TOOL_ID,
            name=PLATFORM_GET_WORKFLOW_RUN_TOOL_ID,
            description="Fetch CI/CD workflow run status from the configured backend.",
            description_short="Get workflow run.",
            input_schema=PlatformGetWorkflowRunInput,
            output_schema=PlatformWorkflowRunOutput,
            error_mapping={},
            side_effects=False,
            category="platform",
            risk_level=ToolRiskLevel.LOW,
            tags=("platform", "cicd"),
        ),
        PlatformGetWorkflowRunHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=PLATFORM_LIST_CHECK_SUITES_TOOL_ID,
            name=PLATFORM_LIST_CHECK_SUITES_TOOL_ID,
            description="List recent CI check suites for a git ref.",
            description_short="List check suites.",
            input_schema=PlatformListCheckSuitesInput,
            output_schema=PlatformListCheckSuitesOutput,
            error_mapping={},
            side_effects=False,
            category="platform",
            risk_level=ToolRiskLevel.LOW,
            tags=("platform", "cicd"),
        ),
        PlatformListCheckSuitesHandler(ctx),
    )
