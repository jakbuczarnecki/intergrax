# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import (
    SideEffectRetrySafety,
    ToolContract,
    ToolRiskLevel,
)
from intergrax.tools.providers.http.contracts import HttpRequestInput, HttpRequestOutput
from intergrax.tools.providers.http.handlers import HttpRequestHandler
from intergrax.tools.providers.http.idempotent_post_qual import (
    HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
    register_http_idempotent_post_qual_tool,
)
from intergrax.tools.providers.http.service import HTTP_REQUEST_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

HTTP_BUNDLE_ID = "http"
HTTP_TOOL_IDS: tuple[str, ...] = (
    HTTP_REQUEST_TOOL_ID,
    HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
)


def register_http_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=HTTP_REQUEST_TOOL_ID,
            name=HTTP_REQUEST_TOOL_ID,
            description="Execute one allowlisted HTTP request via configured HttpClientBackend.",
            description_short="HTTP request.",
            input_schema=HttpRequestInput,
            output_schema=HttpRequestOutput,
            error_mapping={},
            side_effects=True,
            side_effect_retry_safety=SideEffectRetrySafety.NOT_RETRY_SAFE,
            category="http",
            risk_level=ToolRiskLevel.HIGH,
            tags=("http", "integration", "network"),
        ),
        HttpRequestHandler(ctx),
    )
    register_http_idempotent_post_qual_tool(registry, ctx)
