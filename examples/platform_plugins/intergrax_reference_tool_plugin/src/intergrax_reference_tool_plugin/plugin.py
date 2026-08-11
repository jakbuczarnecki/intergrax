# © Artur Czarnecki. All rights reserved.

"""Reference :class:`ToolPlugin` for third-party wheel delivery (PLATFORM-PLUGIN-8)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

REFERENCE_PREFIX_ECHO_TOOL_ID = "reference_prefix_echo.ping"
_EXTRAS_PREFIX_KEY = "echo_prefix"


class ReferencePrefixEchoInput(BaseModel):
    message: str = Field(default="pong")


class ReferencePrefixEchoOutput(BaseModel):
    message: str


def _reference_prefix_echo_contract() -> ToolContract:
    return ToolContract(
        tool_id=REFERENCE_PREFIX_ECHO_TOOL_ID,
        name="reference_prefix_echo.ping",
        description="Reference external echo tool consuming host-provided ToolWiringContext.extras.",
        input_schema=ReferencePrefixEchoInput,
        output_schema=ReferencePrefixEchoOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        tags=("reference", "platform-plugin-8"),
    )


def _prefix_echo_service(ctx: ToolWiringContext, request: ReferencePrefixEchoInput) -> ReferencePrefixEchoOutput:
    prefix = str(ctx.extras.get(_EXTRAS_PREFIX_KEY, ""))
    body = request.message
    if prefix:
        body = f"{prefix}:{body}"
    return ReferencePrefixEchoOutput(message=body)


class ReferencePrefixEchoHandler(
    ServiceToolHandler[ReferencePrefixEchoInput, ReferencePrefixEchoOutput],
):
    _service = _prefix_echo_service


class ReferencePrefixEchoToolPlugin:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id="reference_prefix_echo",
            tool_ids=(REFERENCE_PREFIX_ECHO_TOOL_ID,),
            status=ToolBundleStatus.BETA,
            description="Reference external tool bundle for PLATFORM-PLUGIN-8.",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        registry.register(_reference_prefix_echo_contract(), ReferencePrefixEchoHandler(ctx))
