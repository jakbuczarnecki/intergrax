# © Artur Czarnecki. All rights reserved.

"""Host-embedded ToolPlugin example — same contract as the external reference package."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

LOCAL_PREFIX_ECHO_TOOL_ID = "local_prefix_echo.ping"
_EXTRAS_PREFIX_KEY = "echo_prefix"


class LocalPrefixEchoInput(BaseModel):
    message: str = Field(default="pong")


class LocalPrefixEchoOutput(BaseModel):
    message: str


def _local_prefix_echo_contract() -> ToolContract:
    return ToolContract(
        tool_id=LOCAL_PREFIX_ECHO_TOOL_ID,
        name="local_prefix_echo.ping",
        description="Host-embedded echo tool consuming ToolWiringContext.extras.",
        input_schema=LocalPrefixEchoInput,
        output_schema=LocalPrefixEchoOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        tags=("reference", "platform-plugin-8", "host-embedded"),
    )


def _prefix_echo_service(ctx: ToolWiringContext, request: LocalPrefixEchoInput) -> LocalPrefixEchoOutput:
    prefix = str(ctx.extras.get(_EXTRAS_PREFIX_KEY, ""))
    body = request.message
    if prefix:
        body = f"{prefix}:{body}"
    return LocalPrefixEchoOutput(message=body)


class LocalPrefixEchoHandler(ServiceToolHandler[LocalPrefixEchoInput, LocalPrefixEchoOutput]):
    _service = _prefix_echo_service


class LocalPrefixEchoToolPlugin:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id="local_prefix_echo",
            tool_ids=(LOCAL_PREFIX_ECHO_TOOL_ID,),
            status=ToolBundleStatus.BETA,
            description="Host-embedded reference tool bundle for PLATFORM-PLUGIN-8.",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        registry.register(_local_prefix_echo_contract(), LocalPrefixEchoHandler(ctx))
