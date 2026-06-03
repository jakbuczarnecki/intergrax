# © Artur Czarnecki. All rights reserved.

"""Reference :class:`ToolPlugin` for external tool packages."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CUSTOM_ECHO_TOOL_ID = "custom_echo.ping"


class CustomEchoInput(BaseModel):
    message: str = Field(default="pong")


class CustomEchoOutput(BaseModel):
    message: str


def _custom_echo_contract() -> ToolContract:
    return ToolContract(
        tool_id=CUSTOM_ECHO_TOOL_ID,
        name="custom_echo.ping",
        description="Example external echo tool for third-party authors.",
        input_schema=CustomEchoInput,
        output_schema=CustomEchoOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        tags=("example",),
    )


def _echo_service(_ctx: ToolWiringContext, request: CustomEchoInput) -> CustomEchoOutput:
    return CustomEchoOutput(message=request.message)


class CustomEchoHandler(ServiceToolHandler[CustomEchoInput, CustomEchoOutput]):
    _service = _echo_service


class CustomEchoToolPlugin:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id="custom_echo",
            tool_ids=(CUSTOM_ECHO_TOOL_ID,),
            status=ToolBundleStatus.BETA,
            description="Example external tool bundle.",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        registry.register(_custom_echo_contract(), CustomEchoHandler(ctx))
