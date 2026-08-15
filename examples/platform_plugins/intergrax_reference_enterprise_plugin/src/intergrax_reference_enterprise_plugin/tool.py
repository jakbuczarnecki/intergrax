# © Artur Czarnecki. All rights reserved.

"""Reference ToolPlugin surface for the enterprise multi-capability package."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

REFERENCE_ENTERPRISE_ECHO_TOOL_ID = "reference_enterprise.echo"


class ReferenceEnterpriseEchoInput(BaseModel):
    message: str = Field(default="enterprise")


class ReferenceEnterpriseEchoOutput(BaseModel):
    message: str


def _reference_enterprise_echo_contract() -> ToolContract:
    return ToolContract(
        tool_id=REFERENCE_ENTERPRISE_ECHO_TOOL_ID,
        name="reference_enterprise.echo",
        description="Reference echo tool bundled in the multi-capability enterprise plugin.",
        input_schema=ReferenceEnterpriseEchoInput,
        output_schema=ReferenceEnterpriseEchoOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        tags=("reference", "platform-plugin-docs-6"),
    )


def _echo_service(_ctx: ToolWiringContext, request: ReferenceEnterpriseEchoInput) -> ReferenceEnterpriseEchoOutput:
    return ReferenceEnterpriseEchoOutput(message=request.message)


class ReferenceEnterpriseEchoHandler(
    ServiceToolHandler[ReferenceEnterpriseEchoInput, ReferenceEnterpriseEchoOutput],
):
    _service = _echo_service


class ReferenceEnterpriseEchoToolPlugin:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id="reference_enterprise_echo",
            tool_ids=(REFERENCE_ENTERPRISE_ECHO_TOOL_ID,),
            status=ToolBundleStatus.BETA,
            description="Reference tool bundle in intergrax-reference-enterprise-plugin.",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        registry.register(_reference_enterprise_echo_contract(), ReferenceEnterpriseEchoHandler(ctx))
