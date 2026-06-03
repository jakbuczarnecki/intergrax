# © Artur Czarnecki. All rights reserved.

"""Entry-point tool plugin for catalog fixture tests."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

FIXTURE_ECHO_TOOL_ID = "fixture_ep.echo"


class FixtureEchoInput(BaseModel):
    message: str = Field(default="ok")


class FixtureEchoOutput(BaseModel):
    message: str


def _fixture_echo_contract() -> ToolContract:
    return ToolContract(
        tool_id=FIXTURE_ECHO_TOOL_ID,
        name="fixture_ep.echo",
        description="Fixture echo tool for entry-point catalog tests.",
        input_schema=FixtureEchoInput,
        output_schema=FixtureEchoOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        tags=("fixture",),
    )


def _echo_execute(_ctx: ToolWiringContext, request: FixtureEchoInput) -> FixtureEchoOutput:
    return FixtureEchoOutput(message=request.message)


class FixtureEchoHandler(ServiceToolHandler[FixtureEchoInput, FixtureEchoOutput]):
    _service = _echo_execute


class FixtureEchoToolPlugin:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id="fixture_ep",
            tool_ids=(FIXTURE_ECHO_TOOL_ID,),
            status=ToolBundleStatus.BETA,
            description="Fixture entry-point tool bundle for pytest.",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        registry.register(_fixture_echo_contract(), FixtureEchoHandler(ctx))
