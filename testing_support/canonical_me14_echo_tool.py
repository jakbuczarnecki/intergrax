# © Artur Czarnecki. All rights reserved.

"""Deterministic reference Tool for ME-14 marketplace lifecycle execution proofs."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from intergrax.contracts.capability_catalog import CapabilityReleaseIdentity
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

ME14_TOOL_LOGICAL_ID: Final = "tools.me14.canonical-echo"
ME14_PACKAGE_REFERENCE_V1: Final = "pkg:tool/me14-canonical-echo"
ME14_PACKAGE_REFERENCE_V2: Final = "pkg:tool/me14-canonical-echo"
ME14_VERSION_V1: Final = "1.0.0"
ME14_VERSION_V2: Final = "2.0.0"
ME14_DIGEST_V1: Final = "sha256:" + ("a" * 64)
ME14_DIGEST_V2: Final = "sha256:" + ("b" * 64)
ME14_OUTPUT_V1: Final = "canonical-tool-ok"
ME14_OUTPUT_V2: Final = "canonical-tool-ok-v2"


class Me14EchoInput(BaseModel):
    message: str = Field(default="ping")


class Me14EchoOutput(BaseModel):
    result: str


def me14_echo_contract() -> ToolContract:
    return ToolContract(
        tool_id=ME14_TOOL_LOGICAL_ID,
        name="ME-14 Canonical Echo",
        description="Deterministic echo for marketplace tool vertical E2E proofs.",
        input_schema=Me14EchoInput,
        output_schema=Me14EchoOutput,
        error_mapping={},
        side_effects=False,
    )


class Me14EchoToolHandler(ToolHandler[Me14EchoInput, Me14EchoOutput]):
    def __init__(self, *, output: str) -> None:
        self._output = output

    def execute(self, request: ToolExecutionRequest[Me14EchoInput]) -> Me14EchoOutput:
        del request
        return Me14EchoOutput(result=self._output)


def expected_output_for_release(release: CapabilityReleaseIdentity) -> str:
    version = release.version_label
    digest = release.content_digest
    if version == ME14_VERSION_V2 or digest == ME14_DIGEST_V2:
        return ME14_OUTPUT_V2
    if version == ME14_VERSION_V1 or digest == ME14_DIGEST_V1:
        return ME14_OUTPUT_V1
    raise ValueError(
        f"unsupported ME-14 tool release version={version!r} digest={digest!r}",
    )


def register_me14_echo_for_release(
    registry: ToolRegistry,
    release: CapabilityReleaseIdentity,
) -> None:
    if registry.has(ME14_TOOL_LOGICAL_ID):
        raise ValueError("ME-14 echo tool already active in registry")
    output = expected_output_for_release(release)
    registry.register(me14_echo_contract(), Me14EchoToolHandler(output=output))


__all__ = [
    "ME14_DIGEST_V1",
    "ME14_DIGEST_V2",
    "ME14_OUTPUT_V1",
    "ME14_OUTPUT_V2",
    "ME14_PACKAGE_REFERENCE_V1",
    "ME14_PACKAGE_REFERENCE_V2",
    "ME14_TOOL_LOGICAL_ID",
    "ME14_VERSION_V1",
    "ME14_VERSION_V2",
    "Me14EchoInput",
    "Me14EchoOutput",
    "Me14EchoToolHandler",
    "expected_output_for_release",
    "me14_echo_contract",
    "register_me14_echo_for_release",
]
