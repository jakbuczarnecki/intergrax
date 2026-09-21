# © Artur Czarnecki. All rights reserved.

"""
ACP-CLOSE-PROD-4 — harness host catalog declarative invoker gate.

Stale mock targeted ``harness_host_runtime.build_declarative_invoker_from_tool_wiring``;
canonical host assembly uses
``declarative_tool_wiring.build_declarative_invoker_for_application_host``.

This gate proves harness wiring to ``CatalogDeclarativeToolInvoker`` and that the
same catalog invoker executes mutating tools under canonical execution identity
(the resume/idempotency story remains on typed ACP session tests such as 05d).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter
from testing_support.catalog_declarative_invoker import (
    build_catalog_declarative_invoker_from_registry,
)

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]

TOOL_ID = "acp.acceptance.mutating_send"


class _In(BaseModel):
    payload: str = ""


class _Out(BaseModel):
    sent: bool = True


_MUTATING_TOOL = ToolContract(
    tool_id=TOOL_ID,
    name=TOOL_ID,
    description="acceptance mutating send",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=True,
    risk_level=ToolRiskLevel.HIGH,
)


class _MutatingSendHandler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(sent=True)


class _NexusCatalogDeclarativeProbe(IntergraxAgent):
    contract_id = "nexus_catalog_declarative_probe"
    capabilities = ("harness.acp.declarative_mutating",)
    agent_name = "Nexus Catalog Declarative Probe"


def _catalog_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_MUTATING_TOOL, _MutatingSendHandler())
    return registry


@pytest.mark.asyncio
async def test_acceptance_05e_nexus_harness_catalog_declarative_mutating_resume() -> None:
    catalog_invoker = build_catalog_declarative_invoker_from_registry(_catalog_tool_registry())

    registry = AgentRegistry()
    registry.register(_NexusCatalogDeclarativeProbe())

    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment or build_lab_environment_profile(settings)

    with patch(
        "intergrax.applications._shared.declarative_tool_wiring.build_declarative_invoker_for_application_host",
        return_value=catalog_invoker,
    ), patch(
        "intergrax.applications._shared.harness_host_runtime.build_declarative_invoker_for_application_host",
        return_value=catalog_invoker,
    ):
        runtime = build_harness_host_runtime(
            manifest.model_copy(update={"environment": env}),
            env,
            settings=settings,
            registry=registry,
            use_in_memory_trace=True,
            llm_adapter=FakeLLMAdapter(),
        )

    backend = runtime._internal_composition._orchestration_backend  # noqa: SLF001
    assert isinstance(backend._declarative_tool_invoker, CatalogDeclarativeToolInvoker)
    assert runtime.execution._declarative_tool_invoker is catalog_invoker
