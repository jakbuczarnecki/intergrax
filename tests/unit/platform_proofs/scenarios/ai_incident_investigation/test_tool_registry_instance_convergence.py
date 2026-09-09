# © Artur Czarnecki. All rights reserved.

"""DS-E2E-14.2B — canonical ToolRegistry instance convergence regression gates."""

from __future__ import annotations

import inspect
from pathlib import Path
import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from platform_proofs.scenarios.ai_incident_investigation.application import investigator_agent
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    INVESTIGATOR_AGENT_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_TELEMETRY_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_runtime_bundle,
)

pytestmark = pytest.mark.unit

_PARITY_TOOL_IDS = (
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_TELEMETRY_READ,
)


def _runtime_request() -> RuntimeRequest:
    return RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        message="investigate",
    )


def test_bundle_registry_is_canonical_merged_instance() -> None:
    bundle = build_runtime_bundle()
    composition = bundle.runtime_composition
    platform_registry = composition.platform.env_wiring.tool_wiring.registry

    assert bundle.registry is composition.tool_registry
    assert composition.tool_registry is platform_registry


def test_planner_and_dispatch_share_canonical_registry_instance() -> None:
    bundle = build_runtime_bundle()
    ctx = bundle.investigator.build_context(_runtime_request())
    tool_invoker = ctx.config.tool_invoker
    assert tool_invoker is not None
    assert isinstance(tool_invoker, RuntimeToolInvoker)

    planner_registry = bundle.runtime_composition.tool_registry
    dispatch_registry = tool_invoker.registry

    assert planner_registry is dispatch_registry
    assert bundle.registry is planner_registry


def test_tool_contract_parity_between_planner_and_dispatch_registries() -> None:
    bundle = build_runtime_bundle()
    ctx = bundle.investigator.build_context(_runtime_request())
    tool_invoker = ctx.config.tool_invoker
    assert tool_invoker is not None
    dispatch_registry = tool_invoker.registry
    planner_registry = bundle.runtime_composition.tool_registry

    for tool_id in _PARITY_TOOL_IDS:
        planner_registration = planner_registry.get(tool_id)
        dispatch_registration = dispatch_registry.get(tool_id)
        assert planner_registration.contract == dispatch_registration.contract
        assert planner_registration.handler is dispatch_registration.handler


def test_investigator_does_not_retain_private_planner_registry() -> None:
    bundle = build_runtime_bundle()
    investigator = bundle.investigator

    assert "_registry" not in investigator.__dict__

    source = Path(investigator_agent.__file__).read_text(encoding="utf-8")
    assert "self._registry" not in source
    assert "self._runtime_composition.tool_registry" in source


def test_investigator_constructor_has_no_registry_parameter() -> None:
    signature = inspect.signature(investigator_agent.IncidentInvestigatorAgent.__init__)
    assert "registry" not in signature.parameters
