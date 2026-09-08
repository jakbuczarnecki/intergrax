# © Artur Czarnecki. All rights reserved.

"""DS-E2E-14.2 — strict adapter parity for fixture and planner integration."""

from __future__ import annotations

import json

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    StrictToolArgumentConformanceError,
)
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    PLANNER_ROUND_TOOL_ID,
    build_atomic_planner_round_tool_definition,
)
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerProtocolConfig,
    NativePlannerProtocolMode,
)
from intergrax.runtime.nexus.tools.tool_planning_service import (
    ToolPlanningService,
    build_tool_planning_schema,
)
from intergrax.tools.registry.runtime import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    TOOL_WORKLOAD_READ,
    register_scenario_tools,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.lab_planner_llm import (
    FixtureDrivenIncidentInvestigationLLM,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from testing_support.strict_tool_contract_validator import (
    StrictToolContractValidationError,
    validate_tool_calls_against_canonical_definitions,
)

pytestmark = pytest.mark.unit


def _incident_registry() -> ToolRegistry:
    fixture_bundle = build_fixture_runtime_bundle()
    registry = ToolRegistry()
    register_scenario_tools(registry, fixture_bundle.bundle.operational_data)
    return registry


def _incident_round_definition(registry: ToolRegistry):
    business_schemas = build_tool_planning_schema(registry)
    return build_atomic_planner_round_tool_definition(business_schemas)


def _planner_messages() -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "Investigation phase: initial\n"
                "Objective: gather distinguishing operational evidence."
            ),
        ),
        ChatMessage(
            role="user",
            content="Investigate the incident by requesting production evidence tools within scope.",
        ),
    ]


def test_fixture_adapter_declares_strict_capability() -> None:
    adapter = FixtureDrivenIncidentInvestigationLLM()
    assert adapter.supports_strict_tool_argument_conformance() is True


def test_fixture_adapter_accepts_valid_arguments() -> None:
    adapter = FixtureDrivenIncidentInvestigationLLM()
    registry = _incident_registry()
    definition = _incident_round_definition(registry)
    response = adapter.generate_with_tools(
        _planner_messages(),
        [definition],
    )
    assert response.tool_calls
    assert response.tool_calls[0].name == PLANNER_ROUND_TOOL_ID
    payload = json.loads(response.tool_calls[0].arguments_json)
    assert payload["actions"][0]["tool_id"] == TOOL_WORKLOAD_READ


def test_fixture_adapter_rejects_invalid_arguments() -> None:
    registry = _incident_registry()
    definition = _incident_round_definition(registry)
    invalid_call = LLMToolCall(
        id="tc-invalid",
        name=PLANNER_ROUND_TOOL_ID,
        arguments_json=json.dumps(
            {
                "actions": [
                    {
                        "tool_id": "production.staffing.attendance.read",
                        "arguments": {"line_id": "line4"},
                    }
                ]
            }
        ),
    )
    with pytest.raises(StrictToolContractValidationError, match="oneOf validation failed"):
        validate_tool_calls_against_canonical_definitions([invalid_call], [definition])


def test_atomic_planner_accepts_strict_capable_fixture() -> None:
    adapter = FixtureDrivenIncidentInvestigationLLM()
    planner = ToolPlanningService(adapter, _incident_registry())
    protocol = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
    )
    round_result = planner.plan_native_round(
        _planner_messages(),
        protocol_config=protocol,
    )
    assert round_result.response.tool_calls
    assert round_result.tool_plan.calls
    assert round_result.tool_plan.calls[0].tool_id == TOOL_WORKLOAD_READ


def test_non_strict_adapter_fails_closed_before_dispatch() -> None:
    class _StrictlessAdapter(LLMAdapter):
        provider = "strictless"
        model = "strictless"

        @property
        def context_window_tokens(self) -> int:
            return 8192

        def supports_tools(self) -> bool:
            return True

        def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
            return build_adapter_response(content="unused")

        def generate_with_tools(
            self,
            messages,
            tools_schema,
            *,
            temperature=None,
            max_tokens=None,
            tool_choice=None,
            run_id=None,
        ):
            _ = messages, tools_schema, temperature, max_tokens, tool_choice, run_id
            return build_adapter_response(content="unused")

    planner = ToolPlanningService(_StrictlessAdapter(), _incident_registry())
    protocol = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
    )
    with pytest.raises(StrictToolArgumentConformanceError, match="does not support"):
        planner.plan_native_round(
            _planner_messages(),
            protocol_config=protocol,
        )
