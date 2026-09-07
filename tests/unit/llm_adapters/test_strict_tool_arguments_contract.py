# © Artur Czarnecki. All rights reserved.

"""DS-E2E-13B — provider-neutral strict tool argument conformance contract."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    CanonicalFunctionToolDefinition,
    StrictToolArgumentConformanceError,
    ToolArgumentConformance,
    ToolDispatchRequirements,
    assert_strict_tool_argument_conformance_supported,
    coerce_canonical_tool_definitions,
    tool_definitions_require_strict_argument_conformance,
)
from intergrax.llm_adapters.providers.openai_responses_adapter import (
    _map_tools_to_responses_api,
)
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    PLANNER_ROUND_TOOL_ID,
    build_atomic_planner_round_schema,
    build_atomic_planner_round_tool_definition,
)
from testing_support.atomic_planner_round_transport import poc_business_tool_schemas

pytestmark = pytest.mark.unit


class _StrictCapableAdapter(LLMAdapter):
    provider = "strict-capable"
    model = "strict-capable"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_strict_tool_argument_conformance(self) -> bool:
        return True

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        raise NotImplementedError


class _ToolsOnlyAdapter(LLMAdapter):
    provider = "tools-only"
    model = "tools-only"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return True

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        raise NotImplementedError


def _same_name_wire_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "same.name",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_planner_round_tool_definition_declares_strict_dispatch_requirements() -> None:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    assert definition.requires_strict_argument_conformance is True
    assert (
        definition.dispatch_requirements.argument_conformance
        is ToolArgumentConformance.STRICT
    )
    assert (
        "requires_strict_argument_conformance"
        not in definition.wire_schema["function"]
    )


def test_planner_round_wire_schema_has_no_magic_strict_field() -> None:
    schema = build_atomic_planner_round_schema(poc_business_tool_schemas())
    assert "requires_strict_argument_conformance" not in schema["function"]


def test_regular_tool_schema_does_not_require_strict_conformance() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "production.telemetry.read",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    (definition,) = coerce_canonical_tool_definitions([tool])
    assert tool_definitions_require_strict_argument_conformance([definition]) is False


def test_request_scoped_requirements_do_not_leak_across_same_name_tools() -> None:
    wire_schema = _same_name_wire_schema()
    strict_definition = CanonicalFunctionToolDefinition(
        wire_schema=wire_schema,  # type: ignore[arg-type]
        dispatch_requirements=ToolDispatchRequirements(
            argument_conformance=ToolArgumentConformance.STRICT
        ),
    )
    default_definition = CanonicalFunctionToolDefinition(
        wire_schema=wire_schema,  # type: ignore[arg-type]
        dispatch_requirements=ToolDispatchRequirements(
            argument_conformance=ToolArgumentConformance.DEFAULT
        ),
    )

    strict_mapped = _map_tools_to_responses_api([strict_definition])[0]
    default_mapped = _map_tools_to_responses_api([default_definition])[0]

    assert strict_mapped.get("strict") is True
    assert "strict" not in default_mapped


def test_assert_strict_support_passes_for_capable_adapter() -> None:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    assert_strict_tool_argument_conformance_supported(
        _StrictCapableAdapter(),
        [definition],
    )


def test_assert_strict_support_fails_closed_for_unsupported_adapter() -> None:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    with pytest.raises(StrictToolArgumentConformanceError, match="does not support"):
        assert_strict_tool_argument_conformance_supported(
            _ToolsOnlyAdapter(),
            [definition],
        )


def test_tools_schema_requires_strict_detects_request_scoped_member() -> None:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    plain_tool = {
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object"}},
    }
    plain_definition = coerce_canonical_tool_definitions([plain_tool])[0]
    assert tool_definitions_require_strict_argument_conformance([plain_definition]) is False
    assert (
        tool_definitions_require_strict_argument_conformance(
            [definition, plain_definition],
        )
        is True
    )


def test_atomic_round_min_items_unchanged_in_canonical_schema() -> None:
    schema = build_atomic_planner_round_schema(poc_business_tool_schemas())
    parameters = schema["function"]["parameters"]
    actions = parameters["properties"]["actions"]
    assert actions["minItems"] == 1
    assert schema["function"]["name"] == PLANNER_ROUND_TOOL_ID


def test_canonical_binding_keeps_schema_requirements_and_guidance_together() -> None:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    (coerced,) = coerce_canonical_tool_definitions([definition])
    assert coerced is definition
    mapped = _map_tools_to_responses_api([coerced])[0]
    assert mapped["strict"] is True
    assert coerced.argument_guidance_text is not None
    assert "production.staffing.attendance.read:" in mapped["parameters"]["properties"]["actions"]["items"]["properties"]["arguments_json"]["description"]


def test_partial_canonical_binding_dict_fails_closed() -> None:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    with pytest.raises(ValueError, match="partial canonical binding"):
        coerce_canonical_tool_definitions(
            [{"wire_schema": definition.wire_schema, "dispatch_requirements": definition.dispatch_requirements}]
        )
