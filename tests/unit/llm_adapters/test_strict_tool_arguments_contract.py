# © Artur Czarnecki. All rights reserved.

"""DS-E2E-13B — provider-neutral strict tool argument conformance contract."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    REQUIRES_STRICT_ARGUMENT_CONFORMANCE_FIELD,
    StrictToolArgumentConformanceError,
    assert_strict_tool_argument_conformance_supported,
    function_tool_requires_strict_argument_conformance,
    tools_schema_requires_strict_argument_conformance,
)
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    PLANNER_ROUND_TOOL_ID,
    build_atomic_planner_round_schema,
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


def test_planner_round_schema_declares_strict_argument_conformance() -> None:
    schema = build_atomic_planner_round_schema(poc_business_tool_schemas())
    fn = schema["function"]
    assert fn[REQUIRES_STRICT_ARGUMENT_CONFORMANCE_FIELD] is True
    assert function_tool_requires_strict_argument_conformance(schema) is True


def test_regular_tool_schema_does_not_require_strict_conformance() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "production.telemetry.read",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert function_tool_requires_strict_argument_conformance(tool) is False


def test_assert_strict_support_passes_for_capable_adapter() -> None:
    schema = build_atomic_planner_round_schema(poc_business_tool_schemas())
    assert_strict_tool_argument_conformance_supported(_StrictCapableAdapter(), [schema])


def test_assert_strict_support_fails_closed_for_unsupported_adapter() -> None:
    schema = build_atomic_planner_round_schema(poc_business_tool_schemas())
    with pytest.raises(StrictToolArgumentConformanceError, match="does not support"):
        assert_strict_tool_argument_conformance_supported(_ToolsOnlyAdapter(), [schema])


def test_tools_schema_requires_strict_detects_any_member() -> None:
    strict_tool = build_atomic_planner_round_schema(poc_business_tool_schemas())
    plain_tool = {
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object"}},
    }
    assert tools_schema_requires_strict_argument_conformance([plain_tool]) is False
    assert tools_schema_requires_strict_argument_conformance([strict_tool, plain_tool]) is True


def test_atomic_round_min_items_unchanged_in_canonical_schema() -> None:
    schema = build_atomic_planner_round_schema(poc_business_tool_schemas())
    parameters = schema["function"]["parameters"]
    actions = parameters["properties"]["actions"]
    assert actions["minItems"] == 1
    assert schema["function"]["name"] == PLANNER_ROUND_TOOL_ID
