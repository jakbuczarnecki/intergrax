# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — deterministic tests for discriminated atomic planner round transport."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    StrictWireProjectionKind,
    ToolArgumentConformance,
)
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    PLANNER_ROUND_TOOL_ID,
    AtomicPlannerRoundError,
    atomic_round_schema_byte_size,
    build_atomic_planner_round_parameters_schema,
    build_atomic_planner_round_schema,
    build_atomic_planner_round_tool_definition,
    compute_atomic_planner_round_schema_hash,
    decode_atomic_planner_round_arguments_json_envelope,
    encode_atomic_planner_round_arguments_json_envelope,
    extract_business_tool_schema_entries,
    materialize_atomic_round_to_tool_plan,
    mint_materialized_tool_calls_from_plan,
    parse_atomic_planner_round_call,
    parse_atomic_planner_round_payload,
    resolve_atomic_planner_round_calls,
)
from intergrax.llm_adapters.providers._openai_schema import (
    project_atomic_planner_round_parameters_for_openai_strict,
)
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerProtocolConfig,
    NativePlannerProtocolMode,
)
from intergrax.runtime.nexus.tools.native_tool_plan_alignment import (
    validate_atomic_tool_plan_alignment,
)
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.atomic_planner_round_transport import poc_business_tool_schemas
from testing_support.builder import tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _TelemetryIn(BaseModel):
    pass


class _TelemetryOut(BaseModel):
    status: str = "ok"


class _AttendanceIn(BaseModel):
    line_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class _AttendanceOut(BaseModel):
    line_id: str
    window: str
    present: int


class _FilterClause(BaseModel):
    field: str
    operator: str
    value: str


class _MetricsIn(BaseModel):
    metric_name: str = Field(min_length=1)
    filters: list[_FilterClause] = Field(default_factory=list)


class _MetricsOut(BaseModel):
    metric_name: str
    value: float


class _StubHandler:
    def execute(self, request):
        _ = request
        raise NotImplementedError


def _registry(*extra: tuple[str, type[BaseModel], type[BaseModel]]) -> ToolRegistry:
    registry = ToolRegistry()
    defaults = (
        ("production.telemetry.read", _TelemetryIn, _TelemetryOut),
        ("production.staffing.attendance.read", _AttendanceIn, _AttendanceOut),
        ("production.metrics.query", _MetricsIn, _MetricsOut),
    )
    for tool_id, input_model, output_model in (*defaults, *extra):
        registry.register(
            tools_agent_make_contract(tool_id, input_model, output_model),
            _StubHandler(),
        )
    return registry


def _round_call(payload: dict[str, object], *, call_id: str = "round-1") -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name=PLANNER_ROUND_TOOL_ID,
        arguments_json=json.dumps(payload),
    )


def _protocol_with_prior() -> NativePlannerProtocolConfig:
    return NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
        available_evidence_references=("obs.ref.a",),
        _reference_index_items=(("obs.ref.a", "obs.ref.a"),),
    )


def test_extract_business_tool_schema_entries_fail_closed_on_malformed() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="missing function.parameters"):
        extract_business_tool_schema_entries(
            [{"type": "function", "function": {"name": "probe.read"}}]
        )


def test_discriminated_schema_uses_one_of_per_tool() -> None:
    schemas = poc_business_tool_schemas()
    params = build_atomic_planner_round_parameters_schema(schemas)
    round_definition = build_atomic_planner_round_tool_definition(schemas)
    round_schema = round_definition.wire_schema
    assert "requires_strict_argument_conformance" not in round_schema["function"]
    assert round_definition.requires_strict_argument_conformance is True
    assert (
        round_definition.dispatch_requirements.strict_wire_projection
        is StrictWireProjectionKind.ATOMIC_PLANNER_DISCRIMINATED_ACTIONS
    )
    properties = params["properties"]
    assert isinstance(properties, dict)
    actions = properties["actions"]
    assert isinstance(actions, dict)
    assert actions["minItems"] == 1
    items = actions["items"]
    assert isinstance(items, dict)
    one_of = items["oneOf"]
    assert isinstance(one_of, list)
    assert len(one_of) == 3


@pytest.mark.parametrize("tool_count", [1, 3, 10])
def test_schema_branch_count_scales_linearly(tool_count: int) -> None:
    schemas: list[dict[str, object]] = []
    for index in range(tool_count):
        tool_id = f"production.probe.{index:02d}.read"
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool_id,
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    params = build_atomic_planner_round_parameters_schema(schemas)
    actions = params["properties"]["actions"]
    assert isinstance(actions, dict)
    items = actions["items"]
    assert isinstance(items, dict)
    one_of = items["oneOf"]
    assert isinstance(one_of, list)
    assert len(one_of) == tool_count


@pytest.mark.parametrize("tool_count", [1, 3, 10])
def test_openai_strict_projection_scales_linearly_without_one_of(tool_count: int) -> None:
    schemas: list[dict[str, object]] = []
    for index in range(tool_count):
        tool_id = f"production.probe.{index:02d}.read"
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool_id,
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    canonical = build_atomic_planner_round_parameters_schema(schemas)
    projected = project_atomic_planner_round_parameters_for_openai_strict(canonical)
    action_item = projected["properties"]["actions"]["items"]
    assert "oneOf" not in action_item
    assert len(action_item["properties"]["tool_id"]["enum"]) == tool_count


def test_provider_roundtrip_preserves_canonical_semantics_for_three_tools() -> None:
    canonical_payloads = [
        {
            "actions": [
                {"tool_id": "production.telemetry.read", "arguments": {}},
            ],
        },
        {
            "actions": [
                {
                    "tool_id": "production.staffing.attendance.read",
                    "arguments": {"line_id": "L1", "window": "last_hour"},
                },
            ],
        },
        {
            "actions": [
                {
                    "tool_id": "production.metrics.query",
                    "arguments": {
                        "metric_name": "error_rate",
                        "filters": [
                            {"field": "service", "operator": "eq", "value": "checkout"},
                        ],
                    },
                },
            ],
        },
    ]
    for canonical_payload in canonical_payloads:
        provider_payload = encode_atomic_planner_round_arguments_json_envelope(
            canonical_payload
        )
        recovered = decode_atomic_planner_round_arguments_json_envelope(provider_payload)
        decision = parse_atomic_planner_round_payload(recovered)
        materialize_atomic_round_to_tool_plan(decision, _registry())
        assert recovered == canonical_payload


def test_provider_decode_rejects_malformed_arguments_json() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="arguments_json is malformed"):
        decode_atomic_planner_round_arguments_json_envelope(
            {
                "actions": [
                    {
                        "tool_id": "production.telemetry.read",
                        "arguments_json": "{not-json",
                    }
                ],
            }
        )


def test_provider_decode_rejects_missing_arguments_payload() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="missing arguments_json"):
        decode_atomic_planner_round_arguments_json_envelope(
            {
                "actions": [
                    {"tool_id": "production.telemetry.read"},
                ],
            }
        )


def test_strict_dispatch_metadata_declares_discriminated_actions_projection() -> None:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    assert (
        definition.dispatch_requirements.argument_conformance
        is ToolArgumentConformance.STRICT
    )
    assert (
        definition.dispatch_requirements.strict_wire_projection
        is StrictWireProjectionKind.ATOMIC_PLANNER_DISCRIMINATED_ACTIONS
    )


def test_schema_ordering_and_hash_are_stable() -> None:
    schemas = poc_business_tool_schemas()
    first_schema = build_atomic_planner_round_schema(schemas)
    second_schema = build_atomic_planner_round_schema(list(reversed(schemas)))
    assert first_schema == second_schema
    first_hash = compute_atomic_planner_round_schema_hash(first_schema)
    second_hash = compute_atomic_planner_round_schema_hash(second_schema)
    assert first_hash == second_hash
    assert atomic_round_schema_byte_size(first_schema) > 0


def test_parse_and_materialize_single_action() -> None:
    payload = {
        "action_context": {
            "evidence_basis_references": ["obs.ref.a"],
            "purpose": "Correlate telemetry with staffing.",
        },
        "actions": [
            {
                "tool_id": "production.telemetry.read",
                "arguments": {},
            }
        ],
    }
    decision = parse_atomic_planner_round_call(_round_call(payload))
    plan = materialize_atomic_round_to_tool_plan(decision, _registry())
    assert len(plan.calls) == 1
    assert plan.calls[0].tool_id == "production.telemetry.read"
    validate_atomic_tool_plan_alignment(decision.actions, plan)


def test_materialize_multi_action_preserves_order() -> None:
    payload = {
        "action_context": {
            "evidence_basis_references": ["obs.ref.a"],
            "purpose": "Cross-check staffing and metrics.",
        },
        "actions": [
            {
                "tool_id": "production.staffing.attendance.read",
                "arguments": {"line_id": "L1", "window": "last_hour"},
            },
            {
                "tool_id": "production.metrics.query",
                "arguments": {
                    "metric_name": "error_rate",
                    "filters": [
                        {"field": "service", "operator": "eq", "value": "checkout"},
                    ],
                },
            },
        ],
    }
    decision = parse_atomic_planner_round_call(_round_call(payload))
    plan = materialize_atomic_round_to_tool_plan(decision, _registry())
    assert [call.tool_id for call in plan.calls] == [
        "production.staffing.attendance.read",
        "production.metrics.query",
    ]


def test_duplicate_actions_not_deduplicated() -> None:
    payload = {
        "actions": [
            {"tool_id": "production.telemetry.read", "arguments": {}},
            {"tool_id": "production.telemetry.read", "arguments": {}},
        ],
    }
    decision = parse_atomic_planner_round_call(_round_call(payload))
    plan = materialize_atomic_round_to_tool_plan(decision, _registry())
    assert len(plan.calls) == 2


def test_invalid_arguments_fail_pydantic_validation() -> None:
    payload = {
        "actions": [
            {
                "tool_id": "production.staffing.attendance.read",
                "arguments": {"line_id": "L1"},
            }
        ],
    }
    decision = parse_atomic_planner_round_call(_round_call(payload))
    with pytest.raises(Exception):
        materialize_atomic_round_to_tool_plan(decision, _registry())


def test_unknown_tool_id_rejected_before_plan() -> None:
    payload = {
        "actions": [{"tool_id": "unknown.tool", "arguments": {}}],
    }
    decision = parse_atomic_planner_round_call(_round_call(payload))
    with pytest.raises(AtomicPlannerRoundError, match="tool not registered"):
        materialize_atomic_round_to_tool_plan(decision, _registry())


def test_resolve_rejects_sibling_business_calls() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="sibling business"):
        resolve_atomic_planner_round_calls(
            (
                _round_call(
                    {
                        "action_context": {
                            "evidence_basis_references": ["obs.ref.a"],
                            "purpose": "probe",
                        },
                        "actions": [
                            {"tool_id": "production.telemetry.read", "arguments": {}},
                        ],
                    }
                ),
                LLMToolCall(
                    id="biz-1",
                    name="production.telemetry.read",
                    arguments_json="{}",
                ),
            ),
            protocol_config=_protocol_with_prior(),
            admitted_tool_ids=frozenset({"production.telemetry.read"}),
        )


def test_resolve_requires_action_context_with_prior_evidence() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="requires action_context"):
        resolve_atomic_planner_round_calls(
            (
                _round_call(
                    {
                        "actions": [
                            {"tool_id": "production.telemetry.read", "arguments": {}},
                        ],
                    }
                ),
            ),
            protocol_config=_protocol_with_prior(),
            admitted_tool_ids=frozenset({"production.telemetry.read"}),
        )


def test_parse_rejects_empty_actions() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="schema validation failed"):
        parse_atomic_planner_round_payload({"actions": []})


def test_resolve_rejects_empty_actions() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="schema validation failed"):
        resolve_atomic_planner_round_calls(
            (_round_call({"actions": []}),),
            protocol_config=_protocol_with_prior(),
            admitted_tool_ids=frozenset({"production.telemetry.read"}),
        )


def test_resolve_rejects_multiple_planner_round_calls() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="cardinality"):
        resolve_atomic_planner_round_calls(
            (
                _round_call({"actions": [{"tool_id": "production.telemetry.read", "arguments": {}}]}, call_id="a"),
                _round_call({"actions": [{"tool_id": "production.telemetry.read", "arguments": {}}]}, call_id="b"),
            ),
            protocol_config=_protocol_with_prior(),
            admitted_tool_ids=frozenset({"production.telemetry.read"}),
        )


def test_atomic_round_schema_hash_is_deterministic() -> None:
    schemas = poc_business_tool_schemas()
    round_schema = build_atomic_planner_round_schema(schemas)
    first = compute_atomic_planner_round_schema_hash(round_schema)
    second = compute_atomic_planner_round_schema_hash(round_schema)
    assert first == second


class _TerminationCapturingAdapter(LLMAdapter):
    provider = "fake-termination"
    model = "fake-termination"

    def __init__(self, *, content: str) -> None:
        super().__init__()
        self._content = content
        self.received_tool_choice: object | None = None

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return True

    def supports_strict_tool_argument_conformance(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return False

    def generate_messages(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        run_id=None,
    ):
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
        self.received_tool_choice = tool_choice
        return build_adapter_response(content=self._content)


def test_atomic_round_fails_closed_when_adapter_lacks_strict_capability() -> None:
    class _StrictlessAdapter(_TerminationCapturingAdapter):
        def supports_strict_tool_argument_conformance(self) -> bool:
            return False

    adapter = _StrictlessAdapter(content="unused")
    planner = ToolPlanningService(adapter, _registry())
    protocol = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
    )
    from intergrax.llm_adapters.contracts.strict_tool_arguments import (
        StrictToolArgumentConformanceError,
    )

    with pytest.raises(StrictToolArgumentConformanceError, match="does not support"):
        planner.plan_native_round(
            [ChatMessage(role="user", content="Plan one round.")],
            protocol_config=protocol,
        )


def test_atomic_round_termination_without_tool_calls() -> None:
    adapter = _TerminationCapturingAdapter(content="Final investigation summary.")
    planner = ToolPlanningService(adapter, _registry())
    protocol = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ATOMIC_ROUND,
    )
    round_result = planner.plan_native_round(
        [ChatMessage(role="user", content="Summarize findings.")],
        protocol_config=protocol,
    )
    assert round_result.response.content == "Final investigation summary."
    assert round_result.tool_plan.calls == []
    assert round_result.materialized_tool_calls == ()


def test_mint_materialized_tool_calls_assigns_identities() -> None:
    payload = {
        "actions": [{"tool_id": "production.telemetry.read", "arguments": {}}],
    }
    decision = parse_atomic_planner_round_call(_round_call(payload))
    plan = materialize_atomic_round_to_tool_plan(decision, _registry())
    materialized = mint_materialized_tool_calls_from_plan(plan)
    assert len(materialized) == 1
    assert materialized[0].id.startswith("toolcall-")
    assert materialized[0].name == "production.telemetry.read"
