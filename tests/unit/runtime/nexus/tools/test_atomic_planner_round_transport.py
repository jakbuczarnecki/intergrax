# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — deterministic tests for discriminated atomic planner round transport."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    PLANNER_ROUND_TOOL_ID,
    AtomicPlannerRoundError,
    atomic_round_schema_byte_size,
    build_atomic_planner_round_parameters_schema,
    build_atomic_planner_round_schema,
    compute_atomic_planner_round_schema_hash,
    extract_business_tool_schema_entries,
    materialize_atomic_round_to_tool_plan,
    mint_materialized_tool_calls_from_plan,
    parse_atomic_planner_round_call,
    resolve_atomic_planner_round_calls,
)
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
    properties = params["properties"]
    assert isinstance(properties, dict)
    actions = properties["actions"]
    assert isinstance(actions, dict)
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


def test_resolve_rejects_empty_actions() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="empty actions"):
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
