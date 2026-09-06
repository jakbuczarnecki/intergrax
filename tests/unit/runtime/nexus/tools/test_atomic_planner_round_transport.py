# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — deterministic tests for atomic planner round transport PoC."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    PLANNER_ROUND_TOOL_ID,
    AtomicPlannerRoundError,
    AtomicPlannerRoundSchemaVariant,
    atomic_round_schema_byte_size,
    build_atomic_planner_round_parameters_schema,
    build_atomic_planner_round_schema,
    compute_atomic_planner_round_schema_hash,
    extract_business_tool_schema_entries,
    materialize_atomic_round_to_tool_plan,
    parse_atomic_planner_round_call,
    resolve_atomic_planner_round_calls,
)
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerProtocolConfig,
    NativePlannerProtocolMode,
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


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("production.telemetry.read", _TelemetryIn, _TelemetryOut),
        _StubHandler(),
    )
    registry.register(
        tools_agent_make_contract(
            "production.staffing.attendance.read",
            _AttendanceIn,
            _AttendanceOut,
        ),
        _StubHandler(),
    )
    registry.register(
        tools_agent_make_contract("production.metrics.query", _MetricsIn, _MetricsOut),
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
        mode=NativePlannerProtocolMode.INVESTIGATION_ACTION_CONTEXT,
        available_evidence_references=("obs.ref.a",),
        _reference_index_items=(("obs.ref.a", "obs.ref.a"),),
    )


def test_extract_business_tool_schema_entries_fail_closed_on_malformed() -> None:
    with pytest.raises(AtomicPlannerRoundError, match="missing function.parameters"):
        extract_business_tool_schema_entries(
            [{"type": "function", "function": {"name": "probe.read"}}]
        )


def test_variant_a_schema_has_generic_arguments() -> None:
    schemas = poc_business_tool_schemas()
    params = build_atomic_planner_round_parameters_schema(
        schemas,
        variant=AtomicPlannerRoundSchemaVariant.GENERIC_ENVELOPE,
    )
    properties = params["properties"]
    assert isinstance(properties, dict)
    actions = properties["actions"]
    assert isinstance(actions, dict)
    items = actions["items"]
    assert isinstance(items, dict)
    properties = items["properties"]
    assert isinstance(properties, dict)
    tool_id_schema = properties["tool_id"]
    assert isinstance(tool_id_schema, dict)
    assert "enum" in tool_id_schema


def test_variant_b_schema_uses_one_of_per_tool() -> None:
    schemas = poc_business_tool_schemas()
    params = build_atomic_planner_round_parameters_schema(
        schemas,
        variant=AtomicPlannerRoundSchemaVariant.DISCRIMINATED_ACTIONS,
    )
    properties = params["properties"]
    assert isinstance(properties, dict)
    actions = properties["actions"]
    assert isinstance(actions, dict)
    items = actions["items"]
    assert isinstance(items, dict)
    one_of = items["oneOf"]
    assert isinstance(one_of, list)
    assert len(one_of) == 3


def test_variant_b_schema_larger_than_variant_a() -> None:
    schemas = poc_business_tool_schemas()
    schema_a = build_atomic_planner_round_schema(
        schemas,
        variant=AtomicPlannerRoundSchemaVariant.GENERIC_ENVELOPE,
        strict_provider_schema=False,
    )
    schema_b = build_atomic_planner_round_schema(
        schemas,
        variant=AtomicPlannerRoundSchemaVariant.DISCRIMINATED_ACTIONS,
        strict_provider_schema=False,
    )
    assert atomic_round_schema_byte_size(schema_b) > atomic_round_schema_byte_size(schema_a)


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


def test_atomic_round_schema_hash_is_deterministic() -> None:
    schemas = poc_business_tool_schemas()
    round_schema = build_atomic_planner_round_schema(
        schemas,
        variant=AtomicPlannerRoundSchemaVariant.GENERIC_ENVELOPE,
        strict_provider_schema=False,
    )
    first = compute_atomic_planner_round_schema_hash(round_schema)
    second = compute_atomic_planner_round_schema_hash(round_schema)
    assert first == second
