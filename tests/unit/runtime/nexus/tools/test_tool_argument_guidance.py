# © Artur Czarnecki. All rights reserved.

"""DS-E2E-13D — provider-neutral business argument guidance tests."""

from __future__ import annotations

import copy

import pytest
from pydantic import BaseModel, ConfigDict, Field

from intergrax.llm_adapters.providers._openai_schema import (
    project_atomic_planner_round_parameters_for_openai_strict,
)
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    build_atomic_planner_argument_guidance_text,
    build_atomic_planner_round_parameters_schema,
    build_atomic_planner_round_tool_definition,
)
from intergrax.runtime.nexus.tools.tool_argument_guidance import (
    ToolArgumentGuidanceProjectionError,
    build_tool_argument_guidance,
    guidance_text_byte_size,
    render_atomic_planner_argument_guidance,
)
from intergrax.tools.exporters.openai import contract_to_openai_tool, compute_openai_tools_schema_hash
from intergrax.tools.exporters.schema import pydantic_parameters_schema
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    LineWindowInput,
    StaffingAttendanceInput,
    TelemetryInput,
)
from testing_support.atomic_planner_round_transport import poc_business_tool_schemas
from testing_support.builder import tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _WorkloadOut(BaseModel):
    line_id: str
    window: str


def _incident_business_schemas() -> list[dict[str, object]]:
    return [
        contract_to_openai_tool(
            tools_agent_make_contract(
                "production.staffing.attendance.read",
                StaffingAttendanceInput,
                StaffingAttendanceInput,
            )
        ),
        contract_to_openai_tool(
            tools_agent_make_contract(
                "production.telemetry.read",
                TelemetryInput,
                TelemetryInput,
            )
        ),
        contract_to_openai_tool(
            tools_agent_make_contract(
                "production.workload.read",
                LineWindowInput,
                _WorkloadOut,
            )
        ),
    ]


def test_staffing_guidance_lists_required_fields() -> None:
    schemas = _incident_business_schemas()
    guidance_text = build_atomic_planner_argument_guidance_text(schemas)
    staffing_section = guidance_text.split("production.telemetry.read:")[0]

    assert "production.staffing.attendance.read:" in guidance_text
    assert "line_id: string" in staffing_section
    assert "shift_id: string" in staffing_section
    assert "window: string" in staffing_section
    assert "  required:" in staffing_section


def test_telemetry_guidance_has_station_id_and_window_without_shift_id() -> None:
    schemas = _incident_business_schemas()
    guidance_text = build_atomic_planner_argument_guidance_text(schemas)
    telemetry_start = guidance_text.index("production.telemetry.read:")
    workload_start = guidance_text.index("production.workload.read:")
    telemetry_section = guidance_text[telemetry_start:workload_start]

    assert "station_id: string" in telemetry_section
    assert "window: string" in telemetry_section
    assert "shift_id" not in telemetry_section


def test_workload_guidance_has_line_id_and_window_without_staffing_fields() -> None:
    schemas = _incident_business_schemas()
    guidance_text = build_atomic_planner_argument_guidance_text(schemas)
    workload_section = guidance_text.split("production.workload.read:", 1)[1]

    assert "line_id: string" in workload_section
    assert "window: string" in workload_section
    assert "shift_id" not in workload_section
    assert "station_id" not in workload_section


def test_field_isolation_between_tools() -> None:
    schemas = _incident_business_schemas()
    guidance_text = build_atomic_planner_argument_guidance_text(schemas)

    staffing_end = guidance_text.index("production.telemetry.read:")
    telemetry_end = guidance_text.index("production.workload.read:")
    staffing = guidance_text[:staffing_end]
    telemetry = guidance_text[staffing_end:telemetry_end]
    workload = guidance_text[telemetry_end:]

    assert "shift_id" in staffing
    assert "shift_id" not in telemetry
    assert "shift_id" not in workload
    assert "station_id" in telemetry
    assert "station_id" not in staffing
    assert "station_id" not in workload


def test_guidance_is_deterministic_for_identical_schema() -> None:
    schemas = _incident_business_schemas()
    first = build_atomic_planner_argument_guidance_text(schemas)
    second = build_atomic_planner_argument_guidance_text(list(copy.deepcopy(schemas)))
    assert first == second


def test_guidance_repeat_build_from_registry_order_is_identical() -> None:
    schemas_a = _incident_business_schemas()
    schemas_b = list(reversed(schemas_a))
    assert build_atomic_planner_argument_guidance_text(schemas_a) == (
        build_atomic_planner_argument_guidance_text(schemas_b)
    )


class _NestedProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    child_id: str = Field(min_length=1, pattern=r"^[a-z]+$")


class _ConstraintProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_name: str = Field(min_length=1, description="Required label.")
    optional_name: str | None = Field(default=None, max_length=8)
    mode: str = Field(description="Mode selector.")
    count: int = Field(ge=1, le=9)
    nested: _NestedProbe
    tags: list[str] = Field(default_factory=list)


def test_constraint_fidelity_for_synthetic_schema() -> None:
    parameters = pydantic_parameters_schema(_ConstraintProbe)
    guidance = build_tool_argument_guidance("probe.tool", parameters)
    rendered = render_atomic_planner_argument_guidance((guidance,))

    assert "required_name: string" in rendered
    assert "optional_name: string" in rendered
    assert "  optional:" in rendered
    assert "mode: string" in rendered
    assert "count: integer" in rendered
    assert "min: 1" in rendered
    assert "max: 9" in rendered
    assert "nested: object" in rendered
    assert "nested.child_id: string" in rendered
    assert "pattern:" in rendered
    assert "tags: array<string>" in rendered
    assert "Required label." in rendered


def test_openai_projection_embeds_argument_guidance_description() -> None:
    schemas = _incident_business_schemas()
    canonical = build_atomic_planner_round_parameters_schema(schemas)
    guidance_text = build_atomic_planner_argument_guidance_text(schemas)
    projected = project_atomic_planner_round_parameters_for_openai_strict(
        canonical,
        argument_guidance_description=guidance_text,
    )

    arguments_json = projected["properties"]["actions"]["items"]["properties"]["arguments_json"]
    assert arguments_json["type"] == "string"
    assert "description" in arguments_json
    assert "shift_id: string" in arguments_json["description"]
    assert "station_id: string" in arguments_json["description"]


def test_tool_definition_carries_provider_neutral_guidance_text() -> None:
    definition = build_atomic_planner_round_tool_definition(_incident_business_schemas())
    assert definition.argument_guidance_text is not None
    assert "production.staffing.attendance.read:" in definition.argument_guidance_text
    assert "shift_id: string" in definition.argument_guidance_text


def test_business_schema_hash_unchanged_when_guidance_is_derived_deterministically() -> None:
    schemas = _incident_business_schemas()
    before = compute_openai_tools_schema_hash(schemas)
    build_atomic_planner_argument_guidance_text(schemas)
    after = compute_openai_tools_schema_hash(schemas)
    assert before == after


def test_guidance_scale_is_linear_for_synthetic_tools() -> None:
    def _schema(index: int) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": f"production.synthetic.tool_{index:03d}",
                "description": f"Synthetic tool {index}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_id": {"type": "string"},
                        "window": {"type": "string"},
                    },
                    "required": ["line_id", "window"],
                    "additionalProperties": False,
                },
            },
        }

    size_3 = guidance_text_byte_size(
        build_atomic_planner_argument_guidance_text(poc_business_tool_schemas())
    )
    size_10 = guidance_text_byte_size(
        build_atomic_planner_argument_guidance_text([_schema(i) for i in range(10)])
    )
    size_100 = guidance_text_byte_size(
        build_atomic_planner_argument_guidance_text([_schema(i) for i in range(100)])
    )

    assert size_3 < size_10 < size_100
    assert size_100 < 250_000


def test_unsupported_oneof_field_fails_closed() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "branch": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ]
            }
        },
        "required": ["branch"],
        "additionalProperties": False,
    }
    with pytest.raises(ToolArgumentGuidanceProjectionError, match="oneOf"):
        build_tool_argument_guidance("unsupported.tool", parameters)
