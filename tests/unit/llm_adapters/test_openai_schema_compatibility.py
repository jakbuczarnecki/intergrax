# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for OpenAI strict JSON Schema compatibility projection."""

from __future__ import annotations

import copy
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from intergrax.llm_adapters.providers._openai_schema import (
    prepare_openai_strict_generation_schema,
    project_atomic_planner_round_parameters_for_openai_strict,
    project_json_schema_for_openai_strict,
    project_json_schema_for_openai_strict_tool_parameters,
)
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    build_atomic_planner_round_parameters_schema,
)
from testing_support.atomic_planner_round_transport import poc_business_tool_schemas
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    ClaimProposal,
    IncidentReasoningProposal,
)

pytestmark = pytest.mark.unit


class _NestedChild(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="child", max_length=64)
    score: int | None = Field(default=None)


class _NestedRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=32)
    child: _NestedChild
    tags: tuple[str, ...] = ()
    note: str | None = Field(default=None, max_length=128)


def test_base_model_subclass_is_accepted_for_strict_projection() -> None:
    projected = prepare_openai_strict_generation_schema(_NestedRoot)
    assert projected["type"] == "object"
    assert "$defs" in projected


def test_input_schema_is_not_mutated() -> None:
    canonical = _NestedRoot.model_json_schema()
    original = copy.deepcopy(canonical)

    project_json_schema_for_openai_strict(canonical)

    assert canonical == original


def test_defaulted_and_optional_fields_become_required_without_defaults() -> None:
    projected = prepare_openai_strict_generation_schema(_NestedRoot)
    child = projected["$defs"]["_NestedChild"]

    assert child["required"] == ["label", "score"]
    assert "default" not in child["properties"]["label"]
    assert child["additionalProperties"] is False

    root_required = set(projected["required"])
    assert root_required == {"name", "child", "tags", "note"}
    assert "default" not in projected["properties"]["tags"]


def test_claim_proposal_rationale_is_required_for_openai_strict() -> None:
    canonical = ClaimProposal.model_json_schema()
    assert "rationale" not in (canonical.get("required") or [])

    projected = project_json_schema_for_openai_strict(canonical)

    assert projected["required"] == [
        "claim_kind",
        "hypothesis_id",
        "rationale",
        "replaces_prior_claim",
        "statement",
    ]
    assert "default" not in projected["properties"]["rationale"]
    assert projected["additionalProperties"] is False


def test_incident_reasoning_proposal_nested_defs_are_normalized() -> None:
    projected = prepare_openai_strict_generation_schema(IncidentReasoningProposal)
    claim = projected["$defs"]["ClaimProposal"]

    assert "rationale" in claim["required"]
    assert "replaces_prior_claim" in claim["required"]
    assert projected["additionalProperties"] is False
    assert "information_gaps" in projected["required"]
    assert "follow_up_objective" in projected["required"]
    assert "unresolved_reason" in projected["required"]


def test_nullable_optional_field_keeps_anyof_shape() -> None:
    projected = prepare_openai_strict_generation_schema(_NestedRoot)
    note = projected["properties"]["note"]

    assert "anyOf" in note
    assert "default" not in note


def test_array_and_tuple_items_are_processed() -> None:
    projected = prepare_openai_strict_generation_schema(_NestedRoot)
    tags = projected["properties"]["tags"]

    assert tags["type"] == "array"
    assert tags["items"] == {"type": "string"}


class _BranchA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["a"] = "a"
    detail: str = Field(default="")


class _BranchB(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["b"] = "b"
    count: int = Field(default=0)


class _UnionRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: _BranchA | _BranchB


def test_defs_references_are_normalized_recursively() -> None:
    projected = prepare_openai_strict_generation_schema(_UnionRoot)
    branch_a = projected["$defs"]["_BranchA"]
    branch_b = projected["$defs"]["_BranchB"]

    assert branch_a["required"] == ["detail", "kind"]
    assert branch_b["required"] == ["count", "kind"]
    assert "default" not in branch_a["properties"]["detail"]


def test_atomic_planner_tool_parameters_projection_preserves_semantics() -> None:
    canonical = build_atomic_planner_round_parameters_schema(poc_business_tool_schemas())
    projected = project_atomic_planner_round_parameters_for_openai_strict(canonical)

    actions = projected["properties"]["actions"]
    assert actions["minItems"] == 1
    assert "oneOf" not in actions["items"]
    action_item = actions["items"]
    assert action_item["additionalProperties"] is False
    assert action_item["required"] == ["arguments_json", "tool_id"]
    tool_id_schema = action_item["properties"]["tool_id"]
    assert tool_id_schema["type"] == "string"
    assert set(tool_id_schema["enum"]) == {
        "production.metrics.query",
        "production.staffing.attendance.read",
        "production.telemetry.read",
    }
    arguments_json = action_item["properties"]["arguments_json"]
    assert arguments_json["type"] == "string"
    assert "description" not in arguments_json

    action_context = projected["properties"]["action_context"]
    assert "anyOf" in action_context
    assert "action_context" in projected["required"]
    assert projected["additionalProperties"] is False
