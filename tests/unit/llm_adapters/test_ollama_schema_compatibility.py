# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for Ollama JSON Schema compatibility projection."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intergrax.llm_adapters.providers._ollama_schema import (
    OLLAMA_UNSUPPORTED_MAX_LENGTH_THRESHOLD,
    RecursiveSchemaReferenceError,
    convert_oneof_to_anyof,
    inline_refs,
    prepare_ollama_generation_schema,
    project_json_schema_for_ollama,
    remove_defaults,
    remove_discriminator_metadata,
    remove_non_validation_metadata,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "applications"))
from local_workspace_application.conversation.interaction_models import (  # noqa: E402
    ConversationInteractionPlan,
)

pytestmark = pytest.mark.unit


class _LongTextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_field: str = Field(max_length=128)
    long_field: str = Field(max_length=2000)
    very_long_field: str = Field(max_length=16_000)


def _collect_max_lengths(node: object) -> list[int]:
    values: list[int] = []
    if isinstance(node, dict):
        max_length = node.get("maxLength")
        if isinstance(max_length, int):
            values.append(max_length)
        for value in node.values():
            values.extend(_collect_max_lengths(value))
    elif isinstance(node, list):
        for item in node:
            values.extend(_collect_max_lengths(item))
    return values


def test_input_schema_is_not_mutated() -> None:
    canonical = _LongTextModel.model_json_schema()
    original = copy.deepcopy(canonical)

    project_json_schema_for_ollama(canonical)

    assert canonical == original


def test_deterministic_output() -> None:
    canonical = _LongTextModel.model_json_schema()

    first = project_json_schema_for_ollama(canonical)
    second = project_json_schema_for_ollama(canonical)

    assert first == second


def test_metadata_removal_helper_only_removes_non_validation_keys() -> None:
    schema = {
        "title": "Example",
        "description": "desc",
        "type": "object",
        "properties": {"name": {"type": "string", "title": "Name"}},
        "required": ["name"],
    }

    projected = remove_non_validation_metadata(schema)

    assert "title" not in projected
    assert "description" not in projected
    assert projected["required"] == ["name"]
    assert "title" not in projected["properties"]["name"]
    assert schema["title"] == "Example"


def test_production_projection_preserves_metadata() -> None:
    canonical = _LongTextModel.model_json_schema()
    projected = project_json_schema_for_ollama(canonical)

    assert projected["title"] == canonical["title"]


def test_reference_inlining_resolves_nested_defs() -> None:
    schema = {
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        },
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "required": ["inner"],
    }

    projected = inline_refs(schema)

    assert "$defs" not in projected
    assert projected["properties"]["inner"] == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }


def test_repeated_references_are_inlined_in_each_location() -> None:
    schema = {
        "$defs": {
            "Node": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
        },
        "type": "object",
        "properties": {
            "left": {"$ref": "#/$defs/Node"},
            "right": {"$ref": "#/$defs/Node"},
        },
        "required": ["left", "right"],
    }

    projected = inline_refs(schema)
    left = projected["properties"]["left"]
    right = projected["properties"]["right"]

    assert left == right
    assert left is not right


def test_recursive_reference_rejection() -> None:
    schema = {
        "$defs": {"Node": {"$ref": "#/$defs/Node"}},
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
    }

    with pytest.raises(RecursiveSchemaReferenceError, match="recursive schema reference"):
        inline_refs(schema)


def test_discriminator_removal_keeps_literal_property_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "item": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "a"}, "a_field": {"type": "string"}},
                        "required": ["kind", "a_field"],
                    },
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "b"}, "b_field": {"type": "integer"}},
                        "required": ["kind", "b_field"],
                    },
                ],
                "discriminator": {"propertyName": "kind"},
            }
        },
    }

    projected = remove_discriminator_metadata(schema)

    assert "discriminator" not in projected["properties"]["item"]
    assert "oneOf" in projected["properties"]["item"]


def test_oneof_conversion_preserves_all_branches() -> None:
    schema = {
        "type": "object",
        "properties": {
            "item": {
                "oneOf": [
                    {"type": "object", "properties": {"kind": {"const": "a"}}},
                    {"type": "object", "properties": {"kind": {"const": "b"}}},
                ]
            }
        },
    }
    original_branches = schema["properties"]["item"]["oneOf"]

    projected = convert_oneof_to_anyof(schema)

    assert "oneOf" not in projected["properties"]["item"]
    assert projected["properties"]["item"]["anyOf"] == original_branches


def test_unsupported_string_max_length_is_removed() -> None:
    canonical = _LongTextModel.model_json_schema()
    projected = project_json_schema_for_ollama(canonical)

    assert 128 in _collect_max_lengths(projected)
    assert 2000 not in _collect_max_lengths(projected)
    assert 16_000 not in _collect_max_lengths(projected)
    assert all(
        value < OLLAMA_UNSUPPORTED_MAX_LENGTH_THRESHOLD
        for value in _collect_max_lengths(projected)
    )


def test_prepare_ollama_generation_schema_from_model() -> None:
    projected = prepare_ollama_generation_schema(_LongTextModel)

    assert projected["type"] == "object"
    assert 2000 not in _collect_max_lengths(projected)


def test_remove_defaults_helper() -> None:
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer", "default": 0}},
    }

    projected = remove_defaults(schema)

    assert "default" not in projected["properties"]["count"]
    assert schema["properties"]["count"]["default"] == 0


def test_canonical_top_level_requirements_preserved() -> None:
    canonical = ConversationInteractionPlan.model_json_schema()
    projected = project_json_schema_for_ollama(canonical)

    for field in ("plan_version", "objects", "actions", "clarifications", "response_mode"):
        assert field in projected["properties"]
    assert set(projected["required"]) == set(canonical["required"])


def test_invalid_generation_payload_fails_original_model_validation() -> None:
    class _ActivateAction(BaseModel):
        model_config = ConfigDict(extra="forbid")

        action_type: Literal["workspace.activate"]
        action_id: str
        workspace: str
        depends_on: tuple[str, ...] = ()
        evidence_quotes: tuple[str, ...] = ()
        evidence_attachment_ids: tuple[str, ...] = ()

    class _Plan(BaseModel):
        model_config = ConfigDict(extra="forbid")

        plan_version: Literal["2"]
        objects: tuple[object, ...] = ()
        actions: tuple[_ActivateAction, ...]
        clarifications: tuple[object, ...] = ()
        response_mode: Literal["aggregate"]

    invalid_payload = {
        "plan_version": "2",
        "objects": [],
        "actions": [
            {
                "action_type": "workspace.activate",
                "action_id": "a1",
                "depends_on": [],
                "evidence_quotes": [],
                "evidence_attachment_ids": [],
            }
        ],
        "clarifications": [],
        "response_mode": "aggregate",
    }

    with pytest.raises(ValidationError):
        _Plan.model_validate(invalid_payload)


def test_invalid_plan_version_type_fails_original_validation() -> None:
    class _Plan(BaseModel):
        model_config = ConfigDict(extra="forbid")

        plan_version: Literal["2"]
        objects: tuple[object, ...] = ()
        actions: tuple[object, ...] = ()
        clarifications: tuple[object, ...] = ()
        response_mode: Literal["aggregate"]

    with pytest.raises(ValidationError):
        _Plan.model_validate({"plan_version": 2})


def test_alias_field_name_is_not_introduced() -> None:
    class _WorkspaceAction(BaseModel):
        model_config = ConfigDict(extra="forbid")

        action_type: Literal["workspace.activate"]
        action_id: str
        workspace: str

    projected = prepare_ollama_generation_schema(_WorkspaceAction)

    assert "workspace" in str(projected["properties"])
    assert "workspace_reference" not in str(projected)

    with pytest.raises(ValidationError):
        _WorkspaceAction.model_validate(
            {
                "action_type": "workspace.activate",
                "action_id": "a1",
                "workspace_reference": "magazyn",
            }
        )
