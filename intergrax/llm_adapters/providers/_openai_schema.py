# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""OpenAI strict JSON Schema compatibility projection for constrained generation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from intergrax.knowledge.contracts.validation import JsonObject, JsonValue
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    AtomicPlannerRoundProjectionError,
    extract_admitted_tool_ids_from_discriminated_actions_schema,
)


def prepare_openai_strict_generation_schema(output_model: type[BaseModel]) -> JsonObject:
    """Build a provider-compatible strict generation schema from a Pydantic output model."""
    canonical_schema = output_model.model_json_schema()
    return project_json_schema_for_openai_strict(canonical_schema)


def project_json_schema_for_openai_strict(schema: Mapping[str, JsonValue]) -> JsonObject:
    """Return a deep-copied schema safe for OpenAI ``strict: true`` structured outputs."""
    projected: JsonObject = copy.deepcopy(dict(schema))
    _normalize_openai_strict_node(projected)
    return projected


def project_json_schema_for_openai_strict_tool_parameters(
    schema: Mapping[str, JsonValue],
) -> JsonObject:
    """Return strict-compatible tool ``parameters`` preserving optional-field semantics."""
    projected: JsonObject = copy.deepcopy(dict(schema))
    _promote_optional_properties_for_openai_strict(projected)
    _normalize_openai_strict_node(projected)
    return projected


def _openai_strict_atomic_planner_action_item_schema(
    tool_ids: Sequence[str],
) -> JsonObject:
    if not tool_ids:
        raise AtomicPlannerRoundProjectionError(
            "atomic planner strict projection requires at least one admitted tool id"
        )
    return {
        "type": "object",
        "properties": {
            "tool_id": {"type": "string", "enum": list(tool_ids)},
            "arguments_json": {"type": "string"},
        },
        "required": ["tool_id", "arguments_json"],
        "additionalProperties": False,
    }


def project_atomic_planner_round_parameters_for_openai_strict(
    canonical_parameters: Mapping[str, object],
) -> JsonObject:
    """Project canonical discriminated actions to OpenAI strict-compatible transport."""
    properties = canonical_parameters.get("properties")
    if not isinstance(properties, Mapping):
        raise AtomicPlannerRoundProjectionError("canonical parameters missing properties")
    actions_schema = properties.get("actions")
    if not isinstance(actions_schema, Mapping):
        raise AtomicPlannerRoundProjectionError("canonical parameters missing actions")
    min_items = actions_schema.get("minItems")
    if not isinstance(min_items, int) or min_items < 1:
        raise AtomicPlannerRoundProjectionError(
            "canonical actions must retain minItems >= 1 during projection"
        )
    admitted_tool_ids = extract_admitted_tool_ids_from_discriminated_actions_schema(
        actions_schema
    )
    projected_properties = {
        key: copy.deepcopy(value) for key, value in properties.items() if key != "actions"
    }
    projected_properties["actions"] = {
        "type": "array",
        "minItems": min_items,
        "items": _openai_strict_atomic_planner_action_item_schema(admitted_tool_ids),
    }
    projected: JsonObject = {
        "type": "object",
        "properties": projected_properties,
        "required": list(canonical_parameters.get("required") or []),
        "additionalProperties": canonical_parameters.get("additionalProperties", False),
    }
    return project_json_schema_for_openai_strict_tool_parameters(projected)


def _promote_optional_properties_for_openai_strict(node: JsonValue) -> None:
    if isinstance(node, dict):
        defs = node.get("$defs")
        if isinstance(defs, dict):
            for value in defs.values():
                _promote_optional_properties_for_openai_strict(value)

        for key in ("properties", "patternProperties", "definitions"):
            properties = node.get(key)
            if isinstance(properties, dict):
                for value in properties.values():
                    _promote_optional_properties_for_openai_strict(value)

        for key in ("items", "additionalItems", "not"):
            child = node.get(key)
            if child is not None:
                _promote_optional_properties_for_openai_strict(child)

        for key in ("prefixItems", "allOf", "anyOf", "oneOf"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    _promote_optional_properties_for_openai_strict(child)

        properties = node.get("properties")
        if isinstance(properties, dict) and properties:
            required = list(node.get("required") or [])
            required_set = set(required)
            for prop_name, prop_schema in properties.items():
                if prop_name in required_set:
                    continue
                if isinstance(prop_schema, dict) and "anyOf" in prop_schema:
                    required_set.add(prop_name)
                    continue
                required_set.add(prop_name)
                properties[prop_name] = {
                    "anyOf": [copy.deepcopy(prop_schema), {"type": "null"}],
                }
            node["required"] = sorted(required_set)
        return

    if isinstance(node, list):
        for item in node:
            _promote_optional_properties_for_openai_strict(item)


def _normalize_openai_strict_node(node: JsonValue) -> None:
    if isinstance(node, dict):
        defs = node.get("$defs")
        if isinstance(defs, dict):
            for value in defs.values():
                _normalize_openai_strict_node(value)

        for key in ("properties", "patternProperties", "definitions"):
            properties = node.get(key)
            if isinstance(properties, dict):
                for value in properties.values():
                    _normalize_openai_strict_node(value)

        for key in ("items", "additionalItems", "not"):
            child = node.get(key)
            if child is not None:
                _normalize_openai_strict_node(child)

        for key in ("prefixItems", "allOf", "anyOf", "oneOf"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    _normalize_openai_strict_node(child)

        if "default" in node:
            del node["default"]

        properties = node.get("properties")
        if isinstance(properties, dict) and properties:
            node["required"] = sorted(properties.keys())
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
        return

    if isinstance(node, list):
        for item in node:
            _normalize_openai_strict_node(item)
