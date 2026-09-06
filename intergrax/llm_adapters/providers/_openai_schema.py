# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""OpenAI strict JSON Schema compatibility projection for constrained generation."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from pydantic import BaseModel

from intergrax.knowledge.contracts.validation import JsonObject, JsonValue


def prepare_openai_strict_generation_schema(output_model: type[BaseModel]) -> JsonObject:
    """Build a provider-compatible strict generation schema from a Pydantic output model."""
    canonical_schema = output_model.model_json_schema()
    return project_json_schema_for_openai_strict(canonical_schema)


def project_json_schema_for_openai_strict(schema: Mapping[str, JsonValue]) -> JsonObject:
    """Return a deep-copied schema safe for OpenAI ``strict: true`` structured outputs."""
    projected: JsonObject = copy.deepcopy(dict(schema))
    _normalize_openai_strict_node(projected)
    return projected


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
