# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""JSON Schema helpers for tool export surfaces (Phase O.6)."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel


def pydantic_parameters_schema(model_cls: Type[BaseModel]) -> dict[str, Any]:
    """OpenAI-compatible function ``parameters`` object from a Pydantic input model."""
    raw = model_cls.model_json_schema()
    properties = dict(raw.get("properties", {}) or {})
    defs = raw.get("$defs", {}) or {}
    for key, prop in list(properties.items()):
        ref = prop.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            def_name = ref.rsplit("/", 1)[-1]
            resolved = defs.get(def_name)
            if isinstance(resolved, dict):
                properties[key] = dict(resolved)
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": raw.get("required", []) or [],
        "additionalProperties": False,
    }
    if defs:
        result["$defs"] = defs
    return result
