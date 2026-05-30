# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""JSON Schema helpers for tool export surfaces (Phase O.6)."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel


def pydantic_parameters_schema(model_cls: Type[BaseModel]) -> dict[str, Any]:
    """OpenAI-compatible function ``parameters`` object from a Pydantic input model."""
    raw = model_cls.model_json_schema()
    return {
        "type": "object",
        "properties": raw.get("properties", {}) or {},
        "required": raw.get("required", []) or [],
        "additionalProperties": False,
    }
