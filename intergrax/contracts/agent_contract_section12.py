# © Artur Czarnecki. All rights reserved.

"""§12 minimum contract fields shared by assembly validation and authoring bases."""

from __future__ import annotations

from typing import Any

DEFAULT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Agent domain input payload",
    "additionalProperties": True,
}

DEFAULT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Agent domain output payload",
    "additionalProperties": True,
}

DEFAULT_VALIDATION_RULES: tuple[str, ...] = ("contract.output_schema",)

DEFAULT_FAILURE_MODES: tuple[str, ...] = (
    "tool_failure",
    "policy_denial",
    "budget_exceeded",
)
