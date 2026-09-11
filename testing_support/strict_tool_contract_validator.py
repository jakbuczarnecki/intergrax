# © Artur Czarnecki. All rights reserved.

"""Canonical strict tool argument validation for test doubles and fixture adapters."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.strict_tool_call_validation import (
    StrictToolContractValidationError,
    validate_json_against_canonical_schema,
    validate_tool_call_against_definition,
    validate_tool_calls_against_canonical_definitions,
)

STRICT_CAPABILITY_BLOCK_REASON = (
    "BLOCKED: provider lacks strict tool argument conformance"
)

__all__ = [
    "STRICT_CAPABILITY_BLOCK_REASON",
    "StrictToolContractValidationError",
    "validate_json_against_canonical_schema",
    "validate_tool_call_against_definition",
    "validate_tool_calls_against_canonical_definitions",
]
