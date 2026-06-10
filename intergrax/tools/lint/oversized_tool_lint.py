# © Artur Czarnecki. All rights reserved.

"""Oversized-tool lint for shipped catalog adoption sweep (AUDIT-IDEAL-11.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.exporters.schema import pydantic_parameters_schema

MAX_DESCRIPTION_CHARS = 1024
MAX_INPUT_PROPERTIES = 20
WORKFLOW_DISGUISE_TAGS = frozenset({"workflow_pack", "multi_step_tool"})


@dataclass(frozen=True, slots=True)
class OversizedToolViolation:
    tool_id: str
    reason: str


def _input_property_count(contract: ToolContract) -> int:
    schema = pydantic_parameters_schema(contract.input_schema)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return 0
    return len(properties)


def lint_tool_contract(contract: ToolContract) -> list[OversizedToolViolation]:
    """Return violations when a tool encodes workflow-scale surface area."""
    violations: list[OversizedToolViolation] = []
    description = contract.llm_description()
    if len(description) > MAX_DESCRIPTION_CHARS:
        violations.append(
            OversizedToolViolation(
                tool_id=contract.tool_id,
                reason=f"description exceeds {MAX_DESCRIPTION_CHARS} chars",
            )
        )
    property_count = _input_property_count(contract)
    if property_count > MAX_INPUT_PROPERTIES:
        violations.append(
            OversizedToolViolation(
                tool_id=contract.tool_id,
                reason=f"input schema has {property_count} properties (max {MAX_INPUT_PROPERTIES})",
            )
        )
    if any(tag in WORKFLOW_DISGUISE_TAGS for tag in contract.tags):
        violations.append(
            OversizedToolViolation(
                tool_id=contract.tool_id,
                reason="workflow-disguised tool tag present",
            )
        )
    return violations


def lint_shipped_tool_contracts(contracts: list[ToolContract]) -> list[OversizedToolViolation]:
    """Lint all shipped contracts and return aggregate violations."""
    violations: list[OversizedToolViolation] = []
    for contract in contracts:
        violations.extend(lint_tool_contract(contract))
    return violations
