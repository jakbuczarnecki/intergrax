# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain-neutral agent execution validation primitives (DS-VER-STAGE-L0).

Extracted from Nexus orchestration so Decision Verification and Nexus can share
deterministic structural checks without Nexus ownership in verification stages.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.validation import ValidationResult

CapabilityValidator = Callable[[AgentExecutionResult], ValidationResult]


@runtime_checkable
class CapabilityValidatorRegistry(Protocol):
    """Lookup capability-scoped validators without exposing mutable registration."""

    def validator_for(self, capability: str) -> CapabilityValidator | None: ...


def apply_agent_validation_rule(
    rule: str,
    execution: AgentExecutionResult,
) -> str | None:
    """Apply one named contract validation rule; return error message or None."""
    if rule == "non_empty_summary" and not (execution.summary or "").strip():
        return "validation_rule: non_empty_summary"
    if rule == "no_errors" and execution.errors:
        return "validation_rule: no_errors"
    return None


def validate_agent_execution(
    execution: AgentExecutionResult,
    *,
    contract: AgentContract,
    capability: str | None = None,
    plan_criteria: tuple[str, ...] = (),
    capability_validators: CapabilityValidatorRegistry | None = None,
) -> ValidationResult:
    """Run deterministic structural validation on one agent execution result."""
    errors: list[str] = list(execution.errors)
    warnings: list[str] = list(execution.warnings)

    if execution.status != AgentExecutionStatus.COMPLETED:
        if not errors:
            errors.append(f"agent status: {execution.status.value}")

    if not (execution.summary or "").strip():
        errors.append("empty summary")

    for rule in contract.validation_rules:
        rule_error = apply_agent_validation_rule(rule, execution)
        if rule_error is not None:
            errors.append(rule_error)

    for criterion in plan_criteria:
        if criterion == "non_empty_summary" and not (execution.summary or "").strip():
            errors.append("plan criterion: non_empty_summary")
        elif criterion.startswith("capability:") and not capability:
            warnings.append(f"capability criterion unverified: {criterion}")

    if capability is not None and capability_validators is not None:
        plug_in = capability_validators.validator_for(capability)
        if plug_in is not None:
            plug_in_result = plug_in(execution)
            errors.extend(plug_in_result.errors)
            warnings.extend(plug_in_result.warnings)
            if not plug_in_result.valid:
                errors.append(f"capability validator failed: {capability}")

    deduped_errors = list(dict.fromkeys(errors))
    deduped_warnings = list(dict.fromkeys(warnings))
    return ValidationResult(
        valid=len(deduped_errors) == 0,
        errors=deduped_errors,
        warnings=deduped_warnings,
    )


@dataclass(frozen=True, slots=True)
class FrozenCapabilityValidators:
    """Immutable capability validator registry backed by a sorted tuple."""

    _entries: tuple[tuple[str, CapabilityValidator], ...]

    def validator_for(self, capability: str) -> CapabilityValidator | None:
        for key, validator in self._entries:
            if key == capability:
                return validator
        return None


def frozen_capability_validators(
    validators: Mapping[str, CapabilityValidator],
) -> FrozenCapabilityValidators:
    """Build an immutable capability validator registry."""
    entries = tuple(sorted(validators.items(), key=lambda item: item[0]))
    return FrozenCapabilityValidators(_entries=entries)
