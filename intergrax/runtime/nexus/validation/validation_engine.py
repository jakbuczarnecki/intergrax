# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.validation import ValidationResult

CapabilityValidator = Callable[[AgentExecutionResult], ValidationResult]


class NexusValidationEngine:
    """
    Nexus-level validation on AgentExecutionResult (§29, Phase B.4).
    """

    def __init__(
        self,
        *,
        capability_validators: Optional[Dict[str, CapabilityValidator]] = None,
    ) -> None:
        self._capability_validators = capability_validators or {}

    def validate(
        self,
        execution: AgentExecutionResult,
        *,
        contract: AgentContract,
        capability: Optional[str] = None,
        plan_criteria: Optional[List[str]] = None,
    ) -> ValidationResult:
        errors: List[str] = list(execution.errors)
        warnings: List[str] = list(execution.warnings)

        if execution.status != AgentExecutionStatus.COMPLETED:
            if not errors:
                errors.append(f"agent status: {execution.status.value}")

        if not (execution.summary or "").strip():
            errors.append("empty summary")

        for rule in contract.validation_rules:
            rule_error = self._apply_rule(rule, execution)
            if rule_error:
                errors.append(rule_error)

        for criterion in plan_criteria or []:
            if criterion == "non_empty_summary" and not (execution.summary or "").strip():
                errors.append("plan criterion: non_empty_summary")
            elif criterion.startswith("capability:") and not capability:
                warnings.append(f"capability criterion unverified: {criterion}")

        if capability and capability in self._capability_validators:
            plug_in = self._capability_validators[capability](execution)
            errors.extend(plug_in.errors)
            warnings.extend(plug_in.warnings)
            if not plug_in.valid:
                errors.append(f"capability validator failed: {capability}")

        deduped_errors = list(dict.fromkeys(errors))
        deduped_warnings = list(dict.fromkeys(warnings))
        return ValidationResult(
            valid=len(deduped_errors) == 0,
            errors=deduped_errors,
            warnings=deduped_warnings,
        )

    @staticmethod
    def _apply_rule(rule: str, execution: AgentExecutionResult) -> Optional[str]:
        if rule == "non_empty_summary" and not (execution.summary or "").strip():
            return "validation_rule: non_empty_summary"
        if rule == "no_errors" and execution.errors:
            return "validation_rule: no_errors"
        return None
