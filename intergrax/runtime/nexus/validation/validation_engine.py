# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.agent_execution_validation import (
    CapabilityValidator,
    apply_agent_validation_rule,
    frozen_capability_validators,
    validate_agent_execution,
)
from intergrax.contracts.validation import ValidationResult


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
        criteria_tuple = tuple(plan_criteria or ())
        capability_registry = (
            frozen_capability_validators(self._capability_validators)
            if self._capability_validators
            else None
        )
        return validate_agent_execution(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=criteria_tuple,
            capability_validators=capability_registry,
        )

    @staticmethod
    def _apply_rule(rule: str, execution: AgentExecutionResult) -> Optional[str]:
        return apply_agent_validation_rule(rule, execution)
