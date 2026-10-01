# © Artur Czarnecki. All rights reserved.

"""Neutral agent execution validation engine port (Nexus implements structurally)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.validation import ValidationResult


@runtime_checkable
class AgentExecutionValidationEnginePort(Protocol):
    """Validate one agent execution result against contract and plan criteria."""

    def validate(
        self,
        execution: AgentExecutionResult,
        *,
        contract: AgentContract,
        capability: str | None = None,
        plan_criteria: list[str] | None = None,
    ) -> ValidationResult: ...


__all__ = ["AgentExecutionValidationEnginePort"]
