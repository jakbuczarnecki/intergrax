# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compensation execution contracts (ERL — execution foundation)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.enterprise_reliability.compensation import (
    CompensationDisposition,
    CompensationPlan,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    contract_declares_compensation,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityStrategyContext,
)

SCHEMA_COMPENSATION_EXECUTION_REQUEST_V1: Final = "compensation_execution_request.v1"
SCHEMA_COMPENSATION_PLUGIN_EXECUTION_RESULT_V1: Final = (
    "compensation_plugin_execution_result.v1"
)
SCHEMA_COMPENSATION_EXECUTION_RESULT_V1: Final = "compensation_execution_result.v1"


class CompensationExecutionOutcome(StrEnum):
    """Platform execution outcome — no provider-specific semantics."""

    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    UNAVAILABLE = "unavailable"


class CompensationExecutionError(ValueError):
    """Compensation execution cannot proceed under contract and plan rules."""


class CompensationExecutionRequest(BaseModel):
    """Provider-neutral compensation invocation — plugins interpret operation refs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_COMPENSATION_EXECUTION_REQUEST_V1
    tenant_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    plugin_id: str = Field(min_length=1, max_length=256)
    compensation_operation_ref: str = Field(min_length=1, max_length=512)
    plan: CompensationPlan
    execution_context: EnterpriseReliabilityStrategyContext
    effect_contract: ExternalEffectContract

    @model_validator(mode="after")
    def _validate_plan_invoke_plugin(self) -> CompensationExecutionRequest:
        if self.plan.disposition is not CompensationDisposition.INVOKE_PLUGIN:
            raise ValueError("execution request requires invoke_plugin plan disposition")
        if self.plan.plugin_id != self.plugin_id:
            raise ValueError("plan plugin_id must match request plugin_id")
        if self.plan.advice is None:
            raise ValueError("plan advice required for execution request")
        if self.plan.advice.compensation_operation_ref != self.compensation_operation_ref:
            raise ValueError("compensation_operation_ref must match plan advice")
        if self.effect_contract.contract_id != self.contract_id:
            raise ValueError("effect_contract contract_id mismatch")
        if self.execution_context.contract_id != self.contract_id:
            raise ValueError("execution_context contract_id mismatch")
        if self.execution_context.correlation_id != self.correlation_id:
            raise ValueError("execution_context correlation_id mismatch")
        return self


class CompensationPluginExecutionResult(BaseModel):
    """Outcome returned by a plugin executor — core maps to platform result only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_COMPENSATION_PLUGIN_EXECUTION_RESULT_V1
    outcome: CompensationExecutionOutcome
    effect_evidence_ref: str | None = Field(default=None, max_length=512)
    rationale: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _validate_terminal_outcome(self) -> CompensationPluginExecutionResult:
        if self.outcome is CompensationExecutionOutcome.UNAVAILABLE:
            raise ValueError("plugins must not return unavailable — platform assigns that")
        return self


class CompensationExecutionResult(BaseModel):
    """Immutable platform compensation execution outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_COMPENSATION_EXECUTION_RESULT_V1
    outcome: CompensationExecutionOutcome
    plan: CompensationPlan
    request: CompensationExecutionRequest | None = None
    plugin_result: CompensationPluginExecutionResult | None = None
    rationale: str = Field(default="", max_length=512)


def build_compensation_execution_request(
    *,
    plan: CompensationPlan,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
    execution_context: EnterpriseReliabilityStrategyContext,
    effect_contract: ExternalEffectContract,
) -> CompensationExecutionRequest:
    """Materialize an execution request from an invoke_plugin compensation plan."""
    if plan.disposition is not CompensationDisposition.INVOKE_PLUGIN:
        raise CompensationExecutionError(
            "execution request requires invoke_plugin disposition",
        )
    if plan.plugin_id is None or plan.advice is None:
        raise CompensationExecutionError("invoke_plugin plan missing plugin_id or advice")
    if not contract_declares_compensation(effect_contract):
        raise CompensationExecutionError(
            "effect contract does not declare compensation support",
        )
    if effect_contract.contract_id != contract_id:
        raise CompensationExecutionError("effect_contract contract_id mismatch")
    contract_ref = effect_contract.compensation_operation_ref
    advice_ref = plan.advice.compensation_operation_ref
    if contract_ref is not None and contract_ref != advice_ref:
        raise CompensationExecutionError(
            "compensation_operation_ref inconsistent with effect contract",
        )
    return CompensationExecutionRequest(
        tenant_id=tenant_id.strip(),
        correlation_id=correlation_id.strip(),
        contract_id=contract_id.strip(),
        plugin_id=plan.plugin_id,
        compensation_operation_ref=advice_ref,
        plan=plan,
        execution_context=execution_context,
        effect_contract=effect_contract,
    )


__all__ = [
    "CompensationExecutionError",
    "CompensationExecutionOutcome",
    "CompensationExecutionRequest",
    "CompensationExecutionResult",
    "CompensationPluginExecutionResult",
    "SCHEMA_COMPENSATION_EXECUTION_REQUEST_V1",
    "SCHEMA_COMPENSATION_EXECUTION_RESULT_V1",
    "SCHEMA_COMPENSATION_PLUGIN_EXECUTION_RESULT_V1",
    "build_compensation_execution_request",
]
