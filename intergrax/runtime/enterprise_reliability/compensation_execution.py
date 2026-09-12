# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compensation execution — plugin gateway, plan validation, immutable outcomes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.compensation import CompensationDisposition
from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationExecutionError,
    CompensationExecutionOutcome,
    CompensationExecutionResult,
    CompensationPluginExecutionResult,
    build_compensation_execution_request,
)
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyStateRecord
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityPluginGateway,
    EnterpriseReliabilityStrategyContext,
)
from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.runtime.enterprise_reliability.compensation_orchestration import (
    CompensationOrchestrationError,
    ExternalEffectCompensationPlanning,
)


class CompensationExecutionFailure(CompensationOrchestrationError):
    """Plugin or platform failure during compensation execution."""


class ExternalEffectCompensationRun(BaseModel):
    """Compensation execution bundle for downstream recovery and observability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    contract_id: str
    execution: CompensationExecutionResult


def _outcome_for_non_executable_plan(
    disposition: CompensationDisposition,
) -> CompensationExecutionOutcome:
    if disposition is CompensationDisposition.ESCALATE_REQUIRED:
        return CompensationExecutionOutcome.ESCALATED
    if disposition is CompensationDisposition.UNAVAILABLE:
        return CompensationExecutionOutcome.UNAVAILABLE
    if disposition is CompensationDisposition.STRATEGY_UNAVAILABLE:
        return CompensationExecutionOutcome.UNAVAILABLE
    return CompensationExecutionOutcome.ESCALATED


def _compensation_execution_context(
    *,
    planning: ExternalEffectCompensationPlanning,
    tenant_id: str,
) -> EnterpriseReliabilityStrategyContext:
    evidence = planning.evidence
    state = planning.state
    return EnterpriseReliabilityStrategyContext(
        tenant_id=tenant_id,
        correlation_id=state.correlation_id,
        contract_id=planning.contract_id,
        effect_outcome=state.effect_outcome,
        lifecycle_phase=state.lifecycle_phase,
        evidence_verdict=evidence.verdict,
        evidence_ref=evidence.evidence_ref,
    )


def execute_external_effect_compensation(
    *,
    planning: ExternalEffectCompensationPlanning,
    effect_contract: ExternalEffectContract,
    gateway: EnterpriseReliabilityPluginGateway,
    tenant_id: str,
) -> ExternalEffectCompensationRun:
    """
    Run one declared compensation plan through the plugin gateway.

    External mutations live in plugin implementations; core records platform outcomes only.
    """
    plan = planning.plan
    state = planning.state

    if plan.disposition is not CompensationDisposition.INVOKE_PLUGIN:
        outcome = _outcome_for_non_executable_plan(plan.disposition)
        return ExternalEffectCompensationRun(
            state=state,
            contract_id=planning.contract_id,
            execution=CompensationExecutionResult(
                outcome=outcome,
                plan=plan,
                rationale=plan.rationale or plan.disposition.value,
            ),
        )

    assert plan.plugin_id is not None
    context = _compensation_execution_context(planning=planning, tenant_id=tenant_id)
    try:
        request = build_compensation_execution_request(
            plan=plan,
            tenant_id=tenant_id,
            correlation_id=state.correlation_id,
            contract_id=planning.contract_id,
            execution_context=context,
            effect_contract=effect_contract,
        )
    except CompensationExecutionError as exc:
        raise CompensationExecutionFailure(str(exc)) from exc

    if not gateway.compensation_executor_registered(plan.plugin_id):
        return ExternalEffectCompensationRun(
            state=state,
            contract_id=planning.contract_id,
            execution=CompensationExecutionResult(
                outcome=CompensationExecutionOutcome.UNAVAILABLE,
                plan=plan,
                request=request,
                rationale="compensation_executor_missing",
            ),
        )

    try:
        plugin_result = gateway.execute_compensation(plan.plugin_id, request)
    except Exception as exc:
        return ExternalEffectCompensationRun(
            state=state,
            contract_id=planning.contract_id,
            execution=CompensationExecutionResult(
                outcome=CompensationExecutionOutcome.FAILED,
                plan=plan,
                request=request,
                rationale=str(exc)[:512],
            ),
        )

    if plugin_result is None:
        return ExternalEffectCompensationRun(
            state=state,
            contract_id=planning.contract_id,
            execution=CompensationExecutionResult(
                outcome=CompensationExecutionOutcome.UNAVAILABLE,
                plan=plan,
                request=request,
                rationale="compensation_executor_missing",
            ),
        )

    outcome = _platform_outcome_from_plugin(plugin_result)
    return ExternalEffectCompensationRun(
        state=state,
        contract_id=planning.contract_id,
        execution=CompensationExecutionResult(
            outcome=outcome,
            plan=plan,
            request=request,
            plugin_result=plugin_result,
            rationale=plugin_result.rationale,
        ),
    )


def _platform_outcome_from_plugin(
    plugin_result: CompensationPluginExecutionResult,
) -> CompensationExecutionOutcome:
    if plugin_result.outcome is CompensationExecutionOutcome.FAILED:
        return CompensationExecutionOutcome.FAILED
    if plugin_result.outcome is CompensationExecutionOutcome.ESCALATED:
        return CompensationExecutionOutcome.ESCALATED
    return CompensationExecutionOutcome.COMPLETED


__all__ = [
    "CompensationExecutionFailure",
    "ExternalEffectCompensationRun",
    "execute_external_effect_compensation",
]
