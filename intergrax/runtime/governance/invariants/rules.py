# © Artur Czarnecki. All rights reserved.

"""Governance representative runtime invariant rules."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantDomain,
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantResult,
    RuntimeInvariantSeverity,
    RuntimeInvariantStatus,
)
from intergrax.runtime.governance.invariants.probe import GovernanceInvariantProbe


def _result(
    *,
    rule_id: str,
    rule_version: str,
    severity: RuntimeInvariantSeverity,
    status: RuntimeInvariantStatus,
    summary: str,
    context: RuntimeInvariantEvaluationContext,
) -> RuntimeInvariantResult:
    return RuntimeInvariantResult(
        rule_id=rule_id,
        domain=RuntimeInvariantDomain.GOVERNANCE,
        rule_version=rule_version,
        severity=severity,
        status=status,
        summary=summary,
        evaluation_id=context.evaluation_id,
        correlation_id=context.correlation_id,
    )


@dataclass(frozen=True, slots=True)
class GovernanceInnerExecutionBindingRule:
    """GOV-INV-001 — meaningful side effect bound to active canonical execution."""

    probe: GovernanceInvariantProbe
    rule_id: str = "GOV-INV-001"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.GOVERNANCE

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        binding = facts.meaningful_side_effect_binding
        if binding is None:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="meaningful side effect binding not in scope",
                context=context,
            )
        if (
            binding.request_task_id == binding.active_task_id
            and binding.request_run_id == binding.active_run_id
            and binding.request_attempt_id == binding.active_attempt_id
            and binding.request_execution_id == binding.active_execution_id
        ):
            pass
        else:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.VIOLATION,
                summary="meaningful side effect not bound to active canonical execution",
                context=context,
            )
        return _result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.PASS,
            summary="meaningful side effect matches active canonical execution",
            context=context,
        )


__all__ = ["GovernanceInnerExecutionBindingRule"]
