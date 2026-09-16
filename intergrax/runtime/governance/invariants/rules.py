# © Artur Czarnecki. All rights reserved.

"""Governance representative runtime invariant rules."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantDomain,
    RuntimeInvariantDomains,
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantRuleEvaluation,
    RuntimeInvariantSeverity,
    RuntimeInvariantStatus,
)
from intergrax.runtime.governance.invariants.probe import GovernanceInvariantProbe


def _decision(
    *,
    status: RuntimeInvariantStatus,
    summary: str,
) -> RuntimeInvariantRuleEvaluation:
    return RuntimeInvariantRuleEvaluation(status=status, summary=summary)


@dataclass(frozen=True, slots=True)
class GovernanceInnerExecutionBindingRule:
    """GOV-INV-001 — meaningful side effect bound to active canonical execution."""

    probe: GovernanceInvariantProbe
    rule_id: str = "GOV-INV-001"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.GOVERNANCE

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        binding = facts.meaningful_side_effect_binding
        if binding is None:
            return _decision(
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="meaningful side effect binding not in scope",
            )
        if (
            binding.request_task_id == binding.active_task_id
            and binding.request_run_id == binding.active_run_id
            and binding.request_attempt_id == binding.active_attempt_id
            and binding.request_execution_id == binding.active_execution_id
        ):
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="meaningful side effect matches active canonical execution",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="meaningful side effect not bound to active canonical execution",
        )


__all__ = ["GovernanceInnerExecutionBindingRule"]
