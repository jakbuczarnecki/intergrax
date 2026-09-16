# © Artur Czarnecki. All rights reserved.

"""Execution engine representative runtime invariant rules."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity_authority import (
    CANONICAL_IDENTITY_AUTHORITY_MODULE,
    CANONICAL_LIFECYCLE_OWNER_MODULE,
)
from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantDomain,
    RuntimeInvariantDomains,
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantRuleEvaluation,
    RuntimeInvariantSeverity,
    RuntimeInvariantStatus,
)
from intergrax.runtime.execution.invariants.probe import ExecutionInvariantProbe


def _decision(
    *,
    status: RuntimeInvariantStatus,
    summary: str,
) -> RuntimeInvariantRuleEvaluation:
    return RuntimeInvariantRuleEvaluation(status=status, summary=summary)


@dataclass(frozen=True, slots=True)
class ExecutionCanonicalIdentityAuthorityRule:
    """EE-INV-001 — canonical identity authority module anchor."""

    probe: ExecutionInvariantProbe
    rule_id: str = "EE-INV-001"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.EXECUTION

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        if facts.identity_authority_module == CANONICAL_IDENTITY_AUTHORITY_MODULE:
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="canonical execution identity authority module",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="execution identity authority module diverged from canonical contract",
        )


@dataclass(frozen=True, slots=True)
class ExecutionCanonicalLifecycleOwnerRule:
    """EE-INV-002 — lifecycle owner module anchor."""

    probe: ExecutionInvariantProbe
    rule_id: str = "EE-INV-002"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.EXECUTION

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        if facts.lifecycle_owner_module == CANONICAL_LIFECYCLE_OWNER_MODULE:
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="canonical execution lifecycle owner module",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="execution lifecycle owner module diverged from canonical contract",
        )


@dataclass(frozen=True, slots=True)
class ExecutionNoSupportedBypassRule:
    """EE-INV-003 — no supported execution bypass path active."""

    probe: ExecutionInvariantProbe
    rule_id: str = "EE-INV-003"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.EXECUTION

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        if not facts.supported_execution_bypass_active:
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="no supported execution bypass active",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="supported execution bypass must not be active",
        )


__all__ = [
    "ExecutionCanonicalIdentityAuthorityRule",
    "ExecutionCanonicalLifecycleOwnerRule",
    "ExecutionNoSupportedBypassRule",
]
