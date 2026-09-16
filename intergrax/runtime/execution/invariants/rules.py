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
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantResult,
    RuntimeInvariantSeverity,
    RuntimeInvariantStatus,
)
from intergrax.runtime.execution.invariants.probe import ExecutionInvariantProbe


def _base_result(
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
        domain=RuntimeInvariantDomain.EXECUTION,
        rule_version=rule_version,
        severity=severity,
        status=status,
        summary=summary,
        evaluation_id=context.evaluation_id,
        correlation_id=context.correlation_id,
    )


@dataclass(frozen=True, slots=True)
class ExecutionCanonicalIdentityAuthorityRule:
    """EE-INV-001 — canonical identity authority module anchor."""

    probe: ExecutionInvariantProbe
    rule_id: str = "EE-INV-001"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.EXECUTION

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        if facts.identity_authority_module == CANONICAL_IDENTITY_AUTHORITY_MODULE:
            return _base_result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.PASS,
                summary="canonical execution identity authority module",
                context=context,
            )
        return _base_result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.VIOLATION,
            summary="execution identity authority module diverged from canonical contract",
            context=context,
        )


@dataclass(frozen=True, slots=True)
class ExecutionCanonicalLifecycleOwnerRule:
    """EE-INV-002 — lifecycle owner module anchor."""

    probe: ExecutionInvariantProbe
    rule_id: str = "EE-INV-002"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.EXECUTION

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        if facts.lifecycle_owner_module == CANONICAL_LIFECYCLE_OWNER_MODULE:
            return _base_result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.PASS,
                summary="canonical execution lifecycle owner module",
                context=context,
            )
        return _base_result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.VIOLATION,
            summary="execution lifecycle owner module diverged from canonical contract",
            context=context,
        )


@dataclass(frozen=True, slots=True)
class ExecutionNoSupportedBypassRule:
    """EE-INV-003 — no supported execution bypass path active."""

    probe: ExecutionInvariantProbe
    rule_id: str = "EE-INV-003"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.EXECUTION

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        if not facts.supported_execution_bypass_active:
            return _base_result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.PASS,
                summary="no supported execution bypass active",
                context=context,
            )
        return _base_result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.VIOLATION,
            summary="supported execution bypass must not be active",
            context=context,
        )


__all__ = [
    "ExecutionCanonicalIdentityAuthorityRule",
    "ExecutionCanonicalLifecycleOwnerRule",
    "ExecutionNoSupportedBypassRule",
]
