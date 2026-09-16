# © Artur Czarnecki. All rights reserved.

"""Delegated provider representative runtime invariant rules."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantDomain,
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantResult,
    RuntimeInvariantSeverity,
    RuntimeInvariantStatus,
)
from intergrax.runtime.execution.delegated_execution.invariants.probe import (
    DelegatedProviderInvariantProbe,
)


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
        domain=RuntimeInvariantDomain.DELEGATED_PROVIDER,
        rule_version=rule_version,
        severity=severity,
        status=status,
        summary=summary,
        evaluation_id=context.evaluation_id,
        correlation_id=context.correlation_id,
    )


@dataclass(frozen=True, slots=True)
class DelegatedProviderNoCanonicalIdentityOwnershipRule:
    """DELEGATION-INV-001 — provider must not own canonical execution identity."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-001"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        if facts.provider_claims_canonical_execution_identity is None:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="provider canonical identity ownership not in scope",
                context=context,
            )
        if not facts.provider_claims_canonical_execution_identity:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.PASS,
                summary="provider does not claim canonical execution identity",
                context=context,
            )
        return _result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.VIOLATION,
            summary="provider must not own canonical execution identity",
            context=context,
        )


@dataclass(frozen=True, slots=True)
class DelegatedCorrelationImmutableRule:
    """DELEGATION-INV-002 — durable correlation binding matches execution key."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-002"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        if facts.correlation_execution_id is None or facts.correlation_binding_execution_id is None:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="correlation evidence not provided",
                context=context,
            )
        if facts.correlation_execution_id == facts.correlation_binding_execution_id:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.PASS,
                summary="durable correlation remains bound to canonical execution id",
                context=context,
            )
        return _result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.VIOLATION,
            summary="durable correlation binding execution id mismatch",
            context=context,
        )


@dataclass(frozen=True, slots=True)
class DelegatedNoCorrelationFallbackSynthesisRule:
    """DELEGATION-INV-003 — no provider correlation fallback synthesis."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-003"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        if facts.provider_correlation_fallback_synthesis_active is None:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="correlation fallback synthesis not in scope",
                context=context,
            )
        if not facts.provider_correlation_fallback_synthesis_active:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.PASS,
                summary="no provider correlation fallback synthesis",
                context=context,
            )
        return _result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.VIOLATION,
            summary="provider correlation fallback synthesis is forbidden",
            context=context,
        )


@dataclass(frozen=True, slots=True)
class DelegatedReattachNoNewWorkRule:
    """DELEGATION-INV-004 — reattach does not create new work."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-004"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        facts = self.probe.read_facts()
        if facts.reattach_creates_new_work is None:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="reattach evidence not provided",
                context=context,
            )
        if not facts.reattach_creates_new_work:
            return _result(
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                severity=self.severity,
                status=RuntimeInvariantStatus.PASS,
                summary="reattach does not create new work",
                context=context,
            )
        return _result(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            severity=self.severity,
            status=RuntimeInvariantStatus.VIOLATION,
            summary="reattach must not create new work",
            context=context,
        )


__all__ = [
    "DelegatedCorrelationImmutableRule",
    "DelegatedNoCorrelationFallbackSynthesisRule",
    "DelegatedProviderNoCanonicalIdentityOwnershipRule",
    "DelegatedReattachNoNewWorkRule",
]
