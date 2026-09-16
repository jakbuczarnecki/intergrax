# © Artur Czarnecki. All rights reserved.

"""Delegated provider representative runtime invariant rules."""

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
from intergrax.runtime.execution.delegated_execution.invariants.probe import (
    DelegatedProviderInvariantProbe,
)


def _decision(
    *,
    status: RuntimeInvariantStatus,
    summary: str,
) -> RuntimeInvariantRuleEvaluation:
    return RuntimeInvariantRuleEvaluation(status=status, summary=summary)


@dataclass(frozen=True, slots=True)
class DelegatedProviderNoCanonicalIdentityOwnershipRule:
    """DELEGATION-INV-001 — provider must not own canonical execution identity."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-001"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.CRITICAL
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        if facts.provider_claims_canonical_execution_identity is None:
            return _decision(
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="provider canonical identity ownership not in scope",
            )
        if not facts.provider_claims_canonical_execution_identity:
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="provider does not claim canonical execution identity",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="provider must not own canonical execution identity",
        )


@dataclass(frozen=True, slots=True)
class DelegatedCorrelationImmutableRule:
    """DELEGATION-INV-002 — durable correlation binding matches execution key."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-002"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        if facts.correlation_execution_id is None or facts.correlation_binding_execution_id is None:
            return _decision(
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="correlation evidence not provided",
            )
        if facts.correlation_execution_id == facts.correlation_binding_execution_id:
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="durable correlation remains bound to canonical execution id",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="durable correlation binding execution id mismatch",
        )


@dataclass(frozen=True, slots=True)
class DelegatedNoCorrelationFallbackSynthesisRule:
    """DELEGATION-INV-003 — no provider correlation fallback synthesis."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-003"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        if facts.provider_correlation_fallback_synthesis_active is None:
            return _decision(
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="correlation fallback synthesis not in scope",
            )
        if not facts.provider_correlation_fallback_synthesis_active:
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="no provider correlation fallback synthesis",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="provider correlation fallback synthesis is forbidden",
        )


@dataclass(frozen=True, slots=True)
class DelegatedReattachNoNewWorkRule:
    """DELEGATION-INV-004 — reattach does not create new work."""

    probe: DelegatedProviderInvariantProbe
    rule_id: str = "DELEGATION-INV-004"
    rule_version: str = "1.0.0"
    severity: RuntimeInvariantSeverity = RuntimeInvariantSeverity.HIGH
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.DELEGATED_PROVIDER

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantRuleEvaluation:
        facts = self.probe.read_facts()
        if facts.reattach_creates_new_work is None:
            return _decision(
                status=RuntimeInvariantStatus.NOT_APPLICABLE,
                summary="reattach evidence not provided",
            )
        if not facts.reattach_creates_new_work:
            return _decision(
                status=RuntimeInvariantStatus.PASS,
                summary="reattach does not create new work",
            )
        return _decision(
            status=RuntimeInvariantStatus.VIOLATION,
            summary="reattach must not create new work",
        )


__all__ = [
    "DelegatedCorrelationImmutableRule",
    "DelegatedNoCorrelationFallbackSynthesisRule",
    "DelegatedProviderNoCanonicalIdentityOwnershipRule",
    "DelegatedReattachNoNewWorkRule",
]
