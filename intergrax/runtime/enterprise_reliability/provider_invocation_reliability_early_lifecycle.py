# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical early-lifecycle reliability evidence emission helpers (GR-7-A8-R1)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityEvidenceObserver,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_evidence import (
    emit_provider_invocation_reliability_fact,
    project_dispatch_attempted,
    project_governance_authorized,
    project_intent_persistence_failed,
    project_intent_persisted,
    project_outcome_persisted,
    project_outcome_persistence_failed,
)


def emit_governance_authorized(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    governance_execution_id: str,
    recorded_at: datetime,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    emit_provider_invocation_reliability_fact(
        project_governance_authorized(
            invocation=invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            governance_execution_id=governance_execution_id,
            recorded_at=recorded_at,
        ),
        observer,
    )


def emit_intent_persisted(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str | None,
    execution_id: str | None,
    attempt_id: str | None,
    recorded_at: datetime,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    emit_provider_invocation_reliability_fact(
        project_intent_persisted(
            invocation=invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            recorded_at=recorded_at,
            execution_id=execution_id,
            attempt_id=attempt_id,
        ),
        observer,
    )


def emit_intent_persistence_failed(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    recorded_at: datetime,
    detail: str,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    emit_provider_invocation_reliability_fact(
        project_intent_persistence_failed(
            invocation=invocation,
            tenant_id=tenant_id,
            recorded_at=recorded_at,
            detail=detail,
        ),
        observer,
    )


def emit_dispatch_attempted(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str | None,
    execution_id: str | None,
    attempt_id: str | None,
    recorded_at: datetime,
    provider_mutation_attempted: bool,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    emit_provider_invocation_reliability_fact(
        project_dispatch_attempted(
            invocation=invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            recorded_at=recorded_at,
            provider_mutation_attempted=provider_mutation_attempted,
            execution_id=execution_id,
            attempt_id=attempt_id,
        ),
        observer,
    )


def emit_outcome_persisted(
    *,
    invocation: ProviderInvocation,
    outcome: ProviderInvocationOutcome,
    tenant_id: str,
    effect_contract_id: str | None,
    execution_id: str | None,
    recorded_at: datetime,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    emit_provider_invocation_reliability_fact(
        project_outcome_persisted(
            invocation=invocation,
            outcome=outcome,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            recorded_at=recorded_at,
            execution_id=execution_id,
        ),
        observer,
    )


def emit_outcome_persistence_failed(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    execution_id: str | None,
    recorded_at: datetime,
    detail: str,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    emit_provider_invocation_reliability_fact(
        project_outcome_persistence_failed(
            invocation=invocation,
            tenant_id=tenant_id,
            recorded_at=recorded_at,
            detail=detail,
            execution_id=execution_id,
        ),
        observer,
    )


__all__ = [
    "emit_dispatch_attempted",
    "emit_governance_authorized",
    "emit_intent_persisted",
    "emit_intent_persistence_failed",
    "emit_outcome_persisted",
    "emit_outcome_persistence_failed",
]
