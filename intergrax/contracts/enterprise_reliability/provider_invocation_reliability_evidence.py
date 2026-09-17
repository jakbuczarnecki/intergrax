# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed provider-invocation reliability evidence (GR-7-A8) — projection only, not source of truth."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.effect_contract import UnknownUncertaintyPosture
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationReason,
    ProviderInvocationReconciliationVerdict,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryReason,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityReason,
    ExternalEffectRepeatEligibilityVerdict,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus

SCHEMA_PROVIDER_INVOCATION_RELIABILITY_FACT_V1: Final = (
    "provider_invocation_reliability_fact.v1"
)
_MAX_REASON = 512


class ProviderInvocationReliabilityTracePhase(StrEnum):
    """Canonical lifecycle phases for end-to-end reliability trace reconstruction."""

    GOVERNANCE_AUTHORIZED = "governance_authorized"
    INTENT_PERSISTED = "intent_persisted"
    INTENT_PERSISTENCE_FAILED = "intent_persistence_failed"
    DISPATCH_ATTEMPTED = "dispatch_attempted"
    OUTCOME_PERSISTED = "outcome_persisted"
    OUTCOME_PERSISTENCE_FAILED = "outcome_persistence_failed"
    UNKNOWN_ADMITTED = "unknown_admitted"
    CRASH_AMBIGUITY_ADMITTED = "crash_ambiguity_admitted"
    REPEAT_ELIGIBILITY_EVALUATED = "repeat_eligibility_evaluated"
    RECONCILIATION_COMPLETED = "reconciliation_completed"
    RECOVERY_DECIDED = "recovery_decided"
    RECOVERY_EXECUTION_COMPLETED = "recovery_execution_completed"
    REPEAT_ATTEMPT_LINKED = "repeat_attempt_linked"
    HITL_ESCALATED = "hitl_escalated"


class ProviderInvocationReliabilityCorrelation(BaseModel):
    """Stable identifiers linking logical effect to physical attempts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=256)
    provider_id: str = Field(min_length=1, max_length=256)
    operation: str = Field(min_length=1, max_length=256)
    invocation_id: str = Field(min_length=1, max_length=256)
    task_id: str = Field(min_length=1, max_length=256)
    run_id: str = Field(min_length=1, max_length=256)
    execution_id: str | None = Field(default=None, max_length=256)
    attempt_id: str | None = Field(default=None, max_length=256)
    idempotency_key: str | None = Field(default=None, max_length=512)
    effect_contract_id: str | None = Field(default=None, max_length=256)
    governance_execution_id: str | None = Field(default=None, max_length=256)
    erl_correlation_id: str | None = Field(default=None, max_length=256)


class ProviderInvocationReliabilityFact(BaseModel):
    """
    One append-only reliability trace record projected from durable domain facts.

    Events/diagnostics consume this shape; durable store remains authoritative.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_PROVIDER_INVOCATION_RELIABILITY_FACT_V1
    phase: ProviderInvocationReliabilityTracePhase
    recorded_at: datetime
    correlation: ProviderInvocationReliabilityCorrelation
    invocation_status: ProviderInvocationStatus | None = None
    dispatch_state: ProviderInvocationRecoveryDispatchState | None = None
    provider_mutation_attempted: bool | None = None
    provider_mutation_count: int | None = Field(default=None, ge=0, le=256)
    repeat_eligibility_verdict: ExternalEffectRepeatEligibilityVerdict | None = None
    repeat_eligibility_reason: ExternalEffectRepeatEligibilityReason | None = None
    repeat_policy_id: str | None = Field(default=None, max_length=256)
    unknown_posture: UnknownUncertaintyPosture | None = None
    reconciliation_verdict: ProviderInvocationReconciliationVerdict | None = None
    reconciliation_reason: ProviderInvocationReconciliationReason | None = None
    reconciliation_plugin_id: str | None = Field(default=None, max_length=256)
    reconciliation_probe_ref: str | None = Field(default=None, max_length=256)
    evidence_ref: str | None = Field(default=None, max_length=512)
    recovery_action: ProviderInvocationRecoveryAction | None = None
    recovery_reason: ProviderInvocationRecoveryReason | None = None
    recovery_policy_id: str | None = Field(default=None, max_length=256)
    recovery_execution_disposition: str | None = Field(default=None, max_length=64)
    recovery_block_reason: str | None = Field(default=None, max_length=64)
    repeat_invocation_id: str | None = Field(default=None, max_length=256)
    continuation_request_id: str | None = Field(default=None, max_length=256)
    detail: str = Field(default="", max_length=_MAX_REASON)


class ProviderInvocationReliabilityEvidenceObserver(Protocol):
    """Pluggable sink for provider-invocation reliability trace projections."""

    def observe_provider_invocation_reliability_fact(
        self,
        fact: ProviderInvocationReliabilityFact,
    ) -> None:
        """Receive a trace fact; must not mutate execution or durable reliability state."""


class NullProviderInvocationReliabilityEvidenceObserver:
    """Default no-op observer preserving existing runtime behavior."""

    def observe_provider_invocation_reliability_fact(
        self,
        fact: ProviderInvocationReliabilityFact,
    ) -> None:
        return None


__all__ = [
    "NullProviderInvocationReliabilityEvidenceObserver",
    "ProviderInvocationReliabilityCorrelation",
    "ProviderInvocationReliabilityEvidenceObserver",
    "ProviderInvocationReliabilityFact",
    "ProviderInvocationReliabilityTracePhase",
    "SCHEMA_PROVIDER_INVOCATION_RELIABILITY_FACT_V1",
]
