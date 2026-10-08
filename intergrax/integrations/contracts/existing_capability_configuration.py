# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed existing-capability configuration realization contracts (INT-CONFIG-REAL-X-P1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
    control_plane_mutation_request_digest,
)
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.integrations.contracts.base import IntegrationCategory

MUTATION_TYPE_INTEGRATION_CONFIGURATION_REALIZE_V1: Final = (
    "integration_configuration.realize.v1"
)
RESOURCE_TYPE_INTEGRATION_CONFIGURATION: Final = "integration_configuration"


class ExistingCapabilityConfigurationRealizationFailureReason(StrEnum):
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    UNSUPPORTED_CONFIGURATION = "UNSUPPORTED_CONFIGURATION"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    UNSUPPORTED_CONFIGURATION_VERSION = "UNSUPPORTED_CONFIGURATION_VERSION"
    MISSING_AUTHORITY_EVIDENCE = "MISSING_AUTHORITY_EVIDENCE"
    AUTHORIZATION_REJECTED = "AUTHORIZATION_REJECTED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    UNSUPPORTED_STRATEGY = "UNSUPPORTED_STRATEGY"
    STRATEGY_AMBIGUITY = "STRATEGY_AMBIGUITY"
    REALIZATION_FAILED = "REALIZATION_FAILED"


class ExistingCapabilityConfigurationRealizationError(Exception):
    """Typed configuration realization failure — reason is the semantic API."""

    def __init__(
        self,
        reason: ExistingCapabilityConfigurationRealizationFailureReason,
        *,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.detail = detail
        message = reason.value if not detail else f"{reason.value}: {detail}"
        super().__init__(message)


@runtime_checkable
class IntegrationConfigurationPayload(Protocol):
    """Platform-owned configuration identity — provider semantics stay in strategies."""

    @property
    def configuration_type(self) -> str: ...

    @property
    def configuration_version(self) -> str: ...

    @property
    def configuration_fingerprint(self) -> str: ...


@dataclass(frozen=True)
class ExistingCapabilityIntegrationTarget:
    """Typed existing-capability identity resolved for one tenant scope."""

    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    current_revision: str


@dataclass(frozen=True)
class ConfiguredCapabilityBinding:
    """Strategy output — typed configured capability identity/reference."""

    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    configuration_type: str
    configuration_version: str
    configuration_fingerprint: str
    realization_evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExistingCapabilityConfigurationRealizationRequest:
    request_id: str
    tenant_id: str
    principal: RequestIdentity
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    configuration: IntegrationConfigurationPayload
    configuration_fingerprint: str
    current_revision: str
    risk_classification: ControlPlaneMutationRisk
    task_id: TaskId | None = None
    run_id: RunId | None = None
    correlation_ref: str | None = None

    def __post_init__(self) -> None:
        validate_realization_request_invariants(self)


@dataclass(frozen=True)
class ExistingCapabilityConfigurationRealizationResult:
    request_id: str
    configured_binding: ConfiguredCapabilityBinding
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence


@runtime_checkable
class ExistingCapabilityIntegrationResolver(Protocol):
    def resolve_existing_integration(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
    ) -> ExistingCapabilityIntegrationTarget: ...


@runtime_checkable
class ExistingCapabilityConfigurationRealizationStrategy(Protocol):
    @property
    def strategy_id(self) -> str: ...

    def can_realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> bool: ...

    def realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> ConfiguredCapabilityBinding: ...


@runtime_checkable
class ExistingCapabilityConfigurationRealizationPort(Protocol):
    """Public production entry — governed realization only."""

    def realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
    ) -> ExistingCapabilityConfigurationRealizationResult: ...


def derive_existing_capability_configuration_realization_request_id(
    *,
    recovery_decision_id: str,
    configuration_ref: str,
) -> str:
    """Deterministic INT-CONFIG request identity for idempotent CONFIGURE_EXISTING fulfillment."""
    decision_id = recovery_decision_id.strip()
    if not decision_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="missing recovery_decision_id",
        )
    ref = configuration_ref.strip()
    if not ref:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="missing configuration_ref",
        )
    return (
        f"existing-capability-configuration-realization:{decision_id}:{ref}"
    )


def validate_realization_request_invariants(
    request: ExistingCapabilityConfigurationRealizationRequest,
) -> None:
    tenant = request.tenant_id.strip()
    if not tenant:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH,
            detail="missing tenant_id",
        )
    if request.principal.tenant_id != tenant:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH,
            detail="principal tenant mismatch",
        )
    provider = request.provider_id.strip()
    if not provider:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="missing provider_id",
        )
    if not request.configuration.configuration_type.strip():
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.INVALID_CONFIGURATION,
            detail="missing configuration_type",
        )
    if not request.configuration.configuration_version.strip():
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_CONFIGURATION_VERSION,
            detail="missing configuration_version",
        )
    fingerprint = request.configuration_fingerprint.strip()
    if not fingerprint:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.INVALID_CONFIGURATION,
            detail="missing configuration_fingerprint",
        )
    if fingerprint != request.configuration.configuration_fingerprint.strip():
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="configuration fingerprint mismatch",
        )
    if not request.current_revision.strip():
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="missing current_revision",
        )
    if (request.task_id is None) ^ (request.run_id is None):
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.INVALID_CONFIGURATION,
            detail="task_id and run_id must both be set or both absent",
        )


def project_control_plane_mutation_request(
    request: ExistingCapabilityConfigurationRealizationRequest,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=request.request_id,
        mutation_type=MUTATION_TYPE_INTEGRATION_CONFIGURATION_REALIZE_V1,
        principal=request.principal,
        resource_scope=request.resource_scope,
        resource_type=RESOURCE_TYPE_INTEGRATION_CONFIGURATION,
        resource_id=request.provider_id,
        current_revision=request.current_revision,
        target_revision=request.configuration_fingerprint,
        risk_classification=request.risk_classification,
        task_id=request.task_id,
        run_id=request.run_id,
    )


def verify_admitted_authorization_evidence(
    *,
    request: ExistingCapabilityConfigurationRealizationRequest,
    governance_request: ControlPlaneMutationRequest,
    evidence: ControlPlaneMutationAuthorizationEvidence,
) -> None:
    from intergrax.contracts.runtime_policy import PolicyAction

    if evidence.policy_action is not PolicyAction.ALLOW:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
            detail="policy_action not ALLOW",
        )
    if evidence.tenant_id != request.tenant_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH,
            detail="evidence tenant mismatch",
        )
    if evidence.mutation_id != governance_request.mutation_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="mutation_id mismatch",
        )
    if evidence.mutation_type != governance_request.mutation_type:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="mutation_type mismatch",
        )
    if evidence.resource_type != governance_request.resource_type:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="resource_type mismatch",
        )
    if evidence.resource_id != governance_request.resource_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="resource_id mismatch",
        )
    if evidence.resource_scope != governance_request.resource_scope:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="resource_scope mismatch",
        )
    if evidence.current_revision != governance_request.current_revision:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="current_revision mismatch",
        )
    if evidence.target_revision != governance_request.target_revision:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="target_revision mismatch",
        )
    expected_task = (
        str(governance_request.task_id)
        if governance_request.task_id is not None
        else None
    )
    expected_run = (
        str(governance_request.run_id)
        if governance_request.run_id is not None
        else None
    )
    if evidence.task_id != expected_task or evidence.run_id != expected_run:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="task_id/run_id mismatch",
        )
    expected_digest = control_plane_mutation_request_digest(governance_request)
    if evidence.request_digest != expected_digest:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="request_digest mismatch",
        )
