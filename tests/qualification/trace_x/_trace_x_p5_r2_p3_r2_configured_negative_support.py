# © Artur Czarnecki. All rights reserved.

"""Shared harness for TRACE-X-P5-R2-P3-R2 configured-path negative E2E proofs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityProfileRef,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    derive_worker_capability_candidate_id,
)
from intergrax.contracts.autonomous_work.ids import mint_worker_instance_id
from intergrax.contracts.autonomous_work.profile_reference import initial_profile_version
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_qualification.configured_capability_execution_subject import (
    ConfiguredCapabilityExecutionSubject,
    derive_configuration_adoption_identity,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.contracts.tools.marketplace_tool_execution_intent import (
    ConfiguredMarketplaceToolExecutionProvenance,
    MarketplaceToolExecutionIntent,
    MarketplaceToolExecutionProvenanceKind,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationResult,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
    ExistingCapabilityConfigurationOpportunity,
    ExistingCapabilityConfigurationOpportunityLookupError,
    ExistingCapabilityConfigurationOpportunityLookupFailureReason,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLiteRelationalStoreConfigurationPayload,
)
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
    execution_target_reference_for_marketplace_qualified_tool,
)
from intergrax.tools.marketplace_tool_execution_routing import (
    MARKETPLACE_TOOL_CONFIGURED_CAPABILITY_BINDING_PROVIDER_ID,
    MARKETPLACE_TOOL_EXECUTION_HANDLER_ID,
    build_marketplace_tool_execution_target,
    derive_marketplace_configured_tool_execution_intent_target_correlation,
    derive_marketplace_configured_tool_execution_target_reference,
)

TRACE_X_P5_R2_P3_R2_START_HEAD = "215f82f855bd6ed5316c1c23f8d35ff076be3526"

_TENANT = "tenant-a"
_CONFIG_REF = ConfigurationOpportunityRef("cfg/opportunity-1")
_TASK_ID = TaskId("task_00000000000000000000000000000001")
_RUN_ID = RunId("run_" + "a" * 32)
_ATTEMPT_ID = AttemptId("attempt_" + "b" * 28)
_NOW = datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ConfiguredNegativeExpectation:
    case_id: str
    detection_layer: str
    disposition_or_reason: str
    execution_exists: bool
    activation_performed: bool
    provider_materialized: bool
    provider_business_call_count: int


def capability_identity() -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tools.database.relational",
    )


def sqlite_payload() -> SQLiteRelationalStoreConfigurationPayload:
    return SQLiteRelationalStoreConfigurationPayload(
        data_dir=Path("/data/tenant/store"),
        relational_db=Path("/data/tenant/store/custom.db"),
    )


def configured_binding(tenant: str = _TENANT) -> ConfiguredCapabilityBinding:
    payload = sqlite_payload()
    return ConfiguredCapabilityBinding(
        tenant_id=tenant,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        configuration_type=payload.configuration_type,
        configuration_version=payload.configuration_version,
        configuration_fingerprint=payload.configuration_fingerprint,
        realization_evidence_refs=("evidence-1",),
    )


def adoption(tenant: str = _TENANT) -> ExecutionIntegrationConfigurationAdoption:
    binding = configured_binding(tenant)
    return ExecutionIntegrationConfigurationAdoption(
        configured_binding=binding,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        resource_scope="scope-a",
    )


def acquisition_decision(
    *,
    tenant_identity: CapabilityIdentityKey | object | None = None,
    configuration_ref: str | None = str(_CONFIG_REF),
) -> WorkerCapabilityAcquisitionDecision:
    worker = mint_worker_instance_id()
    identity = capability_identity() if tenant_identity is None else tenant_identity
    return WorkerCapabilityAcquisitionDecision(
        decision_id="decision-1",
        worker_instance_id=worker,
        obstacle_id=f"{worker}:obs-1",
        recovery_decision_id="recovery-1",
        need_id="need-1",
        capability_profile_ref=CapabilityProfileRef(
            profile_id="profile/default",
            version=initial_profile_version(),
        ),
        disposition=CapabilityAcquisitionDisposition.CONFIGURE_EXISTING,
        reason_code=CapabilityAcquisitionReasonCode.EXISTING_CONFIGURATION_SELECTED,
        selected_candidate=WorkerCapabilityCandidate(
            candidate_id=derive_worker_capability_candidate_id(
                candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
                capability_ref="integration:sqlite",
                configuration_ref=configuration_ref or str(_CONFIG_REF),
            ),
            candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
            capability_ref="integration:sqlite",
            source_domain="integrations",
            operations=("database.query",),
            risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
            evidence_refs=(),
            discovered_at=_NOW,
            configuration_ref=configuration_ref,
            capability_identity=identity,  # type: ignore[arg-type]
        ),
        autonomy_level=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        decided_at=_NOW,
        decision_policy_version="v1",
        evidence_refs=(),
    )


def recovery_outcome(
    *,
    correlation_id: str = "corr-1",
) -> WorkerCapabilityRecoveryOutcome:
    return WorkerCapabilityRecoveryOutcome(
        phase=WorkerCapabilityRecoveryPhase.CONFIGURE_EXISTING_REQUIRED,
        provenance=WorkerCapabilityRecoveryProvenance(
            worker_need_id="need-1",
            canonical_need_id="canonical-1",
            discovery_correlation_id=correlation_id,
            discovery_completion_outcome="CONFIGURE_EXISTING",
        ),
    )


def opportunity(tenant: str = _TENANT) -> ExistingCapabilityConfigurationOpportunity:
    payload = sqlite_payload()
    return ExistingCapabilityConfigurationOpportunity(
        configuration_ref=_CONFIG_REF,
        tenant_id=tenant,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        current_revision="rev-1",
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        risk_classification=ControlPlaneMutationRisk.LOW,
    )


def configured_subject(**overrides: Any) -> ConfiguredCapabilityExecutionSubject:
    fingerprint = sqlite_payload().configuration_fingerprint
    recovery = "recovery-1"
    decision = "decision-1"
    base = {
        "tenant_id": _TENANT,
        "worker_need_id": "need-1",
        "recovery_decision_id": recovery,
        "decision_id": decision,
        "capability_identity": capability_identity(),
        "configuration_adoption_identity": derive_configuration_adoption_identity(
            recovery_decision_id=recovery,
            decision_id=decision,
            configuration_fingerprint=fingerprint,
        ),
        "configuration_fingerprint": fingerprint,
        "selected_operations": ("database.query",),
    }
    base.update(overrides)
    return ConfiguredCapabilityExecutionSubject(**base)


def configured_intent(
    *,
    binding_operation_id: str = "bind-configured-1",
    provenance_kind: MarketplaceToolExecutionProvenanceKind = (
        MarketplaceToolExecutionProvenanceKind.CONFIGURED
    ),
    subject_reference: str | None = None,
) -> MarketplaceToolExecutionIntent:
    subject = configured_subject()
    subject_ref = subject_reference or subject.subject_reference
    if provenance_kind is MarketplaceToolExecutionProvenanceKind.CONFIGURED:
        provenance = ConfiguredMarketplaceToolExecutionProvenance(
            recovery_decision_id=subject.recovery_decision_id,
            acquisition_decision_id=subject.decision_id,
            configured_binding_operation_id=binding_operation_id,
            configured_execution_operation_id="exec-op-configured-1",
            configuration_adoption_identity=subject.configuration_adoption_identity,
        )
    else:
        from intergrax.contracts.tools.marketplace_tool_execution_intent import (
            UcaMarketplaceToolExecutionProvenance,
        )

        provenance = UcaMarketplaceToolExecutionProvenance(
            handoff_id="handoff-uca",
            resume_operation_id="resume-uca",
            uca_qualified_subject_reference=subject_ref,
        )
    return MarketplaceToolExecutionIntent(
        execution_request_id="exec-req-configured-1",
        binding_operation_id=binding_operation_id,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        worker_need_id="need-1",
        subject_reference=subject_ref,
        capability_identity=subject.capability_identity,
        selected_operation="database.query",
        execution_target_correlation=derive_marketplace_configured_tool_execution_intent_target_correlation(
            binding_operation_id,
        ),
        provenance=provenance,
    )


def configured_target(
    *,
    binding_operation_id: str = "bind-configured-1",
    binding_provider_id: str = MARKETPLACE_TOOL_CONFIGURED_CAPABILITY_BINDING_PROVIDER_ID,
    execution_handler_id: str = MARKETPLACE_TOOL_EXECUTION_HANDLER_ID,
    qualified_subject_reference: str | None = None,
) -> QualifiedCapabilityExecutionTarget:
    subject = configured_subject()
    target = build_marketplace_tool_execution_target(
        execution_target_reference=derive_marketplace_configured_tool_execution_target_reference(
            binding_operation_id,
        ),
        binding_provider_id=binding_provider_id,
        qualified_subject_reference=qualified_subject_reference or subject.subject_reference,
    )
    if execution_handler_id != MARKETPLACE_TOOL_EXECUTION_HANDLER_ID:
        target = target.model_copy(update={"execution_handler_id": execution_handler_id})
    return target


def uca_target_on_configured_path() -> QualifiedCapabilityExecutionTarget:
    return build_marketplace_tool_execution_target(
        execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
            "handoff-1",
        ),
        binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
        qualified_subject_reference=configured_subject().subject_reference,
    )


def bound_dispatch(
    target: QualifiedCapabilityExecutionTarget,
    *,
    execution_request_id: str = "exec-req-configured-1",
    tenant_id: str = _TENANT,
) -> BoundCapabilityExecutionDispatchRequest:
    return BoundCapabilityExecutionDispatchRequest(
        execution_request_id=execution_request_id,
        execution_target=target,
        tenant_id=tenant_id,
        task_id=_TASK_ID,
    )


def realization_error_correlation() -> ExistingCapabilityConfigurationRealizationError:
    return ExistingCapabilityConfigurationRealizationError(
        ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
        detail="correlation_mismatch",
    )


def opportunity_tenant_lookup_error() -> ExistingCapabilityConfigurationOpportunityLookupError:
    return ExistingCapabilityConfigurationOpportunityLookupError(
        ExistingCapabilityConfigurationOpportunityLookupFailureReason.TENANT_MISMATCH,
    )


def realization_success() -> ExistingCapabilityConfigurationRealizationResult:
    return ExistingCapabilityConfigurationRealizationResult(
        request_id="req-1",
        configured_binding=configured_binding(),
        authorization_evidence=MagicMock(),
    )


__all__ = [
    "ConfiguredNegativeExpectation",
    "TRACE_X_P5_R2_P3_R2_START_HEAD",
    "_ATTEMPT_ID",
    "_RUN_ID",
    "_TENANT",
    "_TASK_ID",
    "acquisition_decision",
    "adoption",
    "bound_dispatch",
    "configured_intent",
    "configured_subject",
    "configured_target",
    "opportunity",
    "opportunity_tenant_lookup_error",
    "realization_error_correlation",
    "realization_success",
    "recovery_outcome",
    "uca_target_on_configured_path",
]
