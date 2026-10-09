# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CONFIGURE_EXISTING fulfillment — Opportunity read, INT-CONFIG realize, explicit adoption."""

from __future__ import annotations

from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingRequired,
    WorkerPrincipalBindingResolver,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidateKind,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
)
from intergrax.contracts.autonomous_work.worker_configured_capability_fulfillment import (
    WorkerConfiguredCapabilityFulfillmentFailureReason,
    WorkerConfiguredCapabilityFulfillmentPort,
    WorkerConfiguredCapabilityFulfillmentResult,
)
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationPort,
    ExistingCapabilityConfigurationRealizationRequest,
    derive_existing_capability_configuration_realization_request_id,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ExistingCapabilityConfigurationOpportunityLookupError,
    ExistingCapabilityConfigurationOpportunityLookupFailureReason,
    ExistingCapabilityConfigurationOpportunityReadPort,
    validate_configuration_opportunity_ref,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)


class WorkerConfiguredCapabilityFulfillmentService(WorkerConfiguredCapabilityFulfillmentPort):
    """Project Opportunity facts into governed INT-CONFIG realization and adoption."""

    def __init__(
        self,
        *,
        opportunity_read: ExistingCapabilityConfigurationOpportunityReadPort,
        realization: ExistingCapabilityConfigurationRealizationPort,
        principal_binding_resolver: WorkerPrincipalBindingResolver,
    ) -> None:
        self._opportunity_read = opportunity_read
        self._realization = realization
        self._principal_binding_resolver = principal_binding_resolver

    def fulfill_configure_existing(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        recovery: WorkerCapabilityRecoveryOutcome,
        decision: WorkerCapabilityAcquisitionDecision,
    ) -> WorkerConfiguredCapabilityFulfillmentResult:
        correlation_ref = recovery.provenance.discovery_correlation_id
        validation_failure = self._validate_decision(decision)
        if validation_failure is not None:
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=validation_failure,
                correlation_ref=correlation_ref,
            )
        candidate = decision.selected_candidate
        assert candidate is not None
        configuration_ref_raw = candidate.configuration_ref
        if configuration_ref_raw is None:
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=WorkerConfiguredCapabilityFulfillmentFailureReason.INVALID_DECISION,
                correlation_ref=correlation_ref,
            )
        try:
            configuration_ref = validate_configuration_opportunity_ref(
                configuration_ref_raw,
            )
        except (TypeError, ValueError):
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=WorkerConfiguredCapabilityFulfillmentFailureReason.INVALID_DECISION,
                correlation_ref=correlation_ref,
            )
        try:
            opportunity = self._opportunity_read.read_exact(
                tenant_id=request.tenant_id,
                configuration_ref=configuration_ref,
            )
        except ExistingCapabilityConfigurationOpportunityLookupError as exc:
            reason = WorkerConfiguredCapabilityFulfillmentFailureReason.OPPORTUNITY_NOT_FOUND
            if exc.reason is ExistingCapabilityConfigurationOpportunityLookupFailureReason.TENANT_MISMATCH:
                reason = (
                    WorkerConfiguredCapabilityFulfillmentFailureReason.OPPORTUNITY_TENANT_MISMATCH
                )
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=reason,
                correlation_ref=correlation_ref,
            )
        if opportunity.tenant_id != request.tenant_id:
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=WorkerConfiguredCapabilityFulfillmentFailureReason.OPPORTUNITY_TENANT_MISMATCH,
                correlation_ref=correlation_ref,
            )
        try:
            principal = self._resolve_principal(request)
        except WorkerPrincipalBindingRequired:
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=WorkerConfiguredCapabilityFulfillmentFailureReason.PRINCIPAL_MISSING,
                correlation_ref=correlation_ref,
            )
        if principal.tenant_id != request.tenant_id:
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=WorkerConfiguredCapabilityFulfillmentFailureReason.PRINCIPAL_TENANT_MISMATCH,
                correlation_ref=correlation_ref,
            )
        need = request.acquisition_request.need
        task_id = request.task_id if request.run_id is not None else None
        run_id = request.run_id if request.run_id is not None else None
        realization_request = ExistingCapabilityConfigurationRealizationRequest(
            request_id=derive_existing_capability_configuration_realization_request_id(
                recovery_decision_id=need.recovery_decision_id,
                configuration_ref=str(configuration_ref),
            ),
            tenant_id=request.tenant_id,
            principal=principal,
            integration_category=opportunity.integration_category,
            provider_id=opportunity.provider_id,
            resource_scope=opportunity.resource_scope,
            configuration=opportunity.configuration,
            configuration_fingerprint=opportunity.configuration_fingerprint,
            current_revision=opportunity.current_revision,
            risk_classification=opportunity.risk_classification,
            task_id=task_id,
            run_id=run_id,
            correlation_ref=correlation_ref,
        )
        try:
            realization_result = self._realization.realize(realization_request)
        except ExistingCapabilityConfigurationRealizationError as exc:
            failure = WorkerConfiguredCapabilityFulfillmentFailureReason.REALIZATION_FAILED
            if exc.reason in {
                ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
                ExistingCapabilityConfigurationRealizationFailureReason.MISSING_AUTHORITY_EVIDENCE,
            }:
                failure = WorkerConfiguredCapabilityFulfillmentFailureReason.REALIZATION_DENIED
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=failure,
                correlation_ref=correlation_ref,
            )
        binding = realization_result.configured_binding
        binding_failure = self._validate_binding_continuity(
            binding=binding,
            opportunity_tenant_id=opportunity.tenant_id,
            opportunity_category=opportunity.integration_category,
            opportunity_provider_id=opportunity.provider_id,
            opportunity_scope=opportunity.resource_scope,
            opportunity_fingerprint=opportunity.configuration_fingerprint,
        )
        if binding_failure is not None:
            return WorkerConfiguredCapabilityFulfillmentResult(
                adoption=None,
                failure_reason=binding_failure,
                correlation_ref=correlation_ref,
            )
        adoption = ExecutionIntegrationConfigurationAdoption(
            configured_binding=binding,
            integration_category=opportunity.integration_category,
            resource_scope=opportunity.resource_scope,
        )
        return WorkerConfiguredCapabilityFulfillmentResult(
            adoption=adoption,
            correlation_ref=correlation_ref,
        )

    def _resolve_principal(self, request: WorkerCapabilityFulfillmentRequest) -> RequestIdentity:
        resolved = self._principal_binding_resolver.resolve(
            worker_instance_id=request.worker_instance_id,
        )
        return RequestIdentity(
            tenant_id=resolved.tenant_id,
            user_id=resolved.principal_id,
            principal_type=PrincipalType.USER,
            auth_subject=resolved.principal_id,
        )

    @staticmethod
    def _validate_decision(
        decision: WorkerCapabilityAcquisitionDecision,
    ) -> WorkerConfiguredCapabilityFulfillmentFailureReason | None:
        if decision.disposition is not CapabilityAcquisitionDisposition.CONFIGURE_EXISTING:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.INVALID_DECISION
        candidate = decision.selected_candidate
        if candidate is None:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.INVALID_DECISION
        if candidate.candidate_kind is not WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.INVALID_DECISION
        if candidate.configuration_ref is None:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.INVALID_DECISION
        return None

    @staticmethod
    def _validate_binding_continuity(
        *,
        binding: ConfiguredCapabilityBinding,
        opportunity_tenant_id: str,
        opportunity_category,
        opportunity_provider_id: str,
        opportunity_scope: str,
        opportunity_fingerprint: str,
    ) -> WorkerConfiguredCapabilityFulfillmentFailureReason | None:
        if binding.tenant_id != opportunity_tenant_id:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.BINDING_IDENTITY_MISMATCH
        if binding.integration_category != opportunity_category:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.BINDING_IDENTITY_MISMATCH
        if binding.provider_id != opportunity_provider_id:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.BINDING_IDENTITY_MISMATCH
        if binding.resource_scope != opportunity_scope:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.BINDING_IDENTITY_MISMATCH
        if binding.configuration_fingerprint != opportunity_fingerprint:
            return WorkerConfiguredCapabilityFulfillmentFailureReason.BINDING_IDENTITY_MISMATCH
        return None


__all__ = ["WorkerConfiguredCapabilityFulfillmentService"]
