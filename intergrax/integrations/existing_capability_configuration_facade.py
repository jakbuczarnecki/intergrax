# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Governed public entry for existing-capability configuration realization."""

from __future__ import annotations

from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationPort,
    ControlPlaneMutationAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.integrations.contracts.existing_capability_configuration import (
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationPort,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityConfigurationRealizationResult,
    project_control_plane_mutation_request,
    verify_admitted_authorization_evidence,
)
from intergrax.integrations.existing_capability_configuration_service import (
    ExistingCapabilityConfigurationRealizationService,
)


class ExistingCapabilityConfigurationRealizationFacade(
    ExistingCapabilityConfigurationRealizationPort
):
    """Exactly one Governance authorization invocation per public realize call."""

    def __init__(
        self,
        *,
        authorization_port: ControlPlaneMutationAuthorizationPort,
        realization_service: ExistingCapabilityConfigurationRealizationService,
    ) -> None:
        if authorization_port is None:
            raise ValueError("authorization_port is required")
        if realization_service is None:
            raise ValueError("realization_service is required")
        self._authorization_port = authorization_port
        self._realization_service = realization_service

    def realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
    ) -> ExistingCapabilityConfigurationRealizationResult:
        governance_request = project_control_plane_mutation_request(request)
        try:
            authorization_result = self._authorization_port.authorize(
                governance_request
            )
        except Exception as exc:
            raise ExistingCapabilityConfigurationRealizationError(
                ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
                detail="authorization port failure",
            ) from exc
        _require_allow_authorization(authorization_result)
        verify_admitted_authorization_evidence(
            request=request,
            governance_request=governance_request,
            evidence=authorization_result.evidence,
        )
        return self._realization_service.realize_admitted(
            request,
            authorization_evidence=authorization_result.evidence,
        )


def _require_allow_authorization(
    result: ControlPlaneMutationAuthorizationResult,
) -> None:
    if not result.permitted:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
            detail="not permitted",
        )
    if result.decision.action is not PolicyAction.ALLOW:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
            detail=f"decision action {result.decision.action.value}",
        )
