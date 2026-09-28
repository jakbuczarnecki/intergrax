# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure existing-capability configuration realization core (INT-CONFIG-REAL-X-P1)."""

from __future__ import annotations

from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
)
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityConfigurationRealizationResult,
    ExistingCapabilityConfigurationRealizationStrategy,
    ExistingCapabilityIntegrationResolver,
    validate_realization_request_invariants,
)


class ExistingCapabilityConfigurationRealizationService:
    """Internal pure core — no Governance invocation."""

    def __init__(
        self,
        *,
        strategies: tuple[ExistingCapabilityConfigurationRealizationStrategy, ...],
        existing_integration_resolver: ExistingCapabilityIntegrationResolver,
    ) -> None:
        self._strategies = strategies
        self._existing_integration_resolver = existing_integration_resolver

    def realize_admitted(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        *,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence,
    ) -> ExistingCapabilityConfigurationRealizationResult:
        validate_realization_request_invariants(request)
        try:
            existing_target = (
                self._existing_integration_resolver.resolve_existing_integration(
                    request
                )
            )
        except ExistingCapabilityConfigurationRealizationError:
            raise
        except Exception as exc:
            raise ExistingCapabilityConfigurationRealizationError(
                ExistingCapabilityConfigurationRealizationFailureReason.TARGET_NOT_FOUND,
                detail=str(exc),
            ) from exc

        _verify_existing_target_continuity(request, existing_target)
        binding = _select_and_realize(request, existing_target, self._strategies)
        _verify_strategy_output_continuity(request, binding)
        return ExistingCapabilityConfigurationRealizationResult(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            integration_category=request.integration_category,
            provider_id=request.provider_id,
            configuration_type=binding.configuration_type,
            configuration_version=binding.configuration_version,
            configuration_fingerprint=binding.configuration_fingerprint,
            configured_binding=binding.configured_binding,
            authorization_evidence=authorization_evidence,
            realization_evidence_refs=binding.realization_evidence_refs,
        )


def _verify_existing_target_continuity(
    request: ExistingCapabilityConfigurationRealizationRequest,
    existing_target: object,
) -> None:
    from intergrax.integrations.contracts.existing_capability_configuration import (
        ExistingCapabilityIntegrationTarget,
    )

    if not isinstance(existing_target, ExistingCapabilityIntegrationTarget):
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.TARGET_NOT_FOUND,
            detail="resolver returned invalid target type",
        )
    if existing_target.tenant_id != request.tenant_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH,
            detail="target tenant mismatch",
        )
    if existing_target.provider_id != request.provider_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="target provider mismatch",
        )
    if existing_target.integration_category != request.integration_category:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="target category mismatch",
        )
    if existing_target.current_revision != request.current_revision:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="target current_revision mismatch",
        )


def _select_and_realize(
    request: ExistingCapabilityConfigurationRealizationRequest,
    existing_target: object,
    strategies: tuple[ExistingCapabilityConfigurationRealizationStrategy, ...],
) -> ConfiguredCapabilityBinding:
    from intergrax.integrations.contracts.existing_capability_configuration import (
        ExistingCapabilityIntegrationTarget,
    )

    target = existing_target
    assert isinstance(target, ExistingCapabilityIntegrationTarget)
    matching = [
        strategy for strategy in strategies if strategy.can_realize(request, target)
    ]
    if not matching:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_STRATEGY,
        )
    if len(matching) > 1:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.STRATEGY_AMBIGUITY,
        )
    try:
        return matching[0].realize(request, target)
    except ExistingCapabilityConfigurationRealizationError:
        raise
    except Exception as exc:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.REALIZATION_FAILED,
            detail=str(exc),
        ) from exc


def _verify_strategy_output_continuity(
    request: ExistingCapabilityConfigurationRealizationRequest,
    binding: ConfiguredCapabilityBinding,
) -> None:
    if binding.tenant_id != request.tenant_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH,
            detail="strategy output tenant mismatch",
        )
    if binding.provider_id != request.provider_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="strategy output provider mismatch",
        )
    if binding.configuration_fingerprint != request.configuration_fingerprint:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="strategy output fingerprint mismatch",
        )
    if binding.configuration_type != request.configuration.configuration_type:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_CONFIGURATION,
            detail="strategy output configuration_type mismatch",
        )
    if binding.configuration_version != request.configuration.configuration_version:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_CONFIGURATION_VERSION,
            detail="strategy output configuration_version mismatch",
        )
