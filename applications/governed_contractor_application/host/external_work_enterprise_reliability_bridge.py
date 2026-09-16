# © Artur Czarnecki. All rights reserved.

"""External Work observation → ERL admission (GR-7-A2 host composition)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.external_effect_outcome_projection import (
    ExternalWorkSideEffectObservation,
    project_external_work_side_effect_to_effect_outcome,
)
from intergrax.contracts.enterprise_reliability.admission_boundary import (
    ExternalEffectAdmissionCaseError,
    ExternalEffectAdmissionContextError,
    ExternalEffectAdmissionRequest,
    ExternalEffectAdmissionResult,
    ExternalEffectAdmissionSourceContext,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.runtime.enterprise_reliability.admission_boundary import (
    admit_external_effect_into_enterprise_reliability,
)


class ExternalEffectEnterpriseReliabilityAdmissionPort(Protocol):
    """Injectable ERL admission — no provider calls."""

    def admit_external_effect(
        self,
        request: ExternalEffectAdmissionRequest,
    ) -> ExternalEffectAdmissionResult: ...


class _DefaultEnterpriseReliabilityAdmissionPort:
    def admit_external_effect(
        self,
        request: ExternalEffectAdmissionRequest,
    ) -> ExternalEffectAdmissionResult:
        return admit_external_effect_into_enterprise_reliability(request)


@dataclass(frozen=True, slots=True)
class ExternalWorkReliabilityAdmissionOutcome:
    """Bridge result — provider observation truth is independent of admission failures."""

    effect_outcome: ExternalEffectOutcome
    admission: ExternalEffectAdmissionResult | None = None
    admission_error: str | None = None


@dataclass(frozen=True, slots=True)
class GovernedExternalWorkEnterpriseReliabilityBridge:
    """Single canonical path: observation → outcome → existing ERL admission."""

    admission_port: ExternalEffectEnterpriseReliabilityAdmissionPort

    @classmethod
    def production(cls) -> GovernedExternalWorkEnterpriseReliabilityBridge:
        return cls(admission_port=_DefaultEnterpriseReliabilityAdmissionPort())

    def admit_side_effect_observation(
        self,
        *,
        observation: ExternalWorkSideEffectObservation,
        invocation: ProviderInvocation,
        execution_id: str,
        action: str,
        capabilities: ExternalWorkProviderCapabilities,
    ) -> ExternalWorkReliabilityAdmissionOutcome | None:
        effect_outcome = project_external_work_side_effect_to_effect_outcome(observation)
        if effect_outcome is None:
            return None
        contract = external_work_effect_contract_for_action(action, capabilities)
        correlation_id = (
            invocation.correlation_id
            or invocation.idempotency_key
            or invocation.task_id
        )
        external_effect_ref = (
            f"ext:external_work:{invocation.idempotency_key or invocation.invocation_id}"
        )
        request = ExternalEffectAdmissionRequest(
            external_effect_ref=external_effect_ref,
            effect_outcome=effect_outcome,
            correlation_id=correlation_id,
            contract=contract,
            source_context=ExternalEffectAdmissionSourceContext(
                source_kind="governed_external_work_host",
                source_ref=execution_id,
            ),
            reason=observation.reason or observation.adapter_result.reason,
        )
        try:
            admission = self.admission_port.admit_external_effect(request)
        except (
            ExternalEffectAdmissionContextError,
            ExternalEffectAdmissionCaseError,
            ValueError,
        ) as exc:
            return ExternalWorkReliabilityAdmissionOutcome(
                effect_outcome=effect_outcome,
                admission=None,
                admission_error=str(exc),
            )
        return ExternalWorkReliabilityAdmissionOutcome(
            effect_outcome=effect_outcome,
            admission=admission,
        )


__all__ = [
    "ExternalEffectEnterpriseReliabilityAdmissionPort",
    "ExternalWorkReliabilityAdmissionOutcome",
    "GovernedExternalWorkEnterpriseReliabilityBridge",
]
