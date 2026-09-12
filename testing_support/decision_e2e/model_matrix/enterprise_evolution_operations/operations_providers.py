# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution operations provider plugins (DS-E2E-15J-L14)."""

from __future__ import annotations

from dataclasses import dataclass, field

from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    AdaptationOperationalReference,
    EvolutionOperationRequest,
    EvolutionOperationStatus,
    EvolutionOperationType,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.protocol import (
    EvolutionOperationOutcome,
)

_DEFAULT_PROVIDER_ID = "default_enterprise_evolution_operations"
_DEFAULT_PROVIDER_VERSION = "1"

_STATUS_BY_OPERATION: dict[EvolutionOperationType, EvolutionOperationStatus] = {
    EvolutionOperationType.OBSERVE: EvolutionOperationStatus.ACTIVE,
    EvolutionOperationType.PAUSE: EvolutionOperationStatus.PAUSED,
    EvolutionOperationType.RESUME: EvolutionOperationStatus.ACTIVE,
    EvolutionOperationType.MARK_REVIEW_REQUIRED: EvolutionOperationStatus.REVIEW_REQUIRED,
    EvolutionOperationType.DISABLE: EvolutionOperationStatus.DISABLED,
}


@dataclass
class DefaultEnterpriseEvolutionOperationsProvider:
    """Default operational plugin — tracks status transitions, no adaptation creation."""

    _status_by_adaptation: dict[str, EvolutionOperationStatus] = field(
        default_factory=dict
    )

    @property
    def provider_id(self) -> str:
        return _DEFAULT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_PROVIDER_VERSION

    def _status_key(self, reference: AdaptationOperationalReference) -> str:
        return f"{reference.adaptation_id}:{reference.version}"

    def current_status(
        self, adaptation_reference: AdaptationOperationalReference
    ) -> EvolutionOperationStatus:
        key = self._status_key(adaptation_reference)
        return self._status_by_adaptation.get(key, EvolutionOperationStatus.ACTIVE)

    def operate(self, request: EvolutionOperationRequest) -> EvolutionOperationOutcome:
        reference = request.adaptation_reference
        key = self._status_key(reference)
        if request.operation_type is EvolutionOperationType.OBSERVE:
            status = self.current_status(reference)
            return EvolutionOperationOutcome(
                operational_status=status,
                summary=(
                    f"Observed adaptation {reference.adaptation_id} "
                    f"at operational status {status.value}."
                ),
            )
        new_status = _STATUS_BY_OPERATION[request.operation_type]
        self._status_by_adaptation[key] = new_status
        return EvolutionOperationOutcome(
            operational_status=new_status,
            summary=(
                f"Applied operational {request.operation_type.value} "
                f"for adaptation {reference.adaptation_id}."
            ),
        )


__all__ = [
    "DefaultEnterpriseEvolutionOperationsProvider",
]
