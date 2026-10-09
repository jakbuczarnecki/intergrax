# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Thin CONFIGURE_EXISTING durable intent preparation — canonical MarketplaceToolExecutionIntent v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_qualification.configured_capability_execution_subject import (
    ConfiguredCapabilityExecutionSubject,
    derive_configured_capability_binding_operation_id,
    derive_configured_capability_execution_operation_id,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.tools.marketplace_tool_execution_intent import (
    ConfiguredMarketplaceToolExecutionProvenance,
    MarketplaceToolExecutionIntent,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    MarketplaceToolExecutionIntentRepository,
    QualifiedMarketplaceToolExecutionIntentConflictError,
    QualifiedMarketplaceToolExecutionIntentIntegrityError,
    QualifiedMarketplaceToolExecutionIntentUnavailableError,
    QualifiedMarketplaceToolExecutionIntentWriteOutcome,
)
from intergrax.contracts.tools.qualified_marketplace_tool_operation_selection import (
    QualifiedMarketplaceToolOperationSelectionOutcome,
)
from intergrax.tools.marketplace_tool_execution_routing import (
    derive_marketplace_configured_tool_execution_intent_target_correlation,
)
from intergrax.tools.marketplace_tool_operation_selection_core import (
    select_marketplace_tool_operation,
)


class MarketplaceConfiguredToolExecutionIntentPreparationOutcome(StrEnum):
    CREATED = "created"
    ALREADY_RECORDED_IDENTICAL = "already_recorded_identical"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"
    INTEGRITY_FAILURE = "integrity_failure"
    INVALID_OPERATION = "invalid_operation"


@dataclass(frozen=True, slots=True)
class MarketplaceConfiguredToolExecutionIntentPreparationRequest:
    subject: ConfiguredCapabilityExecutionSubject
    execution_request_id: str
    binding_operation_id: str
    tenant_id: str
    task_id: TaskId
    worker_need_id: str


@dataclass(frozen=True, slots=True)
class MarketplaceConfiguredToolExecutionIntentPreparationResult:
    outcome: MarketplaceConfiguredToolExecutionIntentPreparationOutcome
    reason_detail: str = ""


class MarketplaceConfiguredToolExecutionIntentPreparation:
    """Record configured provenance intent — same repository as UCA."""

    def __init__(
        self,
        *,
        intent_repository: MarketplaceToolExecutionIntentRepository,
    ) -> None:
        self._intent_repository = intent_repository

    def prepare(
        self,
        request: MarketplaceConfiguredToolExecutionIntentPreparationRequest,
    ) -> MarketplaceConfiguredToolExecutionIntentPreparationResult:
        subject = request.subject
        identity_failure = _validate_configured_capability_identity(subject.capability_identity)
        if identity_failure is not None:
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail=identity_failure,
            )
        if subject.tenant_id != request.tenant_id:
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail="tenant_id mismatch",
            )

        selection = select_marketplace_tool_operation(
            required_operations=subject.selected_operations,
        )
        if (
            selection.outcome
            is not QualifiedMarketplaceToolOperationSelectionOutcome.SELECTED
        ):
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.INVALID_OPERATION,
                reason_detail=selection.reason_detail or "invalid operation",
            )

        configured_execution_operation_id = derive_configured_capability_execution_operation_id(
            recovery_decision_id=subject.recovery_decision_id,
            decision_id=subject.decision_id,
        )
        configured_binding_operation_id = derive_configured_capability_binding_operation_id(
            configured_execution_operation_id=configured_execution_operation_id,
            subject_reference=subject.subject_reference,
        )
        if request.binding_operation_id != configured_binding_operation_id:
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail="binding_operation_id mismatch",
            )

        intent = MarketplaceToolExecutionIntent(
            execution_request_id=request.execution_request_id,
            binding_operation_id=request.binding_operation_id,
            tenant_id=request.tenant_id,
            task_id=str(request.task_id),
            worker_need_id=request.worker_need_id,
            subject_reference=subject.subject_reference,
            capability_identity=subject.capability_identity,
            selected_operation=selection.selected_operation,
            execution_target_correlation=derive_marketplace_configured_tool_execution_intent_target_correlation(
                request.binding_operation_id,
            ),
            provenance=ConfiguredMarketplaceToolExecutionProvenance(
                recovery_decision_id=subject.recovery_decision_id,
                acquisition_decision_id=subject.decision_id,
                configured_binding_operation_id=configured_binding_operation_id,
                configured_execution_operation_id=configured_execution_operation_id,
                configuration_adoption_identity=subject.configuration_adoption_identity,
            ),
        )

        try:
            write_result = self._intent_repository.record(intent)
        except QualifiedMarketplaceToolExecutionIntentUnavailableError as exc:
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.UNAVAILABLE,
                reason_detail=str(exc),
            )
        except QualifiedMarketplaceToolExecutionIntentConflictError as exc:
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.CONFLICT,
                reason_detail=str(exc),
            )
        except QualifiedMarketplaceToolExecutionIntentIntegrityError as exc:
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail=str(exc),
            )

        if write_result.outcome is QualifiedMarketplaceToolExecutionIntentWriteOutcome.CREATED:
            return MarketplaceConfiguredToolExecutionIntentPreparationResult(
                outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.CREATED,
            )
        return MarketplaceConfiguredToolExecutionIntentPreparationResult(
            outcome=MarketplaceConfiguredToolExecutionIntentPreparationOutcome.ALREADY_RECORDED_IDENTICAL,
        )


def _validate_configured_capability_identity(
    identity: CapabilityIdentityKey,
) -> str | None:
    if type(identity) is not CapabilityIdentityKey:
        return "capability_identity must be CapabilityIdentityKey"
    if identity.kind is not CapabilityKind.TOOL:
        return "configured marketplace intent requires TOOL capability_identity"
    return None


__all__ = [
    "MarketplaceConfiguredToolExecutionIntentPreparation",
    "MarketplaceConfiguredToolExecutionIntentPreparationOutcome",
    "MarketplaceConfiguredToolExecutionIntentPreparationRequest",
    "MarketplaceConfiguredToolExecutionIntentPreparationResult",
]
