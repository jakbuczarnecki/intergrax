# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace Tool pre-EE durable execution intent preparation (S24-GAP-02-P3)."""

from __future__ import annotations

from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubjectKind,
    qualified_capability_subject_from_result,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageRepository,
    MarketplaceQualifiedToolStageUnavailableError,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContextNotFoundError,
    MarketplaceQualifiedToolStageContextResolver,
    MarketplaceQualifiedToolStageContextResolverConflictError,
    MarketplaceQualifiedToolStageContextResolverIntegrityError,
    MarketplaceQualifiedToolStageContextResolverNotSupportedError,
    MarketplaceQualifiedToolStageContextResolverUnavailableError,
)
from intergrax.contracts.tools.qualified_capability_execution_intent_preparation import (
    QualifiedCapabilityExecutionIntentPreparationOutcome,
    QualifiedCapabilityExecutionIntentPreparationRequest,
    QualifiedCapabilityExecutionIntentPreparationResult,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntent,
    QualifiedMarketplaceToolExecutionIntentConflictError,
    QualifiedMarketplaceToolExecutionIntentIntegrityError,
    QualifiedMarketplaceToolExecutionIntentRepository,
    QualifiedMarketplaceToolExecutionIntentUnavailableError,
    QualifiedMarketplaceToolExecutionIntentWriteOutcome,
)
from intergrax.marketplace.acquisition import MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID
from intergrax.tools.qualified_marketplace_tool_operation_selector import (
    DefaultQualifiedMarketplaceToolOperationSelector,
    QualifiedMarketplaceToolOperationSelectionOutcome,
)


def _supports_marketplace_tool_qualification(
    request: QualifiedCapabilityExecutionIntentPreparationRequest,
) -> bool:
    qualification = request.qualification_result
    if qualification.outcome is not CapabilityQualificationOutcome.QUALIFIED:
        return False
    if qualification.strategy_id != MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID:
        return False
    subject = qualified_capability_subject_from_result(qualification)
    if subject is None:
        return False
    if subject.subject_kind is not QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE:
        return False
    evidence = qualification.evidence
    if evidence is None or not evidence.domain_handoff_reference:
        return False
    return evidence.domain_handoff_reference == subject.subject_reference


class MarketplaceQualifiedToolExecutionIntentPreparation:
    """Record durable intent before resume — no activation, EE, or ToolRuntime."""

    def __init__(
        self,
        *,
        intent_repository: QualifiedMarketplaceToolExecutionIntentRepository,
        stage_repository: MarketplaceQualifiedToolStageRepository,
        context_resolver: MarketplaceQualifiedToolStageContextResolver,
        operation_selector: DefaultQualifiedMarketplaceToolOperationSelector | None = None,
    ) -> None:
        self._intent_repository = intent_repository
        self._stage_repository = stage_repository
        self._context_resolver = context_resolver
        self._operation_selector = (
            operation_selector or DefaultQualifiedMarketplaceToolOperationSelector()
        )

    def prepare(
        self,
        request: QualifiedCapabilityExecutionIntentPreparationRequest,
    ) -> QualifiedCapabilityExecutionIntentPreparationResult:
        if not _supports_marketplace_tool_qualification(request):
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.NOT_APPLICABLE,
            )

        qualification = request.qualification_result
        subject = qualified_capability_subject_from_result(qualification)
        assert subject is not None
        subject_ref = subject.subject_reference

        try:
            ctx = self._context_resolver.resolve_for_qualification(
                acquisition_request_id=qualification.acquisition_request_id,
                domain_handoff_reference=subject_ref,
                strategy_id=qualification.strategy_id,
            )
        except MarketplaceQualifiedToolStageContextResolverNotSupportedError:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.NOT_APPLICABLE,
            )
        except MarketplaceQualifiedToolStageContextNotFoundError:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE,
                reason_detail="stage context missing",
            )
        except MarketplaceQualifiedToolStageContextResolverUnavailableError as exc:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE,
                reason_detail=str(exc),
            )
        except (
            MarketplaceQualifiedToolStageContextResolverConflictError,
            MarketplaceQualifiedToolStageContextResolverIntegrityError,
        ) as exc:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail=str(exc),
            )

        if ctx.tenant_id != request.tenant_id:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail="tenant_id mismatch",
            )

        try:
            stage = self._stage_repository.get(
                tenant_id=ctx.tenant_id,
                handoff_id=ctx.handoff_id,
            )
        except MarketplaceQualifiedToolStageUnavailableError as exc:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE,
                reason_detail=str(exc),
            )
        except MarketplaceQualifiedToolStageIntegrityError as exc:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail=str(exc),
            )

        if stage is None:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE,
                reason_detail="staged tool release missing",
            )

        selection = self._operation_selector.select(
            required_operations=request.need.required_operations,
            stage=stage,
            qualified_subject_reference=subject.qualified_subject_reference,
            handoff_id=ctx.handoff_id,
        )
        if (
            selection.outcome
            is not QualifiedMarketplaceToolOperationSelectionOutcome.SELECTED
        ):
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.INVALID_OPERATION,
                reason_detail=selection.reason_detail or "invalid operation",
            )

        intent = QualifiedMarketplaceToolExecutionIntent(
            execution_request_id=request.execution_request_id,
            binding_operation_id=request.binding_operation_id,
            resume_operation_id=request.resume_operation_id,
            tenant_id=request.tenant_id,
            task_id=str(request.task_id),
            worker_need_id=request.worker_need_id,
            qualified_subject_reference=subject.qualified_subject_reference,
            handoff_id=ctx.handoff_id,
            selected_operation=selection.selected_operation,
        )

        try:
            write_result = self._intent_repository.record(intent)
        except QualifiedMarketplaceToolExecutionIntentUnavailableError as exc:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE,
                reason_detail=str(exc),
            )
        except QualifiedMarketplaceToolExecutionIntentConflictError as exc:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.CONFLICT,
                reason_detail=str(exc),
            )
        except QualifiedMarketplaceToolExecutionIntentIntegrityError as exc:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
                reason_detail=str(exc),
            )

        if write_result.outcome is QualifiedMarketplaceToolExecutionIntentWriteOutcome.CREATED:
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.CREATED,
            )
        return QualifiedCapabilityExecutionIntentPreparationResult(
            outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.ALREADY_RECORDED_IDENTICAL,
        )


__all__ = ["MarketplaceQualifiedToolExecutionIntentPreparation"]
