# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default Marketplace qualified Tool operation selector (S24-GAP-02-P3)."""

from __future__ import annotations

from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)
from intergrax.contracts.tools.qualified_marketplace_tool_operation_selection import (
    QualifiedMarketplaceToolOperationSelectionOutcome,
    QualifiedMarketplaceToolOperationSelectionPolicy,
    QualifiedMarketplaceToolOperationSelectionResult,
    QualifiedMarketplaceToolOperationSelector,
)


class DefaultQualifiedMarketplaceToolOperationSelector(
    QualifiedMarketplaceToolOperationSelector,
):
    """Fail closed on 0 or >1 operations unless explicit policy is injected."""

    def __init__(
        self,
        *,
        multi_operation_policy: QualifiedMarketplaceToolOperationSelectionPolicy
        | None = None,
    ) -> None:
        self._multi_operation_policy = multi_operation_policy

    def select(
        self,
        *,
        required_operations: tuple[str, ...],
        stage: MarketplaceQualifiedToolStage,
        qualified_subject_reference: str,
        handoff_id: str,
    ) -> QualifiedMarketplaceToolOperationSelectionResult:
        count = len(required_operations)
        if count == 0:
            return QualifiedMarketplaceToolOperationSelectionResult(
                outcome=QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION,
                reason_detail="no required operations",
            )
        if count == 1:
            return QualifiedMarketplaceToolOperationSelectionResult(
                outcome=QualifiedMarketplaceToolOperationSelectionOutcome.SELECTED,
                selected_operation=required_operations[0],
            )
        if self._multi_operation_policy is None:
            return QualifiedMarketplaceToolOperationSelectionResult(
                outcome=QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION,
                reason_detail="multiple operations require explicit selection policy",
            )
        policy_result = self._multi_operation_policy.select(
            required_operations=required_operations,
            stage=stage,
            qualified_subject_reference=qualified_subject_reference,
            handoff_id=handoff_id,
        )
        if (
            policy_result.outcome
            is not QualifiedMarketplaceToolOperationSelectionOutcome.SELECTED
        ):
            return policy_result
        selected = policy_result.selected_operation
        if selected not in required_operations:
            return QualifiedMarketplaceToolOperationSelectionResult(
                outcome=QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION,
                reason_detail="policy selected operation not in required_operations",
            )
        return policy_result


__all__ = [
    "DefaultQualifiedMarketplaceToolOperationSelector",
]
