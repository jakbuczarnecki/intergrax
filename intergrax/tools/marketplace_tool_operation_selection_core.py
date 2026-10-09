# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared 0/1/N required-operation selection for Marketplace Tool execution paths."""

from __future__ import annotations

from intergrax.contracts.tools.qualified_marketplace_tool_operation_selection import (
    QualifiedMarketplaceToolOperationSelectionOutcome,
    QualifiedMarketplaceToolOperationSelectionPolicy,
    QualifiedMarketplaceToolOperationSelectionResult,
)


def select_marketplace_tool_operation(
    *,
    required_operations: tuple[str, ...],
    multi_operation_policy: QualifiedMarketplaceToolOperationSelectionPolicy | None = None,
    policy_select: (
        QualifiedMarketplaceToolOperationSelectionPolicy | None
    ) = None,
) -> QualifiedMarketplaceToolOperationSelectionResult:
    """Source-neutral operation cardinality gate — UCA policy hook optional."""
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
    policy = multi_operation_policy or policy_select
    if policy is None:
        return QualifiedMarketplaceToolOperationSelectionResult(
            outcome=QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION,
            reason_detail="multiple operations require explicit selection policy",
        )
    return QualifiedMarketplaceToolOperationSelectionResult(
        outcome=QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION,
        reason_detail="multi-operation policy requires UCA stage context adapter",
    )


__all__ = ["select_marketplace_tool_operation"]
