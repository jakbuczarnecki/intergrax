# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualified Marketplace Tool operation selection SPI (S24-GAP-02-P3).

ToolContract denotes a single atomic runtime callable. ``selected_operation`` is the
semantic capability-operation intent (preparation, material, policy, audit, idempotency).
ToolRuntime dispatch uses ``activated_tool_id`` as the atomic callable identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)


class QualifiedMarketplaceToolOperationSelectionOutcome(StrEnum):
    SELECTED = "selected"
    INVALID_OPERATION = "invalid_operation"


@dataclass(frozen=True, slots=True)
class QualifiedMarketplaceToolOperationSelectionResult:
    outcome: QualifiedMarketplaceToolOperationSelectionOutcome
    selected_operation: str = ""
    reason_detail: str = ""


@runtime_checkable
class QualifiedMarketplaceToolOperationSelectionPolicy(Protocol):
    """Optional host policy when need declares multiple required operations."""

    def select(
        self,
        *,
        required_operations: tuple[str, ...],
        stage: MarketplaceQualifiedToolStage,
        qualified_subject_reference: str,
        handoff_id: str,
    ) -> QualifiedMarketplaceToolOperationSelectionResult: ...


@runtime_checkable
class QualifiedMarketplaceToolOperationSelector(Protocol):
    """Pluginable selector for semantic capability-operation intent."""

    def select(
        self,
        *,
        required_operations: tuple[str, ...],
        stage: MarketplaceQualifiedToolStage,
        qualified_subject_reference: str,
        handoff_id: str,
    ) -> QualifiedMarketplaceToolOperationSelectionResult: ...


__all__ = [
    "QualifiedMarketplaceToolOperationSelectionOutcome",
    "QualifiedMarketplaceToolOperationSelectionPolicy",
    "QualifiedMarketplaceToolOperationSelectionResult",
    "QualifiedMarketplaceToolOperationSelector",
]
