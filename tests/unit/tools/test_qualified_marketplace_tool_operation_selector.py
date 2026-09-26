# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffConsumerTarget
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)
from intergrax.contracts.tools.qualified_marketplace_tool_operation_selection import (
    QualifiedMarketplaceToolOperationSelectionOutcome,
    QualifiedMarketplaceToolOperationSelectionPolicy,
    QualifiedMarketplaceToolOperationSelectionResult,
)
from intergrax.tools.qualified_marketplace_tool_operation_selector import (
    DefaultQualifiedMarketplaceToolOperationSelector,
)

pytestmark = pytest.mark.unit


def _stage() -> MarketplaceQualifiedToolStage:
    source = CapabilitySourceIdentity(
        source_id="official.op",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    release = CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tool.op",
            ),
        ),
        version_label="1.0.0",
        content_digest="sha256:op",
        package_reference="pkg://op",
    )
    return MarketplaceQualifiedToolStage(
        handoff_id="handoff-1",
        tenant_id="tenant-a",
        selected_release=release,
        discovery_correlation_id="disc-1",
        selection_id="sel-1",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id="tool.qualification_staging.v1",
        recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=UTC),
    )


def test_one_operation_selected_exact() -> None:
    selector = DefaultQualifiedMarketplaceToolOperationSelector()
    result = selector.select(
        required_operations=("invoke",),
        stage=_stage(),
        qualified_subject_reference="subj",
        handoff_id="handoff-1",
    )
    assert result.outcome is QualifiedMarketplaceToolOperationSelectionOutcome.SELECTED
    assert result.selected_operation == "invoke"


def test_zero_operations_invalid() -> None:
    selector = DefaultQualifiedMarketplaceToolOperationSelector()
    result = selector.select(
        required_operations=(),
        stage=_stage(),
        qualified_subject_reference="subj",
        handoff_id="handoff-1",
    )
    assert result.outcome is QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION


def test_multi_without_policy_invalid() -> None:
    selector = DefaultQualifiedMarketplaceToolOperationSelector()
    result = selector.select(
        required_operations=("a", "b"),
        stage=_stage(),
        qualified_subject_reference="subj",
        handoff_id="handoff-1",
    )
    assert result.outcome is QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION


class _PickFirstPolicy:
    def select(
        self,
        *,
        required_operations: tuple[str, ...],
        stage: MarketplaceQualifiedToolStage,
        qualified_subject_reference: str,
        handoff_id: str,
    ) -> QualifiedMarketplaceToolOperationSelectionResult:
        return QualifiedMarketplaceToolOperationSelectionResult(
            outcome=QualifiedMarketplaceToolOperationSelectionOutcome.SELECTED,
            selected_operation=required_operations[1],
        )


def test_multi_with_policy_selects_explicit() -> None:
    selector = DefaultQualifiedMarketplaceToolOperationSelector(
        multi_operation_policy=_PickFirstPolicy(),
    )
    result = selector.select(
        required_operations=("a", "b"),
        stage=_stage(),
        qualified_subject_reference="subj",
        handoff_id="handoff-1",
    )
    assert result.selected_operation == "b"


class _OutsidePolicy:
    def select(
        self,
        *,
        required_operations: tuple[str, ...],
        stage: MarketplaceQualifiedToolStage,
        qualified_subject_reference: str,
        handoff_id: str,
    ) -> QualifiedMarketplaceToolOperationSelectionResult:
        return QualifiedMarketplaceToolOperationSelectionResult(
            outcome=QualifiedMarketplaceToolOperationSelectionOutcome.SELECTED,
            selected_operation="outside",
        )


def test_policy_operation_not_requested_invalid() -> None:
    selector = DefaultQualifiedMarketplaceToolOperationSelector(
        multi_operation_policy=_OutsidePolicy(),
    )
    result = selector.select(
        required_operations=("a", "b"),
        stage=_stage(),
        qualified_subject_reference="subj",
        handoff_id="handoff-1",
    )
    assert result.outcome is QualifiedMarketplaceToolOperationSelectionOutcome.INVALID_OPERATION
