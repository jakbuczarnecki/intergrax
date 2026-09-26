# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production resolver for Marketplace qualified Tool stage context (S24-GAP-02-P2)."""

from __future__ import annotations

from typing import Final

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.tools.marketplace_handoff_reference import (
    MarketplaceHandoffReferenceError,
    derive_marketplace_gap_tool_handoff_id,
    parse_marketplace_domain_handoff_reference,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
    MarketplaceQualifiedToolStageContextAssociationRepository,
    MarketplaceQualifiedToolStageContextAssociationUnavailableError,
    MarketplaceQualifiedToolStageContextNotFoundError,
    MarketplaceQualifiedToolStageContextResolverConflictError,
    MarketplaceQualifiedToolStageContextResolverIntegrityError,
    MarketplaceQualifiedToolStageContextResolverNotSupportedError,
    MarketplaceQualifiedToolStageContextResolverUnavailableError,
)

_MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID: Final = "marketplace.gap_acquisition.v1"


class MarketplaceQualifiedToolStageContextResolverImpl:
    """Resolve tenant context from domain handoff reference + acquisition facts."""

    def __init__(
        self,
        association_repository: MarketplaceQualifiedToolStageContextAssociationRepository,
    ) -> None:
        self._association_repository = association_repository

    def resolve_for_qualification(
        self,
        *,
        acquisition_request_id: str,
        domain_handoff_reference: str,
        strategy_id: str,
    ) -> MarketplaceQualifiedToolStageContext:
        if strategy_id != _MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID:
            raise MarketplaceQualifiedToolStageContextResolverNotSupportedError(
                "marketplace tool context resolver supports marketplace gap acquisition only",
            )
        normalized_acquisition = require_non_empty_text(
            acquisition_request_id,
            label="acquisition_request_id",
        )
        try:
            handoff_id = parse_marketplace_domain_handoff_reference(
                domain_handoff_reference,
            )
        except (MarketplaceHandoffReferenceError, TypeError, ValueError) as exc:
            raise MarketplaceQualifiedToolStageContextResolverIntegrityError(
                "malformed domain handoff reference",
            ) from exc

        try:
            association = self._association_repository.get_by_handoff_id(handoff_id)
        except MarketplaceQualifiedToolStageContextAssociationUnavailableError as exc:
            raise MarketplaceQualifiedToolStageContextResolverUnavailableError(
                str(exc),
            ) from exc

        if association is None:
            raise MarketplaceQualifiedToolStageContextNotFoundError(
                "no marketplace qualified tool stage context association for handoff",
            )

        if association.acquisition_request_id != normalized_acquisition:
            raise MarketplaceQualifiedToolStageContextResolverConflictError(
                "acquisition_request_id does not match stored association",
            )

        if association.handoff_id != handoff_id:
            raise MarketplaceQualifiedToolStageContextResolverIntegrityError(
                "stored association handoff_id does not match parsed reference",
            )

        expected_handoff = derive_marketplace_gap_tool_handoff_id(
            tenant_id=association.tenant_id,
            operation_id=normalized_acquisition,
        )
        if expected_handoff != handoff_id:
            raise MarketplaceQualifiedToolStageContextResolverIntegrityError(
                "handoff_id does not match tenant-bound v2 derivation",
            )

        return association


__all__ = [
    "MarketplaceQualifiedToolStageContextResolverImpl",
]
