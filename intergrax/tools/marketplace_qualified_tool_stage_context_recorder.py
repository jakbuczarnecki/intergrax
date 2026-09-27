# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool-owned recorder for marketplace qualified stage context (S24-GAP-02-P2-R1)."""

from __future__ import annotations

from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
    MarketplaceQualifiedToolStageContextAssociationRepository,
    MarketplaceQualifiedToolStageContextAssociationWriteResult,
)


class MarketplaceQualifiedToolStageContextRecorderImpl:
    """Delegates authoritative handoff context to the association repository."""

    def __init__(
        self,
        repository: MarketplaceQualifiedToolStageContextAssociationRepository,
    ) -> None:
        self._repository = repository

    def record_tool_handoff_context(
        self,
        *,
        handoff_id: str,
        tenant_id: str,
        acquisition_request_id: str,
    ) -> MarketplaceQualifiedToolStageContextAssociationWriteResult:
        return self._repository.record(
            MarketplaceQualifiedToolStageContext(
                handoff_id=handoff_id,
                tenant_id=tenant_id,
                acquisition_request_id=acquisition_request_id,
            ),
        )


__all__ = ["MarketplaceQualifiedToolStageContextRecorderImpl"]
