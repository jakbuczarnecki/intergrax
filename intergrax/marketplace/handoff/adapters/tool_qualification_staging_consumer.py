# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace handoff consumer that stages exact Tool releases for qualification (S24-GAP-02-P1/P2)."""

from __future__ import annotations

from typing import Final

from pydantic import ValidationError

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerError,
    CapabilityHandoffConsumerFailureDisposition,
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffEnvelope,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
    MarketplaceQualifiedToolStageConflictError,
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageRepository,
    MarketplaceQualifiedToolStageUnavailableError,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContextAssociationRepository,
)

TOOL_QUALIFICATION_STAGING_CONSUMER_ID: Final = "tool.qualification_staging.v1"


def _stage_from_envelope(
    envelope: CapabilityHandoffEnvelope,
) -> MarketplaceQualifiedToolStage:
    tenant_id = envelope.tenant_id
    if tenant_id is None:
        raise ValueError("tenant_id is required for tool qualification staging")
    normalized_tenant = require_non_empty_text(tenant_id, label="tenant_id")
    return MarketplaceQualifiedToolStage(
        handoff_id=envelope.handoff_id,
        tenant_id=normalized_tenant,
        selected_release=envelope.selected_release,
        discovery_correlation_id=envelope.discovery_correlation_id,
        selection_id=envelope.selection_id,
        consumer_target=envelope.consumer_target,
        downstream_consumer_id=envelope.downstream_consumer_id,
        recorded_at=envelope.recorded_at,
    )


class ToolQualificationStagingConsumer:
    """Projects marketplace handoff envelopes into Tool-owned durable staging."""

    def __init__(
        self,
        repository: MarketplaceQualifiedToolStageRepository,
        association_repository: MarketplaceQualifiedToolStageContextAssociationRepository,
    ) -> None:
        self._repository = repository
        self._association_repository = association_repository

    @property
    def consumer_id(self) -> str:
        return TOOL_QUALIFICATION_STAGING_CONSUMER_ID

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        if type(envelope) is not CapabilityHandoffEnvelope:
            raise CapabilityHandoffConsumerError(
                "handoff envelope must be CapabilityHandoffEnvelope",
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            )
        if envelope.consumer_target is not CapabilityHandoffConsumerTarget.TOOL_DOMAIN:
            raise CapabilityHandoffConsumerError(
                "tool qualification staging requires TOOL_DOMAIN consumer target",
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            )
        if envelope.tenant_id is None:
            raise CapabilityHandoffConsumerError(
                "tenant_id is required for tool qualification staging",
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            )
        try:
            normalized_tenant = require_non_empty_text(envelope.tenant_id, label="tenant_id")
        except (TypeError, ValueError) as exc:
            raise CapabilityHandoffConsumerError(
                str(exc),
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            ) from exc
        if envelope.selected_release.discovery.kind is not CapabilityKind.TOOL:
            raise CapabilityHandoffConsumerError(
                "tool qualification staging requires TOOL capability release",
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            )

        association = self._association_repository.get_by_handoff_id(envelope.handoff_id)
        if association is None:
            raise CapabilityHandoffConsumerError(
                "tool qualification staging requires pre-recorded handoff context",
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            )
        if association.tenant_id != normalized_tenant:
            raise CapabilityHandoffConsumerError(
                "handoff context tenant does not match envelope tenant",
                disposition=CapabilityHandoffConsumerFailureDisposition.FAILED,
            )

        try:
            stage = _stage_from_envelope(envelope)
        except (TypeError, ValueError, ValidationError) as exc:
            raise CapabilityHandoffConsumerError(
                str(exc),
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            ) from exc

        try:
            self._repository.stage(stage)
        except MarketplaceQualifiedToolStageConflictError as exc:
            raise CapabilityHandoffConsumerError(
                str(exc),
                disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            ) from exc
        except MarketplaceQualifiedToolStageUnavailableError as exc:
            raise CapabilityHandoffConsumerError(
                str(exc),
                disposition=CapabilityHandoffConsumerFailureDisposition.UNAVAILABLE,
            ) from exc
        except MarketplaceQualifiedToolStageIntegrityError as exc:
            raise CapabilityHandoffConsumerError(
                str(exc),
                disposition=CapabilityHandoffConsumerFailureDisposition.FAILED,
            ) from exc


__all__ = [
    "TOOL_QUALIFICATION_STAGING_CONSUMER_ID",
    "ToolQualificationStagingConsumer",
]
